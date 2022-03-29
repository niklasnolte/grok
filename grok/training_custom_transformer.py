import os
from argparse import ArgumentParser, Namespace
from typing import Dict, Tuple
from copy import deepcopy
import json

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary


from grok.data import (
    DEFAULT_DATA_DIR,
    ArithmeticDataset,
    ArithmeticIterator,
)
from grok.transformer import Transformer

LOG_DIR = "logs"


def generate_causal_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)

class TrainableTransformer:
    def __init__(self, hparams=None, checkpoint=None) -> None:
        if checkpoint is not None:
            assert hparams is None
            cp_path = os.path.dirname(checkpoint)
            with open(os.path.join(cp_path, "hparams.json"), "r") as f:
                self.hparams = Namespace(**json.load(f))
                self.hparams.no_log = True
        else:
            assert hparams is not None
            self.hparams = hparams
        self.device = torch.device(f"cuda:{self.hparams.gpu}")

        self.train_dataset, self.val_dataset = ArithmeticDataset.splits(
            train_pct=self.hparams.train_data_pct,  # type: ignore
            operator=self.hparams.math_operator,  # type: ignore
            operand_length=self.hparams.operand_length,  # type: ignore
            data_dir=self.hparams.datadir,  # type: ignore
            device=self.device,
        )

        self.transformer = (
            Transformer(
                self.hparams.n_layers,
                self.hparams.n_heads,
                self.hparams.d_model,
                self.hparams.dropout,
                self.hparams.max_context_len,
                len(self.train_dataset.tokenizer),
            )
            .float()
            .to(self.device)
        )

        if checkpoint is not None:
            self.transformer.load_state_dict(torch.load(checkpoint))

        if not self.hparams.no_log:
            self.logdir = os.path.join(self.hparams.logdir, LOG_DIR)
            prefix = "version_"
            logs = [
                int(x.split("_")[-1]) for x in os.listdir(self.logdir) if prefix in x
            ]
            self.logdir = os.path.join(
                self.logdir, f"{prefix}{max(logs, default=-1)+1}"
            )
            print("Logging to", self.logdir)
            self.writer = SummaryWriter(log_dir=self.logdir)
            self.writer.add_hparams(vars(self.hparams), {"z": 0}, run_name=".")
            self.checkpoint_path = os.path.join(self.logdir, "checkpoints")
            os.makedirs(self.checkpoint_path, exist_ok=True)
            with open(os.path.join(self.checkpoint_path, "hparams.json"), "w") as f:
                json.dump(vars(self.hparams), f, indent=2)

        if checkpoint is None:
            self.next_epoch_to_eval = -1
            self.next_train_epoch_to_log = 0
            self.grad_norms = dict()
            self.current_epoch = 0
            self.best_val_accuracy = 0
            self.next_checkpoint_val_accuracy = 0
            # save initialization
            torch.save(
                self.transformer.state_dict(),
                os.path.join(self.checkpoint_path, "init.ckpt",),
            )

        self.optimizer, self.scheduler = self.configure_optim()

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser) -> ArgumentParser:
        """
        Defines the hyperparameter arguments needed by instances of this
        class. This is intended to be called when parsing command line
        arguments.

        :param parser: an argparse.ArgumentParser created by the caller
        :returns: the argument parser with the command line arguments added
                  for this class.
        """
        parser.add_argument("--checkpoint", type=str, default=None)

        parser.add_argument(
            "--batchsize",
            type=float,
            default=0,
            help="-1 -> entire dataset, 0 -> auto-calculate, 0<N<1 -> fraction of dataset, N>1 -> N",
        )

        parser.add_argument("--n_layers", type=int, default=2)
        parser.add_argument("--n_heads", type=int, default=4)
        parser.add_argument("--d_model", type=int, default=128)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--max_context_len", type=int, default=50)

        parser.add_argument("--math_operator", type=str, default="+")
        parser.add_argument(
            "--operand_length",
            type=int,
            help="for list operations, the length of the lists",
        )

        parser.add_argument("--train_data_pct", type=float, default=50)
        parser.add_argument("--warmup_steps", type=int, default=10)
        parser.add_argument("--anneal_lr_steps", type=int, default=100000)
        parser.add_argument("--anneal_lr", dest="anneal_lr", action="store_true")
        parser.set_defaults(anneal_lr=False)

        parser.add_argument("--optimizer", type=str, default="adamw")
        parser.add_argument("--lr_multiplier", type=float, default=1e-3)
        parser.add_argument("--decoder_lr", type=float, default=1)
        parser.add_argument("--embedding_lr", type=float, default=1)
        parser.add_argument("--linear_lr", type=float, default=1)
        parser.add_argument("--weight_decay", type=float, default=0)
        parser.add_argument("--weight_decay_kind", type=str, default="to_zero")
        parser.add_argument("--noise_factor", type=float, default=0)
        parser.add_argument(
            "--beta1", type=float, default=0.9, help="Adam beta1, momentum in sgd",
        )
        parser.add_argument(
            "--beta2",
            type=float,
            default=0.999,
            help="Adam beta2, not used when choosing sgd",
        )
        parser.add_argument(
            "--amsgrad",
            action="store_true",
            help="Adam amsgrad, not used when choosing sgd",
        )
        parser.set_defaults(amsgrad=False)

        parser.add_argument(
            "--logdir", type=str, default=".",
        )
        parser.add_argument(
            "--datadir", type=str, default=DEFAULT_DATA_DIR,
        )

        return parser

    def train_dataloader(self) -> ArithmeticIterator:  # type: ignore
        iterator = ArithmeticIterator(
            self.train_dataset, batchsize_hint=self.hparams.batchsize,  # type: ignore
        )
        self.train_batchsize = iterator.batchsize
        self.batches_per_epoch = len(iterator)

        return iterator

    def val_dataloader(self) -> ArithmeticIterator:  # type: ignore
        iterator = ArithmeticIterator(
            self.val_dataset, batchsize_hint=-1,  # no need to batch validation data
        )
        return iterator

    def configure_optim(self):
        optim_str = self.hparams.optimizer.lower()

        optim_args = dict()
        if optim_str in ["adam", "adamw"]:
            optim_fn = torch.optim.AdamW if optim_str == "adamw" else torch.optim.Adam
            optim_args["betas"] = (self.hparams.beta1, self.hparams.beta2)
            optim_args["eps"] = 1e-8
            optim_args["amsgrad"] = self.hparams.amsgrad
        elif optim_str == "sgd":
            optim_fn = torch.optim.SGD
            optim_args["momentum"] = self.hparams.beta1
        else:
            raise ValueError(f"Unknown optimizer {self.hparams.optimizer}")

        param_groups = [
            dict(
                params=self.transformer.decoder.parameters(),
                lr=self.hparams.decoder_lr,
                weight_decay=self.hparams.weight_decay,
            ),
            dict(
                params=self.transformer.embedding.parameters(),
                lr=self.hparams.embedding_lr,
            ),
            dict(
                params=self.transformer.linear.parameters(),
                lr=self.hparams.linear_lr,
                weight_decay=self.hparams.weight_decay,
            ),
        ]

        optimizer = optim_fn(param_groups, **optim_args)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: self.hparams.lr_multiplier
        )

        return optimizer, scheduler

    def _accuracy(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """
        Takes the most likely solution predicted for each equation and
        calculates the frac of equations in the batch for which these
        answers were correct

        :param y_hat: The softmax tensor output of the transformer
        :param y: A tensor of the token ids for the correct answers to each
                  equation in the batch
        :returns: the fraction of equations correctly answered
        """

        # find max prediction from output
        y_hat = torch.max(y_hat, dim=-2).indices  # batchsize x num_rhs_tokens
        row_accuracy = torch.min((y_hat == y), dim=-1).values  # shape: batchsize
        accuracy = row_accuracy.float() * 100  # shape: batchsize
        return accuracy

    def _step(
        self, batch: Dict, train: bool = True, reduction: str = "mean",
    ) -> Tuple[Tensor, Tensor, float, Tensor, Tensor]:
        """
        Performs one forward pass on a training or validation batch

        :param batch: The batch of equations to process
        :param train: True is this is a training batch, false otherwise
        :returns: The loss from the predicted solutions to the equation,
                  The accuracy of the predicted solutions
                  The fraction of this dataset contained in this batch
                  The portion of the input equations left of the equal sign
                  The softmax probilities for the solutions to the equations
                  Margin for this batch
        """
        x = batch["text"]  # shape = batchsize * context_len
        y = batch["target"]  # shape = batchsize * context_len
        y_hat = self.transformer(x=x)  # shape = batchsize * context_len * vocab_size
        y_hat = y_hat.transpose(-2, -1)  # shape = batchsize * vocab_size * context_len

        # Note: each sample must have exactly one '=' and all of them must
        # have it in the same position.
        eq_token_index = self.train_dataset.tokenizer.stoi["="]
        eq_position_t = torch.nonzero(y[0, :] == eq_token_index, as_tuple=False)
        eq_position = int(eq_position_t.squeeze())

        # only calculate loss/accuracy on right hand side of the equation
        y_rhs = y[..., eq_position + 1 :]
        y_hat_rhs = y_hat[..., eq_position + 1 :]
        x_lhs = x[..., : eq_position + 1]

        if train:
            coeff = float(batch["target"].shape[0]) / len(self.train_dataset)
        else:
            coeff = float(batch["target"].shape[0]) / len(self.val_dataset)
        loss = F.cross_entropy(y_hat_rhs, y_rhs, reduction=reduction)

        with torch.no_grad():
            acc = self._accuracy(y_hat_rhs, y_rhs)
            if reduction == "mean":
                acc = acc.mean()

        return loss, acc, coeff, x_lhs, y_hat_rhs

    def training_step(self, batch):
        loss, accuracy, coeff, _, y_hat_rhs = self._step(batch=batch, train=True)

        lr_decoder = self.optimizer.param_groups[0]["lr"]
        lr_embedding = self.optimizer.param_groups[1]["lr"]
        lr_linear = self.optimizer.param_groups[2]["lr"]
        output = {
            "loss": loss,
            "partial_train_loss": (coeff * loss).detach(),
            "partial_train_accuracy": (coeff * accuracy).detach(),
            "lr_decoder": torch.tensor([lr_decoder]).detach(),
            "lr_embedding": torch.tensor([lr_embedding]).detach(),
            "lr_linear": torch.tensor([lr_linear]).detach(),
            "y_hat_rhs": y_hat_rhs.detach(),
        }

        return output

    def training_epoch_end(self, outputs):
        with torch.no_grad():
            loss = torch.stack([x["partial_train_loss"] for x in outputs]).sum()
            accuracy = torch.stack([x["partial_train_accuracy"] for x in outputs]).sum()

        logs = {
            "train_loss": loss,
            "train_accuracy": accuracy,
            "lr_decoder": outputs[0]["lr_decoder"],
            "lr_embedding": outputs[0]["lr_embedding"],
            "lr_linear": outputs[0]["lr_linear"],
            "batches_per_epoch": self.batches_per_epoch,
        }

        # TODO maybe average over the batches?
        for name, param in self.transformer.named_parameters():
            self.grad_norms["gradnorm_" + name] = torch.norm(param.grad, p=2).item()

        self.log_dict(logs)
        self.log_dict(self.grad_norms)

    def log_dict(self, d: Dict) -> None:
        if self.hparams.no_log:
            return None
        for k, v in d.items():
            self.writer.add_scalar(k, v, self.current_epoch)

    def validation_step(self, batch):
        with torch.no_grad():
            loss, accuracy, coeff, x_lhs, y_hat_rhs = self._step(
                batch=batch, train=False
            )
        output = {
            "partial_val_loss": coeff * loss,
            "partial_val_accuracy": coeff * accuracy,
            "y_hat_rhs": y_hat_rhs,
        }
        return output

    def validation_epoch_end(self, outputs):

        if len(outputs) == 0:
            return None

        if len(outputs[0]) == 0:
            return None

        loss = torch.stack([x["partial_val_loss"] for x in outputs]).sum()
        accuracy = torch.stack([x["partial_val_accuracy"] for x in outputs]).sum()

        self.best_val_accuracy = max(accuracy, self.best_val_accuracy)

        logs = {
            "val_loss": loss,
            "val_accuracy": accuracy,
        }
        for name, param in self.transformer.named_parameters():
            # n parameters
            n_params = param.numel()
            # get the l2 norm of the parameter
            logs["paramnorm_" + name] = torch.norm(
                param, 2
            ).detach().cpu().numpy() / np.sqrt(n_params)

        # train accuracy
        train_data = self.train_dataset.data
        training_data = {"text": train_data[:, :-1], "target": train_data[:, 1:]}
        with torch.no_grad():
            tr_loss, tr_acc, *_ = self._step(training_data, 0)
            logs["full_train_loss"] = tr_loss
            logs["full_train_acc"] = tr_acc

        self.log_dict(logs)
        if self.best_val_accuracy >= self.next_checkpoint_val_accuracy:
            torch.save(
                self.transformer.state_dict(),
                os.path.join(self.checkpoint_path, f"epoch_{self.current_epoch}.ckpt",),
            )
            self.next_checkpoint_val_accuracy += 5
        return logs


def train(hparams: Namespace) -> None:
    """
    This is the main trainer_method. This sets up and runs experiment with
    the defined hyperparameters

    :param hparams: An argparse.Namespace with all of the relevant hyperparameters
    """

    # Process the args
    hparams.logdir = os.path.abspath(hparams.logdir)

    # Make sure d_model, heads, and d_key are compatible
    assert hparams.d_model % hparams.n_heads == 0, (
        "n_heads=%s does not evenly divide d_model=%s"
        % (hparams.n_heads, hparams.d_model,)
    )
    hparams.d_key = hparams.d_model / hparams.n_heads

    # Set up the RNGs for repeatability
    if hparams.random_seed != -1:
        torch.manual_seed(hparams.random_seed)
        torch.cuda.manual_seed(hparams.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create the model
    model: TrainableTransformer = TrainableTransformer(hparams)

    bar = tqdm(range(hparams.max_epochs))
    for epoch in bar:
        train_data = model.train_dataloader()
        val_data = model.val_dataloader()
        outputs = []
        for train_data_i in train_data:
            model.optimizer.zero_grad()
            output = model.training_step(train_data_i)
            output["loss"].backward()
            model.optimizer.step()
            model.scheduler.step()
            outputs.append(output)
        model.training_epoch_end(outputs)

        outputs = []
        for val_data_i in val_data:
            output = model.validation_step(val_data_i)
            outputs.append(output)
        logs = model.validation_epoch_end(outputs)
        tl = logs["full_train_loss"]
        ta = logs["full_train_acc"]
        vl = logs["val_loss"]
        va = logs["val_accuracy"]

        bar.set_description(
            f"train loss: {tl:.3e}, train acc: {ta:.3f}, val loss: {vl:.3e}, val acc: {va:.3f}"
        )
        model.current_epoch += 1
        if model.best_val_accuracy >= 100:
            break


def add_args(parser=None) -> Namespace:
    """
    Parses the command line arguments

    :returns: an argparse.Namespace with all of the needed arguments
    """
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=-1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=20000)
    parser.add_argument("--no_log", action="store_true")
    parser.set_defaults(no_log=False)
    parser = TrainableTransformer.add_model_specific_args(parser)
    return parser


if __name__ == "__main__":
    parser = add_args()
    parser.set_defaults(logdir=os.environ.get("GROK_LOGDIR", "."))
    hparams = parser.parse_args()
    hparams.datadir = os.path.abspath(hparams.datadir)
    hparams.logdir = os.path.abspath(hparams.logdir)
    print(hparams)
    train(hparams)
