#!/usr/bin/env python

import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Tuple
import time

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

from grok.data import (
    DEFAULT_DATA_DIR,
    ArithmeticDataset,
    ArithmeticIterator,
)
from grok.transformer import Transformer

DEFAULT_LOG_DIR = "logs"


class TrainableTransformer(LightningModule):
    """
    Adds training methods to train a generic transformer on arithmetic equations
    """

    def __init__(self, **hparams: Dict) -> None:
        """
        :param hparams: An argparse.Namespace with parameters defined in
                        self.add_model_specific_args().
        """
        super().__init__()

        self.save_hyperparameters(hparams)
        self.prepare_data()

        self.transformer = Transformer(
            self.hparams.n_layers,
            self.hparams.n_heads,
            self.hparams.d_model,
            self.hparams.dropout,
            self.hparams.max_context_len,
            len(self.train_dataset.tokenizer),
            self.hparams.non_linearity,
            weight_noise=self.hparams.weight_noise,
        ).to(self.device)

        self.next_epoch_to_eval = -1
        self.next_train_epoch_to_log = 0
        self.grad_norms = dict()

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
        parser.add_argument(
            "--batchsize",
            type=float,
            # default=0.25,
            default=0,
            help="-1 -> entire dataset, 0 -> auto-calculate, 0<N<1 -> fraction of dataset, N>1 -> N",
        )

        parser.add_argument("--n_layers", type=int, default=2)
        parser.add_argument("--n_heads", type=int, default=4)
        parser.add_argument("--d_model", type=int, default=128)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--weight_noise", type=float, default=0.0)
        parser.add_argument("--non_linearity", type=str, default="relu")
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
            "--beta1",
            type=float,
            default=0.9,
            help="Adam beta1, not used when choosing sgd",
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
            "--save_activations", dest="save_activations", action="store_true"
        )
        parser.set_defaults(save_activations=False)
        parser.add_argument("--save_outputs", dest="save_outputs", action="store_true")
        parser.set_defaults(save_outputs=False)

        parser.add_argument(
            "--logdir", type=str, default=DEFAULT_LOG_DIR,
        )
        parser.add_argument(
            "--datadir", type=str, default=DEFAULT_DATA_DIR,
        )

        return parser

    def prepare_data(self) -> None:
        """
        Used by pytorch_lighting

        Loads training data to self.train_dataset
        Loads validation data to self.val_dataset
        """
        (self.train_dataset, self.val_dataset,) = ArithmeticDataset.splits(
            train_pct=self.hparams.train_data_pct,  # type: ignore
            operator=self.hparams.math_operator,  # type: ignore
            operand_length=self.hparams.operand_length,  # type: ignore
            data_dir=self.hparams.datadir,  # type: ignore
            device=self.device,
        )

    def train_dataloader(self) -> ArithmeticIterator:  # type: ignore
        """
        Used by pytorch_lighting

        :returns: an iterator for self.train_dataset
        """
        iterator = ArithmeticIterator(
            self.train_dataset, batchsize_hint=self.hparams.batchsize,  # type: ignore
        )
        self.train_batchsize = iterator.batchsize
        self.batches_per_epoch = len(iterator)

        return iterator

    def val_dataloader(self) -> ArithmeticIterator:  # type: ignore
        """
        Used by pytorch_lighting

        :returns: an iterator for self.train_dataset
        """
        iterator = ArithmeticIterator(
            self.val_dataset, batchsize_hint=-1,  # no need to batch validation data
        )
        return iterator

    def test_dataloader(self) -> ArithmeticIterator:  # type: ignore
        """
        Used by pytorch_lighting

        :returns: an iterator for self.train_dataset
        """
        iterator = ArithmeticIterator(
            self.val_dataset, device, batchsize_hint=-1  # type: ignore
        )
        return iterator

    def _scheduler_lr(self, step: int) -> float:
        """
        Used by pytorch_lighting

        :returns: the learning_rate for this training step
        """
        max_lr = self.hparams.lr_multiplier  # type: ignore
        return max_lr
        min_lr = self.hparams.lr_multiplier / 10  # type: ignore
        warmup_steps = self.hparams.warmup_steps  # type: ignore
        if not self.hparams.anneal_lr:
            if step <= warmup_steps:
                lr = (float(step) / max(warmup_steps, 1)) * max_lr
            else:
                lr = max_lr
        else:
            if step <= warmup_steps:
                lr = (float(step) / max(warmup_steps, 1)) * max_lr
            elif step <= self.hparams.anneal_lr_steps + warmup_steps:
                effective_step = step - warmup_steps
                t = effective_step / self.hparams.anneal_lr_steps
                cos = (1 + np.cos(np.pi * t)) / 2
                lr = min_lr + (max_lr - min_lr) * cos
            else:
                lr = min_lr
        return lr

    def configure_optimizers(self) -> Tuple[List[Any], List[Dict]]:
        """
        Used by pytorch_lighting

        :returns: optimizers and schedulers.
        """
        optim_str = self.hparams.optimizer.lower()

        optim_args = dict()
        if optim_str in ["adam", "adamw"]:
            optim_fn = torch.optim.AdamW if optim_str == "adamw" else torch.optim.Adam
            optim_args["betas"] = (self.hparams.beta1, self.hparams.beta2)
            optim_args["eps"] = 1e-8
            optim_args["amsgrad"] = self.hparams.amsgrad
        elif optim_str == "sgd":
            optim_fn = torch.optim.SGD
            optim_args["momentum"] = 0.9
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
        schedulers = [
            {
                "scheduler": LambdaLR(optimizer, lr_lambda=self._scheduler_lr),
                "interval": "step",
                "frequency": 1,
            }
        ]
        return [optimizer], schedulers

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

    def on_after_backward(self) -> None:
        """ record gradients """
        for name, param in self.named_parameters():
            self.grad_norms["gradnorm_" + name] = torch.norm(param.grad, p=2).item()

        return super().on_after_backward()

    def _step(
        self,
        batch: Dict,
        batch_idx: int,
        train: bool = True,
        reduction: str = "mean",
        grads: bool = False,
    ) -> Tuple[Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor]:
        """
        Performs one forward pass on a training or validation batch

        :param batch: The batch of equations to process
        :param batch_idx: which batch this is in the epoch.
        :param train: True is this is a training batch, false otherwise
        :returns: The loss from the predicted solutions to the equation,
                  The accuracy of the predicted solutions
                  The fraction of this dataset contained in this batch
                  The portion of the input equations left of the equal sign
                  The softmax probilities for the solutions to the equations
                  A list lists of attention matrices by layer and head
                  A list lists of value matrices by layer and head
                  Margin for this batch
        """
        x = batch["text"]  # shape = batchsize * context_len
        y = batch["target"]  # shape = batchsize * context_len
        y_hat, attentions, values = self(
            x=x, save_activations=self.hparams.save_activations  # type: ignore
        )  # shape = batchsize * context_len * vocab_size
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

        return loss, acc, coeff, x_lhs, y_hat_rhs, attentions, values

    def _save_inputs(self, outputs: Dict, ds: str) -> None:
        """
        Saves the input equations to disk for analysis later

        :param outputs: a list of tuples from self.training_step()
        :param ds: a string ('train' or 'val') naming which dataset
                   these inputs are from.
        :param train: True is this is a training batch, false otherwise
        """
        logdir = self.hparams.logdir + "/inputs/" + ds  # type: ignore
        os.makedirs(logdir, exist_ok=True)
        pickle_file = logdir + f"/{ds}.pt"

        x_lhs = torch.cat([x["x_lhs"] for x in outputs])
        with open(pickle_file, "wb") as fh:
            torch.save(x_lhs, fh)

    def _merge_batch_activations(
        self, partial_activations: List[List[Tensor]]
    ) -> List[List[Tensor]]:
        """
        Merges the head_attentions / head_values from all batches in
        this epoch.

        :param partial_activations: A list of
                                   (lists of lists of activations by layer and head)
        :returns: A lists of lists of activations by layer and head
        """
        # num_batches = len(partial_activations)
        num_layers = len(partial_activations[0])
        num_heads = len(partial_activations[0][0])
        activations: List = []
        for _ in range(num_layers):
            activations.append([])
            for _ in range(num_heads):
                activations[-1].append([])

        for minibatch_activations in partial_activations:
            for l, layer_activations in enumerate(minibatch_activations):
                for h, head_attn in enumerate(layer_activations):
                    # # print(f"head_attn = {head_attn}")
                    activations[l][h].append(head_attn)

        for l in range(num_layers):
            for h in range(num_heads):
                activations[l][h] = torch.cat(activations[l][h])

        return activations

    def _save_activations(self, outputs: Dict, ds: str) -> None:
        """
        Saves activations out to disk for analysis later

        :param outputs: a list of tuples from self.training_step()
        """

        output: Dict[str, Any] = {}
        if self.hparams.save_outputs:  # type: ignore
            y_hat_rhs = torch.cat([x["y_hat_rhs"] for x in outputs])
            output["y_hat_rhs"] = y_hat_rhs
        if self.hparams.save_activations:  # type: ignore
            partial_attentions = list([o["partial_attentions"] for o in outputs])
            attentions = self._merge_batch_activations(partial_attentions)
            partial_values = list([o["partial_values"] for o in outputs])
            values = self._merge_batch_activations(partial_values)
            output["attentions"] = attentions
            output["values"] = values
        if self.hparams.save_outputs or self.hparams.save_activations:  # type: ignore
            logdir = self.hparams.logdir + "/outputs/" + ds  # type: ignore
            os.makedirs(logdir, exist_ok=True)
            pickle_file = logdir + f"/epoch_{self.current_epoch:010}.pt"
            with open(pickle_file, "wb") as fh:
                torch.save(output, fh)

    def training_step(self, batch, batch_idx):
        """
        Used by pytorch_lightning
        Runs one forward training pass on one batch

        :param batch: The batch of equations to process
        :param batch_idx: which batch this is in the epoch.
        :returns: a dict with loss, accuracy, lr, probabilities of solutions,
                  attentions, and values
        """
        if batch_idx == 0:
            self.training_epoch_start_time = time.time()
            self.fwd_time_in_epoch = 0

        start = time.time()
        loss, accuracy, coeff, x_lhs, y_hat_rhs, attentions, values = self._step(
            batch=batch, batch_idx=batch_idx, train=True
        )
        self.fwd_time_in_epoch += time.time() - start

        schedulers = self.trainer.lr_schedulers[0]
        # if self.current_epoch != self.next_train_epoch_to_log:
        # return {"loss": loss}
        lr_decoder = schedulers["scheduler"].optimizer.param_groups[0]["lr"]
        lr_embedding = schedulers["scheduler"].optimizer.param_groups[1]["lr"]
        lr_linear = schedulers["scheduler"].optimizer.param_groups[2]["lr"]
        print(loss)
        output = {
            "loss": loss,
            "partial_train_loss": (coeff * loss).detach(),
            "partial_train_accuracy": (coeff * accuracy).detach(),
            "lr_decoder": torch.tensor([lr_decoder]).detach(),
            "lr_embedding": torch.tensor([lr_embedding]).detach(),
            "lr_linear": torch.tensor([lr_linear]).detach(),
            "y_hat_rhs": y_hat_rhs.detach(),
            "partial_attentions": attentions,
            "partial_values": values,
        }
        if self.current_epoch == 0:
            output["x_lhs"] = x_lhs

        return output

    def training_epoch_end(self, outputs):
        """
        Used by pytorch_lightning
        Accumulates results of all forward training passes in this epoch

        :param outputs: a list of dicts from self.training_step()
        :param batch_idx: which batch this is in the epoch.
        :returns: a dict with loss, accuracy, lr, probabilities of solutions,
                  attentions, and values
        """
        # epoch_is_to_be_logged = self.current_epoch == self.next_train_epoch_to_log
        # if epoch_is_to_be_logged:
        #     self.next_train_epoch_to_log = max(
        #         int(1.01 * self.next_train_epoch_to_log),
        #         self.next_train_epoch_to_log + 1,
        #     )
        with torch.no_grad():
            loss = torch.stack([x["partial_train_loss"] for x in outputs]).sum()
            accuracy = torch.stack([x["partial_train_accuracy"] for x in outputs]).sum()

        if self.hparams.save_activations or self.hparams.save_outputs:
            if self.current_epoch == 0:
                self._save_inputs(outputs, ds="train")
            self._save_activations(outputs, ds="train")

        logs = {
            "train_loss": loss,
            "train_accuracy": accuracy,
            "lr_decoder": outputs[0]["lr_decoder"],
            "lr_embedding": outputs[0]["lr_embedding"],
            "lr_linear": outputs[0]["lr_linear"],
            "batches_per_epoch": self.batches_per_epoch,
            "time_per_epoch": time.time() - self.training_epoch_start_time,
            "fwd_time_in_epoch": self.fwd_time_in_epoch,
        }
        self.log_dict(logs)
        self.log_dict(self.grad_norms)

    def validation_step(self, batch, batch_idx):
        """
        Used by pytorch_lightning
        Runs one forward validation pass on one batch

        :param batch: The batch of equations to process
        :param batch_idx: which batch this is in the epoch.
        :returns: a dict with val_loss, val_accuracy, probabilities of solutions,
                  attentions, and values
        """
        # if self.next_epoch_to_eval < self.current_epoch:
        #     self.next_epoch_to_eval = self.current_epoch
        # if self.current_epoch != self.next_epoch_to_eval:
        #     return {}
        with torch.no_grad():
            loss, accuracy, coeff, x_lhs, y_hat_rhs, attentions, values = self._step(
                batch=batch, batch_idx=batch_idx, train=False
            )
        output = {
            "partial_val_loss": coeff * loss,
            "partial_val_accuracy": coeff * accuracy,
            "y_hat_rhs": y_hat_rhs,
            "partial_attentions": attentions,
            "partial_values": values,
        }
        if self.current_epoch == 0:
            output["x_lhs"] = x_lhs
        return output

    def validation_epoch_end(self, outputs):
        """
        Used by pytorch_lightning
        Accumulates results of all forward validation passes in this epoch

        :param outputs: a list of dicts from self.validation_step()
        :param batch_idx: which batch this is in the epoch.
        :returns: a dict with val_loss, val_accuracy
        """

        # validation_is_real = len(outputs[0]) != 0 # raises an error if validation_step returns {}
        validation_is_real = len(outputs) != 0
        if validation_is_real:
            validation_is_real = len(outputs[0]) != 0

        # if validation_is_real:
        # self.next_epoch_to_eval = max(
        #     int(1.02 * self.next_epoch_to_eval), self.next_epoch_to_eval + 1
        # )
        if validation_is_real:
            loss = torch.stack([x["partial_val_loss"] for x in outputs]).sum()
            accuracy = torch.stack([x["partial_val_accuracy"] for x in outputs]).sum()

            if self.hparams.save_activations or self.hparams.save_outputs:
                if self.current_epoch == 0:
                    self._save_inputs(outputs, ds="val")
                self._save_activations(outputs, ds="val")

            logs = {
                "val_loss": loss,
                "val_accuracy": accuracy,
            }
            print(f"\nStep = {self.global_step}")
            print(f"val_loss = {loss}")
            print(f"val_accuracy = {accuracy}")
            for name, param in self.named_parameters():
                # n parameters
                n_params = param.numel()
                # get the l2 norm of the parameter
                logs["paramnorm_" + name] = torch.norm(
                    param, 2
                ).detach().cpu().numpy() / np.sqrt(n_params)

            # train accuracy
            device = self.transformer.embedding.weight.device
            train_data = self.train_dataset.data.to(device)
            training_data = {"text": train_data[:, :-1], "target": train_data[:, 1:]}
            with torch.no_grad():
                tr_loss, tr_acc, *_ = self._step(training_data, 0)
                logs["full_train_loss"] = tr_loss
                logs["full_train_acc"] = tr_acc

            self.log_dict(logs)
            # save a checkpoint if the epoch is a power of 2
            if (
                self.current_epoch > 0
                and int(2 ** (int(np.log(self.current_epoch) / np.log(2))))
                == self.current_epoch
            ):
                self.trainer.save_checkpoint(
                    os.path.join(
                        self.hparams.checkpoint_path,
                        "epoch_" + str(self.current_epoch) + ".ckpt",
                    )
                )

    def test_step(self, batch, batch_idx):
        """
        Used by pytorch_lightning
        Runs one forward validation pass on one batch

        :param batch: The batch of equations to process
        :param batch_idx: which batch this is in the epoch.
        :returns: a dict with val_loss, val_accuracy, probabilities of solutions,
                  attentions, and values
        """

        loss, accuracy, coeff, x_lhs, y_hat_rhs, attentions, values = self._step(
            batch=batch, batch_idx=batch_idx, train=False, reduction="none"
        )
        output = {
            "partial_test_loss": coeff * loss,
            "partial_test_accuracy": coeff * accuracy,
            "y_hat_rhs": y_hat_rhs,
            "partial_attentions": attentions,
            "partial_values": values,
        }
        if self.current_epoch == 0:
            output["x_lhs"] = x_lhs

        return output

    def test_epoch_end(self, outputs):
        """
        Used by pytorch_lightning
        Accumulates results of all forward validation passes in this epoch

        :param outputs: a list of dicts from self.validation_step()
        :param batch_idx: which batch this is in the epoch.
        :returns: a dict with val_loss, val_accuracy
        """
        loss = torch.cat([x["partial_test_loss"] for x in outputs], dim=0)  # .sum()
        # loss = list([x["partial_test_loss"] for x in outputs])  # .sum()
        accuracy = torch.cat([x["partial_test_accuracy"] for x in outputs], dim=0)

        logs = {
            "test_loss": loss,
            "test_accuracy": accuracy,
        }

        return {"test_loss": loss, "log": logs}

    def forward(self, *args, **kwargs) -> Any:
        """Passes all arguments directly to Tranformer.forward()"""
        return self.transformer(*args, **kwargs)


def train(hparams: Namespace) -> None:
    """
    This is the main trainer_method. This sets up and runs experiment with
    the defined hyperparameters

    :param hparams: An argparse.Namespace with all of the relevant hyperparameters
    """

    # Process the args
    if hparams.logdir is None:
        hparams.logdir = os.environ.get("LOGDIR", ".")
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

    checkpoint_path = hparams.logdir + "/checkpoints"
    os.makedirs(checkpoint_path, exist_ok=True)
    hparams.checkpoint_path = checkpoint_path

    # Create the model
    device = torch.device(f"cuda:{hparams.gpu}" if torch.cuda.is_available() else "cpu")
    model = TrainableTransformer(**vars(hparams)).float().to(device)

    torch.save(model, os.path.join(checkpoint_path, "init.pt"))

    if not hparams.no_log:
        logger = TensorBoardLogger(hparams.logdir)
    else:
        logger = None

    # checkpointer = ModelCheckpoint(
    #     filepath=checkpoint_path,
    #     monitor="save_ckpt",
    #     mode="max",
    #     save_top_k=len(hparams.ckpt_epochs),
    #     verbose=False,
    # )

    trainer_args = {
        "max_steps": hparams.max_steps,
        "min_steps": hparams.max_steps,
        "max_epochs": hparams.max_epochs,
        # "val_check_interval": 1.0,
        "profiler": False,
        # "checkpoint_callback": checkpointer,
        "logger": logger,
        "log_every_n_steps": 1,
        # "flush_logs_every_n_steps": 1000, # deprecated in lightning 1.5, will be removed in 1.7
    }
    if torch.cuda.is_available() and hparams.gpu >= 0:
        trainer_args["gpus"] = [hparams.gpu]

    trainer = Trainer(**trainer_args)

    trainer.fit(model=model)  # type: ignore
    return hparams.logdir


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
    parser.add_argument("--max_steps", type=int, default=1000000)
    parser.add_argument("--no_log", action="store_true")
    parser.set_defaults(no_log=False)
    # parser.add_argument("--checkpoint_period", type=int, default=1)
    parser = TrainableTransformer.add_model_specific_args(parser)
    return parser
