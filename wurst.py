import torch
from torch import nn
from itertools import combinations_with_replacement
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


torch.manual_seed(2)

P = 50
NTOKENS = 2 * P - 1
LATENT_DIM = 2
HIDDEN_DIM = 256
NHEADS = 2
EPOCHS = 10001
BATCHSIZE = 512
SPLIT = 0.7
PLOTFREQ = 100

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def generate_causal_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)

class DecoderModel(torch.nn.Module):
    def __init__(
        self,
        make_decoder_layer,
        make_linear,
        n_decoders=2,
    ):
        super().__init__()
        self.decoder_layers = torch.nn.ModuleList(
            [make_decoder_layer() for _ in range(n_decoders)]
        )
        self.linear = make_linear()

    def forward(self, x):
        causal_mask = generate_causal_mask(x.shape[1]).to(x.device)
        for layer in self.decoder_layers:
            x = layer(x, causal_mask)
        x = self.linear(x)
        return F.softmax(x, dim=-1)


class ChadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Embedding(P, LATENT_DIM)
        self.decoder = DecoderModel(
            lambda: torch.nn.TransformerEncoderLayer(
                d_model=HIDDEN_DIM,
                nhead=NHEADS,
                dim_feedforward=HIDDEN_DIM,
                #dropout=0.1,
                activation="relu",
                batch_first=True,
            ),
            lambda: torch.nn.Linear(HIDDEN_DIM, NTOKENS),
        )
        # self.decoder = nn.Sequential(
        #     nn.Linear(LATENT_DIM, HIDDEN_DIM),
        #     nn.SELU(),
        #     nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        #     nn.SELU(),
        #     nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        #     nn.SELU(),
        #     nn.Linear(HIDDEN_DIM, NTOKENS),
        # )

    def forward(self, x):
        return self.decoder(self.encoder(x[:, 0]) + self.encoder(x[:, 1]))


# data size
print(P * (P + 1) // 2)

# try to learn one hot encoding of the input
data = torch.tensor(list(combinations_with_replacement(range(P), 2))).long()
target = F.one_hot(data.sum(axis=1)).float()


# shuffle data and target
idx = torch.randperm(len(data))
data = data[idx].to(device)
target = target[idx].to(device)

# split data into train and test
split = int(SPLIT * len(data))
data_train = DataLoader(
    TensorDataset(data[:split], target[:split]), batch_size=BATCHSIZE
)
X_test = data[split:]
Y_test = target[split:]

model = ChadModel().to(device)


def plot_shiz(savepath, EPOCH, train_loss, train_acc, test_loss, test_acc):
    # scatter plot of embeddings
    points = model.encoder.weight.detach().cpu().numpy()
    if LATENT_DIM == 1:
        # make up second dimension
        points = np.concatenate([points, np.arange(len(points)).reshape(-1, 1)], axis=1)

    plt.scatter(points[:, 0], points[:, 1])
    # annotate with numbers
    for i, p in enumerate(points):
        plt.annotate(str(i), p)

    plt.title(
        f"Epoch {EPOCH}, Train Loss: {train_loss:.2e}, "
        + f"Train Acc: {train_acc:.3f}, Test Loss: {test_loss:.2e}, "
        + f"Test Acc: {test_acc:.3f}",
        fontsize=8,
    )

    plt.savefig(savepath)
    plt.close()


enc_optim = torch.optim.AdamW(model.encoder.parameters(), lr=1e-3)
dec_optim = torch.optim.AdamW(model.decoder.parameters(), lr=1e-4, weight_decay=0)

print("Train loss|Train acc||Test loss|Test acc")
pbar = tqdm(range(EPOCHS))
for i in pbar:
    model.train()
    accs = []
    for X_train, Y_train in data_train:
        enc_optim.zero_grad()
        dec_optim.zero_grad()
        output_train = model(X_train)
        loss_train = F.mse_loss(output_train, Y_train)
        loss_train.backward()
        accs.append(
            accuracy(output_train.detach(), Y_train.argmax(1).view(-1)).cpu().numpy()
        )
        enc_optim.step()
        dec_optim.step()

    acc_train = np.mean(accs)

    model.eval()
    with torch.no_grad():
        output_test = model(X_test)
        loss_test = F.mse_loss(output_test, Y_test)
        acc_test = accuracy(output_test.detach(), Y_test.argmax(1).view(-1)).cpu()
        if i % PLOTFREQ == 0:
            plot_shiz(
                f"plotü/scatter_{i}.png",
                i,
                loss_train.cpu(),
                acc_train,
                loss_test.cpu(),
                acc_test,
            )

    pbar.set_description(
        f"{loss_train.item():.2e}|{acc_train:.4f}||{loss_test.item():.2e}|{acc_test:.4f}"
    )


# combine plots into gif
import imageio

images = []
for i in range(EPOCHS // PLOTFREQ + 1):
    name = f"plotü/scatter_{i*PLOTFREQ}.png"
    images.append(imageio.imread(name))
    os.remove(name)
imageio.mimsave("plotü/scatter.gif", images, duration=0.05)
