import matplotlib.pyplot as plt
import sys
import torch
from torch.nn import functional as F
from sklearn.manifold import TSNE
import itertools
import numpy as np
import os
import imageio
import re

from grok.training_custom_transformer_slim import TrainableTransformer
from grok.data import NUMS, render, ArithmeticTokenizer


def plot_tsne(checkpoint: str, save: str = None, embedding: bool = False):
    np.random.seed(1)
    with torch.no_grad():
        print(checkpoint)
        if "init" in checkpoint:
            epoch = 0
        else:
            (epoch,) = re.findall(r"epoch_(\d+)_", checkpoint)
            epoch = int(epoch)
        transformer = TrainableTransformer(checkpoint=checkpoint)
        print(transformer.hparams)
        val_data = next(transformer.val_dataloader())
        target = val_data["target"]
        y_hat = transformer(val_data["text"])
        loss = F.cross_entropy(y_hat, target).item()
        acc = transformer.accuracy(y_hat, target).mean().item()
        print(f"Loss: {loss}, Acc: {acc}")

        transformer.hparams.batchsize = -1  # full batch
        tokenizer = ArithmeticTokenizer()
        if "s5" in transformer.hparams.math_operator:
            x = [render(n) for n in itertools.permutations(range(5))]  # s5
        else:
            x = [render(n) for n in NUMS]
        x = tokenizer.encode(x).to(transformer.device)
        if embedding:
            # embeddings
            emb = transformer.transformer.embedding(x).squeeze(1).cpu().numpy()
        else:
            # last layer weights
            emb = transformer.transformer.linear.weight[x.flatten()].cpu().numpy()
            emb = emb / np.linalg.norm(emb, ord=1, axis=1, keepdims=True)
        tsned = TSNE(
            n_components=2,
            verbose=True,
            learning_rate="auto",
            perplexity=5,
            n_iter=10000,
            n_iter_without_progress=1000,
            init="pca",
        ).fit_transform(emb)
        plt.scatter(tsned[:, 0], tsned[:, 1], c=x.cpu().numpy())
        for i, txt in enumerate(map(str, NUMS)):
            plt.annotate(txt, tsned[i])
        plt.title(f"EPOCH: {epoch}, Loss: {loss:.3e}, Acc: {acc:.2f}")
        plt.colorbar()
        if save is None:
            plt.show()
        else:
            plt.savefig(save)
        plt.close()

def savepath_from_ckpt(checkpoint: str, embedding: bool):
    return (
        checkpoint.replace("ckpt", "")
        + f"tsne_{'emb' if embedding else 'penultimate'}.png"
    )


def get_checkpoints_in_path(basepath: str):
    ckpts = {0: os.path.join(basepath, "init.ckpt")}
    for f in os.listdir(basepath):
        if f.endswith("ckpt") and f.startswith("epoch"):
            epoch = int(f.split("_")[1])
            ckpts[epoch] = os.path.join(basepath, f)
    return tuple(dict(sorted(ckpts.items())).values())


if __name__ == "__main__":
    path = sys.argv[1]
    embedding = "emb" in sys.argv[2:]
    if path.endswith("ckpt"):
        plot_tsne(path, save=savepath_from_ckpt(path, embedding), embedding=embedding)
    else:
        ckpts = get_checkpoints_in_path(path)
        for ckpt in ckpts:
            plot_tsne(
                ckpt, save=savepath_from_ckpt(ckpt, embedding), embedding=embedding
            )

        images = []
        for ckpt in ckpts:
            images += [imageio.imread(savepath_from_ckpt(ckpt, embedding),)]
        imageio.mimsave(
            f"{path}/tsne_{'emb' if embedding else 'penultimate'}.gif",
            images,
            duration=1,
        )

