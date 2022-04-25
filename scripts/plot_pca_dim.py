import matplotlib.pyplot as plt
import sys
import torch
from torch.nn import functional as F
from sklearn.decomposition import PCA
import itertools
import numpy as np
import os
import re
import json

from grok.training_custom_transformer_slim import TrainableTransformer
from grok.data import NUMS, render, ArithmeticTokenizer


def get_pca_dim(checkpoint: str, embedding: bool = False):
    np.random.seed(1)
    with torch.no_grad():
        print(checkpoint)
        if "init" in checkpoint:
            epoch = 0
        else:
            (epoch,) = re.findall(r"epoch_(\d+)_", checkpoint)
            epoch = int(epoch)
        transformer = TrainableTransformer(checkpoint=checkpoint)
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
        
        pca = PCA().fit(emb)
        return np.argmax(np.cumsum(pca.explained_variance_ratio_) > .8), acc, loss

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
    # read hparams.json
    hparams = json.load(open(os.path.join(path, "hparams.json")))
       
    embedding = "emb" in sys.argv[2:]
    ckpts = get_checkpoints_in_path(path)
    accs = []
    losss = []
    expl_vars = []
    for ckpt in ckpts:
        expl_var,acc,loss = get_pca_dim(
            ckpt, embedding=embedding
        )
        accs.append(acc)
        expl_vars.append(expl_var)
        losss.append(loss)

    plt.scatter(accs, expl_vars, c=np.log(losss), alpha=.5)
    plt.xlabel("accuracy")
    plt.ylabel("n_components to explain 80% variance")
    plt.colorbar()
    plt.title(f"weight decay {hparams['weight_decay']}")
    plt.show()
