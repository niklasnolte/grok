import matplotlib.pyplot as plt
import sys
import torch
from sklearn.manifold import TSNE
import itertools
import numpy as np

from grok.training_custom_transformer import TrainableTransformer
from grok.data import NUMS, render, ArithmeticTokenizer


def main():
    np.random.seed(1)
    with torch.no_grad():
        transformer = TrainableTransformer(checkpoint=sys.argv[1])
        print(transformer.hparams)
        val_data = next(transformer.val_dataloader())
        out = transformer.validation_step(val_data)
        print(f"Loss: {out['partial_val_loss'].item()}, Acc: {out['partial_val_accuracy'].item()}")
        
        transformer.hparams.batchsize = -1  # full batch
        tokenizer = ArithmeticTokenizer()
        if "s5" in transformer.hparams.math_operator:
          x = [render(n) for n in itertools.permutations(range(5))]  # s5
        else:
          x = [render(n) for n in NUMS]
        x = tokenizer.encode(x).to(transformer.device)
        if "emb" in sys.argv:
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
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    main()

