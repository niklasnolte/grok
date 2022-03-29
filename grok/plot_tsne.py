import matplotlib.pyplot as plt
import sys
import torch
from sklearn.manifold import TSNE
import itertools

from grok.training_custom_transformer import TrainableTransformer
from grok.data import NUMS, render, ArithmeticTokenizer


def main():
    with torch.no_grad():
        transformer = TrainableTransformer(checkpoint=sys.argv[1])
        print(transformer.hparams)
        transformer.hparams.batchsize = -1  # full batch
        tokenizer = ArithmeticTokenizer()
        if "s5" in transformer.hparams.math_operator:
          x = [render(n) for n in itertools.permutations(range(5))]  # s5
        else:
          x = [render(n) for n in NUMS]
        x = tokenizer.encode(x).to(transformer.device)
        emb = transformer.transformer.linear.weight[x.flatten()].cpu().numpy()
        # emb = transformer.transformer.embedding(x).squeeze(1).cpu().numpy()
        emb = TSNE(
            n_components=2,
            verbose=True,
            learning_rate="auto",
            perplexity=30,
            n_iter=10000,
            n_iter_without_progress=1000,
            init="pca",
        ).fit_transform(emb)
        plt.scatter(emb[:, 0], emb[:, 1], c=x.cpu().numpy())
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    main()

