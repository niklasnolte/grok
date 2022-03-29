import matplotlib.pyplot as plt
import sys
import torch
from sklearn.manifold import TSNE

from grok.training_custom_transformer import TrainableTransformer
from grok.data import NUMS


def main():
    with torch.no_grad():
        transformer = TrainableTransformer(checkpoint=sys.argv[1])
        transformer.hparams.batchsize = -1  # full batch
        x = torch.tensor(NUMS).to(transformer.device).view(-1, 1)
        emb = transformer.transformer.embedding(x).squeeze(1).cpu().numpy()
        emb = TSNE(
            n_components=2,
            verbose=True,
            learning_rate="auto",
            perplexity=10,
            init="pca",
        ).fit_transform(emb)
        plt.scatter(emb[:, 0], emb[:, 1], c=x.cpu().numpy())
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    main()

