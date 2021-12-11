from argparse import Namespace
import os
import json
import yaml


from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from grok.training import TrainableTransformer

model_path = './checkpoints/epoch_512.ckpt'

model = TrainableTransformer.load_from_checkpoint(model_path)

data = model.transformer.linear.weight.data.cpu().numpy()[23:120]

mods = [5,7]
colors = [np.arange(data.shape[0]) % i for i in mods]

# print(colors)

# perform t-SNE
model = TSNE(n_components=2, perplexity=30, learning_rate=200, verbose=True)
# model = PCA(n_components=2)
transformed_data = model.fit_transform(data / np.linalg.norm(data, ord=1, axis=1, keepdims=True))

for m, color in zip(mods, colors):
    plt.figure(figsize=(10,10))
    plt.title(f"Numbers mod {m}")
    plt.scatter(transformed_data[:,0], transformed_data[:,1], alpha=0.5, c=color, cmap='viridis')
    for i, txt in enumerate(map(lambda x: str(x), np.arange(data.shape[0]))):
        plt.annotate(txt, (transformed_data[i,0], transformed_data[i,1]))
    plt.show()