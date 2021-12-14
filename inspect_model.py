from argparse import Namespace
import os
import json
import yaml


from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from grok.training import TrainableTransformer

# ckpt_dir = "./openai_grok-scripts/e62ioupl/checkpoints"
# ckpt_dir = "./openai_grok-scripts/3gpmnogc/checkpoints"
# ckpt_dir = "./openai_grok-scripts/cyss152e/checkpoints"
ckpt_dir = "./openai_grok-scripts/1co466b4/checkpoints"
model_path = os.path.join(ckpt_dir, next(os.walk(ckpt_dir))[-1][0])
print(model_path)
# model_path = './checkpoints/epoch_512.ckpt'

model = TrainableTransformer.load_from_checkpoint(model_path)

data = model.transformer.linear.weight.data.cpu().numpy()[22:119] # not sure if this should be 23:120 instead

mods = list(range(2,30))
colors = [np.arange(data.shape[0]) % i for i in mods]

# print(colors)

# perform t-SNE
model = TSNE(n_components=2, perplexity=30, learning_rate=200, verbose=True)
# model = PCA(n_components=2)
transformed_data = model.fit_transform(data / np.linalg.norm(data, ord=1, axis=1, keepdims=True))

for m, color in zip(mods, colors):
    # compute order to connect points
    max_num = data.shape[0]-1
    order = [0]
    cur = m
    while cur != 0:
        order.append(cur)
        cur = (cur + m) % 97

    plt.figure(figsize=(10,10))
    plt.title(f"Numbers mod {m}")
    plt.scatter(transformed_data[:,0], transformed_data[:,1], alpha=0.5, c=color, cmap='viridis')
    for i, txt in enumerate(map(lambda x: str(x), np.arange(data.shape[0]))):
        plt.annotate(txt, (transformed_data[i,0], transformed_data[i,1]))
        
        if i < data.shape[0]-1:
            plt.plot(transformed_data[order][i:i+2,0], transformed_data[order][i:i+2,1], alpha=0.3, c='grey')
        else:
            plt.plot([transformed_data[order][-1,0], transformed_data[order][0,0]], [transformed_data[order][i,1], transformed_data[order][0,1]], alpha=0.3, c='grey')
    plt.savefig(f"tSNE_weights_mod_{m}.png")
    plt.close()