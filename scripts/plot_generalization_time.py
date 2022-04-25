# %%
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import os
from IPython import embed
# %% 
weight_decays = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 5]
paths = [f"../logs/weight_decay_{wd}_seed_2_old/version_0/" for wd in weight_decays]
files = [f for file in paths for f in os.listdir(file) if f.endswith(".0")]

assert len(paths) == len(files)
files = [os.path.join(p,f) for f,p in zip(files, paths)]
# %%

train_ttgs = []
val_ttgs = []

for file in files:
  ea = event_accumulator.EventAccumulator(file)
  ea.Reload()
  val_accs = np.array([[x.step, x.value] for x in ea.Scalars("val_accuracy")])
  train_accs = np.array([[x.step, x.value] for x in ea.Scalars("train_accuracy")])
  val_ttg = int(val_accs[np.argmax(val_accs[:,1] > 99.9),0])
  train_ttg = int(train_accs[np.argmax(train_accs[:,1] > 99.9),0])
  if val_ttg == 0:
    val_ttg = ea.Scalars("val_accuracy")[-1].step
  val_ttgs.append(val_ttg)
  train_ttgs.append(train_ttg)

plt.plot(weight_decays, val_ttgs, label="val", alpha=.7)
plt.plot(weight_decays, train_ttgs, label="train", alpha=.7)
plt.plot(weight_decays, [v - t  for t,v in zip(train_ttgs, val_ttgs)], label="val - train", alpha=.7)
plt.hlines([0], 0, 1, linestyles="dashed")
plt.semilogx()
plt.legend()
plt.show()
embed()
