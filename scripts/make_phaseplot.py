import os
import matplotlib.pyplot as plt
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), "grok.mplstyle"))
from matplotlib import ticker
import numpy as np
import json
from utils import decoder_lrs_phaseplot, weight_decays_phaseplot


seeds = [1,2,3]
lrs, wds = np.meshgrid(decoder_lrs_phaseplot, weight_decays_phaseplot)

train_ttgs = []
val_ttgs = []
diff_ttgs = []

for lr,wd in zip(lrs.flatten(), wds.flatten()):
  #read json file
  with open(f"logs/phaseplot_weight_decay_{wd}_decoder_lr_{lr}_seed_1/version_0/performance_info.json") as f:
    data = json.load(f)
  val_accs = np.array(data["val_accuracy"])
  train_accs = np.array(data["train_accuracy"])
  val_ttg = int(val_accs[np.argmax(val_accs[:,1] > 99.9),0])
  train_ttg = int(train_accs[np.argmax(train_accs[:,1] > 99.9),0])
  if val_ttg == 0:
    val_ttg = data["val_accuracy"][-1][0] # last step
  if train_ttg == 0:
    train_ttg = val_ttg
  if train_ttg == val_ttg == data["val_accuracy"][-1][0]:
    diff_ttgs.append(train_ttg) 
  else:
    diff_ttgs.append(max(val_ttg - train_ttg + 1, 1))
  val_ttgs.append(val_ttg)
  train_ttgs.append(train_ttg)

val_ttgs = np.array(val_ttgs).reshape(-1, len(weight_decays_phaseplot))
train_ttgs = np.array(train_ttgs).reshape(-1, len(weight_decays_phaseplot))
diff_ttgs = np.array(diff_ttgs).reshape(-1, len(weight_decays_phaseplot))

fig, axs = plt.subplots(1,3, figsize=(13, 4), sharey=True)
for ttgs, ax, title in zip([val_ttgs, train_ttgs, diff_ttgs], axs, ["Validation", "Training", "Val - Train"]):
  im = ax.contourf(lrs, wds, np.log10(ttgs), levels=10)
  ax.set_title(title)
  ax.loglog()

cbar = fig.colorbar(im, ax=axs.ravel().tolist())
cbar.set_label("log$_{10}$(time till accuracy $\geq 99.9\%$)")
plt.setp(axs[0], ylabel='decoder weight decay')
plt.setp(axs[1], xlabel='decoder learning rate')

#plt.show()
plt.savefig(f"generalization_time_weight_decay_vs_decoder_lr.pdf", bbox_inches="tight")
