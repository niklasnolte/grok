import os
import matplotlib.pyplot as plt
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), "grok.mplstyle"))
import numpy as np
import json
import sys
from utils import weight_decays, dropouts, esam_rhos, decoder_lrs

if len(sys.argv) != 2:
    print("Usage: python plot_generalization_time.py [weight_decay|dropout|esam_rho|decoder_lr]")
    sys.exit(1)

seeds = [1,2,3]
if "weight_decay" == sys.argv[1]:
  files = [f"logs/weight_decay_{wd}_seed_{seed}/version_0/performance_info.json" for wd in weight_decays for seed in seeds]
  Xs = [wd for wd in weight_decays for _ in seeds]
elif "dropout" == sys.argv[1]:
  files = [f"logs/dropout_{dropout}_seed_{seed}/version_0/performance_info.json" for dropout in dropouts for seed in seeds]
  Xs = [dropout for dropout in dropouts for _ in seeds]
elif "esam_rho" == sys.argv[1]:
  files = [f"logs/esam_rho_{rho}_beta_1_seed_{seed}/version_0/performance_info.json" for rho in esam_rhos for seed in seeds]
  Xs = [rho for rho in esam_rhos for _ in seeds]
elif "decoder_lr" == sys.argv[1]:
  files = [f"logs/decoder_lr_{lr}_multiplier_0.001_seed_{seed}/version_0/performance_info.json" for lr in decoder_lrs for seed in seeds]
  Xs = [lr for lr in decoder_lrs for _ in seeds]


train_ttgs = []
val_ttgs = []
diff_ttgs = []

for file in files:
  #read json file
  with open(file) as f:
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
    diff_ttg = val_ttg
  else:
    diff_ttg = val_ttg - train_ttg
  val_ttgs.append(val_ttg)
  train_ttgs.append(train_ttg)
  diff_ttgs.append(diff_ttg)

val_ttgs_median = np.median(np.array(val_ttgs).reshape(-1, len(seeds)), axis=1)
train_ttgs_median = np.median(np.array(train_ttgs).reshape(-1, len(seeds)), axis=1)
diff_ttgs_median = np.median(np.array(diff_ttgs).reshape(-1, len(seeds)), axis=1)

plt.scatter(Xs, val_ttgs, label="validation set", alpha=.5)
plt.scatter(Xs, train_ttgs, label="training set", alpha=.5)
plt.scatter(Xs, diff_ttgs, label="validation - training", alpha=.5)

plt.plot(Xs[::3], val_ttgs_median)
plt.plot(Xs[::3], train_ttgs_median)
plt.plot(Xs[::3], diff_ttgs_median)

plt.ylabel('epochs to 99.9\% accuracy')

if "weight_decay" == sys.argv[1]:
  plt.xlabel('decoder weight decay')
  plt.loglog()
elif "dropout" == sys.argv[1]:
  plt.xlabel('dropout rate')
  plt.semilogy()
elif "esam_rho" == sys.argv[1]:
  plt.xlabel(r'ESAM $\rho$ parameter')
  plt.loglog()
elif "decoder_lr" == sys.argv[1]:
  plt.xlabel('decoder learning rate')  
  plt.loglog()
  
if "dropout" == sys.argv[1]:
  plt.legend(loc="upper center")
else:
  plt.legend(loc="best")

plt.tight_layout()
#plt.show()
plt.savefig(f"generalization_time_{sys.argv[1]}.pdf")
