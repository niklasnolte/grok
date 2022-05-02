# %%
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
from utils import weight_decays, dropouts, esam_rhos, decoder_lrs
# %% 
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
# %%

train_ttgs = []
val_ttgs = []

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
  val_ttgs.append(val_ttg)
  train_ttgs.append(train_ttg)

plt.scatter(Xs, val_ttgs, label="val", alpha=.5)
plt.scatter(Xs, train_ttgs, label="train", alpha=.5)
plt.scatter(Xs, [v - t  for t,v in zip(train_ttgs, val_ttgs)], label="val - train", alpha=.5)
plt.ylabel('epochs to 99.9% accuracy')

if "weight_decay" == sys.argv[1]:
  plt.xlabel('decoder weight decay')
elif "dropout" == sys.argv[1]:
  plt.xlabel('dropout')
elif "esam_rho" == sys.argv[1]:
  plt.xlabel('esam rho')
elif "decoder_lr" == sys.argv[1]:
  plt.xlabel('decoder learning rate')  
  
plt.loglog()
plt.legend()
plt.tight_layout()
plt.show()
