#!/usr/bin/env python
import os

if os.environ.get("CH", "0") == "1":
  from grok import training_ch as training
elif os.environ.get("CH", "0") == "2":
  from grok import training_custom_transformer as training
else:
  from grok import training


parser = training.add_args()
parser.set_defaults(logdir=os.environ.get("GROK_LOGDIR", "."))
hparams = parser.parse_args()
hparams.datadir = os.path.abspath(hparams.datadir)
hparams.logdir = os.path.abspath(hparams.logdir)


print(hparams)
print(training.train(hparams))
