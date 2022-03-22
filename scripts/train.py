#!/usr/bin/env python
import os

if os.environ.get("CH", "0") != "0":
  from grok import training_ch as training
else:
  from grok import training


parser = training.add_args()
parser.set_defaults(logdir=os.environ.get("GROK_LOGDIR", "."))
hparams = parser.parse_args()
hparams.datadir = os.path.abspath(hparams.datadir)
hparams.logdir = os.path.abspath(hparams.logdir)


print(hparams)
print(training.train(hparams))
