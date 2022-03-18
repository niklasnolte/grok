#/usr/bin/env sh

# nominal, grokking observed
python scripts/train.py --gpu 1 --weight_decay 0.1 --optimizer adamw --lr_multiplier 1e-3

# adamw with different lrs
python scripts/train.py --gpu 1 --weight_decay 0.1 --optimizer adamw --lr_multiplier 1e-2
python scripts/train.py --gpu 1 --weight_decay 0.1 --optimizer adamw --lr_multiplier 1e-4

# sgd with different lrs
python scripts/train.py --gpu 1 --weight_decay 0.1 --optimizer sgd --lr_multiplier 1e-2
python scripts/train.py --gpu 1 --weight_decay 0.1 --optimizer sgd --lr_multiplier 1e-3
python scripts/train.py --gpu 1 --weight_decay 0.1 --optimizer sgd --lr_multiplier 1e-4

# adamw with different decoder lrs
python scripts/train.py --gpu 1 --weight_decay 0.1 --optimizer adamw --lr_multiplier 1e-3 --decoder_lr 1e-1
python scripts/train.py --gpu 1 --weight_decay 0.1 --optimizer adamw --lr_multiplier 1e-3 --decoder_lr 1e-2
python scripts/train.py --gpu 1 --weight_decay 0.1 --optimizer adamw --lr_multiplier 1e-2 --decoder_lr 1e-1
python scripts/train.py --gpu 1 --weight_decay 0.1 --optimizer adamw --lr_multiplier 1e-2 --decoder_lr 1e-2

# no weight decay
python scripts/train.py --gpu 1 --weight_decay 0 --optimizer adamw --lr_multiplier 1e-3
python scripts/train.py --gpu 1 --weight_decay 0 --optimizer adamw --lr_multiplier 1e-3 --decoder_lr 1e-2

# asymmetric operator
python scripts/train.py --gpu 1 --weight_decay 0.1 --optimizer adamw --lr_multiplier 1e-3 --math_operator -
