#/usr/bin/env sh

# nominal, grokking observed
python grok/training_custom_transformer.py --gpu 1 --weight_decay 0.1 --optimizer adamw --lr_multiplier 1e-3 --math_operator / --max_epochs 50000 --random_seed 1

# adamw with different lr
python grok/training_custom_transformer.py --gpu 1 --weight_decay 0.1 --optimizer adamw --lr_multiplier 1e-4 --math_operator / --max_epochs 50000 --random_seed 1

# adamw with different decoder lrs
python grok/training_custom_transformer.py --gpu 1 --weight_decay 0.1 --optimizer adamw --lr_multiplier 1e-3 --decoder_lr 1e-1 --math_operator / --max_epochs 50000 --random_seed 1
python grok/training_custom_transformer.py --gpu 1 --weight_decay 0.1 --optimizer adamw --lr_multiplier 1e-3 --decoder_lr 1e-2 --math_operator / --max_epochs 50000 --random_seed 1
python grok/training_custom_transformer.py --gpu 1 --weight_decay 0.1 --optimizer adamw --lr_multiplier 1e-2 --decoder_lr 1e-1 --math_operator / --max_epochs 50000 --random_seed 1
python grok/training_custom_transformer.py --gpu 1 --weight_decay 0.1 --optimizer adamw --lr_multiplier 1e-2 --decoder_lr 1e-2 --math_operator / --max_epochs 50000 --random_seed 1

# no weight decay
python grok/training_custom_transformer.py --gpu 1 --weight_decay 0 --optimizer adamw --lr_multiplier 1e-3 --math_operator / --max_epochs 50000 --random_seed 1
python grok/training_custom_transformer.py --gpu 1 --weight_decay 0 --optimizer adamw --lr_multiplier 1e-3 --decoder_lr 1e-2 --math_operator / --max_epochs 50000 --random_seed 1

# no weight decay but dropout
python grok/training_custom_transformer.py --gpu 1 --weight_decay 0 --optimizer adamw --lr_multiplier 1e-3 --math_operator / --max_epochs 50000 --random_seed 1 --dropout .1
python grok/training_custom_transformer.py --gpu 1 --weight_decay 0 --optimizer adamw --lr_multiplier 1e-3 --decoder_lr 1e-2 --math_operator / --max_epochs 50000 --random_seed 1 --dropout .1
