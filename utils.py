import numpy as np

def train_cmd(
    weight_decay=0,
    optimizer="adamw",
    lr_multiplier=1e-3,
    decoder_lr=1,
    max_epochs=100000,
    random_seed=1,
    train_data_pct=50,
    dropout=0,
    esam_rho=.05,
    esam_beta=1,
    esam_gamma=1,
):
    return " ".join(
        [
            f"python grok/training_custom_transformer_slim.py",
            f"--weight_decay {weight_decay}",
            f"--optimizer {optimizer}",
            f"--lr_multiplier {lr_multiplier}",
            f"--decoder_lr {decoder_lr}",
            f"--math_operator +",
            f"--max_epochs {max_epochs}",
            f"--random_seed {random_seed}",
            f"--train_data_pct {train_data_pct}",
            f"--dropout {dropout}",
            f"--esam_rho {esam_rho}",
            f"--esam_beta {esam_beta}",
            f"--esam_gamma {esam_gamma}",
        ]
    )


weight_decays = [0] + list(np.exp(np.linspace(-4, 1, 19)))
