import torch
import json
import argparse
import wandb
import optuna

from randman_dataset import data_split_randman
from train_model import objective

models = [
    "SNN",
    "ANN_with_LIF_output",
    "Hybrid_RNN_SNN_rec",
    "Hybrid_RNN_SNN_V1_same_layer",
]
    
def main():
    parser = argparse.ArgumentParser(description="Optuna + WandB tuning for SNN models")
    parser.add_argument(
        "--data_config_path",
        type=str,
        default="/scratch/nar8991/snn/snn_ann_hybrid/randman_config.json",
        help="Path to the configuration JSON file.",
    )
    args = parser.parse_args()

    with open(args.data_config_path, "r") as f:
        data_config = json.load(f)

    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load data
    train_loader, test_loader, val_loader = data_split_randman(data_config, device)

    # Run Optuna optimization for each model
    for model_name in models:
        allowed_recurrents = [True, False]
        # Special case: Hybrid_RNN_SNN_V1_same_layer is always recurrent
        if model_name == "Hybrid_RNN_SNN_V1_same_layer":
            allowed_recurrents = [True]

        for recurrent_setting in allowed_recurrents:
            print(f"Running optimization for model: {model_name}, recurrent={recurrent_setting}")
            
            study = optuna.create_study(direction="maximize")

            study.optimize(
                lambda trial: objective(
                    trial,
                    model_name,
                    data_config,
                    device,
                    train_loader,
                    val_loader,
                    test_loader,
                    recurrent_setting,
                    "Optuna_randman_v2" # pass fixed recurrent setting
                ),
                n_trials=20,
            )

            print(f"Best trial for {model_name} (recurrent={recurrent_setting}):")
            print(study.best_trial)

if __name__ == "__main__":
    main()
