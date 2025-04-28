import torch
import json
import argparse
import wandb
import optuna
import numpy as np
import os
import pickle

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

    for model_name in models:
        best_val_acc = -np.inf
        best_history = None
        best_config = None
        allowed_recurrents = [True, False]
        if model_name == "Hybrid_RNN_SNN_V1_same_layer":
            allowed_recurrents = [True]

        for recurrent_setting in allowed_recurrents:
            print(f"Running optimization for model: {model_name}, recurrent={recurrent_setting}")

            study = optuna.create_study(direction="maximize")

            def wrapped_objective(trial):
                return objective(
                    trial,
                    model_name,
                    data_config,
                    device,
                    train_loader,
                    val_loader,
                    test_loader,
                    recurrent_setting,
                    "Optuna_randman_v4",
                    # save_dir,
                )

            study.optimize(wrapped_objective, n_trials=20)

            best_trial = study.best_trial
            print(f"Best trial for {model_name} (recurrent={recurrent_setting}):")
            print(best_trial)

            if best_trial.value > best_val_acc:
                print(f"New best for {model_name} with val acc {best_trial.value:.4f}")
                best_val_acc = best_trial.value
                best_history = best_trial.user_attrs["history"]
                best_weights = best_trial.user_attrs["weights"]
                print(f"Best trial user attributes: {best_trial.user_attrs}")
                best_config = {
                    "model_name": model_name,
                    "recurrent_setting": recurrent_setting,
                    **best_trial.params,
                    "best_val_acc": best_val_acc
                }
            save_dir = "/scratch/nar8991/snn/snn_ann_hybrid/optuna_results/randman"
            os.makedirs(save_dir, exist_ok=True)  # Create directory if missing
            # After finishing allowed recurrents for this model
            model_name_adj = model_name + f"_rec_{recurrent_setting}"
            if best_history and best_config:
                model_save_dir = os.path.join(save_dir, model_name_adj)
                os.makedirs(model_save_dir, exist_ok=True)

                # Save history
                history_path = os.path.join(model_save_dir, "history.pkl")
                with open(history_path, "wb") as f:
                    pickle.dump(best_history, f)
                print(f"Saved history to {history_path}")
                # Save weights
                weights_path = os.path.join(model_save_dir, "weights.pkl")
                with open(weights_path, "wb") as f:
                    pickle.dump(best_weights, f)
                print(f"Saved weights to {weights_path}")
                # Save config
                config_path = os.path.join(model_save_dir, "config.json")
                with open(config_path, "w") as f:
                    json.dump(best_config, f, indent=2)
                print(f"Saved config to {config_path}")
            else:
                print(f"No valid results for {model_name}")

if __name__ == "__main__":
    main()
