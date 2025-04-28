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
                print(f"Best trial user attributes: {best_trial.user_attrs}")
                best_config = {
                    "model_name": model_name,
                    "recurrent_setting": recurrent_setting,
                    **best_trial.params,
                    "best_val_acc": best_val_acc
                }
            save_dir = "/scratch/nar8991/snn/snn_ann_hybrid/optuna_results"
            os.makedirs(save_dir, exist_ok=True)  # Create directory if missing
            # After finishing allowed recurrents for this model
            if best_history is not None:
                path = os.path.join(save_dir, f"{model_name}_best.pkl")
                try:
                    with open(path, "wb") as f:
                        pickle.dump(best_history, f)
                    print(f"Saved best model for {model_name} at {path}")
                except PermissionError:
                    print(f"Permission denied for {path}")
                except Exception as e:
                    print(f"Error saving {path}: {str(e)}")

                # Also save the config file for the best run
                if best_config is not None:
                    config_path = os.path.join(save_dir, f"{model_name}_config.json")
                    with open(config_path, "w") as f:
                        json.dump(best_config, f, indent=4)  # Save as JSON with indentation for readability
                    print(f"Saved best config for {model_name} at {config_path}")
            else:
                print("History not found for this model.")
if __name__ == "__main__":
    main()
