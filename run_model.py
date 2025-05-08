import torch
import json
import argparse
import wandb
import optuna
import numpy as np
import os
import pickle

# from nmnist_dataset import data_split_nmnist
from randman_dataset import data_split_randman
from shd_dataset import data_split_shd

from train_model import objective

models = [
    "SNN",
    "ANN_with_LIF_output",
    "Hybrid_RNN_SNN_rec",
    "Hybrid_RNN_SNN_V1_same_layer",
]

data_loaders = {
    "randman": data_split_randman,
    "shd": data_split_shd,
}


def arg_parser():
    parser = argparse.ArgumentParser(description="Optuna + WandB tuning for SNN models")
    parser.add_argument(
        "--data", type=str, help="Dataset to use: randman or shd", default="randman"
    )
    parser.add_argument(
        "--dim_manifold", type=int, default=None, help="only relevant for randman"
    )
    parser.add_argument(
        "--n_trials", type=int, default=20, help="Should be >100 for full sweep"
    )
    return parser.parse_args()


def main():
    args = arg_parser()
    data, dim_manifold, n_trials = args.data, args.dim_manifold, args.n_trials
    with open(f"{data}_config.json", "r") as f:
        data_config = json.load(f)

    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load data
    train_loader, test_loader, val_loader = data_loaders[data](
        data_config, device, dim_manifold=dim_manifold
    )

    # For saving the models
    save_dir = (
        f"/scratch/nar8991/snn/snn_ann_hybrid/optuna_results/{data}/{dim_manifold}_d"
    )
    os.makedirs(save_dir, exist_ok=True)  # Create directory if missing

    for model_name in models:
        allowed_recurrents = [True, False]
        if model_name == "Hybrid_RNN_SNN_V1_same_layer":
            allowed_recurrents = [True]

        for recurrent_setting in allowed_recurrents:
            best_history = None
            best_config = None
            best_weights = None
            best_3d_landscape = None
            print(
                f"Running optimization for model: {model_name}, recurrent={recurrent_setting}"
            )

            # Retrieve the best model to compare it to where available
            model_name_adj = model_name + f"_rec_{recurrent_setting}"
            model_save_dir = os.path.join(save_dir, model_name_adj)
            os.makedirs(model_save_dir, exist_ok=True)
            config_path = os.path.join(model_save_dir, "config.json")

            # Get the best val acc if it exists
            best_val_acc = -np.inf
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    best_config = json.load(f)
                best_val_acc = best_config["best_val_acc"]
                print(f"Best val acc from previous model: {best_val_acc:.4f}")
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
                    f"Optuna_{data}_v4_{dim_manifold}d_with_landscape",
                    # save_dir,
                )

            study.optimize(wrapped_objective, n_trials=n_trials)

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
                    "best_val_acc": best_val_acc,
                }
                best_3d_landscape = best_trial.user_attrs["3d_landscape"]

            if best_history and best_config and best_weights and best_3d_landscape:
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
                # Save 3D landscape
                loss_landscape_path = os.path.join(model_save_dir, "3d_loss_surface.png")
                best_3d_landscape.savefig(loss_landscape_path)
                print(f"Saved 3D landscape to {config_path}")
            else:
                print(f"No valid results for {model_name}")


if __name__ == "__main__":
    main()
