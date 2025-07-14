import torch
import json
import argparse
import wandb
import optuna
import numpy as np
import io
import os
import pickle
import pytorch_lightning as pl

from maren_data.helpers import choose_data_params
# from data_construction.nmnist_dataset import data_split_nmnist
from data_construction.randman_dataset import data_split_randman
from data_construction.shd_dataset import data_split_shd

from class_based_implementation.train_model import objective, MODEL_CLASSES, LOSS_FUNCTIONS # Import mappings too

# from class_based_implementation.hparams_config import (
#     hyperparameter_definitions,
#     get_grid_search_space,
#     suggest_tpe_params
# )

# Define which models to run experiments for
models_to_run = [
    "Hybrid_RNN_SNN_V1_same_layer",
    "SNN",
    "ANN_with_LIF_output",
    "Hybrid_RNN_SNN_rec",
]

# Map dataset names to their data loading functions
data_loaders_map = {
    "randman": data_split_randman,
    # "shd": data_split_shd,
    # "nmnist": data_split_nmnist,
}

# Helper for unpickling models trained on different devices
class DynamicDeviceUnpickler(pickle.Unpickler):
    def __init__(self, file, target_device):
        super().__init__(file)
        self.target_device = target_device

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=self.target_device)
        else:
            return super().find_class(module, name)


def arg_parser():
    parser = argparse.ArgumentParser(description="Optuna + WandB tuning for SNN models")
    parser.add_argument(
        "--data", type=str, help="Dataset to use: randman or shd", default="randman"
    )
    parser.add_argument(
        "--dim_manifold", type=int, default=None, help="only relevant for randman"
    )
    parser.add_argument(
        "--num_classes", type=int, default=None, help="only relevant for randman"
    )
    parser.add_argument(
        "--n_trials", type=int, default=20, help="Number of trials for Bayesian search or max trials for Grid search"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse",
        choices=list(LOSS_FUNCTIONS.keys()),
        help="Loss function type: " + ", ".join(LOSS_FUNCTIONS.keys()),
    )
    parser.add_argument(
        "--sampler_type",
        type=str,
        default="tpe",
        choices=["tpe", "grid"],
        help="Type of Optuna sampler: 'bayesian' (TPE) or 'grid'",
    )
    parser.add_argument("--sweep_seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, help="Number of workers for data loading", default=4)
    parser.add_argument("--chosen_model", type=str, help="Model to run, otherwise all models", default=None)
    # Add more arguments as needed for your specific use case
    return parser.parse_args()

# def create_sampler(sampler_type, sweep_seed):
#     """Creates the appropriate Optuna sampler based on type."""
#     if sampler_type == "tpe":
#         print("Using TPESampler (Bayesian Optimization)")
#         return optuna.samplers.TPESampler(seed=sweep_seed)
#     elif sampler_type == "grid":
#         # Get the grid search space from the config file
#         grid_search_space = get_grid_search_space()

#         # Calculate total trials for the grid (for informative print)
#         total_grid_trials = 1
#         for param_values_list in grid_search_space.values():
#             total_grid_trials *= len(param_values_list)
#         print(f"Using GridSampler with {total_grid_trials} total combinations.")
#         return optuna.samplers.GridSampler(grid_search_space, seed=sweep_seed)
#     else:
#         raise ValueError(f"Unknown sampler type: {sampler_type}. Choose 'tpe' or 'grid'.")


def main():
    global models_to_run
    args = arg_parser()
    data, dim_manifold, n_trials, loss_type, sampler_type, sweep_seed, num_workers, chosen_model = (
        args.data, args.dim_manifold, args.n_trials, args.loss_type, args.sampler_type, args.sweep_seed, args.num_workers, args.chosen_model
    )
    num_classes = args.num_classes
    if chosen_model is not None:
        if chosen_model not in models_to_run:
            raise ValueError(f"Chosen model {chosen_model} is not in the list of models to run: {models_to_run}")
        models_to_run = [chosen_model]
    config_file_path = f"data_construction/{data}_config.json"
    if data == "shd_old":
        config_file_path = "data_construction/shd_config.json"
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found: {config_file_path}")
    with open(config_file_path, "r") as f:
        data_config = json.load(f)

    # if num_classes is not None:
    #     data_config["nb_outputs"] = num_classes # This logic isn't my favorite but we can revisit
    # else:
    if num_classes is None:
        num_classes = data_config["nb_outputs"] 
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    save_dir_base = (
        f"/scratch/nar8991/snn/snn_ann_hybrid/optuna_results/{data}/{dim_manifold}_d/{num_classes}_classes/{data_config['nb_inputs']}"
    )
    os.makedirs(save_dir_base, exist_ok=True)
    save_dir = os.path.join(save_dir_base, loss_type)
    print(f"Results will be saved to: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    # NEW FILE - Updated line
    if data == "shd":
        settings = data_config.copy()
        settings["device"] = device
        settings["dtype"] = torch.float32
        # settings["max_time"] = data_config.get("max_time", 1.0) # Default to 1.0 if not specified
        pre_path_data = f"/scratch/nar8991/snn/snn_ann_hybrid/maren_data/shd/{data_config['nb_inputs']}_inputs/{data_config['max_time']}_max_time/{data_config['noise']}_noise"
        # if data_config["percent_data"] < 1.0:
        #     pre_path_data += f"/{data_config['percent_data']}_percent_data"
        train_loader, val_loader, test_loader = choose_data_params(
        data, settings, num_workers=num_workers,pre_path=pre_path_data
        )
        search_space_grid_and_tpe_params = {
            "lr": [1e-3], # This should actually be 1e-3
            "optimizer": ["Adam"], # For grid search, you can also add "adamw"
            # "momentum": [0, 0.5, 0.99], # Only relevant for SGD
            "l2_lower": [100], #use SHD paper instead
            "v2_lower": [0.001],
            "l1_upper": [0.06],
            "v1_upper": [100, 15],
            "l2_upper": [0],
            "v2_upper": [0],
            "spike_grad_scale": [10],
            "zenke_enabled": [True], # Zenke regularization is used in SHD paper
            # "attention_loss": [False]
        }
        if models_to_run == ["ANN"]:
            search_space_grid_and_tpe_params['l2_lower'] = [None] # No regularization for ANN
            search_space_grid_and_tpe_params['v2_lower'] = [None]
            search_space_grid_and_tpe_params['l1_upper'] = [None]
            search_space_grid_and_tpe_params['v1_upper'] = [None]
            search_space_grid_and_tpe_params['l2_upper'] = [None]
            search_space_grid_and_tpe_params['v2_upper'] = [None]
            search_space_grid_and_tpe_params['spike_grad_scale'] = [None] # No spike grad scale for ANN
            search_space_grid_and_tpe_params['zenke_enabled'] = [None] # No zenke enabled for ANN

    elif data == "shd_old":
        data_loaders_path = os.path.join(save_dir_base, "data_loaders.pkl")
        if os.path.exists(data_loaders_path):
            print(f"Loading existing data loaders from {data_loaders_path}")
            with open(data_loaders_path, "rb") as f:
                data_loaders_dict = DynamicDeviceUnpickler(f, device).load()
                train_loader = data_loaders_dict["train_loader"]
                val_loader = data_loaders_dict["val_loader"]
                test_loader = data_loaders_dict["test_loader"]
        else:
            train_loader, test_loader, val_loader = data_split_shd(
                data_config, device, dim_manifold=dim_manifold
            )
            with open(data_loaders_path, "wb") as f:
                pickle.dump(
                    {
                        "train_loader": train_loader,
                        "val_loader": val_loader,
                        "test_loader": test_loader,
                    },
                    f,
                )
            print(f"Data loaders saved to {data_loaders_path}")
    elif data == "randman":
        data_loaders_path = os.path.join(save_dir_base, "data_loaders.pkl")

        if os.path.exists(data_loaders_path):
            print(f"Loading existing data loaders from {data_loaders_path}")
            with open(data_loaders_path, "rb") as f:
                data_loaders_dict = DynamicDeviceUnpickler(f, device).load()
                train_loader = data_loaders_dict["train_loader"]
                val_loader = data_loaders_dict["val_loader"]
                test_loader = data_loaders_dict["test_loader"]
        else:
            train_loader, test_loader, val_loader = data_loaders_map[data](
                data_config, device, dim_manifold=dim_manifold
            )
            with open(data_loaders_path, "wb") as f:
                pickle.dump(
                    {
                        "train_loader": train_loader,
                        "val_loader": val_loader,
                        "test_loader": test_loader,
                    },
                    f,
                )
            print(f"Data loaders saved to {data_loaders_path}")
        # search_space_grid_and_tpe_params = {
        #     # "hidden_features": [32, 64, 128], # Must be even for hybrid models
        #     # "alpha": [0.7, 0.8, 0.9],
        #     # "beta": [0.7, 0.8, 0.9],
        #     "lr": [1e-4, 1e-3, 1e-2],
        #     "optimizer": ["Adam", "SGD"], # For grid search, you can also add "adamw"
        #     "momentum": [0, 0.5, 0.99], # Only relevant for SGD
        #     "l2_lower": [4],
        #     "v2_lower": [1e-2],
        #     "l1_upper": [1, 2, 4],
        #     "v1_upper": [0, 20, 40],
        #     "l2_upper": [0, 1, 4],
        #     "v2_upper": [0, 2, 4],
        #     "zenke_enabled": [True],
        # }
        search_space_grid_and_tpe_params = {
            # "hidden_features": [32, 64, 128], # Must be even for hybrid models
            # "alpha": [0.7, 0.8, 0.9],
            # "beta": [0.7, 0.8, 0.9],
            "adam_lr": [1e-2],
            "optimizer": ["Adam"], # For grid search, you can also add "adamw"
            # "momentum": [0, 0.5, 0.99], # Only relevant for SGD
            "l2_lower": [100],
            "v2_lower": [10e-3],
            "l1_upper": [1, 100],
            "v1_upper": [100],
            "l2_upper": [0],
            "v2_upper": [0],
            "zenke_enabled": [True],
            "spike_grad_scale": [10]
        }
    else:
        raise ValueError(f"Unsupported dataset: {data}. Supported datasets are: {list(data_loaders_map.keys())}")
    # Define the search space for Grid Search and parameter ranges for Bayesian (TPE)


    # Main optimization loop
    for model_name in models_to_run:
        allowed_recurrents = [True, False]
        if model_name in ["Hybrid_RNN_SNN_V1_same_layer"]:
            allowed_recurrents = [True] # These models are designed to be recurrent

        for recurrent_setting in allowed_recurrents:
            # These saves are not used for now, but could be useful later
            best_history = None
            best_config = None
            best_weights = None
            best_3d_landscape = None # These were from the custom train_and_evaluate, not used here
            best_clustering_coefficients = None

            print(
                f"Running optimization for model: {model_name}, recurrent={recurrent_setting}"
            )
        
            model_name_adj = model_name + f"_rec_{recurrent_setting}"
            
            # More saving logic that's not really used right now
            model_save_dir = os.path.join(save_dir, model_name_adj)
            os.makedirs(model_save_dir, exist_ok=True)
            config_path = os.path.join(model_save_dir, "config.json")
            best_metric_value = -np.inf # Optuna maximizes, so -loss is maximized
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    prev_best_config = json.load(f)
                # Assuming "best_val_acc" stored was actually the negative loss (the metric Optuna maximizes)
                best_metric_value = prev_best_config.get("best_val_acc", -np.inf)
                print(f"Best metric from previous model: {best_metric_value:.4f}")

            # Sampler and optimization setup
            if sampler_type == "grid":
                sampler = optuna.samplers.GridSampler(search_space=search_space_grid_and_tpe_params)
                num_combinations = 1
                for values in search_space_grid_and_tpe_params.values():
                    num_combinations *= len(values)
                print(f"Using Grid Search with {num_combinations} combinations.")
            else: # Default to Bayesian (TPE)
                sampler = optuna.samplers.TPESampler()
                print("Using Bayesian Optimization (TPE).")
            pl.seed_everything(sweep_seed)  
            study = optuna.create_study(
                direction="minimize", # or "maximize" if your objective is accuracy
                pruner=optuna.pruners.MedianPruner(
                    n_warmup_steps=5 # Potentially change this to patience
                ),
                study_name = f"model_search_{data}_{loss_type}_{model_name}_seed_{sweep_seed}_{sampler_type}",
                sampler=sampler
            )
            def wrapped_objective_with_args(trial):
                return objective(
                    trial=trial,
                    model_name_str=model_name,
                    data_config=data_config,
                    device=device,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader, # Still passing test_loader for consistency, though objective doesn't use it directly for training
                    recurrent_setting=recurrent_setting,
                    wandb_project_name=f"Class_based_sweeps_{data}_{dim_manifold}d_with_reg_{num_classes}classes_{loss_type}",
                    loss_type=loss_type,
                    sampler_type=sampler_type,
                )

            study.optimize(wrapped_objective_with_args, n_trials=n_trials) # Maybe eventually set to None for grid search (or num combinations)


            # Other saving logic that isn't really used right now
            best_trial = study.best_trial
            print(f"Best trial for {model_name} (recurrent={recurrent_setting}):")
            print(best_trial)

            if best_trial.value > best_metric_value:
                print(f"New best for {model_name} with value {best_trial.value:.4f}")
                best_metric_value = best_trial.value

                # Note: 'history', 'weights', '3d_landscape', 'hidden_layer_clustering'
                # are currently set as user_attrs in your provided objective code snippet.
                # However, the current `objective` function (which uses PL.Trainer)
                # does NOT produce these directly. If you need these, you'd have
                # to add custom logic to the PL.Trainer or model callbacks to extract and store them.
                # For now, these lines will access attributes that might not be set by the PL Trainer.
                best_history = best_trial.user_attrs.get("history")
                best_weights = best_trial.user_attrs.get("weights")
                best_3d_landscape = best_trial.user_attrs.get("3d_landscape")
                # best_clustering_coefficients = best_trial.user_attrs.get("clustering_coefficients")


                best_config = {
                    "model_name": model_name,
                    "recurrent_setting": recurrent_setting,
                    **best_trial.params,
                    "best_val_metric": best_metric_value, # Storing the maximized metric (negative total loss)
                    "data_config": data_config,
                    "loss_type": loss_type,
                }
                print(f"Best trial user attributes: {best_trial.user_attrs}")

                # Save best config and other artifacts if they exist
                if best_config:
                    config_path = os.path.join(model_save_dir, "config.json")
                    with open(config_path, "w") as f:
                        json.dump(best_config, f, indent=2)
                    print(f"Saved config to {config_path}")

                    # These are only saved if the custom train_and_evaluate
                    # logic (which produces them) is re-integrated, or
                    # if you add custom PL Callbacks to generate and store them.
                    if best_history:
                        history_path = os.path.join(model_save_dir, "history.pkl")
                        with open(history_path, "wb") as f:
                            pickle.dump(best_history, f)
                        print(f"Saved history to {history_path}")

                    if best_weights:
                        weights_path = os.path.join(model_save_dir, "weights.pkl")
                        # You'd need to convert PL model state_dict to 'weights' format if custom format is desired
                        # For a PL model, it's typically model.state_dict()
                        # This current 'best_weights' would only be set if the provided train_and_evaluate was used
                        with open(weights_path, "wb") as f:
                            pickle.dump(best_weights, f)
                        print(f"Saved weights to {weights_path}")

                    if best_3d_landscape:
                        loss_landscape_path = os.path.join(model_save_dir, "3d_loss_surface.png")
                        best_3d_landscape.savefig(loss_landscape_path)
                        print(f"Saved 3D landscape to {loss_landscape_path}")

                    # if best_clustering_coefficients:
                    #     clustering_path = os.path.join(model_save_dir, "clustering_coefficients.json")
                    #     with open(clustering_path, "w") as f:
                    #         json.dump(best_clustering_coefficients, f, indent=2)
                    #     print(f"Saved clustering coefficients to {clustering_path}")

            else:
                print(f"No valid results or no improvement for {model_name}")


if __name__ == "__main__":
    main()