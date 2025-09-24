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
# from data_construction.randman_dataset import data_split_randman
# from data_construction.shd_dataset import data_split_shd

from class_based_implementation.train_model import objective, MODEL_CLASSES, LOSS_FUNCTIONS # Import mappings too

import sys

# Import the randman stuff from DarwinNeuron instead
sys.path.append("/scratch/nar8991/snn/DarwinNeuron")
from src.RandmanFunctions import RandmanConfig, split_and_load, split_test_and_load




# Define which models to run experiments for
models_to_run = [
    "Hybrid_RNN_SNN_rec",
    "Hybrid_NSN_SNN_rec",
    "Hybrid_RNN_SNN_V1_same_layer",
    "Hybrid_NSN_SNN_V1_same_layer",
    "SNN",
    "ANN_with_LIF_output",
    "NSN_with_LIF_output",
    "Hybrid_NSN_SNN_V1_Flexible_Spiking"
]

# Map dataset names to their data loading functions
data_loaders_map = {
    # "randman": data_split_randman,
    # "shd": data_split_shd,
    # "nmnist": data_split_nmnist,
}

percent_data_list = [1.0, 0.5, 0.25]

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
    parser.add_argument(
        "--no_non_recurrent",
        action="store_true",  # Set to True if the flag is present
        help="Flag to disable recurrent connections (or some other boolean option)",
    )    # Add more arguments as needed for your specific use case
    parser.add_argument(
        "--nb_hidden",
        action="store_true",  # Set to True if the flag is present
        help="Sweep for nb_hidden sizes",
    )    # Add more arguments as needed for your specific use case
    parser.add_argument(
        "--percent_data",
        action="store_true",  # Set to True if the flag is present
        help="Sweep for percent_data sizes",
    )    # Add more arguments as needed for your specific use case
    parser.add_argument(
        "--gamma_init_type",
        type=str,
        default="rand",
        choices=["rand", "const_zero", "const_one", "const_half"],
        help="How to initialize gamma: rand, const_zero, const_one, const_half"
    )
    parser.add_argument(
        "--gamma_fixed",
        action="store_true",
        help="If set, gamma will be fixed (not trainable)"
    )
    parser.add_argument(
        "--no_reset_nsn",
        dest="reset_nsn",
        action="store_false",
        help="If set, NSN neurons will NOT reset after firing (default: True for NSN models)"
    )
    return parser.parse_args()

def run_optuna_study(
    model_name,
    recurrent_setting,
    seed,
    data_config,
    device,
    train_loader,
    val_loader,
    test_loader,
    wandb_project_name,
    loss_type,
    sampler_type,
    n_trials,
    study_name_suffix,
    sampler,
    local_checkpoint_dir_for_study: str, # This is the specific directory for THIS study
    percent_snn: float = None,
    gamma_init_type: str = None,
    gamma_fixed: bool = None,
    reset_nsn: bool = None

):
    """
    Encapsulates the Optuna study creation and optimization.
    """
    print(
        f"Running optimization for model: {model_name}, recurrent={recurrent_setting}, {study_name_suffix}"
    )

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        study_name=f"model_search_{data_config['data_name']}_{loss_type}_{model_name}_seed_{seed}_{sampler_type}_{study_name_suffix}_{gamma_init_type}_{gamma_fixed}_{reset_nsn}",
        sampler=sampler
    )

    # List to store (trial_value, checkpoint_path) for each trial
    trial_results = []

    def wrapped_objective_with_args(trial):
        # The objective function now returns (loss, best_checkpoint_path)
        loss, best_checkpoint_path = objective(
            trial=trial,
            model_name_str=model_name,
            data_config=data_config,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            recurrent_setting=recurrent_setting,
            wandb_project_name=wandb_project_name,
            loss_type=loss_type,
            sampler_type=sampler_type,
            local_checkpoint_dir=local_checkpoint_dir_for_study, # Pass the specific local dir for this study
            percent_snn=percent_snn,
            gamma_init_type=gamma_init_type,
            gamma_fixed=gamma_fixed,
            reset_nsn=reset_nsn
        )
        trial_results.append((loss, best_checkpoint_path)) # Store the result
        return loss # Optuna optimizes based on the first returned value

    study.optimize(wrapped_objective_with_args, n_trials=n_trials)

    # --- Log Best Checkpoint Path for the Study ---
    best_trial = study.best_trial
    best_loss = best_trial.value

    # Find the corresponding checkpoint path for the best trial
    best_checkpoint_for_study = "N/A"
    for loss, path in trial_results:
        if loss == best_loss: # This assumes unique best loss, or picks first if multiple
            best_checkpoint_for_study = path
            break
    
    # NEW: Create a standardized symbolic link to the best model
    standard_symlink_filename = "best_model_of_study.ckpt"
    standard_symlink_full_path = os.path.join(local_checkpoint_dir_for_study, standard_symlink_filename)

    if best_checkpoint_for_study != "N/A" and os.path.exists(best_checkpoint_for_study):
        try:
            # Remove existing symlink if it exists to avoid FileExistsError
            if os.path.exists(standard_symlink_full_path) or os.path.islink(standard_symlink_full_path):
                os.remove(standard_symlink_full_path)
            
            # Create a relative symlink for better portability if the base directory moves
            # Calculate relative path from symlink location to target file
            relative_path_to_best_ckpt = os.path.relpath(best_checkpoint_for_study, local_checkpoint_dir_for_study)
            os.symlink(relative_path_to_best_ckpt, standard_symlink_full_path)
            print(f"Created symbolic link to best model: {standard_symlink_full_path} -> {relative_path_to_best_ckpt}")
        except Exception as e:
            print(f"Warning: Could not create symbolic link for best model: {e}")
            standard_symlink_full_path = "Symlink Creation Failed"
    else:
        print("Warning: No valid best model checkpoint path found to create symlink.")
        standard_symlink_full_path = "N/A (No valid checkpoint)"


    # Initialize a new Wandb run for the study summary
    # study_wandb_run_name = f"Study_Summary_{model_name}_rec_{recurrent_setting}_{loss_type}_{study_name_suffix}"
    # wandb.init(
    #     project=wandb_project_name,
    #     name=study_wandb_run_name,
    #     reinit=True,
    #     job_type="study_summary",
    #     config={
    #         "model_name": model_name,
    #         "recurrent_setting": recurrent_setting,
    #         "data_config": data_config,
    #         "loss_type": loss_type,
    #         "sampler_type": sampler_type,
    #         "n_trials": n_trials,
    #         "study_name_suffix": study_name_suffix,
    #         "best_trial_number": best_trial.number,
    #         "best_trial_loss": best_loss,
    #         "best_trial_params": best_trial.params,
    #         "local_checkpoint_dir_for_study": local_checkpoint_dir_for_study,
    #         "standardized_best_model_symlink": standard_symlink_full_path # NEW: Log the symlink path
    #     }
    # )
    # wandb.log({"best_model_checkpoint_path_for_study": best_checkpoint_for_study})
    # wandb.finish() # End the study summary Wandb run

def main():
    global models_to_run
    args = arg_parser()
    data, dim_manifold, n_trials, loss_type, sampler_type, sweep_seed, num_workers, chosen_model, no_non_recurrent, nb_hidden, percent_data, gamma_init_type, gamma_fixed, reset_nsn = (
        args.data, args.dim_manifold, args.n_trials, args.loss_type, args.sampler_type, args.sweep_seed, args.num_workers, args.chosen_model, args.no_non_recurrent, args.nb_hidden, args.percent_data,
        args.gamma_init_type, args.gamma_fixed, args.reset_nsn
    )
    num_classes = args.num_classes
    if chosen_model is not None:
        # if chosen_model not in models_to_run:
        #     raise ValueError(f"Chosen model {chosen_model} is not in the list of models to run: {models_to_run}")
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
    
    # Adjust these parameters based on sweep
    # if nb_hidden:
       
    # if percent_data:
    #     data_config['percent_data'] = percent_data


    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    save_dir_base = (
        f"/vast/nar8991/snn/training_results/{data}/{dim_manifold}_d/{num_classes}_classes/{data_config['nb_inputs']}"
    )
    os.makedirs(save_dir_base, exist_ok=True)
    save_dir = os.path.join(save_dir_base, loss_type)
    print(f"Results will be saved to: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    # NEW FILE - Updated line
    if data == "shd":
        nb_hidden_list = [256, 64, 128]
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
            "adam_lr": [1e-3], # SNN default - only SNN now
            "optimizer": ["Adam"], # For grid search, you can also add "adamw"
            # "momentum": [0, 0.5, 0.99], # Only relevant for SGD
            "l2_lower": [15], #use SHD paper instead
            "v2_lower": [0.001],
            "l1_upper": [0.06],
            # "v1_upper": [15, 100],
            "v1_upper": [15],            
            "l2_upper": [0],
            "v2_upper": [0],
            "spike_grad_scale": [10],
            "zenke_enabled": [True], # Zenke regularization is used in SHD paper
            # "attention_loss": [False]
        }
        if models_to_run == ["ANN_with_LIF_output"] or models_to_run == ["NSN_with_LIF_output"]:
            search_space_grid_and_tpe_params['l2_lower'] = [None] # No regularization for ANN
            search_space_grid_and_tpe_params['v2_lower'] = [None]
            search_space_grid_and_tpe_params['l1_upper'] = [None]
            search_space_grid_and_tpe_params['v1_upper'] = [None]
            search_space_grid_and_tpe_params['l2_upper'] = [None]
            search_space_grid_and_tpe_params['v2_upper'] = [None]
            search_space_grid_and_tpe_params['spike_grad_scale'] = [None] # No spike grad scale for ANN
            search_space_grid_and_tpe_params['zenke_enabled'] = [None] # No zenke enabled for ANN
            search_space_grid_and_tpe_params['adam_lr'] = [2e-4]
        elif models_to_run == ["Hybrid_RNN_SNN_V1_same_layer"] or models_to_run == ["Hybrid_NSN_SNN_V1_same_layer"]:
            search_space_grid_and_tpe_params['adam_lr'] = [5e-4]       
        elif models_to_run == ["Hybrid_RNN_SNN_rec"] or models_to_run == ["Hybrid_NSN_SNN_rec"]:
            search_space_grid_and_tpe_params['adam_lr'] = [1e-3]
        if nb_hidden or percent_data:
            search_space_grid_and_tpe_params['v1_upper'] = [100] # Reduces final number of sweeps a bit
    elif data == "randman":
        data_loader_config = {
            'randman_id': 0,
            'randman_dir': '/scratch/nar8991/snn/DarwinNeuron/data/randman'
        }
        randman = RandmanConfig.lookup_by_id(data_loader_config['randman_id'], os.path.join(data_loader_config['randman_dir'], "meta-data.csv"))
        dataset = randman.read_dataset(data_loader_config['randman_dir'])
        train_loader, val_loader = split_and_load(dataset, batch_size=data_config["batch_size"])
        test_loader = split_test_and_load(dataset, batch_size=data_config["batch_size"])
        data_config["nb_inputs"] = randman.nb_units
        data_config["nb_outputs"] = randman.nb_classes
        nb_hidden_list = [20]

        search_space_grid_and_tpe_params = {
            "adam_lr": [0.05],
            "optimizer": ["Adam"], # For grid search, you can also add "adamw"
            "l2_lower": [100],
            "v2_lower": [10e-3],
            # "l1_upper": [0.06],
            "l1_upper": [1],
            "v1_upper": [50],
            "l2_upper": [0],
            "v2_upper": [0],
            "zenke_enabled": [True],
            "spike_grad_scale": [10],
            "gradient_clip_val": [None],
            # "percent_snn": [None]
            # "percent_snn": [0.25, 0.5, 0.75]
        }
        if models_to_run == ["ANN_with_LIF_output"]:
            search_space_grid_and_tpe_params['l2_lower'] = [None] # No regularization for ANN
            search_space_grid_and_tpe_params['v2_lower'] = [None]
            search_space_grid_and_tpe_params['l1_upper'] = [None]
            search_space_grid_and_tpe_params['v1_upper'] = [None]
            search_space_grid_and_tpe_params['l2_upper'] = [None]
            search_space_grid_and_tpe_params['v2_upper'] = [None]
            search_space_grid_and_tpe_params['spike_grad_scale'] = [None] # No spike grad scale for ANN
            search_space_grid_and_tpe_params['zenke_enabled'] = [None] # No zenke enabled for ANN
            search_space_grid_and_tpe_params['adam_lr'] = [1e-3, 0.005]
            search_space_grid_and_tpe_params['gradient_clip_val'] = [3.25, 3.75]
        elif models_to_run == ["Hybrid_NSN_SNN_V1_same_layer"]:
            search_space_grid_and_tpe_params['gradient_clip_val'] = [1, 2, 3, 4, 5]
        elif models_to_run == ["Hybrid_RNN_SNN_V1_same_layer"]:
            search_space_grid_and_tpe_params['gradient_clip_val'] = [1, 2, 4, 9]
            search_space_grid_and_tpe_params['l1_upper'] = [1, 100]
            search_space_grid_and_tpe_params['v1_upper'] = [50, 100]
        elif models_to_run == ["Hybrid_RNN_SNN_rec"]:
            search_space_grid_and_tpe_params['gradient_clip_val'] = [1, 2, 4, 5]
            search_space_grid_and_tpe_params['l1_upper'] = [1, 100]
            search_space_grid_and_tpe_params['v1_upper'] = [50, 100]
    else:
        raise ValueError(f"Unsupported dataset: {data}. Supported datasets are: {list(data_loaders_map.keys())}")

    # Add back in these if statements if still useful...
    # Percent data and nb hidden sweeps
    # else:
    # percent_data_for_run = percent_data_list

    # Set the sweep stuff
    if sampler_type == "grid":
        sampler = optuna.samplers.GridSampler(search_space=search_space_grid_and_tpe_params)
        num_combinations = 1
        for values in search_space_grid_and_tpe_params.values():
            num_combinations *= len(values)
        print(f"Using Grid Search with {num_combinations} combinations.")
        # Reassign this so it's not a pain: n_trials
        n_trials = num_combinations
    else: # Default to Bayesian (TPE)
        sampler = optuna.samplers.TPESampler()
        print("Using Bayesian Optimization (TPE).")
    pl.seed_everything(sweep_seed)  

    # Main optimization loop
    for model_name in models_to_run:

        if model_name in ["Hybrid_RNN_SNN_V1_same_layer"] or no_non_recurrent == True:
            allowed_recurrents = [True] # These models are designed to be recurrent
        else:
            allowed_recurrents = [True, False]
        # if data == "randman" and model_name == "Hybrid_NSN_SNN_V1_same_layer":
        #     percent_snn = [0.75]
        if "Hybrid" in model_name and "Flexible" not in model_name:
            # These models can use percent SNN
            percent_snn = [0.9, 0.85, 0.8, 0.75, 0.5, 0.25] 
        else:
            percent_snn = [None]

        for recurrent_setting in allowed_recurrents:
            extra_gamma_dir = []
            # REVISIT BASED ON FINDINGS
            # if "NSN" in model_name or model_name == "Hybrid_NSN_SNN_V1_Flexible_Spiking":
            #     extra_gamma_dir.append(f"reset_nsn_{reset_nsn}")
            if model_name == "Hybrid_NSN_SNN_V1_Flexible_Spiking":
                extra_gamma_dir = [f"{gamma_init_type}_gamma_init", f"gamma_fixed_{gamma_fixed}", f"reset_nsn_{reset_nsn}"]
            for p_s in percent_snn:

                # First -> % data normal, nb hidden reduced
                # This should probably be removed from the config eventually but don't worry for now...
                if nb_hidden:
                    data_config['percent_data'] = percent_data_list[0]
                    # Do this logic so we don't have to run hypothesis 1 again - to adjust in the morning for the other sweep!
                    for nb_h in nb_hidden_list[1:]: # Skip the first run
                        data_config['nb_hidden'] = nb_h
                        current_study_local_checkpoint_dir = os.path.join(
                            save_dir_base,
                            loss_type,
                            model_name,
                            f"recurrent_{recurrent_setting}",
                            f"seed_{sweep_seed}",
                            f"{sampler_type}_sampler",
                            f"{data_config['nb_hidden']}_hidden", # Use the current nb_hidden
                            f"{data_config['percent_data']}_pct_data", # Use the current percent_data
                            f"{p_s}_percent_snn", # Use the current percent_snn
                            *extra_gamma_dir
                        )
                        os.makedirs(current_study_local_checkpoint_dir, exist_ok=True)

                        run_optuna_study(
                            model_name=model_name,
                            recurrent_setting=recurrent_setting,
                            data_config=data_config, # Pass the current data_config
                            device=device,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            test_loader=test_loader,
                            seed=sweep_seed,
                            wandb_project_name=f"Class_based_sweeps_{data}_{dim_manifold}d_with_reg_{num_classes}classes_{loss_type}",
                            loss_type=loss_type,
                            sampler_type=sampler_type,
                            n_trials=n_trials,
                            study_name_suffix=f"{nb_h}_neurons",
                            sampler=sampler,
                            local_checkpoint_dir_for_study=current_study_local_checkpoint_dir,
                            percent_snn=p_s,
                            gamma_init_type=gamma_init_type,
                            gamma_fixed=gamma_fixed,
                            reset_nsn=reset_nsn
                        )
                elif percent_data:
                    data_config['nb_hidden'] = nb_hidden_list[0]
                    for p_d in percent_data_list[1:]: # Skip the first run (default)
                        data_config['percent_data'] = p_d
                        current_study_local_checkpoint_dir = os.path.join(
                            save_dir_base,
                            loss_type,
                            model_name,
                            f"recurrent_{recurrent_setting}",
                            f"seed_{sweep_seed}",
                            f"{sampler_type}_sampler",
                            f"{data_config['nb_hidden']}_hidden", # Use the current nb_hidden
                            f"{data_config['percent_data']}_pct_data", # Use the current percent_data
                            f"{p_s}_percent_snn", # Use the current percent_snn
                            *extra_gamma_dir
                        )
                        os.makedirs(current_study_local_checkpoint_dir, exist_ok=True)
                        run_optuna_study(
                            model_name=model_name,
                            recurrent_setting=recurrent_setting,
                            data_config=data_config, # Pass the current data_config
                            device=device,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            test_loader=test_loader,
                            seed=sweep_seed,
                            wandb_project_name=f"Class_based_sweeps_{data}_{dim_manifold}d_with_reg_{num_classes}classes_{loss_type}",
                            loss_type=loss_type,
                            sampler_type=sampler_type,
                            n_trials=n_trials,
                            study_name_suffix=f"{p_d}_pct_data",
                            sampler=sampler,
                            local_checkpoint_dir_for_study=current_study_local_checkpoint_dir,
                            percent_snn=p_s,
                            gamma_init_type=gamma_init_type,
                            gamma_fixed=gamma_fixed,
                            reset_nsn=reset_nsn
                        )
                else:
                    current_study_local_checkpoint_dir = os.path.join(
                        save_dir_base,
                        loss_type,
                        model_name,
                        f"recurrent_{recurrent_setting}",
                        f"seed_{sweep_seed}",
                        f"{sampler_type}_sampler",
                        f"{data_config['nb_hidden']}_hidden", # Use the current nb_hidden
                        f"{data_config['percent_data']}_pct_data", # Use the current percent_data
                        f"{p_s}_percent_snn", # Use the current percent_snn
                        *extra_gamma_dir
                    )
                    print(current_study_local_checkpoint_dir)
                    os.makedirs(current_study_local_checkpoint_dir, exist_ok=True)

                    run_optuna_study(
                        model_name=model_name,
                        recurrent_setting=recurrent_setting,
                        data_config=data_config, # Pass the current data_config
                        device=device,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        seed=sweep_seed,
                        wandb_project_name=f"Class_based_sweeps_{data}_{dim_manifold}d_with_reg_{num_classes}classes_{loss_type}",
                        loss_type=loss_type,
                        sampler_type=sampler_type,
                        n_trials=n_trials,
                        study_name_suffix=None,
                        sampler=sampler,
                        local_checkpoint_dir_for_study=current_study_local_checkpoint_dir,
                        percent_snn=p_s,
                        gamma_init_type=gamma_init_type,
                        gamma_fixed=gamma_fixed,
                        reset_nsn=reset_nsn
                    )

                

if __name__ == "__main__":
    main()