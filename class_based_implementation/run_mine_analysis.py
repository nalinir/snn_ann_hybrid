import torch
import pytorch_lightning as pl
import wandb
import argparse
import json
import os
import io
import pickle
import optuna # NEW IMPORT for MINE hyperparameter sweep
from functools import partial # NEW IMPORT for passing args to objective
from torch.utils.data import DataLoader

# Import your BaseTemporalModel and other model classes
from class_based_implementation.models import BaseTemporalModel, SNN, ANN_with_LIF_output, Hybrid_RNN_SNN_rec, Hybrid_RNN_SNN_V1_same_layer, MIEstimator, MINEDataTupleDataset
from class_based_implementation.surr_grad import SurrGradSpike
from class_based_implementation.train_model import objective, MODEL_CLASSES, LOSS_FUNCTIONS # Import mappings too

# --- Data Loading Helpers (Copied from your main.py for self-containment) ---
from maren_data.helpers import choose_data_params
from data_construction.randman_dataset import data_split_randman
from data_construction.shd_dataset import data_split_shd

import torch.multiprocessing as mp # <<< NEW: Import multiprocessing
# --- CRITICAL: Set the start method for multiprocessing at the top level ---
# This must be done BEFORE any CUDA operations (like .to('cuda')) or
# DataLoaders with num_workers > 0 are initialized.
# The 'if __name__ == "__main__":' block below will handle the execution.
# This ensures it's set only once when the script is run directly.
# --- CRITICAL: Set the start method for multiprocessing at the top level ---
try:
    # Only set if not already set, to avoid RuntimeError in some environments.
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
except RuntimeError as e:
    print(f"Could not set multiprocessing start method: {e}. It might already be set or you are not running in __main__.")

# Map dataset names to their data loading functions
data_loaders_map = {
    "randman": data_split_randman,
    # "shd": data_split_shd,
    # "nmnist": data_split_nmnist,
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
    return parser.parse_args()


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

# Define your MODEL_CLASSES (should match train_model.py)
MODEL_CLASSES = {
    "SNN": SNN,
    "ANN_with_LIF_output": ANN_with_LIF_output,
    "Hybrid_RNN_SNN_rec": Hybrid_RNN_SNN_rec,
    "Hybrid_RNN_SNN_V1_same_layer": Hybrid_RNN_SNN_V1_same_layer,
    "BaseTemporalModel": BaseTemporalModel # Include BaseTemporalModel for direct loading if needed
}

# # Dummy Loss class for loading the model, as it's a required arg for BaseTemporalModel
# class DummyLoss(torch.nn.Module):
#     def forward(self, input, target):
#         return torch.tensor(0.0)

# --- NEW HELPER FUNCTION FOR DATA PREPROCESSING ---
def _prepare_h_variants_batch_optimized(h_raw_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepares the different H variants by permuting dimensions to 'remove' information
    for a single batch, using vectorized operations.
    h_raw_batch expected shape: (batch_size, time_steps, hidden_features)
    """
    batch_size, time_steps, hidden_features = h_raw_batch.shape
    device = h_raw_batch.device

    H_full_batch = h_raw_batch.clone()

    # Optimized Spatial Shuffle (shuffles features within each (batch, timestep))
    h_flat_bt = h_raw_batch.view(-1, hidden_features) # (B*T, F)
    idx = torch.rand(h_flat_bt.shape, device=device).argsort(dim=1) # Permutations for each row
    H_no_spatial_batch_flat = torch.gather(h_flat_bt, 1, idx)
    H_no_spatial_batch = H_no_spatial_batch_flat.view(batch_size, time_steps, hidden_features)

    # Optimized Temporal Shuffle (shuffles timesteps within each (batch, feature))
    h_transposed_bf = h_raw_batch.transpose(1, 2).contiguous() # (B, F, T)
    h_flat_bf = h_transposed_bf.view(-1, time_steps) # (B*F, T)
    idx_temporal = torch.rand(h_flat_bf.shape, device=device).argsort(dim=1)
    H_no_temporal_batch_flat = torch.gather(h_flat_bf, 1, idx_temporal)
    H_no_temporal_batch = H_no_temporal_batch_flat.view(batch_size, hidden_features, time_steps).transpose(1, 2)

    # Optimized Only Activation Shuffle (shuffles all elements within each batch item's (T, F) block)
    h_flat_per_batch = h_raw_batch.view(batch_size, -1) # (B, T*F)
    idx_activation = torch.rand(h_flat_per_batch.shape, device=device).argsort(dim=1)
    H_only_activation_batch_flat = torch.gather(h_flat_per_batch, 1, idx_activation)
    H_only_activation_batch = H_only_activation_batch_flat.view(batch_size, time_steps, hidden_features)

    return H_full_batch, H_no_spatial_batch, H_no_temporal_batch, H_only_activation_batch


# --- NEW HELPER FUNCTION FOR DATA PREPROCESSING ---
def preprocess_mine_data(
    base_model: pl.LightningModule, # This is the pre-trained SNN/ANN model
    original_train_loader: torch.utils.data.DataLoader,
    original_val_loader: torch.utils.data.DataLoader,
    original_test_loader: torch.utils.data.DataLoader,
    input_features_per_timestep: int, # X_dim
    hidden_features: int, # H_dim
    device: torch.device,
    num_workers_for_mine_dataloaders: int # This can now be >0
) -> dict:
    
    print("\n--- Starting MINE data preprocessing ---")
    base_model.eval() # Ensure model is in evaluation mode
    base_model.to(device) # Ensure model is on the correct device for inference

    processed_data_loaders = {}
    
    for loader_name, original_loader in zip(
        ["train", "val", "test"],
        [original_train_loader, original_val_loader, original_test_loader]
    ):
        print(f"Processing {loader_name} data...")
        # Lists to store processed X and H variants
        all_x_data = []
        all_h_full = []
        all_h_no_spatial = []
        all_h_no_temporal = []
        all_h_only_activation = []

        with torch.no_grad():
            for i, (inputs, _) in enumerate(original_loader):
                inputs = inputs.to(device)
                
                # IMPORTANT: This call MUST return (batch_size, time_steps, hidden_features)
                # Adjust your base_model's method if it returns more or less.
                # If your model returns output, states, x_raw, h_raw as per your commented
                # out MINEDataLoader, then this should be:
                # _, _, x_current_batch, h_raw_batch = base_model(inputs)
                # Assuming `x_current_batch` is the X to be used for MI.
                # Otherwise, it might be just:
                # h_raw_batch = base_model.get_hidden_states(inputs)
                # x_current_batch = inputs # If inputs are the raw X
                
                # Based on your commented code:
                _, _, x_current_batch, h_raw_batch = base_model(inputs)
                # x_current_batch will be (batch, time, x_feature_dim)
                # h_raw_batch will be (batch, time, h_feature_dim)
                
                # Get all H variants for this batch using the optimized function
                H_full, H_no_spatial, H_no_temporal, H_only_activation = \
                    _prepare_h_variants_batch_optimized(h_raw_batch)
                
                # Move all processed tensors to CPU to conserve GPU memory, as they are accumulated.
                # MINEDataTupleDataset will then yield CPU tensors, which MINELightningModule
                # will move to its device (GPU/CPU) for training the critic.
                all_x_data.append(x_current_batch.cpu())
                all_h_full.append(H_full.cpu())
                all_h_no_spatial.append(H_no_spatial.cpu())
                all_h_no_temporal.append(H_no_temporal.cpu())
                all_h_only_activation.append(H_only_activation.cpu())

                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(original_loader)} batches for {loader_name}.")

        # Concatenate all batches into single tensors
        x_tensor = torch.cat(all_x_data, dim=0)
        h_full_tensor = torch.cat(all_h_full, dim=0)
        h_no_spatial_tensor = torch.cat(all_h_no_spatial, dim=0)
        h_no_temporal_tensor = torch.cat(all_h_no_temporal, dim=0)
        h_only_activation_tensor = torch.cat(all_h_only_activation, dim=0)
        
        # Create TensorDatasets and DataLoaders
        processed_data_loaders[f"{loader_name}_full"] = DataLoader(
            MINEDataTupleDataset(x_tensor, h_full_tensor),
            batch_size=original_loader.batch_size,
            shuffle=True if loader_name == "train" else False,
            num_workers=num_workers_for_mine_dataloaders, # Now safe to use workers
            pin_memory=True if device.type == 'cuda' else False
        )
        processed_data_loaders[f"{loader_name}_no_spatial"] = DataLoader(
            MINEDataTupleDataset(x_tensor, h_no_spatial_tensor),
            batch_size=original_loader.batch_size,
            shuffle=True if loader_name == "train" else False,
            num_workers=num_workers_for_mine_dataloaders,
            pin_memory=True if device.type == 'cuda' else False
        )
        processed_data_loaders[f"{loader_name}_no_temporal"] = DataLoader(
            MINEDataTupleDataset(x_tensor, h_no_temporal_tensor),
            batch_size=original_loader.batch_size,
            shuffle=True if loader_name == "train" else False,
            num_workers=num_workers_for_mine_dataloaders,
            pin_memory=True if device.type == 'cuda' else False
        )
        processed_data_loaders[f"{loader_name}_only_activation"] = DataLoader(
            MINEDataTupleDataset(x_tensor, h_only_activation_tensor),
            batch_size=original_loader.batch_size,
            shuffle=True if loader_name == "train" else False,
            num_workers=num_workers_for_mine_dataloaders,
            pin_memory=True if device.type == 'cuda' else False
        )

    print("--- MINE data preprocessing complete ---")
    return processed_data_loaders

# --- MINE Optuna Objective Function ---
def mine_optuna_objective(
    trial: optuna.Trial,
    chosen_model: str, # This is the type of base model, not the instance
    checkpoint_path: str, # Path to the trained base model checkpoint
    data_config: dict,
    device: torch.device,
    wandb_project_name: str, # For Optuna sweep and summary
    preprocessed_mine_dataloaders: dict # Now this is a dictionary of DataLoaders
) -> float:
    """
    Optuna objective function for optimizing MINE hyperparameters.
    """
    # Suggest MINE hyperparameters for this trial
    mine_epochs = trial.suggest_int("mine_epochs", 50, 500, step=50)
    mine_lr = trial.suggest_float("mine_lr", 1e-4, 1e-2, log=True)
    mine_hidden_size = trial.suggest_categorical("mine_hidden_size", [64, 128, 256])

    # --- Initialize Wandb for this MINE trial (Optuna Trial's run) ---
    try:
        wandb_run = wandb.init(
            project=wandb_project_name,
            group=f"MINE_Optuna_Sweep_{chosen_model}_{os.path.basename(checkpoint_path).replace('.ckpt', '')}",
            name=f"MINE_trial_{trial.number}_epochs{mine_epochs}_lr{mine_lr:.1e}_h{mine_hidden_size}",
            reinit=True,
            config={
                "model_name": chosen_model,
                "checkpoint_path": checkpoint_path,
                "data_config": data_config,
                "mine_epochs": mine_epochs,
                "mine_lr": mine_lr,
                "mine_hidden_size": mine_hidden_size,
                "device": str(device)
            }
        )
    except Exception as e:
        print(f"Error initializing wandb for MINE trial {trial.number}: {e}")
        raise optuna.exceptions.TrialPruned(f"Wandb initialization failed: {e}")

    # --- MINE dimensions from pre-processed data ---
    # Retrieve dimensions from the first processed DataLoader's dataset
    # All variants should have the same X and H dimensions
    sample_x_data = preprocessed_mine_dataloaders['train_full'].dataset.x_data
    sample_h_data = preprocessed_mine_dataloaders['train_full'].dataset.h_data
    
    x_feature_dim = sample_x_data.shape[-1] # Assuming (N, T, Features)
    h_feature_dim = sample_h_data.shape[-1] # Assuming (N, T, Features)
    
    print(f"MINE critic will be initialized with x_feature_dim: {x_feature_dim}, h_feature_dim: {h_feature_dim}")


    # --- Initialize MIEstimator (no base_model needed here) ---
    mine_estimator = MIEstimator(
        x_feature_dim=x_feature_dim,
        h_feature_dim=h_feature_dim,
        mine_hidden_size=mine_hidden_size,
        mine_lr=mine_lr,
        chosen_model=chosen_model, # This is the type of model, not an instance
    )

    # --- Run MINE Estimation ---
    print("\nStarting MINE estimation for all variants...")
    try:
        mi_results = mine_estimator.estimate_all_mi_and_rates(
            train_loader_full=preprocessed_mine_dataloaders["train_full"],
            train_loader_no_spatial=preprocessed_mine_dataloaders["train_no_spatial"],
            train_loader_no_temporal=preprocessed_mine_dataloaders["train_no_temporal"],
            train_loader_only_activation=preprocessed_mine_dataloaders["train_only_activation"],
            val_loader_full=preprocessed_mine_dataloaders["val_full"],
            val_loader_no_spatial=preprocessed_mine_dataloaders["val_no_spatial"],
            val_loader_no_temporal=preprocessed_mine_dataloaders["val_no_temporal"],
            val_loader_only_activation=preprocessed_mine_dataloaders["val_only_activation"],
            test_loader_full=preprocessed_mine_dataloaders["test_full"],
            test_loader_no_spatial=preprocessed_mine_dataloaders["test_no_spatial"],
            test_loader_no_temporal=preprocessed_mine_dataloaders["test_no_temporal"],
            test_loader_only_activation=preprocessed_mine_dataloaders["test_only_activation"],
            mine_epochs=mine_epochs,
            mine_analysis_project_name="MINE_results", # MINE training runs log here
        )

        # Log final MI results for this Optuna trial
        wandb_run.log({ # Use the stored run object
            "final_mi_full": mi_results["mi_full"],
            "final_mi_no_spatial": mi_results["mi_no_spatial"],
            "final_mi_no_temporal": mi_results["mi_no_temporal"],
            "final_mi_only_activation": mi_results["mi_only_activation"],
            "final_ser": mi_results.get("exploitation_rates", {}).get("SER", None),
            "final_ter": mi_results.get("exploitation_rates", {}).get("TER", None),
            "final_aer": mi_results.get("exploitation_rates", {}).get("AER", None),
        })
        print(f"MINE results for trial {trial.number} logged to Wandb.")

        # Optuna's objective is to maximize MI_full (so return negative value)
        if mi_results["mi_full"] is None:
            print(f"Warning: MI_full is None for trial {trial.number}. Returning float('-inf') to prune.")
            # If MI_full is None or invalid, it's a bad trial. Prune it.
            raise optuna.exceptions.TrialPruned(f"MI_full could not be calculated for trial {trial.number}")
        
        # Use mi_full as the optimization target (maximize), so return its negative.
        # Add a small epsilon to avoid issues with MI=0 in some cases if it affects pruning
        return -(mi_results["mi_full"] + 1e-9)

    except Exception as e:
        print(f"MINE estimation failed for trial {trial.number}: {e}")
        import traceback
        traceback.print_exc()
        wandb_run.log({"mine_estimation_failed": True, "mine_error_message": str(e)})
        raise optuna.exceptions.TrialPruned(f"MINE estimation failed: {e}")

    finally:
        wandb.finish() # Ensure wandb.finish() is called for the trial run

def main():
    args = arg_parser()
    data, dim_manifold, n_trials, loss_type, sampler_type, sweep_seed, num_workers, chosen_model, no_non_recurrent, nb_hidden, percent_data = (
        args.data, args.dim_manifold, args.n_trials, args.loss_type, args.sampler_type, args.sweep_seed, args.num_workers, args.chosen_model, args.no_non_recurrent, args.nb_hidden, args.percent_data
    )

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_config_path = f"data_construction/{data}_config.json"
    # --- Load Data Configuration ---
    if not os.path.exists(data_config_path):
        raise FileNotFoundError(f"Data config file not found at: {data_config_path}")
    with open(data_config_path, 'r') as f:
        data_config = json.load(f)

    # --- Load Actual DataLoaders ---
    # This section is copied from your main.py to ensure consistent data loading
    # You might need to adjust paths like /scratch/nar8991/snn/snn_ann_hybrid/maren_data/shd/
    
    # Infer necessary data config parameters for data loading functions
    # data = data_config.get("data_name", "randman") # Assuming data_name is in data_config
    # dim_manifold = data_config.get("dim_manifold")
    num_classes = data_config["nb_outputs"]
    nb_inputs = data_config["nb_inputs"]
    nb_hidden = data_config["nb_hidden"]

    train_loader, val_loader, test_loader = None, None, None
    model_checkpoint_path = f'optuna_results/{data}/{dim_manifold}_d/{num_classes}_classes/{nb_inputs}/cross_entropy/{chosen_model}/recurrent_True/seed_{sweep_seed}/grid_sampler/{nb_hidden}_hidden/1.0_pct_data/best_model_of_study.ckpt'

    if data == "shd":
        settings = data_config.copy()
        settings["device"] = device
        settings["dtype"] = torch.float32
        pre_path_data = f"/scratch/nar8991/snn/snn_ann_hybrid/maren_data/shd/{nb_inputs}_inputs/{data_config.get('max_time', 1.0)}_max_time/{data_config.get('noise', 0.0)}_noise"
        train_loader, val_loader, test_loader = choose_data_params(
            data, settings, num_workers=args.num_workers, pre_path=pre_path_data
        )
    elif data == "shd_old": # Assuming shd_old also exists
        save_dir_base_shd_old = f"/scratch/nar8991/snn/snn_ann_hybrid/optuna_results/{data}/{dim_manifold}_d/{num_classes}_classes/{nb_inputs}"
        data_loaders_path = os.path.join(save_dir_base_shd_old, "data_loaders.pkl")
        if os.path.exists(data_loaders_path):
            print(f"Loading existing data loaders from {data_loaders_path}")
            with open(data_loaders_path, "rb") as f:
                data_loaders_dict = DynamicDeviceUnpickler(f, device).load()
                train_loader = data_loaders_dict["train_loader"]
                val_loader = data_loaders_dict["val_loader"]
                test_loader = data_loaders_dict["test_loader"]
        else:
            # Fallback if pkl not found, generate them
            train_loader, test_loader, val_loader = data_split_shd(
                data_config, device, dim_manifold=dim_manifold
            )
            os.makedirs(save_dir_base_shd_old, exist_ok=True)
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
        save_dir_base_randman = f"/scratch/nar8991/snn/snn_ann_hybrid/optuna_results/{data}/{dim_manifold}_d/{num_classes}_classes/{nb_inputs}"
        data_loaders_path = os.path.join(save_dir_base_randman, "data_loaders.pkl")
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
            os.makedirs(save_dir_base_randman, exist_ok=True)
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
    else:
        raise ValueError(f"Unsupported dataset: {data}. Supported datasets are: {list(data_loaders_map.keys())}")

    if train_loader is None or val_loader is None or test_loader is None:
        raise RuntimeError("Failed to load one or more data loaders.")
    model_checkpoint_path = (
        f'optuna_results/{data}/{dim_manifold}_d/{num_classes}_classes/{nb_inputs}/'
        f'{loss_type}/{chosen_model}/recurrent_True/seed_{sweep_seed}/'
        f'{sampler_type}_sampler/{data_config["nb_hidden"]}_hidden/1.0_pct_data/best_model_of_study.ckpt'
    )

    # --- Load Trained Base Model (once) ---
    model_class = MODEL_CLASSES.get(chosen_model)
    if model_class is None:
        raise ValueError(f"Unknown model name: {chosen_model}. Available: {list(MODEL_CLASSES.keys())}")
    
    print(f"Loading base model '{chosen_model}' from checkpoint: {model_checkpoint_path}")
    try:
        base_model_for_preprocessing = model_class.load_from_checkpoint(model_checkpoint_path)
        base_model_for_preprocessing.to(device) # Move to device for efficient inference
        base_model_for_preprocessing.eval()
        print("Base model loaded successfully for preprocessing.")
    except Exception as e:
        print(f"Error loading base model from checkpoint for preprocessing: {e}")
        raise RuntimeError(f"Failed to load base model: {e}")

    # --- Pre-process MINE Data (The Big Speedup!) ---
    # Pass the original num_workers here, as the new DataLoaders can use them.
    preprocessed_mine_dataloaders = preprocess_mine_data(
        base_model=base_model_for_preprocessing,
        original_train_loader=train_loader,
        original_val_loader=val_loader,
        original_test_loader=test_loader,
        input_features_per_timestep=nb_inputs,
        hidden_features=data_config["nb_hidden"], # Pass the inferred hidden features
        device=device,
        num_workers_for_mine_dataloaders=num_workers # Use global num_workers for new Dataloaders
    )

    # --- Optuna Study for MINE Hyperparameters ---
    study_name = f"MINE_HP_Sweep_{chosen_model}_{os.path.basename(model_checkpoint_path).replace('.ckpt', '')}"
    print(f"Starting Optuna study for MINE hyperparameters: {study_name}")
    
    study = optuna.create_study(
        direction="maximize", # Maximize MI_full
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=args.sweep_seed)
    )

    # Use functools.partial to pass fixed arguments to the objective function
    func = partial(
        mine_optuna_objective,
        chosen_model=chosen_model,
        checkpoint_path=model_checkpoint_path,
        data_config=data_config,
        device=device,
        wandb_project_name="MINE_results",
        preprocessed_mine_dataloaders=preprocessed_mine_dataloaders
    )

    study.optimize(func, n_trials=args.n_trials)

    print("\n--- MINE Hyperparameter Optimization Complete ---")
    print(f"Best trial: {study.best_trial.value} (negative MI_full)")
    print(f"Best parameters: {study.best_trial.params}")

    # --- Re-run MINE estimation with best parameters to get full results ---
    # This is necessary because mine_optuna_objective only returns negative MI_full
    # and we want all MI and exploitation rates for the best configuration.
    print("\n--- Re-running MINE estimation for best hyperparameters to log full results ---")
    
    # Instantiate MIEstimator with the best parameters found by Optuna
    best_mine_epochs = study.best_trial.params.get("mine_epochs")
    best_mine_lr = study.best_trial.params.get("mine_lr")
    best_mine_hidden_size = study.best_trial.params.get("mine_hidden_size")

    # Load the base model again (or ensure it's available)
    model_class = MODEL_CLASSES.get(chosen_model)
    input_features_per_timestep = data_config.get("nb_inputs")
    output_features = data_config.get("nb_outputs")
    hidden_features = data_config.get("nb_hidden")

    # dummy_model_args = {
    #     "input_features": input_features_per_timestep,
    #     "hidden_features": hidden_features,
    #     "output_features": output_features,
    #     "data_config": data_config,
    #     "recurrent": True, # Needs to match original model's setting
    #     "learning_rate": 1e-3,
    #     "loss_fn": DummyLoss(),
    #     "zenke_config": None,
    #     "optimizer_name": "Adam",
    #     "spike_grad_scale": 100.0,
    #     "model_type": chosen_model,
    #     "spike_fn": SurrGradSpike.apply if "SNN" in chosen_model else None
    # }
    model = model_class.load_from_checkpoint(model_checkpoint_path)
    # model = model_class.load_from_checkpoint(model_checkpoint_path, **dummy_model_args)
    model.to(device)
    model.eval()

    best_mine_estimator = MIEstimator(
        x_feature_dim=input_features_per_timestep,
        h_feature_dim=hidden_features,
        mine_hidden_size=best_mine_hidden_size,
        mine_lr=best_mine_lr,
        chosen_model=chosen_model
    )

    # Run the full estimation with the best parameters
    final_mi_results_for_logging = best_mine_estimator.estimate_all_mi_and_rates(
        base_model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        mine_epochs=best_mine_epochs,
        mine_analysis_project_name="MINE_results",
    )

    # Log best MINE parameters and the full MI results to the main MINE analysis project
    wandb.init(
        project="MINE_results",
        name=f"MINE_Best_HP_Summary_{chosen_model}",
        reinit=True,
        job_type="mine_hp_summary",
        config={
            "model_name": chosen_model,
            "checkpoint_path": model_checkpoint_path,
            "best_mine_epochs": best_mine_epochs,
            "best_mine_lr": best_mine_lr,
            "best_mine_hidden_size": best_mine_hidden_size,
            "best_negative_mi_full": study.best_trial.value,
            "corresponding_mi_full": -study.best_trial.value,
            "data_config_used": data_config,
            # Log the full results here
            "final_mi_full_best_hp": final_mi_results_for_logging["mi_full"],
            "final_mi_no_spatial_best_hp": final_mi_results_for_logging["mi_no_spatial"],
            "final_mi_no_temporal_best_hp": final_mi_results_for_logging["mi_no_temporal"],
            "final_mi_only_activation_best_hp": final_mi_results_for_logging["mi_only_activation"],
            "final_ser_best_hp": final_mi_results_for_logging["exploitation_rates"]["SER"],
            "final_ter_best_hp": final_mi_results_for_logging["exploitation_rates"]["TER"],
            "final_aer_best_hp": final_mi_results_for_logging["exploitation_rates"]["AER"],
        }
    )
    wandb.log({"best_mine_hyperparameters": study.best_trial.params})
    wandb.finish()


if __name__ == "__main__":
    main()
