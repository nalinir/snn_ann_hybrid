# class_based_implementation/train_model.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy
import wandb
import optuna # Make sure optuna is imported
import numpy as np # For pl.seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import os

# Import your models (assuming these are defined in snn_hybrid_models.py)
# from class_based_implementation.snn_hybrid_models import SNN, ANN_with_LIF_output, Hybrid_RNN_SNN_rec, Hybrid_RNN_SNN_V1_same_layer, SurrGradSpike

# Streamlined 
from class_based_implementation.models import SNN, SNN_2_Layer, ANN_with_LIF_output, Hybrid_RNN_SNN_rec, Hybrid_RNN_SNN_V1_same_layer, NSN_with_LIF_output, Hybrid_NSN_SNN_rec, Hybrid_NSN_SNN_V1_same_layer, Hybrid_NSN_SNN_V1_Flexible_Spiking
from class_based_implementation.surr_grad import SurrGradSpike

from old_implementation.loss_landscape import visualize_loss_landscape_3d

# Define your MODEL_CLASSES and LOSS_FUNCTIONS dictionaries here
MODEL_CLASSES = {
    "SNN": SNN,
    "ANN_with_LIF_output": ANN_with_LIF_output,
    "NSN_with_LIF_output": NSN_with_LIF_output,
    "Hybrid_RNN_SNN_rec": Hybrid_RNN_SNN_rec,
    "Hybrid_NSN_SNN_rec": Hybrid_NSN_SNN_rec,
    "Hybrid_RNN_SNN_V1_same_layer": Hybrid_RNN_SNN_V1_same_layer,
    "Hybrid_NSN_SNN_V1_same_layer": Hybrid_NSN_SNN_V1_same_layer,
    "Hybrid_NSN_SNN_V1_Flexible_Spiking": Hybrid_NSN_SNN_V1_Flexible_Spiking,
    "SNN_2_Layer": SNN_2_Layer,
}

# TO REMOVE/ADD LOGIC TO MAKE WORK -- right now only cross_entropy is correct
LOSS_FUNCTIONS = {
    "cross_entropy": nn.CrossEntropyLoss(),
    # "mse": nn.MSELoss(),
    # "nll": nn.NLLLoss(),
    # Add other loss functions if you use them
}

def objective(
    trial: optuna.Trial,
    model_name_str: str,
    data_config: dict,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader, # Kept for consistency, though not strictly used in current objective logic
    recurrent_setting: bool,
    wandb_project_name: str,
    loss_type: str,
    sampler_type: str,
    local_checkpoint_dir: str, # NEW ARGUMENT: Directory for local checkpoints
    percent_snn: float = None,
    gamma_init_type: str = None,
    gamma_fixed: bool = None,
    reset_nsn: bool = None,
):
    """
    Objective function for Optuna hyperparameter optimization.
    """
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    # Conditional hyperparameter suggestion based on optimizer
    if optimizer_name == "Adam":
        # learning_rate = trial.suggest_float("adam_lr", 1e-4, 2e-3, log=False)
        learning_rate = trial.suggest_float("adam_lr", 1e-4, 1e-2, log=False)
        momentum = 0.0
    elif optimizer_name == "SGD":
        learning_rate = trial.suggest_float("sgd_lr", 1e-2, 0.5, log=False)
        momentum = trial.suggest_float("momentum", 0.0, 0.99)
    elif optimizer_name == "Adamax":
        # Adamax specific learning rate
        if "ANN" in model_name_str:
            learning_rate = trial.suggest_float("adamax_lr", 1e-3, 1e-3, log=False)
        else:
            learning_rate = trial.suggest_float("adamax_lr", 1e-3, 1e-3, log=False)
        momentum = 0.0
    
    if model_name_str not in ["ANN_with_LIF_output", "NSN_with_LIF_output"]:        # Zenke regularization specific hyperparameters
        l2_lower = trial.suggest_float("l2_lower", 100, 100, log=False)
        v2_lower = trial.suggest_float("v2_lower", 1e-3, 1e-3, log=False)
        l1_upper = trial.suggest_float("l1_upper", 0.06, 0.06, log=False)
        v1_upper = trial.suggest_float("v1_upper", 0, 1000, log=False)
        l2_upper = trial.suggest_categorical("l2_upper", [0])
        v2_upper = trial.suggest_float("v2_upper", 0, 0, log=False)
        # Store Zenke config in a dict
        zenke_config = {
            "l2_lower": l2_lower,
            "v2_lower": v2_lower,
            "l1_upper": l1_upper,
            "v1_upper": v1_upper,
            "l2_upper": l2_upper,
            "v2_upper": v2_upper,
        }
        spike_grad_scale = trial.suggest_float("spike_grad_scale", 10.0, 10.0, log=False)
        spike_fn = SurrGradSpike.apply
    else:
        zenke_config = None
        spike_grad_scale = None
        spike_fn = None
    if model_name_str == "Hybrid_NSN_SNN_V1_Flexible_Spiking":
        v1_upper = trial.suggest_float("v1_upper", 100, 1000, log=False)
        

    if model_name_str == "ANN_with_LIF_output" : #To think about whether to apply this to the RNN models
        gradient_clip_algorithm = "norm"
        gradient_clip_val = trial.suggest_float("gradient_clip_val", 0, 10, log=False)
    elif (model_name_str == "Hybrid_NSN_SNN_V1_same_layer" or model_name_str == "Hybrid_RNN_SNN_V1_same_layer" or model_name_str == "Hybrid_RNN_SNN_rec") and data_config['data_name'] == "randman":
        gradient_clip_algorithm = "norm"
        gradient_clip_val = trial.suggest_float("gradient_clip_val", 0, 10, log=False)
    else:
        gradient_clip_algorithm = None
        gradient_clip_val = None
        
    # if "Hybrid" in model_name_str:
    #     # percent_snn = trial.suggest_float("percent_snn", 0, 1, log=False)
    # # Initialize wandb for this trial
    # else:
    #     # percent_snn = None
    wandb.init(
        project=wandb_project_name,
        group=f"{model_name_str}_rec_{recurrent_setting}_{loss_type}",
        name=f"{model_name_str}_trial_{trial.number}_params_{wandb.util.generate_id()}", # Add a unique ID
        reinit=True, # Allows reinitializing if a previous run didn't finish cleanly
        config={**trial.params, "sampler_type": sampler_type},
    )

    # --- Get Model Class and Loss Function Instance ---
    model_class = MODEL_CLASSES.get(model_name_str)
    if model_class is None:
        raise ValueError(f"Unknown model name: {model_name_str}")

    loss_fn_instance = LOSS_FUNCTIONS.get(loss_type)
    if loss_fn_instance is None:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # --- Model Instantiation ---
    model_args = {
        "input_features": data_config["nb_inputs"],
        "hidden_features": data_config["nb_hidden"],
        "output_features": data_config["nb_outputs"],
        "data_config": data_config,
        "recurrent": recurrent_setting,
        "learning_rate": learning_rate,
        "loss_fn": loss_fn_instance,
        "zenke_config": zenke_config,
        "optimizer_name": optimizer_name,
        "spike_grad_scale": spike_grad_scale,
        "model_type": model_name_str,
        "spike_fn": spike_fn,
        "gradient_clip_val": gradient_clip_val,
        "gradient_clip_algorithm": gradient_clip_algorithm,
        "percent_snn": percent_snn,
        "gamma_fixed": gamma_fixed,
        "gamma_init_type": gamma_init_type,
        "reset_nsn": reset_nsn,
    }

    # if model_name_str in ["SNN", "Hybrid_RNN_SNN_rec", "Hybrid_RNN_SNN_V1_same_layer"]:
    #     model_args["spike_fn"] = SurrGradSpike.apply # Use SurrGradSpike.apply directly
    # elif model_name_str == "ANN_with_LIF_output":
    #     model_args["spike_fn"] = None # No spikes for hidden layer in ANN
    #     # model_args["zenke_config"] = None # I think maybe this wasn't run fully but to revisit
        # model_args["spike_grad_scale"] = None # No spike grad scale for ANN
    model = model_class(**model_args)

    model.to(device)

    os.makedirs(local_checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=local_checkpoint_dir, # UPDATED: Save checkpoints to the local directory
        filename=f"{model_name_str}-trial_{trial.number}-best-{{epoch:02d}}-{{val_total_loss:.4f}}", # Added trial.number to filename
        monitor="val_total_loss", # Metric to monitor
        mode="min", # Minimize this metric
        save_top_k=1, # Save only the best model
        save_last=True, # Also save the last model
        verbose=False # Set to True for more logging during checkpointing
    )

    # After model instantiation, before training
    # print("Model expects input_features:", model.input_features)
    # print("Model w1 shape:", model.w1.shape)

    # # Get a batch from your train_loader to check input shape
    # xb, yb = next(iter(train_loader))
    # print("Sample batch xb shape:", xb.shape)
    # print("Sample batch yb shape:", yb.shape)
    # # --- PyTorch Lightning Trainer Setup ---
    trainer_args = {
        "max_epochs": data_config["epochs"],
        "accelerator": str(device.type),
        "devices": 1,
        "logger": pl.loggers.WandbLogger(log_model=True, project=wandb_project_name),
        "enable_checkpointing": True,
        "callbacks": [
            optuna.integration.PyTorchLightningPruningCallback(trial, monitor="val_total_loss"),
            checkpoint_callback,
        ],
    }

    # Conditionally add gradient clipping arguments
    if gradient_clip_val is not None:
        trainer_args["gradient_clip_val"] = gradient_clip_val
        trainer_args["gradient_clip_algorithm"] = "norm"

    trainer = pl.Trainer(**trainer_args)

    # --- Training ---
    try:
        trainer.fit(model, train_loader, val_loader)
    except Exception as e:
        print(f"Trainer fit failed for trial {trial.number}: {e}")
        # Prune the trial if fitting fails (e.g., due to NaNs)
        raise optuna.exceptions.TrialPruned()

    # --- Evaluation ---
    # CLEAN UP NAMING AT END
    val_total_loss = trainer.callback_metrics.get("val_total_loss")
    if val_total_loss is None:
        val_total_loss = trainer.callback_metrics.get("val_loss", torch.tensor(float('inf'))).item()
    else:
        val_total_loss = val_total_loss.item()

    # --- Test Evaluation ---
    test_results = trainer.test(model, dataloaders=test_loader)
    # Optionally log test results to wandb
    if test_results and isinstance(test_results, list):
        wandb.log({f"{k}": v for k, v in test_results[0].items()})

    final_epoch = trainer.max_epochs

    # --- Loss Landscape Visualization ---
    try:
        # Train loss landscape
        fig3d_train, fig2d_train = visualize_loss_landscape_3d(
            model, loss_fn_instance, train_loader, device=device, wandb_run=wandb.run
        )
        if fig3d_train:
            wandb.log({"3d_landscape_train": wandb.Image(fig3d_train), "epoch": final_epoch})
            print("3D train loss landscape visualization logged to wandb at epoch", final_epoch)
        if fig2d_train:
            wandb.log({"2d_landscape_train": wandb.Image(fig2d_train), "epoch": final_epoch})
            print("2D train loss landscape visualization logged to wandb at epoch", final_epoch)

        # Validation loss landscape
        fig3d_val, fig2d_val = visualize_loss_landscape_3d(
            model, loss_fn_instance, val_loader, device=device, wandb_run=wandb.run
        )
        if fig3d_val:
            wandb.log({"3d_landscape_val": wandb.Image(fig3d_val), "epoch": final_epoch})
            print("3D val loss landscape visualization logged to wandb at epoch", final_epoch)
        if fig2d_val:
            wandb.log({"2d_landscape_val": wandb.Image(fig2d_val), "epoch": final_epoch})
            print("2D val loss landscape visualization logged to wandb at epoch", final_epoch)

        # Test loss landscape
        fig3d_test, fig2d_test = visualize_loss_landscape_3d(
            model, loss_fn_instance, test_loader, device=device, wandb_run=wandb.run
        )
        if fig3d_test:
            wandb.log({"3d_landscape_test": wandb.Image(fig3d_test), "epoch": final_epoch})
            print("3D test loss landscape visualization logged to wandb at epoch", final_epoch)
        if fig2d_test:
            wandb.log({"2d_landscape_test": wandb.Image(fig2d_test), "epoch": final_epoch})
            print("2D test loss landscape visualization logged to wandb at epoch", final_epoch)

    except Exception as e:
        print(f"Loss landscape visualization failed: {e}")


    wandb.log({"final_val_total_loss": val_total_loss})
    wandb.finish()

    best_checkpoint_path = checkpoint_callback.best_model_path
    if best_checkpoint_path is None:
        print("Warning: No best model checkpoint was saved. This might happen if training was pruned early.")
        # Fallback: if no best model was saved, return the path to the last saved checkpoint if it exists
        if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback.last_model_path:
            best_checkpoint_path = trainer.checkpoint_callback.last_model_path
        else:
            best_checkpoint_path = "N/A" # Indicate no path found

    # ==========================================================
    # 🌟 CRITICAL NEW LINE FOR PERSISTENCE AND SYMLINK RETRIEVAL 🌟
    # Store the path in the Optuna trial's metadata
    trial.set_user_attr("best_checkpoint_path", best_checkpoint_path)
    # ==========================================================
    return val_total_loss, best_checkpoint_path
