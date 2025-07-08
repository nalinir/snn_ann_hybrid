# class_based_implementation/train_model.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy
import wandb
import optuna # Make sure optuna is imported
import numpy as np # For pl.seed_everything

# Import your models (assuming these are defined in snn_hybrid_models.py)
from class_based_implementation.snn_hybrid_models import SNN, ANN_with_LIF_output, Hybrid_RNN_SNN_rec, Hybrid_RNN_SNN_V1_same_layer, SurrGradSpike

from old_implementation.loss_landscape import visualize_loss_landscape_3d

# Define your MODEL_CLASSES and LOSS_FUNCTIONS dictionaries here
MODEL_CLASSES = {
    "SNN": SNN,
    "ANN_with_LIF_output": ANN_with_LIF_output,
    "Hybrid_RNN_SNN_rec": Hybrid_RNN_SNN_rec,
    "Hybrid_RNN_SNN_V1_same_layer": Hybrid_RNN_SNN_V1_same_layer,
}

LOSS_FUNCTIONS = {
    "cross_entropy": nn.CrossEntropyLoss(),
    "mse": nn.MSELoss(),
    "nll": nn.NLLLoss(),
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
    sampler_type: str
):
    """
    Objective function for Optuna hyperparameter optimization.
    """
    # --- Trial-Level Seed ---
    # This ensures model initialization and data loading are reproducible for this specific trial.
    # We pass 'sweep_seed' from main.py implicitly via the global state if needed, or explicitly.
    # The 'trial.number' is key for trial-level seeding.
    # Let's assume sweep_seed is implicitly handled by pl.seed_everything(None) or a global variable
    # If you want to explicitly pass sweep_seed, add it to objective's signature.
    
    # For robust trial-level seeding with sweep_seed:
    # Assuming `sweep_seed` is passed to objective, if not, you need to add it to the signature.
    # The `sweep_seed` is now passed through the `objective_wrapper`.
    # Let's adjust objective signature slightly for explicit sweep_seed.
    
    # If the `sweep_seed` is NOT passed to `objective` directly:
    # `pl.seed_everything(trial.number, workers=True)` would ensure individual trial reproducibility
    # regardless of sweep_seed, but the sequence of trial seeds would always be `0, 1, 2...`
    # for each separate sweep.

    # Re-integrating explicit trial_seed based on sweep_seed (passed via outer objective_wrapper)
    # The `sweep_seed` is actually defined in `main.py` and used by `pl.seed_everything` there.
    # For a deterministic *trial-level* seed that is also affected by the *sweep-level* seed,
    # the trial_seed calculation should happen BEFORE this objective is called, or we assume
    # Optuna's internal seeding for trial.suggest_* is enough and `pl.seed_everything(None)` handles the rest.

    # Let's assume `pl.seed_everything(trial_seed)` happens in `objective_wrapper` (as in previous main.py).
    # Then `trial.suggest_*` will be overridden by GridSampler.
    # The `torch.cuda.manual_seed(trial.number)` etc. are more specific than `pl.seed_everything`.
    # It's generally best to let `pl.seed_everything` handle all of them.
    
    # If you want to keep trial-specific manual seeds *inside* objective:
    # The most consistent way is to just use `pl.seed_everything(trial.number)` or derive
    # a unique seed for the trial *within* this objective.
    # Given the previous `objective_wrapper` had the `pl.seed_everything(trial_seed, workers=True)` call,
    # we should rely on that and remove the `torch.cuda.manual_seed` lines here for consistency.
    
    # Let's keep `pl.seed_everything` call in `objective_wrapper` in `main.py`
    # and remove the manual torch seeds from here. This makes the `objective` cleaner.

    # --- Hyperparameter Retrieval from trial.suggest_ functions ---
    # These will define the search space for Optuna
    # alpha = trial.suggest_float("alpha", 0.7, 0.95)
    # beta = trial.suggest_float("beta", 0.7, 0.95)
    
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adamax"])
    # Conditional hyperparameter suggestion based on optimizer
    if optimizer_name == "Adam":
        # learning_rate = trial.suggest_float("adam_lr", 1e-4, 2e-3, log=False)
        learning_rate = trial.suggest_float("adam_lr", 1e-3, 1e-3, log=False)
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
        
    # Zenke regularization specific hyperparameters
    l2_lower = trial.suggest_float("l2_lower", 100, 100, log=False)
    v2_lower = trial.suggest_float("v2_lower", 1e-3, 1e-3, log=False)
    l1_upper = trial.suggest_float("l1_upper", 0.06, 0.06, log=False)
    # l1_upper = trial.suggest_float("l1_upper", 1, 100, log=False)
    # v1_upper = 0.06
    v1_upper = trial.suggest_float("v1_upper", 0, 1000, log=False)
    l2_upper = trial.suggest_categorical("l2_upper", [0])
    # trial.suggest_categorical("l2_upper", [0, 1, data_config["nb_hidden"]])
    v2_upper = trial.suggest_float("v2_upper", 0, 0, log=False)
    # trial.suggest_float("v2_upper", 0, data_config["nb_hidden"], log=False)

    # Store Zenke config in a dict
    zenke_config = {
        "l2_lower": l2_lower,
        "v2_lower": v2_lower,
        "l1_upper": l1_upper,
        "v1_upper": v1_upper,
        "l2_upper": l2_upper,
        "v2_upper": v2_upper,
    }
    spike_grad_scale = trial.suggest_float("spike_grad_scale", 50.0, 50.0, log=False)

    # Initialize wandb for this trial
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
        "hidden_features": data_config["nb_hidden"], # Ensure nb_hidden is in data_config
        "output_features": data_config["nb_outputs"],
        "data_config": data_config,
        # "alpha": alpha,
        # "beta": beta,
        "recurrent": recurrent_setting,
        "learning_rate": learning_rate,
        "loss_fn": loss_fn_instance,
        "zenke_config": zenke_config,
        "optimizer_name": optimizer_name,
        "spike_grad_scale": spike_grad_scale
    }

    if model_name_str in ["SNN", "Hybrid_RNN_SNN_rec", "Hybrid_RNN_SNN_V1_same_layer"]:
        model_args["spike_fn"] = SurrGradSpike.apply # Use SurrGradSpike.apply directly
    elif model_name_str == "ANN_with_LIF_output":
        model_args["spike_fn"] = None # No spikes for hidden layer in ANN

    model = model_class(**model_args)

    model.to(device)
    # After model instantiation, before training
    print("Model expects input_features:", model.input_features)
    print("Model w1 shape:", model.w1.shape)

    # Get a batch from your train_loader to check input shape
    xb, yb = next(iter(train_loader))
    print("Sample batch xb shape:", xb.shape)
    print("Sample batch yb shape:", yb.shape)
    # --- PyTorch Lightning Trainer Setup ---
    trainer = pl.Trainer(
        max_epochs=data_config["epochs"],
        accelerator=str(device.type),
        devices=1,
        logger=pl.loggers.WandbLogger(log_model=True, project=wandb_project_name),
        enable_checkpointing=False,
        callbacks=[
            optuna.integration.PyTorchLightningPruningCallback(trial, monitor="val_total_loss"),
        ],
    )

    # --- Training ---
    try:
        trainer.fit(model, train_loader, val_loader)
    except Exception as e:
        print(f"Trainer fit failed for trial {trial.number}: {e}")
        # Prune the trial if fitting fails (e.g., due to NaNs)
        raise optuna.exceptions.TrialPruned()

    # --- Evaluation ---
    # I DON'T THINK WE NEED THIS, REMOVE?
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

    return val_total_loss