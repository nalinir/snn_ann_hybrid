import torch
import json
import argparse
import wandb
import optuna

from randman_dataset import data_split_randman
from models import (
    SNN,
    ANN_with_LIF_output,
    Hybrid_RNN_SNN_rec,
    Hybrid_RNN_SNN_V1_same_layer,
)
from train_model import train_and_val

function_mappings = {
    "SNN": SNN,
    "ANN_with_LIF_output": ANN_with_LIF_output,
    "Hybrid_RNN_SNN_rec": Hybrid_RNN_SNN_rec,
    "Hybrid_RNN_SNN_V1_same_layer": Hybrid_RNN_SNN_V1_same_layer,
}

def objective(trial, model_name, data_config, device, train_loader, val_loader, test_loader, recurrent_setting):
    # Use original distribution styles from the WandB sweep config
    config = {
        "model_name": model_name,
        "l2_lower": data_config["nb_hidden"],
        "v2_lower": 1e-2,
        "l1_upper": trial.suggest_int("l1_upper", 1, data_config["nb_hidden"]),
        "v1_upper": trial.suggest_int("v1_upper", 0, data_config["nb_hidden"] * data_config["nb_outputs"]),
        "l2_upper": trial.suggest_categorical("l2_upper", [0, 1, data_config["nb_hidden"]]),
        "v2_upper": trial.suggest_int("v2_upper", 0, data_config["nb_hidden"]),
        "learning_rate": 2e-3,
        "epochs": 150,
        "regularization": True,
        "optimizer": trial.suggest_categorical("optimizer", ["Adam"]),
        "recurrent": recurrent_setting,  # <- use the fixed value
        "zenke_actual": True,
    }

    run_name = (
        f"{model_name}-recurrent_{config['recurrent']}-"
        f"l2_lower_{config['l2_lower']}-v2_lower_{config['v2_lower']}-"
        f"l1_upper_{config['l1_upper']}-v1_upper_{config['v1_upper']}-"
        f"l2_upper_{config['l2_upper']}-v2_upper_{config['v2_upper']}-"
        f"regularization_{config['regularization']}"
    )

    with wandb.init(project="SNN_Optuna_Tuning", config=config, name=run_name):
        model_class = function_mappings[model_name]
        result = train_and_val(
            model_class,
            train_loader,
            val_loader,
            test_loader,
            wandb.run,
            data_config,
            device,
            trial,
            config
        )
    return result
    
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
    for model_name in function_mappings:
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
                    recurrent_setting  # pass fixed recurrent setting
                ),
                n_trials=20,
            )

            print(f"Best trial for {model_name} (recurrent={recurrent_setting}):")
            print(study.best_trial)

if __name__ == "__main__":
    main()
