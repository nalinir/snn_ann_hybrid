import torch
from shd_dataset import data_split_shd
from models import (
    SNN,
    ANN_with_LIF_output,
    Hybrid_RNN_SNN_rec,
    Hybrid_RNN_SNN_V1_same_layer,
)
from train_model import train_and_val
import json
import wandb
import argparse

function_mappings = {
    "SNN": SNN,
    "ANN_with_LIF_output": ANN_with_LIF_output,
    "Hybrid_RNN_SNN_rec": Hybrid_RNN_SNN_rec,
    "Hybrid_RNN_SNN_V1_same_layer": Hybrid_RNN_SNN_V1_same_layer,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run Wandb sweep for SNN regularization."
    )
    parser.add_argument(
        "--data_config_path",
        type=str,
        default="/scratch/nar8991/snn/snn_ann_hybrid/shd_config.json",
        help="Path to the configuration JSON file.",
    )
    args = parser.parse_args()

    with open(args.data_config_path, "r") as f:
        data_config = json.load(f)

    sweep_config = {
        "method": "grid",
        "name": "SNN Regularization Sweep",
        "metric": {"name": "val_accuracy", "goal": "maximize"},
        "parameters": {
            "l1": {"values": [1e-6, 5e-6, 1e-5]},
            "l2": {"values": [1e-6, 5e-6, 1e-5]},
            "learning_rate": {
                "value": 2e-3  # Keep learning rate constant for this sweep
            },
            "epochs": {"value": 1},
            "regularization": {"value": True},
            "optimizer": {"value": "Adam"},
            "model_name": {"value": "SNN"},
            "recurrent": {"value": True},
        },
    }
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    data_name = "shd"
    train_loader, test_loader, val_loader = data_split_shd(data_config)
    ## Regularization parameterization

    def train_wandb():
        with wandb.init() as run:
            config = wandb.config
            run.name = f"{model_name}-l1_{config['l1']}-l2_{config['l2']}-{data_name}-recurrent_{config['recurrent']}-regularization_{config['regularization']}"
            model = function_mappings[config.model_name]
            train_and_val(
                model, train_loader, val_loader, test_loader, run, data_config, device
            )

    for model_name, model_func in function_mappings.items():

        sweep_config["parameters"]["model_name"]["value"] = model_name
        if model_name != "Hybrid_RNN_SNN_V1_same_layer":
            # Set recurrent step to both True and False for each model (exception is our V1)
            for i in [True, False]:
                sweep_config["parameters"]["recurrent"]["value"] = i
                # Initialize a Wandb sweep for the current model
                sweep_id = wandb.sweep(sweep_config, project="SNN_test_reg_optimize")

                # Run the sweep agent
                wandb.agent(
                    sweep_id,
                    train_wandb,
                    count=9,
                )  # Run all 3 x 3 combinations
        else:
            sweep_config["parameters"]["recurrent"]["value"] = True
            # Initialize a Wandb sweep for the current model
            sweep_id = wandb.sweep(sweep_config, project="SNN_test_reg_optimize_shd")

            # Run the sweep agent
            wandb.agent(
                sweep_id,
                train_wandb,
                count=9,
            )  # Run all 3 x 3 combinations

    ## Later -> Initialization parameterization with the loss function (using data_split_randman)


if __name__ == "__main__":
    main()
