import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import sys
import argparse
import os
import pandas as pd
import sys

sys.path.append("/scratch/nar8991/snn/snn_ann_hybrid")
from class_based_implementation.models import SNN, ANN_with_LIF_output, Hybrid_RNN_SNN_rec, Hybrid_RNN_SNN_V1_same_layer, NSN_with_LIF_output, Hybrid_NSN_SNN_rec, Hybrid_NSN_SNN_V1_same_layer
from class_based_implementation.surr_grad import SurrGradSpike
from maren_data.helpers import choose_data_params
sys.path.append("/scratch/nar8991/snn/DarwinNeuron")
from src.RandmanFunctions import RandmanConfig, split_and_load, split_test_and_load
from loss_landscapes.utils import get_shd_train_data, get_randman_data


def compute_sam_sharpness(model, dataloader, rho=0.05, device='cuda'):
    # ... (same as your original function) ...
    model.to(device)
    model.zero_grad()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        predictions, _, _, _ = model(inputs)
        logit_output, _ = torch.max(predictions, dim=1)
        loss = criterion(logit_output, targets)
        loss.backward()
        total_loss += loss.item()
    grad = torch.cat([p.grad.flatten().to(device) for p in model.parameters() if p.grad is not None])
    grad_norm = torch.norm(grad)
    if grad_norm == 0:
        return 0.0, 0.0
    normalized_grad = grad / grad_norm
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None:
                continue
            numel = p.numel()
            epsilon = rho * normalized_grad[offset:offset+numel].view_as(p)
            p.add_(epsilon)
            offset += numel
    sam_total_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        predictions, _, _, _ = model(inputs)
        logit_output, _ = torch.max(predictions, dim=1)
        sam_loss = criterion(logit_output, targets)
        sam_total_loss += sam_loss.item()
    avg_sam = sam_total_loss / len(dataloader)
    diff_sam = (sam_total_loss - total_loss) / len(dataloader)
    return avg_sam, diff_sam

MODEL_CLASSES = {
    "NSN_with_LIF_output": NSN_with_LIF_output, "SNN": SNN, "ANN_with_LIF_output": ANN_with_LIF_output,
    "Hybrid_RNN_SNN_rec": Hybrid_RNN_SNN_rec, "Hybrid_NSN_SNN_rec": Hybrid_NSN_SNN_rec,
    "Hybrid_RNN_SNN_V1_same_layer": Hybrid_RNN_SNN_V1_same_layer, "Hybrid_NSN_SNN_V1_same_layer": Hybrid_NSN_SNN_V1_same_layer,
}

def calculate_single_sam_sharpness(data, model_name, recurrent, percent, seed, rho):
    if data == 'randman':
        BASE_PATH = "/scratch/nar8991/snn/snn_ann_hybrid/optuna_results/randman/1_d/2_classes/3/cross_entropy"
        train_loader = get_randman_data()
        hidden_neurons = 20
    elif data == 'shd':
        BASE_PATH = "/scratch/nar8991/snn/snn_ann_hybrid/optuna_results/shd/None_d/20_classes/700/cross_entropy"
        train_loader = get_shd_train_data()
        hidden_neurons = 256
    
    model_class = MODEL_CLASSES.get(model_name)
    if not model_class:
        print(f"Model class for {model_name} not found.")
        return

    if percent is not None:
        model_path_suffix = f"{percent}_percent_snn/best_model_of_study.ckpt"
    else:
        model_path_suffix = "best_model_of_study.ckpt"
    
    base_path = f"{BASE_PATH}/{model_name}/recurrent_{recurrent}/seed_{seed}/grid_sampler/{hidden_neurons}_hidden/1.0_pct_data"
    model_file_path = f"{base_path}/{model_path_suffix}"

    if not os.path.exists(model_file_path):
        print(f"Model file not found: {model_file_path}")
        return

    model = model_class.load_from_checkpoint(model_file_path)
    avg_sam, diff_sam = compute_sam_sharpness(model, train_loader, rho=rho)
    
    result_data = {
        "data": [data], "model_name": [model_name], "recurrent": [recurrent],
        "percent": [percent], "seed": [seed], "rho": [rho],
        "avg_sam": [avg_sam], "diff_sam": [diff_sam]
    }
    df = pd.DataFrame(result_data)
    
    output_dir = "/scratch/nar8991/snn/output/sam_results_csv_final"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"sam_result_{os.environ['SLURM_ARRAY_TASK_ID']}.csv")
    if os.path.exists(output_file):
        df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        df.to_csv(output_file, index=False)
    
    print(f"Results saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate SAM sharpness for a single model.")
    parser.add_argument('--data', type=str, choices=['shd', 'randman'], required=True, help="Dataset to use: 'shd' or 'randman'")
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--recurrent', type=str, required=True)
    parser.add_argument('--percent', type=str, default="None")
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--rho', type=float, required=True)
    args = parser.parse_args()
    if args.percent == "None":
        args.percent = None
    else:
        args.percent = float(args.percent)
    calculate_single_sam_sharpness(args.data, args.model_name, args.recurrent, args.percent, args.seed, args.rho)