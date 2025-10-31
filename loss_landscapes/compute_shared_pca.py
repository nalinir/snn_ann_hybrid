import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
import os
import argparse
import sys

sys.path.append("/scratch/nar8991/snn/snn_ann_hybrid")
from class_based_implementation.models import SNN, ANN_with_LIF_output, Hybrid_RNN_SNN_rec, Hybrid_RNN_SNN_V1_same_layer, NSN_with_LIF_output, Hybrid_NSN_SNN_rec, Hybrid_NSN_SNN_V1_same_layer
from class_based_implementation.surr_grad import SurrGradSpike
from loss_landscapes.utils import get_shd_train_data, get_randman_train_data, _flatten_params

MODEL_CLASSES = {
    "NSN_with_LIF_output": NSN_with_LIF_output,
    "SNN": SNN,
    "ANN_with_LIF_output": ANN_with_LIF_output,
    "Hybrid_RNN_SNN_rec": Hybrid_RNN_SNN_rec,
    "Hybrid_NSN_SNN_rec": Hybrid_NSN_SNN_rec,
    "Hybrid_RNN_SNN_V1_same_layer": Hybrid_RNN_SNN_V1_same_layer,
    "Hybrid_NSN_SNN_V1_same_layer": Hybrid_NSN_SNN_V1_same_layer,
}

def compute_shared_pca(models):
    flat_params_list = []
    for model in tqdm(models, desc="Flattening parameters"):
        params = list(model.parameters())
        flat_vec, _ = _flatten_params(params)
        flat_params_list.append(flat_vec.cpu().numpy())
    flat_matrix = np.stack(flat_params_list)
    pca = PCA(n_components=2)
    pca.fit(flat_matrix)
    d1 = torch.tensor(pca.components_[0], dtype=torch.float32)
    d2 = torch.tensor(pca.components_[1], dtype=torch.float32)
    print("Shared PCA directions computed.")
    return d1, d2

def get_all_models(data):
    if data == 'randman': # TO DO - Allow for all randman problems eventually 
        BASE_PATH="/vast/nar8991/snn/training_results/randman/1_d/2_classes/3/cross_entropy"
        hidden_neurons = 20
    elif data == 'shd':
        BASE_PATH="/vast/nar8991/snn/training_results/shd/None_d/20_classes/700/cross_entropy"
        hidden_neurons = 256
    
    all_models = []
    for model_name, model_class in MODEL_CLASSES.items():
        percent_list = [0.25, 0.5, 0.75] if 'Hybrid' in model_name else [None]
        for percent in percent_list:
            for seed in range(1, 5):
                base_path = f"{BASE_PATH}/{model_name}/recurrent_True/seed_{seed}/tpe_sampler/{hidden_neurons}_hidden/1.0_pct_data"
                model_path_suffix = f"{percent}_percent_snn/best_model_of_study.ckpt" # if percent else "best_model_of_study.ckpt"
                model_file_path = f"{base_path}/{model_path_suffix}"
                if os.path.exists(model_file_path):
                    model = model_class.load_from_checkpoint(model_file_path)
                    all_models.append(model)
                else:
                    print("Model not found for loading: ", model_file_path)
    return all_models, BASE_PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute and save shared PCA directions.")
    parser.add_argument('--data', type=str, choices=['shd', 'randman'], required=True, help="Dataset to use: 'shd' or 'randman'")
    args = parser.parse_args()
    
    all_models, BASE_PATH = get_all_models(args.data)
    d1, d2 = compute_shared_pca(all_models)
    
    pca_output_dir = os.path.join(BASE_PATH, "pca_directions")
    os.makedirs(pca_output_dir, exist_ok=True)
    torch.save({"d1": d1, "d2": d2}, os.path.join(pca_output_dir, "shared_pca_directions.pt"))
    print(f"PCA directions saved to {os.path.join(pca_output_dir, 'shared_pca_directions.pt')}")

