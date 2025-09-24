import os
import torch
import numpy as np
import sys

sys.path.append("/scratch/nar8991/snn/snn_ann_hybrid")
from maren_data.helpers import choose_data_params

sys.path.append("/scratch/nar8991/snn/DarwinNeuron")
from src.RandmanFunctions import RandmanConfig, split_and_load, split_test_and_load


def get_shd_train_data():
    pre_path_data = f"/scratch/nar8991/snn/snn_ann_hybrid/maren_data/shd/700_inputs/1.0_max_time/False_noise"
    settings = {"max_time": 1, "noise": False, "time_step": 0.002, "nb_inputs": 700, "nb_outputs": 20, "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.float32, "batch_size": 256, "percent_data": 1.0}
    train_loader, _, _ = choose_data_params("shd", settings, num_workers=4, pre_path=pre_path_data)
    return train_loader

def get_randman_train_data():
    data_loader_config = {'randman_id': 0, 'randman_dir': '/scratch/nar8991/snn/DarwinNeuron/data/randman'}
    batch_size=516
    randman = RandmanConfig.lookup_by_id(data_loader_config['randman_id'], os.path.join(data_loader_config['randman_dir'], "meta-data.csv"))
    dataset = randman.read_dataset(data_loader_config['randman_dir'])
    train_loader, _ = split_and_load(dataset, batch_size=batch_size)
    return train_loader

def _flatten_params(params):
    vecs = [p.detach().reshape(-1) for p in params]
    shapes = [p.shape for p in params]
    return torch.cat(vecs), shapes

def _unflatten_params(vec, shapes):
    tensors = []
    idx = 0
    for shape in shapes:
        numel = np.prod(shape)
        tensors.append(vec[idx:idx+numel].reshape(shape))
        idx += numel
    return tensors

def _assign_flat_params(params, flat_vec, shapes, device=None):
    tensors = _unflatten_params(flat_vec, shapes)
    for p, t in zip(params, tensors):
        t = t.to(p.device if device is None else device)
        p.data.copy_(t)
