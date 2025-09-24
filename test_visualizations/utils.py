
import torch
import sys
import os
from tqdm import tqdm

from maren_data.helpers import choose_data_params

sys.path.append("/scratch/nar8991/snn/DarwinNeuron")
from src.RandmanFunctions import RandmanConfig, split_and_load, split_test_and_load


def get_shd_test_data():
    pre_path_data = f"/scratch/nar8991/snn/snn_ann_hybrid/maren_data/shd/700_inputs/1.0_max_time/False_noise"
    settings = {}
    settings["max_time"] = 1
    settings["noise"] = False
    settings["time_step"] = 0.002
    settings["nb_inputs"] = 700
    settings["nb_outputs"] = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings["device"] = device
    settings["dtype"] = torch.float32
    settings["batch_size"] = 256
    settings["percent_data"] = 1.0  # Use all data

    # UPDATING THIS BUT DON'T WANT STUFF TO CRASH
    _, _, test_loader = choose_data_params(
        "shd", settings, num_workers=4, pre_path=pre_path_data
        )
    return test_loader

def get_randman_test_data():
    data_loader_config = {'randman_id': 0, 'randman_dir': '/scratch/nar8991/snn/DarwinNeuron/data/randman'}
    batch_size=516
    randman = RandmanConfig.lookup_by_id(data_loader_config['randman_id'], os.path.join(data_loader_config['randman_dir'], "meta-data.csv"))
    dataset = randman.read_dataset(data_loader_config['randman_dir'])
    test_loader = split_test_and_load(dataset, batch_size=batch_size)
    return test_loader


def compute_accuracy(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.test_accuracy.reset()
    model.to(device)
    model.zero_grad()
    # criterion = nn.CrossEntropyLoss()
    for inputs, targets in tqdm(test_loader, desc="Testing"):
        inputs, targets = inputs.to(device), targets.to(device)
        predictions, _, _, _ = model(inputs)
        logit_output, _ = torch.max(predictions, dim=1)
        model.test_accuracy.update(logit_output, targets)
    test_acc = model.test_accuracy.compute().item()
    return test_acc