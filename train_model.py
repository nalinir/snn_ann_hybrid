import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from surrogate_gradient import SurrGradSpike
from models import (
    SNN,
    ANN_with_LIF_output,
    Hybrid_RNN_SNN_rec,
    Hybrid_RNN_SNN_V1_same_layer,
)
import optuna
import wandb
import pickle
import os

function_mappings = {
    "SNN": SNN,
    "ANN_with_LIF_output": ANN_with_LIF_output,
    "Hybrid_RNN_SNN_rec": Hybrid_RNN_SNN_rec,
    "Hybrid_RNN_SNN_V1_same_layer": Hybrid_RNN_SNN_V1_same_layer,
}


# --- Utility Functions for Spike & Voltage Metrics ---
def neurons_spiking_per_timestep(spk_tensor):
    """
    Given spk_tensor of shape (T, batch, N), returns array of length T with number of neurons spiking per timestep."""
    spk_np = spk_tensor.detach().cpu().numpy()
    return spk_np.sum(axis=(1, 2))


def total_spike_count(spk_tensor):
    """
    Total spikes across all timesteps, batch, and neurons."""
    return int(torch.sum(spk_tensor).item())


# --- Regularization Modules ---
def bound_regularizer(spk, v_t, l, l1, upper_bound=True, population_level=True):
    multiplier = 1 if upper_bound else -1
    cnt = torch.sum(spk, dim=0)
    if population_level:
        cnt = torch.mean(spk, dim=0)
    reg = torch.relu(multiplier * (cnt - v_t))
    return l * (torch.mean(torch.abs(reg)) if l1 else torch.mean(torch.square(reg)))


def regularization_loss_zenke(spks, config):
    lower_l2 = bound_regularizer(spks, config['v2_lower'], config['l2_lower'], l1=False, upper_bound=False, population_level=False)
    upper_l1 = bound_regularizer(spks, config['v1_upper'], config['l1_upper'], l1=True)
    upper_l2 = bound_regularizer(spks, config['v2_upper'], config['l2_upper'], l1=False)
    return lower_l2 + upper_l1 + upper_l2


def regularization_loss_original(spks, config):
    loss = config['l1'] * torch.sum(spks)
    loss += config['l2'] * torch.mean(torch.sum(torch.sum(spks, dim=0), dim=0) ** 2)
    return loss


# --- Core Training/Evaluation Functions ---
def run_epoch(model, data_loader, weights, device, config, data_config, mode='train', optimizer=None):
    """
    Runs one epoch. Returns:
      avg_loss, accuracy,
      per_timestep_spikes (np.array), total_spikes (int),
      voltage_tensors (list of recs[0]), spike_tensors (list of recs[1]), recurrent_tensors (list of recs[3]).
    """
    w1, w2, v1 = weights
    alpha = float(np.exp(-data_config['time_step'] / data_config['tau_syn']))
    beta = float(np.exp(-data_config['time_step'] / data_config['tau_mem']))
    spike_fn = SurrGradSpike.apply
    log_softmax = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()

    # if mode == 'train':
    #     model.train()
        # optimizer = torch.optim.Adam(weights, lr=config['learning_rate'])
    # else:
    #     model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []
    per_timestep_spike_counts = []
    total_spikes = 0
    voltage_tensors, spike_tensors, recurrent_tensors = [], [], []

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        output, recs = model(x, w1, w2, v1, alpha, beta, spike_fn, device, config['recurrent'], None)
        if model == ANN_with_LIF_output:
            rec_out = recs[0]
            volt = None
            spks = None
        elif model == SNN:
            volt, spks = recs[0], recs[1]
            rec_out = None
        else:
            volt, spks = recs[0], recs[1], recs[2]

        # compute loss
        m, _ = torch.max(output, dim=1)
        logp = log_softmax(m)
        loss = loss_fn(logp, y)
        if config.get('regularization', False):
            bin_spks = (spks > 0).float()
            loss += (regularization_loss_zenke(bin_spks, config)
                     if config.get('zenke_actual', False)
                     else regularization_loss_original(bin_spks, config))

        # track metrics
        per_timestep_spike_counts.append(neurons_spiking_per_timestep(spks))
        total_spikes += total_spike_count(spks)
        if volt != None:
            voltage_tensors.append(volt.detach().cpu())
        if spks != None:
            spike_tensors.append(spks.detach().cpu())
        if rec_out != None:
            recurrent_tensors.append(rec_out.detach().cpu())

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, pred = torch.max(m, 1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    per_timestep_spike_counts = np.concatenate(per_timestep_spike_counts)

    return avg_loss, accuracy, per_timestep_spike_counts, total_spikes, voltage_tensors, spike_tensors, recurrent_tensors


def train_and_evaluate(model, train_loader, val_loader, test_loader, config, data_config, device, wandb_run=None):
    # weight init
    def init_weight(shape):
        t = torch.empty(shape, device=device, requires_grad=True)
        nn.init.normal_(t, mean=0.0, std=1.0)
        return t

    w1 = init_weight((data_config['nb_inputs'], data_config['nb_hidden']))
    w2 = init_weight((data_config['nb_hidden'], data_config['nb_outputs']))
    v1 = init_weight((data_config['nb_hidden'], data_config['nb_hidden']))

    history = {k: [] for k in ['train', 'val']}
    for split in ['train', 'val']:
        history[split] = {
            'loss': [], 'acc': [], 'spikes_per_t': [], 'total_spikes': [],
            'voltage_tensors': [], 'spike_tensors': [], 'recurrent_tensors': []
        }
    history['test'] = {**history['train']}
    optimizer = torch.optim.Adam([w1, w2, v1], lr=config['learning_rate'])

    for epoch in range(config['epochs']):
        t_metrics = run_epoch(model, train_loader, (w1, w2, v1), device, config, data_config, mode='train', optimizer=optimizer)
        v_metrics = run_epoch(model, val_loader,   (w1, w2, v1), device, config, data_config, mode='eval')

        for split, metrics in zip(['train','val'], [t_metrics, v_metrics]):
            history[split]['loss'].append(metrics[0])
            history[split]['acc'].append(metrics[1])
            history[split]['spikes_per_t'].append(metrics[2])
            history[split]['total_spikes'].append(metrics[3])
            history[split]['voltage_tensors'].append(metrics[4])
            history[split]['spike_tensors'].append(metrics[5])
            history[split]['recurrent_tensors'].append(metrics[6])
            if wandb_run:
                wandb_run.log({
                    f"{split}_loss": metrics[0],
                    f"{split}_acc": metrics[1],
                    f"{split}_spikes_per_t": metrics[2],
                    f"{split}_total_spikes": metrics[3],
                })

    # test
    test_metrics = run_epoch(model, test_loader, (w1, w2, v1), device, config, data_config, mode='eval')
    for key, val in zip(['loss','acc','spikes_per_t','total_spikes','voltage_tensors','spike_tensors','recurrent_tensors'], test_metrics):
        history['test'][key] = val

    # if wandb_run:
    #     # Save full history as a pickle file
    #     save_path = "history.pkl"
    #     with open(save_path, "wb") as f:
    #         pickle.dump(history, f)

    #     # Create W&B artifact and log the file
    #     artifact = wandb.Artifact("full_training_history", type="results")
    #     artifact.add_file(save_path)
    #     wandb_run.log_artifact(artifact)

    #     # Optional: cleanup local file
    #     os.remove(save_path)

    print(f"Train Acc: {history['train']['acc'][-1]*100:.2f}%, Val Acc: {history['val']['acc'][-1]*100:.2f}%, Test Acc: {history['test']['acc']*100:.2f}%")
    wandb_run.log({
        "final_train_acc": history['train']['acc'][-1],
        "final_val_acc": history['val']['acc'][-1],
        "final_test_acc": history['test']['acc'],
    })

    return history

def objective(trial, model_name, data_config, device, train_loader, val_loader, test_loader, recurrent_setting, project_name):
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
        "epochs": 300,
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

    with wandb.init(project=project_name, config=config, name=run_name):
        model_class = function_mappings[model_name]
        history = train_and_evaluate(
            model_class,
            train_loader,
            val_loader,
            test_loader,
            config,
            data_config,
            device,
            wandb_run=wandb.run,
        )

    final_val_acc = history['val']['acc'][-1]

    # Save history if it's better
    trial.set_user_attr("history", history)  # store the history inside the trial for later if needed

    return final_val_acc
