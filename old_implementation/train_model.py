import torch
import numpy as np
import torch.nn as nn
from loss_landscape import visualize_loss_landscape_3d, HybridNet, clustering_metrics_calc
from functools import partial
from surrogate_gradient import SurrGradSpike
from models import (
    SNN,
    ANN_with_LIF_output,
    Hybrid_RNN_SNN_rec,
    Hybrid_RNN_SNN_V1_same_layer,
)
import wandb

function_mappings = {
    # "SNN": SNN,
    # "ANN_with_LIF_output": ANN_with_LIF_output,
    # "Hybrid_RNN_SNN_rec": Hybrid_RNN_SNN_rec,
    "Hybrid_RNN_SNN_V1_same_layer": Hybrid_RNN_SNN_V1_same_layer,
}
# -- Percent Spiking Metrics --
# Sample level
def percent_of_neurons_spiking_per_sample(spk, model):
    # Aggregate spikes for each neuron across the batch
    # spk should be of shape (T, batch, N)
    print(f"spk shape: {spk.shape}")
    spk_sum = torch.sum(spk, dim=0)  # Shape: (batch, N)
    print(f"spk_sum shape: {spk_sum.shape}")
    spk_count_greater_than_zero = (spk_sum > 0).float()  # Shape: (batch, N)
    spk_sum_greater_0 = torch.mean(spk_count_greater_than_zero, dim=1)  # Shape: (batch,)
    print(f"spk_sum_greater_0 shape: {spk_sum_greater_0.shape}")
    if model == Hybrid_RNN_SNN_rec or model == Hybrid_RNN_SNN_V1_same_layer:
        # For hybrid models, multiply by 2 since half will not be spiking
        spk_sum_greater_0 *= 2
    return spk_sum_greater_0

# --- Attention Mechanism Loss ---
def parameter_free_attention(mem, n):
    # Input current should be the membrane potential for a given neuron
    # Threshold is 1 for this model (just don't do mthr)
    # Other should be 0


    # The membrane potential for each timestep has shape (batch, nb_hidden)
    first_part = (1-mem)**2
    # This is just the resulting membrane potential from all other neurons other than a given neuron
    second_part_interim = (0-mem)**2
    total_sum_mem = torch.sum(second_part_interim)
    second_part = (total_sum_mem - second_part_interim)/(n-1)
    # We take all except a given index and then take the mean
    return second_part + first_part

def attention_loss(mem, w1, n, config):
    """
    Computes the attention loss with L2 regularization.
    """
    # mem should be of shape (batch, nb_hidden)
    # n is the number of neurons in the layer
    ## To allow for L2 regularization for model
    if mem is None:
        attn_loss = 0
    else:
        attn_loss = torch.sum(parameter_free_attention(mem, n))
    l2_loss = torch.sum(config["l2"] * (w1**2))
    return attn_loss + l2_loss

# --- Utility Functions for Spike & Voltage Metrics ---
def neurons_spiking_per_timestep(spk_tensor):
    """
    Given spk_tensor of shape (T, batch, N), returns array of length T with number of neurons spiking per timestep.
    """
    spk_np = spk_tensor.detach().cpu().numpy()
    return spk_np.sum(axis=(1, 2))


def total_spike_count(spk_tensor):
    """
    Total spikes across all timesteps, batch, and neurons."""
    return int(torch.sum(spk_tensor).item())


# --- Regularization Modules ---
def bound_regularizer(spk, v_t, l_t, l1, upper_bound=True, population_level=True):
    multiplier = 1 if upper_bound else -1
    cnt = torch.sum(spk, dim=0)
    if population_level:
        cnt = torch.mean(spk, dim=0)
    reg = torch.relu(multiplier * (cnt - v_t))
    return l_t * (torch.mean(torch.abs(reg)) if l1 else torch.mean(torch.square(reg)))
    


def regularization_loss_zenke(spks, config):
    lower_l2 = bound_regularizer(
        spks,
        config["v2_lower"],
        config["l2_lower"],
        l1=False,
        upper_bound=False,
        population_level=False,
    )
    upper_l1 = bound_regularizer(spks, config["v1_upper"], config["l1_upper"], l1=True)
    upper_l2 = bound_regularizer(spks, config["v2_upper"], config["l2_upper"], l1=False)
    return lower_l2 + upper_l1 + upper_l2


def regularization_loss_original(spks, config):
    loss = config["l1"] * torch.sum(spks)
    loss += config["l2"] * torch.mean(torch.sum(torch.sum(spks, dim=0), dim=0) ** 2)
    return loss


# --- Core Training/Evaluation Functions ---
def run_epoch(
    model,
    data_loader,
    weights,
    device,
    config,
    data_config,
    mode="train",
    optimizer=None,
    save_spikes=False,
    save_voltages=False,
):
    w1, w2, v1 = weights
    alpha = float(np.exp(-data_config["time_step"] / data_config["tau_syn"]))
    beta = float(np.exp(-data_config["time_step"] / data_config["tau_mem"]))
    spike_fn = SurrGradSpike.apply
    log_softmax = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()

    total_loss = 0.0
    all_preds, all_labels = [], []
    per_timestep_spike_counts = []
    total_spikes = 0
    spike_tensors = [] if save_spikes else None
    voltage_tensors = [] if save_voltages else None
    percent_spiking = []

    snn_mask = torch.zeros((data_config["nb_hidden"],), device=device)
    snn_mask[: data_config["nb_hidden"] // 2] = 1.0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        output, recs = model(
            x, w1, w2, v1, alpha, beta, spike_fn, device, config["recurrent"], snn_mask
        )

        if model == ANN_with_LIF_output:
            volt, spks = None, None
        elif model == SNN:
            volt, spks = recs[0], recs[1]
        else:
            volt, spks = recs[0], recs[1]

        # compute loss
        m, _ = torch.max(output, dim=1)
        logp = log_softmax(m)
        loss = loss_fn(logp, y)
        # This is only spike regularization
        print(config["regularization"], config["zenke_actual"])
        if config["regularization"] == True and model != ANN_with_LIF_output:
            bin_spks = (spks > 0).float()
            loss += (
                regularization_loss_zenke(bin_spks, config)
                if config["zenke_actual"] == True
                else regularization_loss_original(bin_spks, config)
            )
            # track metrics
        if config["parameter_free_attention"]:
            loss += attention_loss(
                volt, w1, data_config["nb_hidden"], config
            )
        if save_spikes and spks is not None:
            spike_tensors.append(spks.detach().cpu())
        if spks is not None:
            per_timestep_spike_counts.append(neurons_spiking_per_timestep(spks))
            total_spikes += total_spike_count(spks)
            percent_spiking.append(percent_of_neurons_spiking_per_sample(spks, model))

        if save_voltages and volt is not None:
            voltage_tensors.append(volt.detach().cpu())

        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, pred = torch.max(m, 1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    
    # Stack all the tensors in the list of shape (batch,)
    percent_spiking = torch.cat(percent_spiking) if percent_spiking else []
    avg_loss = total_loss / len(data_loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    if spks is not None:
        per_timestep_spike_counts = np.concatenate(per_timestep_spike_counts)
        avg_percent_spiking = torch.mean(percent_spiking)
    else:
        avg_percent_spiking = 0.0
    return (
        avg_loss,
        accuracy,
        per_timestep_spike_counts,
        total_spikes,
        spike_tensors,
        voltage_tensors,
        avg_percent_spiking
    )


def train_and_evaluate(
    model,
    train_loader,
    val_loader,
    test_loader,
    config,
    data_config,
    device,
    wandb_run=None,
):
    def init_weight(shape):
        t = torch.empty(shape, device=device, requires_grad=True)
        nn.init.normal_(t, mean=0.0, std=1.0)
        return t

    w1 = init_weight((data_config["nb_inputs"], data_config["nb_hidden"]))
    w2 = init_weight((data_config["nb_hidden"], data_config["nb_outputs"]))
    v1 = init_weight((data_config["nb_hidden"], data_config["nb_hidden"]))

    history = {
        split: {"loss": [], "acc": [], "spikes_per_t": [], "total_spikes": [], "avg_percent_spiking": []}
        for split in ["train", "val"]
    }
    history["test"] = {
        **history["train"],
        "spike_tensors": None,
        "voltage_tensors": None,
    }
    optimizer = torch.optim.Adam([w1, w2, v1], lr=config["learning_rate"])

    for epoch in range(config["epochs"]):
        t_metrics = run_epoch(
            model,
            train_loader,
            (w1, w2, v1),
            device,
            config,
            data_config,
            mode="train",
            optimizer=optimizer,
        )
        v_metrics = run_epoch(
            model, val_loader, (w1, w2, v1), device, config, data_config, mode="eval"
        )

        for split, metrics in zip(["train", "val"], [t_metrics, v_metrics]):
            history[split]["loss"].append(metrics[0])
            history[split]["acc"].append(metrics[1])
            history[split]["spikes_per_t"].append(metrics[2])
            history[split]["total_spikes"].append(metrics[3])
            history[split]["avg_percent_spiking"].append(metrics[6])
            if wandb_run:
                wandb_run.log(
                    {
                        f"{split}_loss": metrics[0],
                        f"{split}_acc": metrics[1],
                        f"{split}_spikes_per_t": np.mean(metrics[2]),
                        f"{split}_total_spikes": metrics[3],
                        f"{split}_avg_percent_spiking": metrics[6],
                    }
                )

        torch.cuda.empty_cache()

    # Final Test
    (
        test_loss,
        test_acc,
        test_spikes_per_t,
        test_total_spikes,
        test_spike_tensors,
        test_voltage_tensors,
        avg_percent_spiking
    ) = run_epoch(
        model,
        test_loader,
        (w1, w2, v1),
        device,
        config,
        data_config,
        mode="eval",
        save_spikes=True,
        save_voltages=True,
    )
    history["test"]["loss"] = test_loss
    history["test"]["acc"] = test_acc
    history["test"]["spikes_per_t"] = test_spikes_per_t
    history["test"]["total_spikes"] = test_total_spikes
    history["test"]["spike_tensors"] = test_spike_tensors
    history["test"]["voltage_tensors"] = test_voltage_tensors
    history["test"]["avg_percent_spiking"] = avg_percent_spiking

    print(
        f"Train Acc: {history['train']['acc'][-1]*100:.2f}%, Val Acc: {history['val']['acc'][-1]*100:.2f}%, Test Acc: {history['test']['acc']*100:.2f}%"
    )
    if wandb_run:
        wandb_run.log(
            {
                "final_train_acc": history["train"]["acc"][-1],
                "final_val_acc": history["val"]["acc"][-1],
                "final_test_acc": history["test"]["acc"],
                "final_test_avg_percent_spiking": history["test"]["avg_percent_spiking"],
            }
        )
    snn_mask = torch.zeros((data_config["nb_hidden"],), device=device)
    snn_mask[: data_config["nb_hidden"] // 2] = 1.0

    base_forward = model
    bound_forward = partial(
        base_forward,
        w1=w1,
        w2=w2,
        v1=v1,
        alpha=float(np.exp(-data_config["time_step"] / data_config["tau_syn"])),
        beta=float(np.exp(-data_config["time_step"] / data_config["tau_mem"])),
        spike_fn=SurrGradSpike.apply,
        device=device,
        recurrent=config['recurrent'],
        snn_mask=snn_mask,
    )
    model_wrapper = HybridNet(
        base_forward,
        w1=w1,
        w2=w2,
        v1=v1,
        alpha=float(np.exp(-data_config["time_step"] / data_config["tau_syn"])),
        beta=float(np.exp(-data_config["time_step"] / data_config["tau_mem"])),
        spike_fn=SurrGradSpike.apply,
        device=device,
        recurrent=config['recurrent'],
        snn_mask=snn_mask,
        ).to(device)
    fig3d, fig2d = visualize_loss_landscape_3d(
        model=model_wrapper,
        criterion=nn.NLLLoss(),
        dataloader=test_loader,
        resolution=31,
        device=device,
        wandb_run=wandb_run,
    )
    _, _, coefficient_metrics = clustering_metrics_calc(
        model=model_wrapper,
        data_config=data_config,
        dataloader=test_loader,
        wandb_run=wandb_run,
    )
    
    # Add coefficient metrics to history
    history["clustering_coefficients"] = coefficient_metrics

    # Save the visualizations to wandb

    return history, w1, w2, v1, fig2d, fig3d
    # ,hidden_layer_clustering


def objective(
    trial,
    model_name,
    data_config,
    device,
    train_loader,
    val_loader,
    test_loader,
    recurrent_setting,
    project_name,
    loss_type
):
    # Use original distribution styles from the WandB sweep config
    config = {}
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    config["optimizer"] = optimizer
    config = {
        "model_name": model_name,
        "l2_lower": data_config["nb_hidden"],
        "v2_lower": 1e-2,
        "l1_upper": trial.suggest_int("l1_upper", 1, data_config["nb_hidden"]),
        "v1_upper": trial.suggest_int(
            "v1_upper", 0, data_config["nb_hidden"] * data_config["nb_outputs"]
        ),
        "l2_upper": trial.suggest_categorical(
            "l2_upper", [0, 1, data_config["nb_hidden"]]
        ),
        "v2_upper": trial.suggest_int("v2_upper", 0, data_config["nb_hidden"]),
        "learning_rate": (
            trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True) 
            if optimizer == "Adam" 
            else trial.suggest_float("learning_rate", 1e-2, 0.5, log=True)
        ),
        "epochs": data_config["epochs"],
        "optimizer": optimizer,
        "regularization": True,
        "momentum": trial.suggest_float("momentum", 0.0, 0.99) if config["optimizer"] == "SGD" else 0.0,
        "recurrent": recurrent_setting,  # <- use the fixed value
        "zenke_actual": True,
        "parameter_free_attention": False
    }
    if loss_type == "attention":
        config["parameter_free_attention"] = True
        config["regularization"] = False # No spike regularization
        # Drop all the l2, v2, l1, v1 upper/lower bounds
        config["l2_lower"] = 0
        config["v2_lower"] = 0
        config["l1_upper"] = 0
        config["v1_upper"] = 0
        # Add a new l2 search parameter
        config["l2"] = trial.suggest_float("l2", 1e-6, 1e-1, log=True)
    # Actual is default, original is not included for now

    run_name = (
        f"{model_name}-recurrent_{config['recurrent']}-"
        f"l2_lower_{config['l2_lower']}-v2_lower_{config['v2_lower']}-"
        f"l1_upper_{config['l1_upper']}-v1_upper_{config['v1_upper']}-"
        f"l2_upper_{config['l2_upper']}-v2_upper_{config['v2_upper']}-"
        f"regularization_{config['regularization']}"
    )

    with wandb.init(project=project_name, config=config, name=run_name):
        model_class = function_mappings[model_name]
        history, w1, w2, v1, fig2d, fig3d = train_and_evaluate(
            model_class,
            train_loader,
            val_loader,
            test_loader,
            config,
            data_config,
            device,
            wandb_run=wandb.run,
        )

    final_val_acc = history["val"]["acc"][-1]

    # Save history if it's better
    trial.set_user_attr(
        "history", history
    )  # store the history inside the trial for later if needed
    trial.set_user_attr(
        "weights", (w1, w2, v1)
    )  # store the weights inside the trial for later if needed
    trial.set_user_attr(
        "3d_landscape", fig3d
    )  # store the model name inside the trial for later if needed
    # trial.set_user_attr(
    #     "hidden_layer_clustering", hidden_layer_clustering
    # )  # store the model name inside the trial for later if needed
    return final_val_acc
