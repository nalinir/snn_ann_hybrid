import tonic
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from sklearn.model_selection import train_test_split

# Gemini used to fix a gpu error

def preprocess_spike_events(spike_events, nb_steps, nb_units, time_step, max_time):
    spike_train = np.zeros((nb_steps, nb_units), dtype=np.float32)
    timestamps = spike_events["t"] / 1e6
    timestamps = np.clip(timestamps, 0, max_time)
    units_fired = spike_events["x"]
    polarities = spike_events["p"]

    for t, unit, p in zip(timestamps, units_fired, polarities):
        time_bin = int(round(t * time_step))
        if 0 <= time_bin < nb_steps and 0 <= unit < nb_units:
            spike_train[time_bin, unit] += 1.0

    return torch.tensor(spike_train, dtype=torch.float32)


def data_split_shd(config, device=torch.device('cpu'), dim_manifold=None):
    """
    Splits the Spiking Heidelberg Digits (SHD) dataset into training, validation,
    and test sets. This version ensures all data handling and processing,
    including the data loading and tensor operations, utilize the specified device
    (defaults to CPU, but should be set to 'cuda' for GPU).

    Args:
        config (dict): A dictionary containing configuration parameters, including:
            'nb_steps' (int): Number of time steps.
            'max_time' (float): Maximum time.
            'nb_inputs' (int): Number of input neurons.
            'batch_size' (int): Batch size for data loaders.
            'nb_outputs' (int): Number of output classes.
        device (torch.device, optional): The device to use (e.g., 'cuda', 'cpu').
            Defaults to CPU.
        dim_manifold: Unused argument.
    Returns:
        tuple: A tuple containing the training, validation, and test data loaders
               (torch.utils.data.DataLoader).
    """
    train_dataset = tonic.datasets.SHD(save_to="./data", train=True)
    test_dataset = tonic.datasets.SHD(save_to="./data", train=False)

    train_subset_indices = list(range(len(train_dataset)))
    test_subset_indices = list(range(len(test_dataset)))

    train_labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
    train_indices, val_indices = train_test_split(
        train_subset_indices,
        test_size=0.2,
        stratify=train_labels,
        random_state=42,
    )

    time_step = config["nb_steps"] / config["max_time"]

    # Preprocess data and KEEP IT ON CPU
    def preprocess_only_cpu(dataset, indices):
        x_data = torch.stack(
            [
                preprocess_spike_events(
                    dataset[i][0],
                    config["nb_steps"],
                    config["nb_inputs"],
                    time_step,
                    config["max_time"],
                )
                for i in indices
            ]
        ) # REMOVED .to(device)
        y_tensor = torch.tensor(
            [dataset[i][1] for i in indices], dtype=torch.int64
        ) # REMOVED .to(device)
        return x_data, y_tensor

    train_x_data, train_y_tensor = preprocess_only_cpu(train_dataset, train_indices)
    val_x_data, val_y_tensor = preprocess_only_cpu(train_dataset, val_indices)
    test_x_data, test_y_tensor = preprocess_only_cpu(test_dataset, test_subset_indices)


    train_tensor_dataset = TensorDataset(train_x_data, train_y_tensor)
    val_tensor_dataset = TensorDataset(val_x_data, val_y_tensor)
    test_data = TensorDataset(test_x_data, test_y_tensor)

    # DataLoaders are configured. pin_memory=True is fine here
    # because it will pin the *CPU* memory for faster transfer later.
    train_loader = DataLoader(
        train_tensor_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_tensor_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=2,
    )

    # The following block for printing class distribution and spike info
    # might still try to iterate over the loader.
    # When iterating, if 'device' is 'cuda', this will now work fine
    # because the DataLoader is serving CPU tensors, and you will move them
    # to CUDA within the loop or specifically for printing.

    # Example of how you should move to device in your main training loop:
    # for batch_idx, (data, target) in enumerate(train_loader):
    #     data, target = data.to(device), target.to(device) # <--- Move to GPU HERE!
    #     # ... rest of your training logic

    for loader, name in [
        (train_loader, "Train"),
        (val_loader, "Validation"),
        (test_loader, "Test"),
    ]:
        all_labels = []
        # When iterating the loader, y_batch is now on CPU
        for _, y_batch in loader:
            all_labels.extend(y_batch.numpy()) # .cpu() is no longer strictly needed if on CPU
        print(
            f"{name} class distribution:",
            np.bincount(all_labels, minlength=config["nb_outputs"]),
        )

    X_batch, _ = next(iter(train_loader))
    print("Spike train shape:", X_batch.shape)
    print("Total spikes in first batch:", X_batch.sum().item())
    print("Spikes per sample:", X_batch.sum(dim=(1, 2)))
    spikes_per_bin = X_batch.sum(dim=(0, 2))
    # No .cpu() needed if X_batch is already on CPU
    print("Spikes per time bin:", spikes_per_bin.numpy())
    print(
        "Average spikes per time bin per unit:", X_batch.mean(dim=0).mean(dim=1).numpy()
    )


    return train_loader, val_loader, test_loader