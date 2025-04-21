import tonic
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from data_utils import get_representative_subset_indices


def preprocess_spike_events(spike_events, nb_steps, nb_units, time_step):
    """
    Convert spike events into a binned spike train.

    Args:
        spike_events: Structured NumPy array of spike events (x, y, t, p).
        nb_steps: Number of time steps to bin the spikes into.
        nb_units: Number of neurons (x * y resolution).

    Returns:
        A binned spike train of shape (nb_steps, nb_units).
    """
    spike_train = np.zeros((nb_steps, nb_units), dtype=np.float32)
    for x, y, t, p in spike_events:
        time_bin = int(t * time_step)  # Convert time to timestep
        if time_bin < nb_steps:
            neuron_id = x + y * int(
                np.sqrt(nb_units)
            )  # Flatten 2D coordinates into 1D neuron ID
            spike_train[
                time_bin, neuron_id
            ] += p  # If it's positive, it'll be 1, if negative, it'll be negative 1

    return torch.tensor(spike_train, dtype=torch.float32)


def data_split_nmnist(config):
    train_limit = 200
    test_limit = train_limit
    val_limit = train_limit
    train_dataset = tonic.datasets.NMNIST(
        save_to="./data", train=True, transform=None, target_transform=None
    )
    test_dataset = tonic.datasets.NMNIST(
        save_to="./data", train=False, transform=None, target_transform=None
    )

    # Calculate indices for train and validation sets
    all_train_indices = np.arange(len(train_dataset))
    np.random.shuffle(all_train_indices)
    val_split_index = int(len(all_train_indices) * 0.8)  # Example: 80% train, 20% val
    train_indices = all_train_indices[:val_split_index]
    val_indices = all_train_indices[val_split_index:]

    # Get representative subsets for the initial train and validation sets
    train_subset_indices = get_representative_subset_indices(
        Subset(train_dataset, train_indices), train_limit, num_inputs=10
    )
    val_subset_indices = get_representative_subset_indices(
        Subset(train_dataset, val_indices), val_limit, num_inputs=10
    )
    test_subset_indices = get_representative_subset_indices(
        test_dataset, test_limit, num_inputs=10
    )

    train_x_data = torch.stack(
        [
            preprocess_spike_events(
                train_dataset[i],
                config["nb_steps"],
                config["nb_inputs"],
                config["time_step"],
            )
            for i in train_indices[train_subset_indices]
        ]
    )
    train_y_tensor = torch.tensor(
        [train_dataset[i][1] for i in train_indices[train_subset_indices]],
        dtype=torch.int64,
    )

    val_x_data = torch.stack(
        [
            preprocess_spike_events(
                train_dataset[i], config["nb_steps"], config["nb_inputs"]
            )
            for i in val_indices[val_subset_indices]
        ]
    )
    val_y_tensor = torch.tensor(
        [train_dataset[i][1] for i in val_indices[val_subset_indices]],
        dtype=torch.int64,
    )

    test_x_data = torch.stack(
        [
            preprocess_spike_events(
                test_dataset[i], config["nb_steps"], config["nb_inputs"]
            )
            for i in test_subset_indices
        ]
    )
    test_y_tensor = torch.tensor(
        [test_dataset[i][1] for i in test_subset_indices], dtype=torch.int64
    )

    train_data = TensorDataset(train_x_data, train_y_tensor)
    val_data = TensorDataset(val_x_data, val_y_tensor)
    test_data = TensorDataset(test_x_data, test_y_tensor)

    train_loader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=config["batch_size"],
        shuffle=False,  # No need to shuffle validation set
        pin_memory=True,
        drop_last=False,  # Might have a smaller last batch
    )

    test_loader = DataLoader(
        test_data,
        batch_size=config["batch_size"],
        shuffle=False,  # No need to shuffle test set
        pin_memory=True,
        drop_last=False,  # Might have a smaller last batch
    )

    return train_loader, val_loader, test_loader
