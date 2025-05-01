import tonic
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from data_utils import get_representative_subset_indices
from sklearn.model_selection import train_test_split


def preprocess_spike_events(spike_events, nb_steps, nb_units, time_step):
    """
    Convert spike events into a binned spike train.

    Args:
        spike_events: Structured NumPy array of spike events with fields ('x', 'y', 't', 'p').
        nb_steps: Number of time steps to bin the spikes into.
        nb_units: Number of neurons (x * y resolution).

    Returns:
        A binned spike train of shape (nb_steps, nb_units).
    """
    spike_train = np.zeros((nb_steps, nb_units), dtype=np.float32)

    # Access each field of the structured array separately
    x_coords = spike_events["x"]
    y_coords = spike_events["y"]
    timestamps = spike_events["t"]
    polarities = spike_events["p"]

    # Iterate through all events simultaneously
    for x, y, t, p in zip(x_coords, y_coords, timestamps, polarities):
        time_bin = int(t * time_step)  # Convert time to timestep
        # if p == 1:  # On spikes only
            time_bin = int(t * time_step)  # Convert time to timestep
            if time_bin < nb_steps:
                neuron_id = x + y * int(np.sqrt(nb_units))  # Flatten 2D coordinates into 1D neuron ID
                spike_train[time_bin, neuron_id] += 1

    return torch.tensor(spike_train, dtype=torch.float32)


def data_split_nmnist(config):
    train_limit = 200
    test_limit = train_limit

    train_dataset = tonic.datasets.NMNIST(
        save_to="./data", train=True, transform=None, target_transform=None
    )
    test_dataset = tonic.datasets.NMNIST(
        save_to="./data", train=False, transform=None, target_transform=None
    )

    train_subset_indices = get_representative_subset_indices(
        train_dataset, train_limit * 2
    )
    test_subset_indices = get_representative_subset_indices(test_dataset, test_limit)

    train_subset = Subset(train_dataset, train_subset_indices)
    test_subset = Subset(test_dataset, test_subset_indices)

    train_indices, val_indices = train_test_split(
        list(range(len(train_subset))),
        test_size=0.5,
        stratify=[train_dataset.targets[i] for i in train_subset.indices],
        random_state=42,
    )

    # Process training data - note we now get both events and label
    train_x_data = torch.stack(
        [
            preprocess_spike_events(
                train_subset[i][0],  # First element is the events
                nb_steps=config["nb_steps"],
                nb_units=config["nb_inputs"],
                time_step=config["time_step"],
            )
            for i in train_indices
        ]
    )

    train_y_tensor = torch.tensor(
        [train_subset[i][1] for i in train_indices], dtype=torch.int64
    )
    train_tensor_dataset = TensorDataset(train_x_data, train_y_tensor)

    # Process validation data
    val_x_data = torch.stack(
        [
            preprocess_spike_events(
                train_subset[i][0],
                nb_steps=config["nb_steps"],
                nb_units=config["nb_inputs"],
                time_step=config["time_step"],
            )
            for i in val_indices
        ]
    )
    val_y_tensor = torch.tensor(
        [train_subset[i][1] for i in val_indices], dtype=torch.int64
    )
    val_tensor_dataset = TensorDataset(val_x_data, val_y_tensor)

    # Process test data
    test_x_data = torch.stack(
        [
            preprocess_spike_events(
                test_subset[i][0],
                nb_steps=config["nb_steps"],
                nb_units=config["nb_inputs"],
                time_step=config["time_step"],
            )
            for i in range(len(test_subset))
        ]
    )
    test_y_tensor = torch.tensor(
        [test_subset[i][1] for i in range(len(test_subset))], dtype=torch.int64
    )
    test_data = TensorDataset(test_x_data, test_y_tensor)

    # Create dataloaders
    train_loader = DataLoader(
        train_tensor_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_tensor_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, val_loader, test_loader
