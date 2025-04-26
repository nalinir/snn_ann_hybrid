import tonic
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from data_utils import get_representative_subset_indices
from sklearn.model_selection import train_test_split


def preprocess_spike_events(spike_events, nb_steps, nb_units, time_step, max_time):
    """
    Convert spike events into a binned spike train with consideration for max_time.

    Args:
        spike_events: Structured NumPy array of spike events with fields ('x', 't', 'p').
        nb_steps: Number of time steps to bin the spikes into.
        nb_units: Number of neurons (700 for SHD).
        time_step: The time resolution to scale spike times (nb_steps / max_time).
        max_time: The maximum time value, defining the end of the time window for discretization.

    Returns:
        A binned spike train of shape (nb_steps, nb_units).
    """
    spike_train = np.zeros((nb_steps, nb_units), dtype=np.float32)

    # Access each field of the structured array separately
    timestamps = spike_events["t"] / 1e6  # Convert microseconds to seconds
    units_fired = spike_events["x"]
    polarities = spike_events["p"]

    # Clip timestamps to max_time to ensure they fit within the time window
    timestamps = np.clip(timestamps, 0, max_time)

    # Iterate through all spike events
    for t, unit, p in zip(timestamps, units_fired, polarities):
        time_bin = int(round(t * time_step))  # Round for better binning precision

        # Ensure the time_bin is within bounds
        if 0 <= time_bin < nb_steps:
            # Ensure that unit ID is within the valid neuron range
            if 0 <= unit < nb_units:
                spike_train[time_bin, unit] = 1.0  # Set to 1.0 (binary)

    return torch.tensor(spike_train, dtype=torch.float32)


def data_split_shd(config):
    train_limit = 400
    test_limit = 100
    val_limit = 100

    train_dataset = tonic.datasets.SHD(
        save_to="./data", train=True, transform=None, target_transform=None
    )
    test_dataset = tonic.datasets.SHD(
        save_to="./data", train=False, transform=None, target_transform=None
    )

    # Select balanced subsets with correct number of classes (20 for SHD)
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    train_subset_indices = get_representative_subset_indices(
        train_labels,
        train_limit + val_limit,
        num_inputs=20  # SHD has 20 classes
    )
    test_labels = [test_dataset[i][1] for i in range(len(test_dataset))]
    test_subset_indices = get_representative_subset_indices(
        test_labels,
        test_limit,
        num_inputs=20
    )

    train_val_subset = Subset(train_dataset, train_subset_indices)
    test_subset = Subset(test_dataset, test_subset_indices)

    # Ensure balanced train/validation split
    labels = [train_val_subset[i][1] for i in range(len(train_val_subset))]
    train_indices, val_indices = train_test_split(
        list(range(len(train_val_subset))),
        test_size=val_limit / (train_limit + val_limit),
        stratify=labels,
        random_state=42,
    )

    # Compute time_step dynamically
    time_step = config["nb_steps"] / config["max_time"]
    print(f"Computed time_step: {time_step}")

    # Process training data
    train_x_data = torch.stack(
        [
            preprocess_spike_events(
                train_val_subset[i][0],
                nb_steps=config["nb_steps"],
                nb_units=config["nb_inputs"],
                time_step=time_step,
                max_time=config["max_time"],
            )
            for i in train_indices
        ]
    )
    train_y_tensor = torch.tensor(
        [train_val_subset[i][1] for i in train_indices], dtype=torch.int64
    )
    train_tensor_dataset = TensorDataset(train_x_data, train_y_tensor)

    # Process validation data
    val_x_data = torch.stack(
        [
            preprocess_spike_events(
                train_val_subset[i][0],
                nb_steps=config["nb_steps"],
                nb_units=config["nb_inputs"],
                time_step=time_step,
                max_time=config["max_time"],
            )
            for i in val_indices
        ]
    )
    val_y_tensor = torch.tensor(
        [train_val_subset[i][1] for i in val_indices], dtype=torch.int64
    )
    val_tensor_dataset = TensorDataset(val_x_data, val_y_tensor)

    # Process test data
    test_x_data = torch.stack(
        [
            preprocess_spike_events(
                test_subset[i][0],
                nb_steps=config["nb_steps"],
                nb_units=config["nb_inputs"],
                time_step=time_step,
                max_time=config["max_time"],
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
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # Debug: Check class distribution
    for loader, name in [(train_loader, "Train"), (val_loader, "Validation"), (test_loader, "Test")]:
        all_labels = []
        for _, y_batch in loader:
            all_labels.extend(y_batch.numpy())
        print(f"{name} class distribution:", np.bincount(all_labels, minlength=20))

    # Debug: Check spike counts
    X_batch, _ = next(iter(train_loader))
    print("Spike train shape:", X_batch.shape)
    print("Total spikes in first batch:", X_batch.sum().item())
    print("Spikes per sample:", X_batch.sum(dim=(1, 2)))

    return train_loader, val_loader, test_loader