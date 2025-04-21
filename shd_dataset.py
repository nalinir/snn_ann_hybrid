import tonic
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np


def dense_data_generator_from_hdf5_spikes(
    X, y, batch_size, nb_steps, nb_units, max_time, shuffle=True
):
    """Generates dense tensors for SNN input from SHD spike data.

    Args:
        X: The data (sample x event x 2) with 'times' and 'units'.
        y: The labels.
        batch_size: Number of samples per batch.
        nb_steps: Number of time steps in the output.
        nb_units: Number of neurons/units (700 for SHD).
        max_time: Maximum time value (1.4s for SHD).
        shuffle: Whether to shuffle the samples.
    """
    labels_ = np.array(y, dtype=np.int32)
    number_of_batches = len(labels_) // batch_size
    sample_index = np.arange(len(labels_))
    firing_times = X["times"]
    units_fired = X["units"]
    time_scale = nb_steps / max_time

    if shuffle:
        np.random.shuffle(sample_index)

    counter = 0
    while counter < number_of_batches:
        batch_index = sample_index[batch_size * counter : batch_size * (counter + 1)]
        X_batch = torch.zeros((batch_size, nb_steps, nb_units), device=device)

        for bc, idx in enumerate(batch_index):
            times = firing_times[idx]
            units = units_fired[idx]
            steps = np.round(times * time_scale).astype(np.int32)
            steps = np.clip(steps, 0, nb_steps - 1)
            valid_units = units[(units >= 0) & (units < nb_units)]
            valid_steps = steps[(units >= 0) & (units < nb_units)]
            if len(valid_units) > 0:
                X_batch[bc, valid_steps, valid_units] = 1.0
            else:
                print(f"Warning: No valid spikes for sample {idx}")

        y_batch = torch.tensor(labels_[batch_index], device=device, dtype=torch.long)
        yield X_batch, y_batch
        counter += 1


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

    train_subset_indices = get_representative_subset_indices(y_train, train_limit)
    test_subset_indices = get_representative_subset_indices(y_test, test_limit)

    train_subset_indices = np.sort(train_subset_indices)
    test_subset_indices = np.sort(test_subset_indices)

    # Create the non-sparse data generators using the NumPy arrays and subset indices
    train_loader = dense_data_generator_from_hdf5_spikes(
        {
            "times": x_train["times"][train_subset_indices],
            "units": x_train["units"][train_subset_indices],
        },
        y_train[train_subset_indices],
        batch_size,
        nb_steps,
        nb_inputs,
        max_time,
        shuffle=True,
    )

    test_loader = dense_data_generator_from_hdf5_spikes(
        {
            "times": x_test["times"][test_subset_indices],
            "units": x_test["units"][test_subset_indices],
        },
        y_test[test_subset_indices],
        batch_size,
        nb_steps,
        nb_inputs,
        max_time,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader
