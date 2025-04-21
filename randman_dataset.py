import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

import sys

sys.path.append("/scratch/nar8991/snn/randman")
import randman


def standardize(x, eps=1e-7):
    mi, _ = x.min(0)
    ma, _ = x.max(0)
    return (x - mi) / (ma - mi + eps)


def make_spiking_dataset(
    nb_classes=10,
    nb_units=100,
    nb_steps=100,
    step_frac=1.0,
    dim_manifold=2,
    nb_spikes=1,
    nb_samples=1000,
    alpha=2.0,
    shuffle=True,
    classification=True,
    seed=None,
):
    """Generates event-based generalized spiking randman classification/regression dataset.
    In this dataset each unit fires a fixed number of spikes. So ratebased or spike count based decoding won't work.
    All the information is stored in the relative timing between spikes.
    For regression datasets the intrinsic manifold coordinates are returned for each target.
    Args:
        nb_classes: The number of classes to generate
        nb_units: The number of units to assume
        nb_steps: The number of time steps to assume
        step_frac: Fraction of time steps from beginning of each to contain spikes (default 1.0)
        nb_spikes: The number of spikes per unit
        nb_samples: Number of samples from each manifold per class
        alpha: Randman smoothness parameter
        shuffe: Whether to shuffle the dataset
        classification: Whether to generate a classification (default) or regression dataset
        seed: The random seed (default: None)
    Returns:
        A tuple of data,labels. The data is structured as numpy array
        (sample x event x 2 ) where the last dimension contains
        the relative [0,1] (time,unit) coordinates and labels.
    """

    data = []
    labels = []
    targets = []

    if seed is not None:
        np.random.seed(seed)

    max_value = np.iinfo(np.int64).max
    randman_seeds = np.random.randint(max_value, size=(nb_classes, nb_spikes))

    for k in range(nb_classes):
        x = np.random.rand(nb_samples, dim_manifold)
        submans = [
            randman.Randman(
                nb_units, dim_manifold, alpha=alpha, seed=randman_seeds[k, i]
            )
            for i in range(nb_spikes)
        ]
        units = []
        times = []
        for i, rm in enumerate(submans):
            y = rm.eval_manifold(x)
            y = standardize(y)
            units.append(
                np.repeat(np.arange(nb_units).reshape(1, -1), nb_samples, axis=0)
            )
            times.append(y.numpy())

        units = np.concatenate(units, axis=1)
        times = np.concatenate(times, axis=1)
        events = np.stack([times, units], axis=2)
        data.append(events)
        labels.append(k * np.ones(len(units)))
        targets.append(x)

    data = np.concatenate(data, axis=0)
    labels = np.array(np.concatenate(labels, axis=0), dtype=np.int64)
    targets = np.concatenate(targets, axis=0)

    if shuffle:
        idx = np.arange(len(data))
        np.random.shuffle(idx)
        data = data[idx]
        labels = labels[idx]
        targets = targets[idx]

    data[:, :, 0] *= nb_steps * step_frac
    # data = np.array(data, dtype=int)

    if classification:
        return data, labels
    else:
        return data, targets


def convert_to_x_data(spike_events, nb_steps, nb_units):
    # Initialize a tensor of zeros (nb_steps, nb_units)
    x_data = torch.zeros((nb_steps, nb_units), dtype=torch.float32)

    # Iterate through each spike event (time, unit)
    for event in spike_events:
        time_step = int(
            event[0]
        )  # Convert time to an index in the range [0, nb_steps-1]
        unit_id = int(event[1])  # Unit index (neuron)
        x_data[time_step, unit_id] = (
            1  # Mark the spike at the correct time step and unit
        )

    return x_data


def create_x_data(data, nb_steps, nb_inputs):
    x_data = []
    for sample in data:
        x_data.append(convert_to_x_data(sample, nb_steps, nb_units=nb_inputs))

    # Convert list of tensors into a single PyTorch tensor (batch of samples)
    x_data = torch.stack(x_data)  # Shape: (num_samples, nb_steps, nb_units)
    return x_data


def final_randman_dataset(nb_outputs, nb_inputs, nb_steps, batch_size):
    data, labels = make_spiking_dataset(
        nb_classes=nb_outputs,
        nb_units=nb_inputs,
        nb_steps=nb_steps,
        dim_manifold=1,
        seed=42,
        nb_samples=int(batch_size / nb_outputs) * 3,
    )
    # Load your data (ensure nb_outputs, device are defined)
    x_data = create_x_data(data, nb_steps, nb_inputs)
    return x_data, labels


def data_split_randman(config, device):
    batch_size_per_class = 128
    x_data, labels = final_randman_dataset(
        config["nb_outputs"],
        config["nb_inputs"],
        config["nb_steps"],
        batch_size_per_class * config["nb_outputs"],
    )
    data_tensor = torch.tensor(x_data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)

    dataset = TensorDataset(data_tensor, labels_tensor)
    total_size = len(dataset)
    third_size = total_size // 3  # Integer division to get the base size

    train_size = third_size
    val_size = third_size
    remaining_size = total_size - train_size - val_size
    test_size = remaining_size  # The rest of the data goes to the test set

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    x_train = torch.stack([data for data, _ in train_dataset]).to(device)
    y_train = torch.tensor([label for _, label in train_dataset], dtype=torch.int64).to(
        device
    )
    x_val = torch.stack([data for data, _ in val_dataset]).to(device)
    y_val = torch.tensor([label for _, label in val_dataset], dtype=torch.int64).to(
        device
    )
    x_test = torch.stack([data for data, _ in test_dataset]).to(device)
    y_test = torch.tensor([label for _, label in test_dataset], dtype=torch.int64).to(
        device
    )
    # Create dataloaders
    train_data = TensorDataset(x_train, y_train)
    val_data = TensorDataset(x_val, y_val)
    test_data = TensorDataset(x_test, y_test)

    train_loader = DataLoader(
        train_data, batch_size=batch_size_per_class * config["nb_outputs"], shuffle=True
    )

    test_loader = DataLoader(
        test_data, batch_size=batch_size_per_class * config["nb_outputs"], shuffle=False
    )

    val_loader = DataLoader(
        val_data, batch_size=batch_size_per_class * config["nb_outputs"], shuffle=False
    )

    return train_loader, test_loader, val_loader
