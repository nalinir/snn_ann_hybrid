import tonic
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_spike_events(spike_events, nb_steps, nb_units, time_step, max_time):
    spike_train = np.zeros((nb_steps, nb_units), dtype=np.float32)
    timestamps = spike_events["t"] / 1e6
    timestamps = np.clip(timestamps, 0, max_time)
    units_fired = spike_events["x"]
    polarities = spike_events["p"]

    # spike_counts = np.zeros(nb_units, dtype=np.int32)
    # max_spikes_per_unit = 30

    for t, unit, p in zip(timestamps, units_fired, polarities):
        # if spike_counts[unit] >= max_spikes_per_unit:
        #     continue
        time_bin = int(round(t * time_step))
        if 0 <= time_bin < nb_steps and 0 <= unit < nb_units:
            spike_train[time_bin, unit] += 1.0
            # spike_counts[unit] += 1

    return torch.tensor(spike_train, dtype=torch.float32)

def data_split_shd(config):
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

    train_x_data = torch.stack([preprocess_spike_events(
        train_dataset[i][0], config["nb_steps"], config["nb_inputs"], time_step, config["max_time"]
    ) for i in train_indices])
    train_y_tensor = torch.tensor([train_dataset[i][1] for i in train_indices], dtype=torch.int64)

    val_x_data = torch.stack([preprocess_spike_events(
        train_dataset[i][0], config["nb_steps"], config["nb_inputs"], time_step, config["max_time"]
    ) for i in val_indices])
    val_y_tensor = torch.tensor([train_dataset[i][1] for i in val_indices], dtype=torch.int64)

    test_x_data = torch.stack([preprocess_spike_events(
        test_dataset[i][0], config["nb_steps"], config["nb_inputs"], time_step, config["max_time"]
    ) for i in test_subset_indices])
    test_y_tensor = torch.tensor([test_dataset[i][1] for i in test_subset_indices], dtype=torch.int64)

    train_tensor_dataset = TensorDataset(train_x_data, train_y_tensor)
    val_tensor_dataset = TensorDataset(val_x_data, val_y_tensor)
    test_data = TensorDataset(test_x_data, test_y_tensor)

    train_loader = DataLoader(
        train_tensor_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True, drop_last=True, num_workers=2
    )
    val_loader = DataLoader(
        val_tensor_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True, drop_last=False, num_workers=2
    )
    test_loader = DataLoader(
        test_data, batch_size=config["batch_size"], shuffle=False, pin_memory=True, drop_last=False, num_workers=2
    )

    for loader, name in [(train_loader, "Train"), (val_loader, "Validation"), (test_loader, "Test")]:
        all_labels = []
        for _, y_batch in loader:
            all_labels.extend(y_batch.numpy())
        print(f"{name} class distribution:", np.bincount(all_labels, minlength=config["nb_outputs"]))

    X_batch, _ = next(iter(train_loader))
    print("Spike train shape:", X_batch.shape)
    print("Total spikes in first batch:", X_batch.sum().item())
    print("Spikes per sample:", X_batch.sum(dim=(1, 2)))
    spikes_per_bin = X_batch.sum(dim=(0, 2))
    print("Spikes per time bin:", spikes_per_bin.numpy())
    print("Average spikes per time bin per unit:", X_batch.mean(dim=0).mean(dim=1).numpy())

    return train_loader, val_loader, test_loader