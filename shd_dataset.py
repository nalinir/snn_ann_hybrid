import tonic
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from sklearn.model_selection import train_test_split
from tonic import transforms


# Gemini used to fix a gpu error


def preprocess_spike_events(spike_events, nb_steps, nb_units, time_step, max_time):
    spike_train = np.zeros((nb_steps, nb_units), dtype=np.float32)
    print(f"Processing spike events with {len(spike_events)} events, ")
    # print(f"Expected shape: ({nb_steps}, {nb_units})")
    # print(f"Shape: {spike_events.shape}, ")

    timestamps = spike_events["t"]/1e6  # Convert microseconds to seconds
    timestamps = np.clip(timestamps, 0, max_time)
    units_fired = spike_events["x"]
    max_unit = np.max(units_fired)
    polarities = spike_events["p"]
    conversion_unit_bin = nb_units/700
    spike_counts = np.zeros(nb_units, dtype=np.int32)
    max_spikes_per_unit = 30

    for t, p, unit in spike_events:
        if spike_counts[unit] >= max_spikes_per_unit:
            continue
        time_bin = int(round(t * time_step))
        # time_bin = int(t)
        unit_bin = int(unit / conversion_unit_bin)
        spike_train[time_bin, unit_bin] += 1.0

    return torch.tensor(spike_train, dtype=torch.float32)


def data_split_shd(config, device=torch.device("cpu"), dim_manifold=None):
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
    # Data = tonic.datasets.hsd.SHD
    # sensor_size = Data.sensor_size[0]
    # print(f"Sensor size: {sensor_size}")
    # transform = transforms.Compose(
    #     [
    #         transforms.Downsample(spatial_factor=config["nb_inputs"] / sensor_size),
    #         transforms.CropTime(
    #             max=(config["max_time"] * 1e6)
    #         ),  # Convert seconds to microseconds
    #         transforms.ToFrame(
    #             sensor_size=(config["nb_inputs"], 1, 1),
    #             time_window=(config["time_step"] * 1e6),
    #             include_incomplete=True,
    #         ),
    #         # For the polarity channel - not used
    #         lambda x: x.squeeze(1),
    #     ]
    # )
    train_dataset = tonic.datasets.SHD(
        save_to="./data", train=True, 
        # transform=transform
    )
    test_dataset = tonic.datasets.SHD(
        save_to="./data", train=False, 
        # transform=transform
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    train_subset_indices = list(range(len(train_dataset)))
    test_subset_indices = list(range(len(test_dataset)))

    sample_x, sample_y = train_dataset[0]    
    print(f"Train dataset sample 0 (x) shape AFTER TRANSFORM: {sample_x.shape}") # Should now be (N, 70)
    print(f"Train dataset sample 0 (y) label: {sample_y}")
    print(f"Number of items in sample 0: {len(train_dataset[0])}")

    train_labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
    train_indices, val_indices = train_test_split(
        train_subset_indices,
        test_size=0.2,
        stratify=train_labels,
        random_state=42,
    )

    # train_subset = Subset(train_dataset, train_indices)
    # val_subset = Subset(train_dataset, val_indices)
    # test_subset = Subset(test_dataset, test_subset_indices)

    actual_nb_steps = int(config["max_time"] / config["time_step"])

    # # Preprocess data and KEEP IT ON CPU
    def preprocess_only_cpu(dataset, indices):
        x_data = torch.stack(
            [
                preprocess_spike_events(
                    dataset[i][0],
                    actual_nb_steps,
                    config["nb_inputs"],
                    config["time_step"],
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
    test_tensor_dataset = TensorDataset(test_x_data, test_y_tensor)

    train_loader = DataLoader(
        train_tensor_dataset,
        # train_subset,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        # num_workers=2,
        # collate_fn=tonic.collation.PadTensors(), # Your target sequence length (e.g., 2799),
    )
    val_loader = DataLoader(
        val_tensor_dataset,
        # val_subset,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        # num_workers=2,
        # collate_fn=tonic.collation.PadTensors(),
    )
    test_loader = DataLoader(
        test_tensor_dataset,
        # test_subset,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        # num_workers=2,
        # collate_fn=tonic.collation.PadTensors(),
    )


    # DataLoaders are configured. pin_memory=True is fine here
    # because it will pin the *CPU* memory for faster transfer later.
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
            all_labels.extend(
                y_batch.numpy()
            )  # .cpu() is no longer strictly needed if on CPU
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
