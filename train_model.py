import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from surrogate_gradient import SurrGradSpike


def train_and_val(
    model,
    train_loader,
    val_loader,
    test_loader,
    wandb_run,
    data_config,
    device,
):
    config = wandb_run.config
    lr = config.learning_rate
    l1 = config.l1
    l2 = config.l2
    epochs = config.epochs
    regularization = config.regularization

    w1 = torch.empty(
        (data_config["nb_inputs"], data_config["nb_hidden"]),
        device=device,
        dtype=torch.float,
        requires_grad=True,
    )
    w2 = torch.empty(
        (data_config["nb_hidden"], data_config["nb_outputs"]),
        device=device,
        dtype=torch.float,
        requires_grad=True,
    )
    v1 = torch.empty(
        (data_config["nb_hidden"], data_config["nb_hidden"]),
        device=device,
        dtype=torch.float,
        requires_grad=True,
    )

    torch.manual_seed(42)
    snn_mask = torch.randint(
        0, 2, (data_config["nb_hidden"],), dtype=torch.float32, device=device
    )

    alpha = float(np.exp(-data_config["time_step"] / data_config["tau_syn"]))
    beta = float(np.exp(-data_config["time_step"] / data_config["tau_mem"]))

    weight_scale = 7 * (1.0 - beta)

    with torch.no_grad():
        torch.nn.init.normal_(
            w1, mean=0.0, std=weight_scale / np.sqrt(data_config["nb_inputs"])
        )
        torch.nn.init.normal_(
            w2, mean=0.0, std=weight_scale / np.sqrt(data_config["nb_hidden"])
        )
        torch.nn.init.normal_(
            v1, mean=0.0, std=weight_scale / np.sqrt(data_config["nb_hidden"])
        )

    spike_fn = SurrGradSpike.apply

    print("init done")

    params = [w1, w2, v1]
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()
    loss_hist_train = []
    loss_hist_val = []
    val_accuracy_hist = []
    spike_count_train_epoch = 0
    spike_count_val_epoch = 0

    for e in range(epochs):
        local_loss = []
        for x_local, y_local in train_loader:
            x_local = x_local.to(device)
            y_local = y_local.to(device)
            # --- Training Phase ---
            output_train, other_recs_train = model(
                x_local,
                w1,
                w2,
                v1,
                alpha,
                beta,
                spike_fn,
                device,
                config.recurrent,
                snn_mask,
            )
            m, _ = torch.max(output_train, 1)
            log_p_y_train = log_softmax_fn(m)
            loss_val_train = loss_fn(log_p_y_train, y_local)

            if regularization:
                spks = other_recs_train[1]
                reg_loss = l1 * torch.sum(spks)
                reg_loss += l2 * torch.mean(
                    torch.sum(torch.sum(spks, dim=0), dim=0) ** 2
                )
                loss_val_train += reg_loss
                spike_count_train_epoch += torch.sum(spks >= 1.0)

            optimizer.zero_grad()
            loss_val_train.backward()
            optimizer.step()
            local_loss.append(loss_val_train.item())

        loss_hist_train.append(np.mean(local_loss))

        # --- Validation Phase ---
        with torch.no_grad():
            local_loss = []
            for x_local, y_local in val_loader:
                x_local = x_local.to(device)
                y_local = y_local.to(device)

                output_val, other_recs_val = model(
                    x_local,
                    w1,
                    w2,
                    v1,
                    alpha,
                    beta,
                    spike_fn,
                    device,
                    config.recurrent,
                    snn_mask,
                )
                m, _ = torch.max(output_val, 1)

                log_p_y_val = log_softmax_fn(m)
                loss_val_val = loss_fn(log_p_y_val, y_local)

                val_accuracy, _, _ = compute_classification_accuracy(
                    val_loader,
                    w1,
                    w2,
                    v1,
                    alpha,
                    beta,
                    spike_fn,
                    snn_mask,
                    data_config,
                    device,
                    config,
                    model,
                )
                val_accuracy_hist.append(val_accuracy)
                local_loss.append(loss_val_train.item())
                if "SNN" in model.__name__ or "Hybrid" in model.__name__:
                    spike_count_val_epoch += (
                        torch.sum(other_recs_val[1][other_recs_val[1] >= 1.0])
                        .detach()
                        .cpu()
                        .numpy()
                    )

            loss_hist_val.append(np.mean(local_loss))

            # Log metrics to Wandb
            if wandb_run:
                wandb_run.log(
                    {
                        "train_loss": loss_val_train.item(),
                        "val_loss": loss_val_val.item(),
                        "val_accuracy": val_accuracy,
                        "epoch": e,
                    }
                )
                if "SNN" in model.__name__ or "Hybrid" in model.__name__:
                    wandb_run.log(
                        {
                            "spike_count_train_epoch": spike_count_train_epoch,
                            "spike_count_val_epoch": spike_count_val_epoch,
                        }
                    )

    # Evaluation after training loop
    train_accuracy, train_conf_matrix, train_spike_ct = compute_classification_accuracy(
        train_loader,
        w1,
        w2,
        v1,
        alpha,
        beta,
        spike_fn,
        snn_mask,
        data_config,
        device,
        config,
        model,
    )
    test_accuracy, test_conf_matrix, test_spike_ct = compute_classification_accuracy(
        test_loader,
        w1,
        w2,
        v1,
        alpha,
        beta,
        spike_fn,
        snn_mask,
        data_config,
        device,
        config,
        model,
    )
    val_accuracy_final, val_conf_matrix, val_spike_ct = compute_classification_accuracy(
        val_loader,
        w1,
        w2,
        v1,
        alpha,
        beta,
        spike_fn,
        snn_mask,
        data_config,
        device,
        config,
        model,
    )

    print(f"Final Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Final Validation Accuracy: {val_accuracy_final * 100:.2f}%")
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

    if wandb_run:
        wandb_run.log(
            {
                "final_train_accuracy": train_accuracy,
                "final_test_accuracy": test_accuracy,
                "final_val_accuracy": val_accuracy,
                "final_train_spike_ct": train_spike_ct,
                "final_test_spike_ct": test_spike_ct,
                "final_val_spike_ct": val_spike_ct,
            }
        )

    # return val_accuracy  # For training sweeps


def compute_classification_accuracy(
    data_loader,
    w1,
    w2,
    v1,
    alpha,
    beta,
    spike_fn,
    snn_mask,
    data_config,
    device,
    config,
    model,
    incl_confusion_matrix=False,
):
    """
    Computes classification accuracy and confusion matrix on supplied data in batches.

    Returns:
        accuracy: Overall classification accuracy.
        conf_matrix: Confusion matrix of shape (num_classes, num_classes).
    """
    all_preds = []
    all_labels = []
    spike_count = 0
    for x_local, y_local in data_loader:
        # Move data to the appropriate device
        x_local, y_local = x_local.to(device), y_local.to(device)

        output, recs = model(
            x_local,
            w1,
            w2,
            v1,
            alpha,
            beta,
            spike_fn,
            device,
            config.recurrent,
            snn_mask,
        )
        m, _ = torch.max(output, 1)
        if "Hybrid" or "SNN" in model.__name__:
            spks = recs[1]
            spks[spks > 0.0] = 1.0
            spks[spks < 0.0] = 0.0
            spike_count += torch.sum(spks).detach().cpu().numpy()
        # Compute training accuracy
        _, am = torch.max(m, 1)
        acc = np.mean((y_local == am).detach().cpu().numpy())
        all_preds.extend(am.cpu().numpy())
        all_labels.extend(y_local.cpu().numpy())
        if incl_confusion_matrix:
            # Compute confusion matrix
            conf_matrix = confusion_matrix(all_labels, all_preds)
        else:
            conf_matrix = None
    return acc, conf_matrix, spike_count
