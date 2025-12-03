import numpy as np
import matplotlib.pyplot as plt


def plot_sharpness_boxplots(sam_results, rho_list, labels):
    """
    Plots box plots for average and difference SAM sharpness.
    Adds the count of seeds for each model as a label on the y-axis.
    """
    # Plotting for Average SAM Sharpness
    fig_avg, axes_avg = plt.subplots(1, len(rho_list), figsize=(18, 8), sharey=True)
    if len(rho_list) == 1:
        axes_avg = [axes_avg]
    for idx, rho in enumerate(rho_list):
        ax = axes_avg[idx]
        data_to_plot = [sam_results['avg'][rho][model_idx] for model_idx in range(len(labels))]
        # Count seeds for each model (length of each list)
        seed_counts = [len(sam_results['diff'][rho][model_idx]) for model_idx in range(len(labels))]
        # Add seed count to label
        labels_with_counts = [f"{label}\n(n={count})" for label, count in zip(labels, seed_counts)]
        ax.boxplot(data_to_plot, vert=False, positions=np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels_with_counts)
        ax.set_title(f"rho={rho}")
        ax.set_xlabel("Average SAM Sharpness")
        if idx == 0:
            ax.set_ylabel("Model | Recurrent\n(Seed count)")
    plt.suptitle("Average SAM Sharpness Across Models and Rhos", y=1.02)
    plt.tight_layout()
    plt.show()

    # Plotting for Difference SAM Sharpness
    fig_diff, axes_diff = plt.subplots(1, len(rho_list), figsize=(18, 8), sharey=True)
    if len(rho_list) == 1:
        axes_diff = [axes_diff]
    for idx, rho in enumerate(rho_list):
        ax = axes_diff[idx]
        data_to_plot = [sam_results['diff'][rho][model_idx] for model_idx in range(len(labels))]
        # Count seeds for each model (length of each list)
        seed_counts = [len(sam_results['diff'][rho][model_idx]) for model_idx in range(len(labels))]
        # Add seed count to label
        labels_with_counts = [f"{label}\n(n={count})" for label, count in zip(labels, seed_counts)]
        ax.boxplot(data_to_plot, vert=False, positions=np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels_with_counts)
        ax.set_title(f"rho={rho}")
        ax.set_xlabel("Difference SAM Sharpness")
        if idx == 0:
            ax.set_ylabel("Model | Recurrent\n(Seed count)")
    plt.suptitle("Difference SAM Sharpness Across Models and Rhos", y=1.02)
    plt.tight_layout()
    plt.show()
