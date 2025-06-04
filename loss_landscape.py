import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
import wandb
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
# from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
global_min = 0
global_max = 1

zlim = (global_min, global_max)

# wrapper registers the parameters
class HybridNet(nn.Module):
    def __init__(
        self,
        base_forward,
        w1,
        w2,
        v1,
        alpha,
        beta,
        spike_fn,
        device,
        recurrent,
        snn_mask,
    ):
        super().__init__()
        self.w1, self.w2, self.v1 = (
            nn.Parameter(w1),
            nn.Parameter(w2),
            nn.Parameter(v1),
        )
        self.alpha, self.beta = alpha, beta
        self.spike_fn = spike_fn
        self.device = device
        self.recurrent = recurrent
        self.register_buffer("snn_mask", snn_mask)
        self.base_forward = base_forward

    def forward(self, x):
        out_rec, _ = self.base_forward(
            x,
            self.w1,
            self.w2,
            self.v1,
            self.alpha,
            self.beta,
            self.spike_fn,
            self.device,
            self.recurrent,
            self.snn_mask,
        )
        m, _ = out_rec.max(dim=1)
        logp = F.log_softmax(m, dim=1)
        return logp
    def hiddenLayer(self, x):
        out_rec, other_recs = self.base_forward(
            x,
            self.w1,
            self.w2,
            self.v1,
            self.alpha,
            self.beta,
            self.spike_fn,
            self.device,
            self.recurrent,
            self.snn_mask,
        )
        ## Clean this up once we convert the model to class format
        result = None
        print("Other recs: ", len(other_recs))
        if len(other_recs) == 3:
            spk_rec, ann_rec = other_recs[1], other_recs[2]
            # We add them for the next step so do the same here
            result = spk_rec + ann_rec
        elif len(other_recs) == 2:
            spk_rec = other_recs[1]
            result = spk_rec
        else:
            ann_recs = other_recs
            result = ann_recs
        return result

def _flatten_params(params):
    """Concatenate all param tensors into one vector, return shapes for unflattening."""
    shapes, parts = [], []
    for p in params:
        shapes.append(p.shape)
        parts.append(p.data.flatten())
    return torch.cat(parts), shapes


@torch.no_grad()
def _assign_flat_params(params, flat_vec, shapes):
    """Overwrite each param in-place from the flat vector (no grad)."""
    idx = 0
    for p, shp in zip(params, shapes):
        n = p.numel()
        p.data.copy_(flat_vec[idx : idx + n].view(shp))
        idx += n


@torch.no_grad()
def visualize_loss_landscape_3d(
    model,
    criterion,
    dataloader,
    device="cpu",
    resolution=21,
    range_lim=1e-2,
    save_path=None,
    wandb_run=None,
):
    """
    Visualize the loss landscape of `model` around its current parameters.
    """
    # 0) Prepare
    model.to(device).eval()
    xb, yb = next(iter(dataloader))
    xb, yb = xb.to(device), yb.to(device)

    # 1) Snapshot & flatten params
    params = list(model.parameters())
    base_vec, shapes = _flatten_params(params)

    # 2) Orthonormal directions
    torch.manual_seed(0)
    d1 = torch.randn_like(base_vec)
    d1 /= d1.norm()
    d2 = torch.randn_like(base_vec)
    d2 -= torch.dot(d2, d1) * d1
    d2 /= d2.norm()

    # 3) Build grid
    alphas = np.linspace(-range_lim, range_lim, resolution)
    betas = np.linspace(-range_lim, range_lim, resolution)
    loss_grid = np.empty((resolution, resolution), dtype=np.float32)

    # 4) Sweep
    for i, a in enumerate(tqdm(alphas, desc="α")):
        for j, b in enumerate(betas):
            offset = a * d1 + b * d2
            _assign_flat_params(params, base_vec + offset, shapes)
            loss_grid[j, i] = criterion(model(xb), yb).item()

    # 5) Restore
    _assign_flat_params(params, base_vec, shapes)

    # 6) Plot
    A, B = np.meshgrid(alphas, betas)

    # 6a) 3D surface
    fig3d = plt.figure(figsize=(6, 5))
    ax3d = fig3d.add_subplot(111, projection="3d")
    surf = ax3d.plot_surface(
        A, B, loss_grid, cmap="viridis", edgecolor="none", alpha=0.9
    )
    ax3d.set_xlabel("α")
    ax3d.set_ylabel("β")
    ax3d.set_zlabel("Loss")
    ax3d.set_title("3D Loss Surface")
    ax3d.set_zlim()
    fig3d.colorbar(surf, shrink=0.6, aspect=10, label="Loss")

    fig3d.savefig("3d_loss_surface.png", dpi=300, bbox_inches="tight")
    if wandb_run is not None:
        wandb_run.log({"3D Loss Surface": wandb.Image("3d_loss_surface.png")})
    plt.show()

    # 6b) Contour
    fig2d, ax2d = plt.subplots(figsize=(5, 4))
    cs = ax2d.contourf(alphas, betas, loss_grid, levels=30, cmap="viridis")
    fig2d.colorbar(cs, ax=ax2d, label="Loss")
    ax2d.set_xlabel("α")
    ax2d.set_ylabel("β")
    ax2d.set_title("2D Loss Contour")

    fig2d.savefig("2d_loss_contour.png", dpi=300, bbox_inches="tight")
    if wandb_run is not None:
        wandb_run.log({"2D Loss Contour": wandb.Image("2d_loss_contour.png")})
    plt.show()

    plt.close(fig3d)
    plt.close(fig2d)
    return fig3d, fig2d 

@torch.no_grad()
def hidden_layer_visualization(
    model,
    data_config,
    dataloader,
    save_path=None,
    wandb_run=None
):
    """
    Visualize the hidden layer representation of the model.
    """
    # 1) Get the hidden layer representation of the data (these are the spikes)
    all_hidden_outputs = []
    all_labels = []

    for data, labels in dataloader:
        hidden_output = model.hiddenLayer(data)
        all_hidden_outputs.append(hidden_output.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    hidden_outputs_np = np.vstack(all_hidden_outputs)
    labels_np = np.concatenate(all_labels)
    print(f"Collected hidden outputs shape: {hidden_outputs_np.shape}")
    print(f"Collected labels shape: {labels_np.shape}")
    hidden_output_for_viz = hidden_outputs_np.reshape(hidden_outputs_np.shape[0], -1)

    print(f"Shape of hidden output for visualization: {hidden_output_for_viz.shape}")

    # 2) Use t-SNE to reduce the dimensionality of the hidden layer representation
    tsne = TSNE(n_components=2, random_state=42)
    hidden_2d_tsne = tsne.fit_transform(hidden_output_for_viz)
    # 3) Visualize
    fig, ax = plt.subplots(figsize=(7, 6)) # You can adjust figsize if needed
    sns.scatterplot(
        x=hidden_2d_tsne[:, 0], y=hidden_2d_tsne[:, 1],
        hue=labels_np, palette=sns.color_palette("tab10", n_colors=data_config['nb_outputs']),
        legend='full', alpha=0.7, s=10,
        ax=ax
    )
    ax.set_title('SNN Hidden Layer Activations (t-SNE)')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Calculate the different scores
    silhouette_avg = silhouette_score(hidden_output_for_viz, labels_np)
    print(f"Silhouette Coefficient (using ground-truth labels): {silhouette_avg:.4f}")
    calinski_harabasz = calinski_harabasz_score(hidden_output_for_viz, labels_np)
    print(f"Calinski-Harabasz Index (using ground-truth labels): {calinski_harabasz:.4f}")
    davies_bouldin = davies_bouldin_score(hidden_output_for_viz, labels_np)
    print(f"Davies-Bouldin Index (using ground-truth labels): {davies_bouldin:.4f}")
    if wandb_run is not None:
        wandb_run.log({
            "t-SNE Visualization": wandb.Image(fig),
            "Silhouette Coefficient": silhouette_avg,
            "Calinski-Harabasz Index": calinski_harabasz,
            "Davies-Bouldin Index": davies_bouldin
        })
    coefficient_metrics = {
        "Silhouette Coefficient": silhouette_avg,
        "Calinski-Harabasz Index": calinski_harabasz,
        "Davies-Bouldin Index": davies_bouldin
    }
    plt.close(fig) # Close the figure to free up memory if not displaying it

    return fig, coefficient_metrics