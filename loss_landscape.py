import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D 
import torch.nn as nn
import wandb
import torch.nn.functional as F
from tqdm import tqdm

# wrapper registers the parameters
class HybridNet(nn.Module):
    def __init__(self, base_forward, w1, w2, v1, alpha, beta,
                 spike_fn, device, recurrent, snn_mask):
        super().__init__()
        self.w1, self.w2, self.v1 = (
            nn.Parameter(w1),
            nn.Parameter(w2),
            nn.Parameter(v1),
        )
        self.alpha, self.beta   = alpha, beta
        self.spike_fn           = spike_fn
        self.device             = device
        self.recurrent          = recurrent
        self.register_buffer("snn_mask", snn_mask)
        self.base_forward       = base_forward

    def forward(self, x):
        out_rec, _ = self.base_forward(
            x,
            self.w1, self.w2, self.v1,
            self.alpha, self.beta,
            self.spike_fn,
            self.device,
            self.recurrent,
            self.snn_mask
        )
        m, _   = out_rec.max(dim=1)
        logp   = F.log_softmax(m, dim=1)
        return logp
    
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
        p.data.copy_(flat_vec[idx:idx+n].view(shp))
        idx += n

@torch.no_grad()
def visualize_loss_landscape_3d(model,
                             criterion,
                             dataloader,
                             device     = 'cpu',
                             resolution = 21,
                             range_lim  = 1e-2,
                             save_path  = None):
    """
    Visualize the loss landscape of `model` around its current parameters.
    """
    # 0) Prepare
    model.to(device).eval()
    xb, yb     = next(iter(dataloader))
    xb, yb     = xb.to(device), yb.to(device)

    # 1) Snapshot & flatten params
    params      = list(model.parameters())
    base_vec, shapes = _flatten_params(params)

    # 2) Orthonormal directions
    torch.manual_seed(0)
    d1 = torch.randn_like(base_vec); d1 /= d1.norm()
    d2 = torch.randn_like(base_vec)
    d2 -= torch.dot(d2, d1) * d1; d2 /= d2.norm()

    # 3) Build grid
    alphas = np.linspace(-range_lim, range_lim, resolution)
    betas  = np.linspace(-range_lim, range_lim, resolution)
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
    fig3d = plt.figure(figsize=(6,5))
    ax3d  = fig3d.add_subplot(111, projection='3d')
    surf = ax3d.plot_surface(A, B, loss_grid, cmap='viridis',
                             edgecolor='none', alpha=0.9)
    ax3d.set_xlabel("α"); ax3d.set_ylabel("β"); ax3d.set_zlabel("Loss")
    ax3d.set_title("3D Loss Surface")
    fig3d.colorbar(surf, shrink=0.6, aspect=10, label="Loss")

    fig3d.savefig("3d_loss_surface.png", dpi=300, bbox_inches="tight")
    wandb.log({"3D Loss Surface": wandb.Image("3d_loss_surface.png")})
    plt.show()

    # 6b) Contour
    fig2d, ax2d = plt.subplots(figsize=(5,4))
    cs = ax2d.contourf(alphas, betas, loss_grid, levels=30, cmap='viridis')
    fig2d.colorbar(cs, ax=ax2d, label="Loss")
    ax2d.set_xlabel("α"); ax2d.set_ylabel("β")
    ax2d.set_title("2D Loss Contour")

    fig2d.savefig("2d_loss_contour.png", dpi=300, bbox_inches="tight")
    wandb.log({"2D Loss Contour": wandb.Image("2d_loss_contour.png")})
    plt.show()

    plt.close(fig3d)
    plt.close(fig2d)
