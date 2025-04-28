import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection
import torch.nn as nn
import wandb

# wrap SNN into a single-arg function
def snn_forward(x):
    return SNN(
        x,             # input
        w1, w2, v1,    # global parameters
        alpha, beta,   # time constants
        spike_fn,      # spike nonlinearity
        device,        # device to run on
        recurrent,     # bool for recurrent connectivity
        snn_mask       # any mask tensor
    )

class HybridNet(nn.Module):
    def __init__(self, forward_fn, w1, w2, v1):
        super().__init__()
        # register the weight parameters
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.v1 = nn.Parameter(v1)
        # everything else is closed over in forward_fn
        self._forward_fn = forward_fn

    def forward(self, x):
        return self._forward_fn(x)

        
def visualize_loss_landscape_3d(model, model_name, w1, w2, v1,
                                    dataloader, loss_fn,
                                    radius=0.1, resolution=21,
                                    device='cpu'):
    # 0) move model and weights onto device
    model = model.to(device)
    w1, w2, v1 = w1.to(device), w2.to(device), v1.to(device)
    model.eval()

    # 1) grab exactly one batch
    inputs, targets = next(iter(dataloader))
    x, y = inputs.to(device), targets.to(device)

    # 2) save originals
    orig_w1, orig_w2, orig_v1 = w1.clone(), w2.clone(), v1.clone()
    # 3) flatten
    base = torch.cat([orig_w1.flatten(),
                      orig_w2.flatten(),
                      orig_v1.flatten()])
    # 4) pick two random orthonormal directions
    d1 = torch.randn_like(base);  d1 /= torch.norm(d1)
    d2 = torch.randn_like(base)
    d2 -= (d2 @ d1) * d1;  d2 /= torch.norm(d2)

    # 5) prepare grid
    alphas = np.linspace(-radius, radius, resolution)
    betas  = np.linspace(-radius, radius, resolution)
    loss_mat = np.zeros((resolution, resolution), dtype=np.float32)

    # 6) sweep the grid
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            offset = a*d1 + b*d2
            idx = 0
            # assign perturbed params
            for tensor, orig in zip((w1, w2, v1),
                                    (orig_w1, orig_w2, orig_v1)):
                n = orig.numel()
                pert = (base + offset)[idx:idx+n].view(orig.shape).to(device)
                tensor.data.copy_(pert)
                idx += n

            # forward & loss
            out, recs = model(x)
            m, _ = out.max(1)
            logp = torch.nn.functional.log_softmax(m, dim=1)
            loss_mat[j, i] = loss_fn(logp, y).item()

    # 7) restore originals
    w1.data.copy_(orig_w1)
    w2.data.copy_(orig_w2)
    v1.data.copy_(orig_v1)

    # 8a) 2D contour
    fig2d, ax2d = plt.subplots(figsize=(5,4))
    cs = ax2d.contourf(alphas, betas, loss_mat, levels=30, cmap='viridis')
    fig2d.colorbar(cs, ax=ax2d, label="Loss")
    ax2d.set_xlabel("α")
    ax2d.set_ylabel("β")
    ax2d.set_title("2D Loss Contour")
    # Save as high-res PNG
    fig2d.savefig(f"{model_name}_2d_loss_contour.png", dpi=300, bbox_inches="tight")
    wandb.log({ "2D Loss Contour": wandb.Image(f"{model_name}_2d_loss_contour.png") })
    plt.show()

    # 8b) 3D surface
    A, B = np.meshgrid(alphas, betas)
    fig3d = plt.figure(figsize=(6,5))
    ax3d  = fig3d.add_subplot(111, projection='3d')
    surf = ax3d.plot_surface(A, B, loss_mat, cmap='viridis',
                            edgecolor='none', alpha=0.9)
    ax3d.set_xlabel("α"); ax3d.set_ylabel("β"); ax3d.set_zlabel("Loss")
    ax3d.set_title("3D Loss Surface")
    fig3d.colorbar(surf, shrink=0.5, aspect=10, label="Loss")
    # Save to PNG
    fig3d.savefig(f"{model_name}_3d_loss_surface.png", dpi=300, bbox_inches="tight")
    wandb.log({ "3D Loss Surface": wandb.Image(f"{model_name}_3d_loss_surface.png") })
    plt.show()


"""
# Usage:
visualize_loss_landscape_loader(
    model=HybridNet(w1, w2, v1, Hybrid_RNN_SNN_V1_same_layer),
    w1=w1, w2=w2, v1=v1,
    dataloader=test_loader,
    loss_fn=torch.nn.NLLLoss(),
    radius=0.05,
    resolution=31,
    device=device
)
"""