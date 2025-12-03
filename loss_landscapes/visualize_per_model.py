import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import argparse
import os

sys.path.append("/scratch/nar8991/snn/snn_ann_hybrid")
from class_based_implementation.models import SNN, ANN_with_LIF_output, Hybrid_RNN_SNN_rec, Hybrid_RNN_SNN_V1_same_layer, NSN_with_LIF_output, Hybrid_NSN_SNN_rec, Hybrid_NSN_SNN_V1_same_layer, Hybrid_NSN_SNN_V1_Flexible_Spiking
from class_based_implementation.train_model import objective, MODEL_CLASSES, LOSS_FUNCTIONS # Import mappings too

from loss_landscapes.utils import get_shd_train_data, get_randman_train_data, _flatten_params, _unflatten_params, _assign_flat_params

# MODEL_CLASSES = {
#     "NSN_with_LIF_output": NSN_with_LIF_output, "SNN": SNN, "ANN_with_LIF_output": ANN_with_LIF_output,
#     "Hybrid_RNN_SNN_rec": Hybrid_RNN_SNN_rec, "Hybrid_NSN_SNN_rec": Hybrid_NSN_SNN_rec,
#     "Hybrid_RNN_SNN_V1_same_layer": Hybrid_RNN_SNN_V1_same_layer, "Hybrid_NSN_SNN_V1_same_layer": Hybrid_NSN_SNN_V1_same_layer,

# }


def get_dataloader(data):
    if data == 'randman':
        return get_randman_train_data()
    elif data == 'shd':
        return get_shd_train_data()

# --- Visualize loss landscape ---
def visualize_loss_landscape(model, dataloader, d1, d2,
                             criterion=torch.nn.CrossEntropyLoss(),
                             device="cuda",
                             resolution=21,
                             range_lim=1.0,
                             save_prefix=None, BASE_PATH=None,
                             exclude_gamma=True, preload_data_to_device=True, use_amp=True, ymax=None):
    # Require GPU device here (caller should have ensured CUDA availability).
    try:
        model.to(device)
    except Exception as e:
        print("ERROR: failed to move model to device:", device)
        print("Exception during device init:", repr(e))
        print("Diagnostics:")
        print("  - CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("  - If using GREENE_GPU_MPS or other MPS env vars, try unsetting them (e.g. unset GREENE_GPU_MPS).")
        print("  - Ensure the job has an allocated GPU and the CUDA/MPS daemon is available.")
        # exit non-zero so SLURM logs show clear failure
        sys.exit(1)
    model.eval()

    # Filter out gamma parameters if requested
    if exclude_gamma:
        params = [p for name, p in model.named_parameters() if 'gamma' not in name.lower()]
        print(f"Excluding gamma parameters. Using {len(params)} parameters for visualization.")
    else:
        params = list(model.parameters())
    base_vec, shapes = _flatten_params(params)
    base_vec = base_vec.to(device)

    alphas = np.linspace(-range_lim, range_lim, resolution)
    betas = np.linspace(-range_lim, range_lim, resolution)
    loss_grid = np.zeros((resolution, resolution), dtype=np.float32)

    # Preload dataloader to device to avoid repeated host->device transfers
    if preload_data_to_device:
        data_on_device = []
        for xb, yb in dataloader:
            xb_dev = xb.to(device, non_blocking=True)
            yb_dev = yb.to(device, non_blocking=True)
            data_on_device.append((xb_dev, yb_dev))
        dataloader = data_on_device

    # Main evaluation loops: disable autograd
    with torch.no_grad():
        for i, a in enumerate(tqdm(alphas, desc="α")):
            for j, b in enumerate(betas):
                offset = (a * d1 + b * d2).to(device)
                # assign new params (existing helper)
                _assign_flat_params(params, base_vec + offset, shapes, device=device)

                total_loss = 0.0
                total_count = 0
                # use AMP context (explicit) only if use_amp True
                with torch.cuda.amp.autocast(enabled=use_amp):
                    for xb, yb in dataloader:
                        outputs = model(xb)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        if outputs.dim() > 2:
                            outputs, _ = torch.max(outputs, dim=1)
                        loss = criterion(outputs, yb)
                        total_loss += float(loss.item()) * xb.size(0)
                        total_count += xb.size(0)
                loss_grid[j, i] = total_loss / total_count if total_count > 0 else float('nan')

    # restore original params
    _assign_flat_params(params, base_vec, shapes, device=device)

    A, B = np.meshgrid(alphas, betas)
    fig3d = plt.figure(figsize=(6,5))
    ax3d = fig3d.add_subplot(111, projection="3d")
    surf = ax3d.plot_surface(A, B, loss_grid, cmap="viridis", edgecolor="none", alpha=0.9)
    ax3d.set_xlabel("α"); ax3d.set_ylabel("β"); ax3d.set_zlabel("Loss")
    ax3d.set_title(f"{save_prefix} - 3D Loss Surface")
    # If a ymax is provided, set the beta (y) axis upper limit (keep lower bound = -range_lim)
    if ymax is not None:
        try:
            ymin = 0
            ax3d.set_ylim(ymin, float(ymax))
        except Exception:
            pass
    fig3d.colorbar(surf, shrink=0.6, aspect=10, label="Loss")
    if save_prefix: fig3d.savefig(f"{BASE_PATH}/{save_prefix}_3d.png", dpi=300, bbox_inches="tight")
    fig2d, ax2d = plt.subplots(figsize=(5,4))
    cs = ax2d.contour(alphas, betas, loss_grid, levels=30, cmap="viridis")
    fig2d.colorbar(cs, ax=ax2d, label="Loss")
    ax2d.set_xlabel("α"); ax2d.set_ylabel("β"); ax2d.set_title(f"{save_prefix} - 2D Loss Contour")
    if ymax is not None:
        try:
            ymin = 0
            ax2d.set_ylim(ymin, float(ymax))
        except Exception:
            pass
    if save_prefix: fig2d.savefig(f"{BASE_PATH}/{save_prefix}_2d.png", dpi=300, bbox_inches="tight")
    return fig3d, fig2d

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize loss landscape for a single model.")
    parser.add_argument('--data', type=str, choices=['shd', 'randman'], required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--recurrent', type=str, required=True)
    parser.add_argument('--percent', type=str, default="None")
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--sampler_type', type=str, default="grid")
    parser.add_argument('--gamma_init_type', type=str, default="rand", choices=["rand", "const_zero", "const_one", "const_half"])
    parser.add_argument('--gamma_fixed', action='store_true', help="If set, gamma was fixed during training")
    parser.add_argument('--no_reset_nsn', dest='reset_nsn', action='store_false', help="If set, NSN neurons do NOT reset after firing")
    parser.add_argument('--range_lim', type=int, default=1)
    parser.add_argument('--ymax', type=float, default=None, help="Optional upper limit for beta (y) axis on plots")
    args = parser.parse_args()

    # require CUDA available and use it; fail early if not present
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU. Request a GPU and retry (e.g. sbatch with --gres=gpu).")
        sys.exit(1)
    runtime_device = "cuda"
    print("Using device:", runtime_device)

    # Get data and correct base path
    dataloader = get_dataloader(args.data)
    if args.data == 'randman':
        BASE_PATH = "/vast/nar8991/snn/training_results/randman/1_d/2_classes/3/cross_entropy"
        hidden_neurons = 20
    elif args.data == 'shd':
        BASE_PATH = "/vast/nar8991/snn/training_results/shd/None_d/20_classes/700/cross_entropy"
        hidden_neurons = 256

    # Load PCA directions and move them to the chosen runtime device (fallback if needed)
    pca_file = os.path.join(BASE_PATH, "pca_directions", "shared_pca_directions.pt")
    if not os.path.exists(pca_file):
        raise FileNotFoundError(f"PCA directions not found. Please run compute_pca.py first. File not found: {pca_file}")
    # load PCA tensors directly onto the runtime device (GPU)
    pca_data = torch.load(pca_file, map_location=runtime_device)
    d1 = pca_data["d1"].to(runtime_device)
    d2 = pca_data["d2"].to(runtime_device)

    # Load the specific model for this task
    model_class = MODEL_CLASSES.get(args.model_name)
    percent = float(args.percent) if args.percent != "None" else None
    base_path = f"{BASE_PATH}/{args.model_name}/recurrent_{args.recurrent}/seed_{args.seed}/{args.sampler_type}_sampler/{hidden_neurons}_hidden/1.0_pct_data"

    percent_str = f"{percent}_percent_snn" if percent else "None_percent_snn"
    extra_dirs = []
    extra_dirs.append(percent_str)

    # Add model-specific directories
    if args.model_name == "Hybrid_NSN_SNN_V1_Flexible_Spiking":
        extra_dirs.extend([
            f"{args.gamma_init_type}_gamma_init",
            f"gamma_fixed_{args.gamma_fixed}"
        ])
        if not args.reset_nsn:
            extra_dirs.append(f"reset_nsn_{args.reset_nsn}")
    elif "NSN" in args.model_name and not args.reset_nsn:
        extra_dirs.append("reset_nsn_False")

    model_file_path = f"{base_path}/{'/'.join(extra_dirs)}/best_model_of_study.ckpt" if extra_dirs else f"{base_path}/best_model_of_study.ckpt"
    if not os.path.exists(model_file_path):
        print(f"Model file not found for visualization: {model_file_path}")
        exit()

    print(f"Loading model for visualization: {model_file_path}")
    # load checkpoint directly onto the runtime GPU to avoid device transfer issues
    try:
        model = model_class.load_from_checkpoint(model_file_path, map_location=runtime_device)
    except Exception as e:
        print("ERROR: failed to load checkpoint directly onto device:", runtime_device)
        print("Exception:", repr(e))
        print("Check that the checkpoint is valid and that the GPU environment is correctly configured.")
        sys.exit(1)

    # Visualize the loss landscape (pass explicit runtime device and disable AMP by default)
    save_prefix = f"{args.model_name}_seed{args.seed}_percent{percent}_reset_nsn{args.reset_nsn}"
    visualize_loss_landscape(
        model, dataloader, d1, d2,
        criterion=torch.nn.CrossEntropyLoss(),
        device=runtime_device, resolution=21, range_lim=args.range_lim,
        save_prefix=save_prefix, BASE_PATH=BASE_PATH,
        use_amp=False, preload_data_to_device=True, ymax=args.ymax
    )