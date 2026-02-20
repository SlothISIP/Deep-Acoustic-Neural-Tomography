"""Generate publication-quality Helmholtz analysis figure for paper.

Produces fig_5_helmholtz_analysis.pdf (replaces old ablation bar chart).
Two-panel figure:
    (a) Scatter: neural nabla^2 p vs physical nabla^2 p  (r = 0.19)
    (b) sigma amplification curve with annotated sigma values

Usage
-----
    python scripts/generate_helmholtz_figure.py
"""

import logging
import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.forward_model import build_transfer_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("helmholtz_fig")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "phase1"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase2"
FIGURE_DIR = PROJECT_ROOT / "results" / "paper_figures"

SPEED_OF_SOUND_M_PER_S: float = 343.0


def compute_neural_laplacian_batch(
    model, x_src, x_rcv, k, sdf_rcv, scene_ids, scale, device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute neural Laplacian for a batch via 2nd-order autograd."""
    x_eval = x_rcv.clone().requires_grad_(True)
    dx = x_eval[:, 0:1] - x_src[:, 0:1]
    dy = x_eval[:, 1:2] - x_src[:, 1:2]
    dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-30)
    inputs = torch.cat(
        [x_src, x_eval, k.unsqueeze(-1), sdf_rcv.unsqueeze(-1), dist, dx, dy],
        dim=-1,
    )

    t_pred = model(inputs, scene_ids=scene_ids)
    t_re = t_pred[:, 0] * scale
    t_im = t_pred[:, 1] * scale

    r = dist.squeeze(-1)
    kr = k * r
    amp = 0.25 * torch.sqrt(2.0 / (math.pi * kr.clamp(min=1.0)))
    phase = kr - math.pi / 4.0
    p_inc_re = amp * torch.sin(phase)
    p_inc_im = -amp * torch.cos(phase)

    p_re = p_inc_re * (1.0 + t_re) - p_inc_im * t_im
    p_im = p_inc_im * (1.0 + t_re) + p_inc_re * t_im

    grad_p_re = torch.autograd.grad(
        p_re.sum(), x_eval, create_graph=True, retain_graph=True,
    )[0]
    d2_re_dx2 = torch.autograd.grad(
        grad_p_re[:, 0].sum(), x_eval, create_graph=False, retain_graph=True,
    )[0][:, 0]
    d2_re_dy2 = torch.autograd.grad(
        grad_p_re[:, 1].sum(), x_eval, create_graph=False, retain_graph=True,
    )[0][:, 1]
    lap_re = (d2_re_dx2 + d2_re_dy2).detach().cpu().numpy()

    grad_p_im = torch.autograd.grad(
        p_im.sum(), x_eval, create_graph=True, retain_graph=True,
    )[0]
    d2_im_dx2 = torch.autograd.grad(
        grad_p_im[:, 0].sum(), x_eval, create_graph=False, retain_graph=True,
    )[0][:, 0]
    d2_im_dy2 = torch.autograd.grad(
        grad_p_im[:, 1].sum(), x_eval, create_graph=False, retain_graph=True,
    )[0][:, 1]
    lap_im = (d2_im_dx2 + d2_im_dy2).detach().cpu().numpy()

    return lap_re, lap_im, p_re.detach().cpu().numpy(), p_im.detach().cpu().numpy()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt = torch.load(CHECKPOINT_DIR / "best_v7.pt", map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_transfer_model(
        d_hidden=cfg.get("d_hidden", 768),
        n_blocks=cfg.get("n_blocks", 6),
        n_fourier=cfg.get("n_fourier", 256),
        fourier_sigma=cfg.get("fourier_sigma", 30.0),
        n_scenes=cfg.get("n_scenes", 15),
        scene_emb_dim=cfg.get("scene_emb_dim", 32),
        d_out=cfg.get("d_out", 2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    scene_scales = ckpt["scene_scales"]
    tsl = cfg.get("trained_scene_list", sorted(scene_scales.keys()))
    sid_map = {sid: idx for idx, sid in enumerate(tsl)}

    # Collect data from 5 scenes
    scenes = [1, 5, 8, 10, 14]
    all_neural_re, all_phys_re = [], []

    for sid in scenes:
        h5_path = DATA_DIR / f"scene_{sid:03d}.h5"
        scale = scene_scales[sid]
        sid_0 = sid_map[sid]

        with h5py.File(h5_path, "r") as f:
            freqs_hz = f["frequencies"][:]
            src_pos = f["sources/positions"][:]
            rcv_pos = f["receivers/positions"][:]
            sdf_gx = f["sdf/grid_x"][:]
            sdf_gy = f["sdf/grid_y"][:]
            sdf_vals = f["sdf/values"][:]

            k_arr = 2.0 * np.pi * freqs_hz / SPEED_OF_SOUND_M_PER_S
            sdf_interp = RegularGridInterpolator(
                (sdf_gx, sdf_gy), sdf_vals,
                method="linear", bounds_error=False, fill_value=1.0,
            )
            sdf_at_rcv = sdf_interp(rcv_pos)

            rng = np.random.RandomState(42 + sid)
            n_freq, n_src, n_rcv = len(freqs_hz), src_pos.shape[0], rcv_pos.shape[0]
            total = n_src * n_freq * n_rcv
            n_draw = min(2000, total)
            indices = rng.choice(total, n_draw, replace=False)

            si = indices // (n_freq * n_rcv)
            rem = indices % (n_freq * n_rcv)
            fi = rem // n_rcv
            ri = rem % n_rcv

            x_src_np = src_pos[si]
            x_rcv_np = rcv_pos[ri]
            k_np = k_arr[fi]
            sdf_np = sdf_at_rcv[ri]

            p_bem = np.array([
                f[f"pressure/src_{si[j]:03d}/field"][fi[j], ri[j]]
                for j in range(n_draw)
            ], dtype=np.complex128)

        phys_re = -(k_np ** 2) * p_bem.real

        chunk = 256
        neural_re_parts = []
        for ci in range(0, n_draw, chunk):
            ce = min(ci + chunk, n_draw)
            lr, _, _, _ = compute_neural_laplacian_batch(
                model,
                torch.tensor(x_src_np[ci:ce], dtype=torch.float32, device=device),
                torch.tensor(x_rcv_np[ci:ce], dtype=torch.float32, device=device),
                torch.tensor(k_np[ci:ce], dtype=torch.float32, device=device),
                torch.tensor(sdf_np[ci:ce], dtype=torch.float32, device=device),
                torch.full((ce - ci,), sid_0, dtype=torch.long, device=device),
                scale, device,
            )
            neural_re_parts.append(lr)

        all_neural_re.append(np.concatenate(neural_re_parts))
        all_phys_re.append(phys_re)

    neural_re = np.concatenate(all_neural_re)
    phys_re = np.concatenate(all_phys_re)
    r_val, _ = pearsonr(neural_re, phys_re)

    # B matrix for sigma analysis
    B = model.encoder.B.cpu().numpy()
    sigma_emp = np.std(B)

    # -----------------------------------------------------------------------
    # Generate publication figure
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))  # ICASSP column width

    # (a) Scatter plot
    ax = axes[0]
    n = len(neural_re)
    idx = np.random.RandomState(42).choice(n, min(3000, n), replace=False)

    ax.scatter(
        phys_re[idx], neural_re[idx],
        s=0.5, alpha=0.25, c="#4878CF", rasterized=True,
    )
    lim = np.percentile(np.abs(np.concatenate([phys_re[idx], neural_re[idx]])), 99.5) * 1.2
    ax.plot([-lim, lim], [-lim, lim], "r-", lw=0.8, alpha=0.7)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"Physical $\nabla^2 p$ ($-k^2 p_{\rm BEM}$)", fontsize=8)
    ax.set_ylabel(r"Neural $\nabla^2 p$ (autodiff)", fontsize=8)
    ax.set_title(f"(a) $r = {r_val:.2f}$: near-zero correlation", fontsize=8, fontweight="bold")
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)

    # (b) sigma amplification curve
    ax2 = axes[1]
    sigmas = np.linspace(0.5, 40, 200)
    amp = 4.0 * np.pi ** 2 * sigmas ** 2

    ax2.plot(sigmas, amp, "k-", lw=1.5)
    ax2.fill_between(sigmas, amp, alpha=0.08, color="gray")

    for sv, color, marker in [(1, "#2CA02C", "o"), (5, "#FF7F0E", "s"),
                               (10, "#D62728", "D"), (30, "#9467BD", "^")]:
        av = 4.0 * np.pi ** 2 * sv ** 2
        ax2.plot(sv, av, marker, ms=6, color=color, zorder=5)
        ax2.annotate(
            f"$\\sigma$={sv}\n{av:.0f}$\\times$",
            (sv, av), textcoords="offset points",
            xytext=(-45, 5) if sv == 30 else (8, -2),
            fontsize=6.5, color=color,
        )

    ax2.set_xlabel(r"Fourier bandwidth $\sigma_{\rm FF}$ [m$^{-1}$]", fontsize=8)
    ax2.set_ylabel(r"$\nabla^2$ amplification $4\pi^2\sigma^2$", fontsize=8)
    ax2.set_title("(b) 2nd-derivative amplification", fontsize=8, fontweight="bold")
    ax2.set_yscale("log")
    ax2.tick_params(labelsize=7)
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim(0, 40)
    ax2.set_ylim(1, 5e5)

    fig.tight_layout(pad=0.5)
    fig.savefig(FIGURE_DIR / "fig_5_helmholtz_analysis.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig_5_helmholtz_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: fig_5_helmholtz_analysis.pdf")

    print(f"\nFigure generated: r = {r_val:.3f}")
    print(f"sigma = {sigma_emp:.1f}, amplification = {4*np.pi**2*sigma_emp**2:.0f}x")


if __name__ == "__main__":
    main()
