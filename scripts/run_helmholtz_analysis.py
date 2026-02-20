"""Helmholtz PDE Analysis: Neural Laplacian vs Physical Laplacian.

Experiment A -- Scatter plot of neural nabla^2 p vs physical nabla^2 p.
    Neural:   nabla^2 p computed via 2nd-order autograd of forward model.
    Physical: nabla^2 p = -k^2 p  (Helmholtz identity on BEM ground truth).

Experiment B -- Fourier feature sigma second-derivative amplification.
    gamma(v) = cos(2*pi*B*v + b)
    d^2 gamma / dv^2 = -(2*pi*B)^2 * gamma(v)
    For B ~ N(0, sigma^2): E[(2*pi*B)^2] = 4*pi^2*sigma^2
    sigma=30 -> amplification ~ 35,530x

Usage
-----
    python scripts/run_helmholtz_analysis.py
    python scripts/run_helmholtz_analysis.py --n-samples 2000 --scenes 1 5 10
"""

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import RegularGridInterpolator
from scipy.special import hankel1
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.forward_model import TransferFunctionModel, build_transfer_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("helmholtz_analysis")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "phase1"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase2"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPEED_OF_SOUND_M_PER_S: float = 343.0


# ---------------------------------------------------------------------------
# Neural Laplacian: 2nd-order autograd through forward model
# ---------------------------------------------------------------------------
def compute_neural_laplacian(
    model: TransferFunctionModel,
    x_src: torch.Tensor,
    x_rcv: torch.Tensor,
    k: torch.Tensor,
    sdf_rcv: torch.Tensor,
    scene_ids: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute neural nabla^2 p via 2nd-order autograd.

    Returns
    -------
    laplacian_re : (B,) -- Re(nabla^2 p_neural)
    laplacian_im : (B,) -- Im(nabla^2 p_neural)
    p_re : (B,) -- Re(p_total) from neural model
    p_im : (B,) -- Im(p_total) from neural model
    """
    B = x_rcv.shape[0]

    # Make receiver positions differentiable
    x_eval = x_rcv.detach().clone().requires_grad_(True)  # (B, 2)

    # Forward model prediction (differentiable in x_eval)
    dx = x_eval[:, 0:1] - x_src[:, 0:1]  # (B, 1)
    dy = x_eval[:, 1:2] - x_src[:, 1:2]  # (B, 1)
    dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-30)  # (B, 1)
    inputs = torch.cat(
        [x_src, x_eval, k.unsqueeze(-1), sdf_rcv.unsqueeze(-1), dist, dx, dy],
        dim=-1,
    )  # (B, 9)

    t_pred = model(inputs, scene_ids=scene_ids)  # (B, 2)
    t_re = t_pred[:, 0] * scale  # (B,)
    t_im = t_pred[:, 1] * scale  # (B,)

    # Incident field (differentiable in x_eval)
    k_flat = k  # (B,)
    r = dist.squeeze(-1)  # (B,)
    kr = k_flat * r  # (B,)
    amp = 0.25 * torch.sqrt(2.0 / (math.pi * kr.clamp(min=1.0)))  # (B,)
    phase = kr - math.pi / 4.0  # (B,)
    p_inc_re = amp * torch.sin(phase)  # (B,)
    p_inc_im = -amp * torch.cos(phase)  # (B,)

    # Total field: p = p_inc * (1 + T)
    p_re = p_inc_re * (1.0 + t_re) - p_inc_im * t_im  # (B,)
    p_im = p_inc_im * (1.0 + t_re) + p_inc_re * t_im  # (B,)

    # Laplacian of Re(p): d^2 p_re / dx^2 + d^2 p_re / dy^2
    # 1st grad needs create_graph=True to enable 2nd differentiation
    grad_p_re = torch.autograd.grad(
        p_re.sum(), x_eval, create_graph=True, retain_graph=True,
    )[0]  # (B, 2)
    d2_p_re_dx2 = torch.autograd.grad(
        grad_p_re[:, 0].sum(), x_eval, create_graph=False, retain_graph=True,
    )[0][:, 0]  # (B,)
    d2_p_re_dy2 = torch.autograd.grad(
        grad_p_re[:, 1].sum(), x_eval, create_graph=False, retain_graph=True,
    )[0][:, 1]  # (B,)
    laplacian_re = d2_p_re_dx2 + d2_p_re_dy2  # (B,)

    # Laplacian of Im(p)
    grad_p_im = torch.autograd.grad(
        p_im.sum(), x_eval, create_graph=True, retain_graph=True,
    )[0]  # (B, 2)
    d2_p_im_dx2 = torch.autograd.grad(
        grad_p_im[:, 0].sum(), x_eval, create_graph=False, retain_graph=True,
    )[0][:, 0]  # (B,)
    d2_p_im_dy2 = torch.autograd.grad(
        grad_p_im[:, 1].sum(), x_eval, create_graph=False, retain_graph=True,
    )[0][:, 1]  # (B,)
    laplacian_im = d2_p_im_dx2 + d2_p_im_dy2  # (B,)

    return (
        laplacian_re.detach(),
        laplacian_im.detach(),
        p_re.detach(),
        p_im.detach(),
    )


# ---------------------------------------------------------------------------
# Experiment A: Neural vs Physical Laplacian
# ---------------------------------------------------------------------------
def run_experiment_a(
    model: TransferFunctionModel,
    scene_scales: Dict[int, float],
    scene_id_map: Dict[int, int],
    device: torch.device,
    scenes: List[int],
    n_samples_per_scene: int = 1000,
) -> Dict:
    """Scatter-compare neural nabla^2 p vs physical nabla^2 p = -k^2 p.

    Physical Laplacian identity:
        If p satisfies Helmholtz (nabla^2 p + k^2 p = 0),
        then nabla^2 p = -k^2 p.
        We use p_BEM as ground truth.

    Returns
    -------
    dict with neural_lap, physical_lap, pearson_r, etc.
    """
    all_neural_lap_re = []
    all_neural_lap_im = []
    all_physical_lap_re = []
    all_physical_lap_im = []
    all_residual_mag = []
    all_k_vals = []

    for sid in scenes:
        h5_path = DATA_DIR / f"scene_{sid:03d}.h5"
        if not h5_path.exists():
            logger.warning("Scene %d not found, skipping", sid)
            continue

        scale = scene_scales.get(sid)
        sid_0idx = scene_id_map.get(sid)
        if scale is None or sid_0idx is None:
            logger.warning("Scene %d missing scale/id, skipping", sid)
            continue

        with h5py.File(h5_path, "r") as f:
            freqs_hz = f["frequencies"][:]  # (F,)
            src_pos = f["sources/positions"][:]  # (S, 2)
            rcv_pos = f["receivers/positions"][:]  # (R, 2)
            sdf_grid_x = f["sdf/grid_x"][:]
            sdf_grid_y = f["sdf/grid_y"][:]
            sdf_values = f["sdf/values"][:]

            n_freq = len(freqs_hz)
            n_src = src_pos.shape[0]
            n_rcv = rcv_pos.shape[0]
            k_arr = 2.0 * np.pi * freqs_hz / SPEED_OF_SOUND_M_PER_S  # (F,)

            sdf_interp = RegularGridInterpolator(
                (sdf_grid_x, sdf_grid_y), sdf_values,
                method="linear", bounds_error=False, fill_value=1.0,
            )
            sdf_at_rcv = sdf_interp(rcv_pos)  # (R,)

            # Sample random (source, freq, receiver) tuples
            rng = np.random.RandomState(42 + sid)
            total_combos = n_src * n_freq * n_rcv
            n_draw = min(n_samples_per_scene, total_combos)
            indices = rng.choice(total_combos, size=n_draw, replace=False)

            si_arr = indices // (n_freq * n_rcv)
            remainder = indices % (n_freq * n_rcv)
            fi_arr = remainder // n_rcv
            ri_arr = remainder % n_rcv

            # Build arrays
            x_src_np = src_pos[si_arr]  # (N, 2)
            x_rcv_np = rcv_pos[ri_arr]  # (N, 2)
            k_np = k_arr[fi_arr]  # (N,)
            sdf_np = sdf_at_rcv[ri_arr]  # (N,)

            # BEM ground truth p_total
            p_bem_list = []
            for idx in range(n_draw):
                si, fi, ri = si_arr[idx], fi_arr[idx], ri_arr[idx]
                p_val = f[f"pressure/src_{si:03d}/field"][fi, ri]
                p_bem_list.append(p_val)
            p_bem = np.array(p_bem_list, dtype=np.complex128)  # (N,)

        # Physical Laplacian: nabla^2 p = -k^2 p (Helmholtz identity)
        physical_lap_re = -(k_np ** 2) * p_bem.real  # (N,)
        physical_lap_im = -(k_np ** 2) * p_bem.imag  # (N,)

        # Neural Laplacian via 2nd-order autograd
        # Process in chunks to avoid OOM
        chunk_size = 256
        neural_lap_re_list = []
        neural_lap_im_list = []
        neural_p_re_list = []
        neural_p_im_list = []

        for ci in range(0, n_draw, chunk_size):
            ce = min(ci + chunk_size, n_draw)
            x_src_t = torch.tensor(x_src_np[ci:ce], dtype=torch.float32, device=device)
            x_rcv_t = torch.tensor(x_rcv_np[ci:ce], dtype=torch.float32, device=device)
            k_t = torch.tensor(k_np[ci:ce], dtype=torch.float32, device=device)
            sdf_t = torch.tensor(sdf_np[ci:ce], dtype=torch.float32, device=device)
            sid_t = torch.full((ce - ci,), sid_0idx, dtype=torch.long, device=device)

            lap_re, lap_im, p_re, p_im = compute_neural_laplacian(
                model, x_src_t, x_rcv_t, k_t, sdf_t, sid_t, scale,
            )
            neural_lap_re_list.append(lap_re.cpu().numpy())
            neural_lap_im_list.append(lap_im.cpu().numpy())
            neural_p_re_list.append(p_re.cpu().numpy())
            neural_p_im_list.append(p_im.cpu().numpy())

        neural_lap_re = np.concatenate(neural_lap_re_list)  # (N,)
        neural_lap_im = np.concatenate(neural_lap_im_list)  # (N,)
        neural_p_re = np.concatenate(neural_p_re_list)  # (N,)
        neural_p_im = np.concatenate(neural_p_im_list)  # (N,)

        # Helmholtz residual: nabla^2 p_neural + k^2 p_neural
        res_re = neural_lap_re + (k_np ** 2) * neural_p_re
        res_im = neural_lap_im + (k_np ** 2) * neural_p_im
        residual_mag = np.sqrt(res_re ** 2 + res_im ** 2)

        all_neural_lap_re.append(neural_lap_re)
        all_neural_lap_im.append(neural_lap_im)
        all_physical_lap_re.append(physical_lap_re)
        all_physical_lap_im.append(physical_lap_im)
        all_residual_mag.append(residual_mag)
        all_k_vals.append(k_np)

        # Per-scene correlation
        r_re, _ = pearsonr(neural_lap_re, physical_lap_re)
        r_im, _ = pearsonr(neural_lap_im, physical_lap_im)
        med_res = np.median(residual_mag)
        logger.info(
            "  Scene %d: r_re=%.4f, r_im=%.4f, median_residual=%.2e (N=%d)",
            sid, r_re, r_im, med_res, n_draw,
        )

    # Aggregate
    neural_re = np.concatenate(all_neural_lap_re)
    neural_im = np.concatenate(all_neural_lap_im)
    phys_re = np.concatenate(all_physical_lap_re)
    phys_im = np.concatenate(all_physical_lap_im)
    residuals = np.concatenate(all_residual_mag)
    k_all = np.concatenate(all_k_vals)

    r_overall_re, _ = pearsonr(neural_re, phys_re)
    r_overall_im, _ = pearsonr(neural_im, phys_im)

    # Normalized residual: |nabla^2 p + k^2 p|^2 / (k^4 |p|^2)
    # Using neural p for normalization
    p_mag_sq = neural_re ** 2 + neural_im ** 2  # approximate
    k4 = k_all ** 4
    norm_res = residuals ** 2 / np.maximum(k4 * np.maximum(p_mag_sq, 1e-30), 1e-30)
    mean_norm_res = np.mean(norm_res)

    logger.info("=" * 60)
    logger.info("Experiment A: Neural vs Physical Laplacian")
    logger.info("  Overall Pearson r (Re): %.4f", r_overall_re)
    logger.info("  Overall Pearson r (Im): %.4f", r_overall_im)
    logger.info("  Mean normalized residual: %.2e", mean_norm_res)
    logger.info("  Median |residual|: %.2e", np.median(residuals))
    logger.info("  N samples: %d", len(neural_re))
    logger.info("=" * 60)

    return {
        "neural_lap_re": neural_re,
        "neural_lap_im": neural_im,
        "physical_lap_re": phys_re,
        "physical_lap_im": phys_im,
        "residuals": residuals,
        "k_vals": k_all,
        "r_re": r_overall_re,
        "r_im": r_overall_im,
        "mean_norm_residual": mean_norm_res,
        "median_residual": float(np.median(residuals)),
    }


# ---------------------------------------------------------------------------
# Experiment B: Fourier Feature sigma amplification
# ---------------------------------------------------------------------------
def run_experiment_b(
    model: TransferFunctionModel,
    device: torch.device,
) -> Dict:
    """Analyze Fourier feature sigma's effect on second derivatives.

    Mathematical analysis:
        gamma(v) = cos(2*pi*B*v)
        d^2 gamma / dv_j^2 = -(2*pi*B_j)^2 * gamma(v)

    For B_ij ~ N(0, sigma^2):
        E[(2*pi*B_ij)^2] = 4*pi^2*sigma^2

    So the Fourier encoding's 2nd derivative has variance that scales as sigma^4.
    """
    # Extract B matrix from model
    B = model.encoder.B.cpu().numpy()  # (n_fourier, d_in)
    n_fourier, d_in = B.shape

    # sigma used in this model
    # Reconstruct from B: sigma ~ std(B) (since B = randn * sigma)
    sigma_empirical = np.std(B)

    # Amplification factor per feature per dimension
    amp_factors = (2.0 * np.pi * B) ** 2  # (n_fourier, d_in)

    # Theoretical amplification: 4*pi^2*sigma^2
    sigma_theoretical = sigma_empirical
    amp_theoretical = 4.0 * np.pi ** 2 * sigma_theoretical ** 2

    # Statistics
    amp_mean = np.mean(amp_factors)
    amp_max = np.max(amp_factors)
    amp_std = np.std(amp_factors)

    # Per-dimension analysis (spatial coords x_rcv are dims 2,3)
    dim_names = ["x_s", "y_s", "x_r", "y_r", "k", "sdf", "dist", "dx", "dy"]
    dim_stats = {}
    for di in range(d_in):
        dim_stats[dim_names[di]] = {
            "mean_amp": float(np.mean(amp_factors[:, di])),
            "max_amp": float(np.max(amp_factors[:, di])),
            "std_amp": float(np.std(amp_factors[:, di])),
        }

    # Empirical measurement: evaluate 2nd derivative of Fourier features
    # at random points
    rng = np.random.RandomState(42)
    test_points = rng.randn(1000, d_in).astype(np.float32) * 0.5  # (1000, d_in)
    test_t = torch.tensor(test_points, device=device, requires_grad=True)

    # Forward through Fourier encoder only
    proj = 2.0 * math.pi * (test_t @ model.encoder.B.T)  # (1000, n_fourier)
    cos_feat = torch.cos(proj)  # (1000, n_fourier)
    sin_feat = torch.sin(proj)  # (1000, n_fourier)

    # 2nd derivative of cos feature w.r.t. x_r (dim 2)
    # d/dx_r cos(2pi B_{:,2} x_r + ...) = -2pi B_{:,2} sin(...)
    # d^2/dx_r^2 = -(2pi B_{:,2})^2 cos(...)
    # So ||d^2 gamma / dx_r^2|| = |(2pi B_{:,2})^2| * |cos(...)|
    B_xr = B[:, 2]  # (n_fourier,) -- x_r dimension
    amp_xr = (2.0 * np.pi * B_xr) ** 2  # (n_fourier,)

    # RMS of 2nd derivative across test points
    cos_np = cos_feat.detach().cpu().numpy()  # (1000, n_fourier)
    d2_cos_dx2 = -amp_xr[None, :] * cos_np  # (1000, n_fourier)
    rms_d2 = np.sqrt(np.mean(d2_cos_dx2 ** 2))

    # Compare with 0th-order RMS
    rms_0 = np.sqrt(np.mean(cos_np ** 2))
    amplification_ratio = rms_d2 / max(rms_0, 1e-30)

    logger.info("=" * 60)
    logger.info("Experiment B: Fourier Feature sigma Amplification")
    logger.info("  B matrix shape: %s", B.shape)
    logger.info("  sigma (empirical): %.2f", sigma_empirical)
    logger.info("  Theoretical amplification (4*pi^2*sigma^2): %.1f", amp_theoretical)
    logger.info("  Empirical mean amplification factor: %.1f", amp_mean)
    logger.info("  Max amplification factor: %.1f", amp_max)
    logger.info("  RMS(d^2 gamma/dx^2) / RMS(gamma): %.1f", amplification_ratio)
    logger.info("  Per-dimension amplification (spatial):")
    for dim_name in ["x_r", "y_r"]:
        ds = dim_stats[dim_name]
        logger.info(
            "    %s: mean=%.1f, max=%.1f", dim_name, ds["mean_amp"], ds["max_amp"]
        )
    logger.info("=" * 60)

    return {
        "B_matrix": B,
        "sigma_empirical": float(sigma_empirical),
        "amp_theoretical": float(amp_theoretical),
        "amp_mean": float(amp_mean),
        "amp_max": float(amp_max),
        "amplification_ratio": float(amplification_ratio),
        "dim_stats": dim_stats,
        "amp_factors": amp_factors,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_experiment_a(results: Dict, output_dir: Path) -> None:
    """Scatter plot: neural nabla^2 p vs physical nabla^2 p."""
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.35)

    neural_re = results["neural_lap_re"]
    neural_im = results["neural_lap_im"]
    phys_re = results["physical_lap_re"]
    phys_im = results["physical_lap_im"]

    # Subsample for plotting (max 5000 points)
    n = len(neural_re)
    if n > 5000:
        idx = np.random.RandomState(42).choice(n, 5000, replace=False)
    else:
        idx = np.arange(n)

    # Panel 1: Re scatter
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(
        phys_re[idx], neural_re[idx],
        s=1, alpha=0.3, c="steelblue", rasterized=True,
    )
    lim = max(np.abs(phys_re[idx]).max(), np.abs(neural_re[idx]).max()) * 1.1
    ax1.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="y=x (ideal)")
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_xlabel(r"Physical $\nabla^2 p_{\rm Re}$ $(-k^2 p_{\rm BEM})$")
    ax1.set_ylabel(r"Neural $\nabla^2 p_{\rm Re}$ (autodiff)")
    ax1.set_title(f"Re: Pearson r = {results['r_re']:.3f}")
    ax1.legend(fontsize=8)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Im scatter
    ax2 = fig.add_subplot(gs[1])
    ax2.scatter(
        phys_im[idx], neural_im[idx],
        s=1, alpha=0.3, c="coral", rasterized=True,
    )
    lim2 = max(np.abs(phys_im[idx]).max(), np.abs(neural_im[idx]).max()) * 1.1
    ax2.plot([-lim2, lim2], [-lim2, lim2], "r--", lw=1, label="y=x (ideal)")
    ax2.set_xlim(-lim2, lim2)
    ax2.set_ylim(-lim2, lim2)
    ax2.set_xlabel(r"Physical $\nabla^2 p_{\rm Im}$ $(-k^2 p_{\rm BEM})$")
    ax2.set_ylabel(r"Neural $\nabla^2 p_{\rm Im}$ (autodiff)")
    ax2.set_title(f"Im: Pearson r = {results['r_im']:.3f}")
    ax2.legend(fontsize=8)
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Residual histogram
    ax3 = fig.add_subplot(gs[2])
    residuals = results["residuals"]
    log_res = np.log10(np.maximum(residuals, 1e-30))
    ax3.hist(log_res, bins=80, color="gray", edgecolor="black", alpha=0.7)
    ax3.axvline(np.median(log_res), color="red", lw=2, label=f"Median: {10**np.median(log_res):.1e}")
    ax3.set_xlabel(r"$\log_{10} |\nabla^2 p + k^2 p|$")
    ax3.set_ylabel("Count")
    ax3.set_title("Helmholtz Residual Distribution")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        "Experiment A: Neural Laplacian vs Physical Laplacian",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(output_dir / "helmholtz_neural_vs_physical.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "helmholtz_neural_vs_physical.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_dir / "helmholtz_neural_vs_physical.pdf")


def plot_experiment_b(results: Dict, output_dir: Path) -> None:
    """Plot Fourier feature amplification analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Distribution of amplification factors
    amp_factors = results["amp_factors"]  # (n_fourier, d_in)
    ax1 = axes[0]

    # Spatial dimensions only (x_r=2, y_r=3)
    for di, (dim_idx, color, label) in enumerate([
        (2, "steelblue", r"$x_r$"),
        (3, "coral", r"$y_r$"),
    ]):
        ax1.hist(
            amp_factors[:, dim_idx], bins=50, alpha=0.6,
            color=color, label=f"{label} (mean={np.mean(amp_factors[:, dim_idx]):.0f})",
        )

    ax1.axvline(
        results["amp_theoretical"], color="red", lw=2, ls="--",
        label=f"Theory: $4\\pi^2\\sigma^2 = {results['amp_theoretical']:.0f}$",
    )
    ax1.set_xlabel(r"$(2\pi B_{ij})^2$ -- 2nd derivative amplification factor")
    ax1.set_ylabel("Count")
    ax1.set_title("Fourier Feature Amplification Distribution")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Amplification vs sigma (theoretical curve)
    ax2 = axes[1]
    sigmas = np.linspace(1, 40, 100)
    amp_curve = 4.0 * np.pi ** 2 * sigmas ** 2

    ax2.plot(sigmas, amp_curve, "k-", lw=2, label=r"$4\pi^2 \sigma^2$")
    ax2.axvline(results["sigma_empirical"], color="red", lw=1.5, ls="--",
                label=f"Current $\\sigma$ = {results['sigma_empirical']:.1f}")

    # Annotate key sigma values
    for sigma_val in [1, 5, 10, 30]:
        amp_val = 4.0 * np.pi ** 2 * sigma_val ** 2
        ax2.plot(sigma_val, amp_val, "o", ms=8, color="steelblue")
        ax2.annotate(
            f"$\\sigma$={sigma_val}\n{amp_val:.0f}$\\times$",
            (sigma_val, amp_val),
            textcoords="offset points", xytext=(10, 5), fontsize=8,
        )

    ax2.set_xlabel(r"Fourier feature $\sigma$ [m$^{-1}$]")
    ax2.set_ylabel(r"2nd derivative amplification factor")
    ax2.set_title(r"$\nabla^2$ Amplification vs $\sigma$")
    ax2.set_yscale("log")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Experiment B: Fourier Feature Second-Derivative Amplification",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(output_dir / "fourier_sigma_amplification.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fourier_sigma_amplification.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_dir / "fourier_sigma_amplification.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Helmholtz PDE Analysis")
    parser.add_argument(
        "--checkpoint", type=str, default="best_v7",
        help="Forward model checkpoint name",
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000,
        help="Samples per scene for Experiment A",
    )
    parser.add_argument(
        "--scenes", nargs="+", type=int, default=None,
        help="Scenes to analyze (default: 1,5,8,10,14)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load forward model
    ckpt_path = CHECKPOINT_DIR / f"{args.checkpoint}.pt"
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    scene_scales = ckpt["scene_scales"]
    trained_scene_list = cfg.get("trained_scene_list", sorted(scene_scales.keys()))
    scene_id_map = {sid: idx for idx, sid in enumerate(trained_scene_list)}

    model = build_transfer_model(
        d_hidden=cfg.get("d_hidden", cfg.get("hidden_dim", 768)),
        n_blocks=cfg.get("n_blocks", cfg.get("n_layers", 6)),
        n_fourier=cfg.get("n_fourier", 256),
        fourier_sigma=cfg.get("fourier_sigma", 30.0),
        dropout=cfg.get("dropout", 0.0),
        n_scenes=cfg.get("n_scenes", 0),
        scene_emb_dim=cfg.get("scene_emb_dim", 32),
        d_out=cfg.get("d_out", 2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info(
        "Model loaded: %s (epoch %d, sigma=%.1f)",
        ckpt_path.name, ckpt["epoch"], cfg.get("fourier_sigma", 30.0),
    )

    scenes = args.scenes or [1, 5, 8, 10, 14]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Experiment A
    logger.info("=" * 60)
    logger.info("Starting Experiment A: Neural vs Physical Laplacian")
    logger.info("=" * 60)
    t0 = time.time()
    results_a = run_experiment_a(
        model, scene_scales, scene_id_map, device,
        scenes=scenes, n_samples_per_scene=args.n_samples,
    )
    logger.info("Experiment A completed in %.1f s", time.time() - t0)
    plot_experiment_a(results_a, RESULTS_DIR)

    # Experiment B
    logger.info("=" * 60)
    logger.info("Starting Experiment B: Fourier sigma Amplification")
    logger.info("=" * 60)
    results_b = run_experiment_b(model, device)
    plot_experiment_b(results_b, RESULTS_DIR)

    # Save summary CSV
    import csv
    csv_path = RESULTS_DIR / "helmholtz_analysis.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["pearson_r_re", f"{results_a['r_re']:.4f}"])
        writer.writerow(["pearson_r_im", f"{results_a['r_im']:.4f}"])
        writer.writerow(["mean_norm_residual", f"{results_a['mean_norm_residual']:.2e}"])
        writer.writerow(["median_residual", f"{results_a['median_residual']:.2e}"])
        writer.writerow(["n_samples", len(results_a["neural_lap_re"])])
        writer.writerow(["sigma_empirical", f"{results_b['sigma_empirical']:.2f}"])
        writer.writerow(["amp_theoretical", f"{results_b['amp_theoretical']:.1f}"])
        writer.writerow(["amp_ratio_empirical", f"{results_b['amplification_ratio']:.1f}"])
    logger.info("Summary: %s", csv_path)

    # Print final summary
    print("\n" + "=" * 60)
    print("HELMHOLTZ ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Experiment A: Neural nabla^2 p vs Physical nabla^2 p")
    print(f"  Pearson r (Re): {results_a['r_re']:.4f}")
    print(f"  Pearson r (Im): {results_a['r_im']:.4f}")
    print(f"  Mean normalized residual: {results_a['mean_norm_residual']:.2e}")
    print(f"  Median |residual|: {results_a['median_residual']:.2e}")
    print()
    print(f"Experiment B: Fourier sigma amplification")
    print(f"  sigma = {results_b['sigma_empirical']:.1f}")
    print(f"  Theoretical amplification (4*pi^2*sigma^2): {results_b['amp_theoretical']:.0f}x")
    print(f"  Empirical RMS(d^2/dx^2) / RMS(value): {results_b['amplification_ratio']:.0f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
