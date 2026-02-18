"""Diagnostic: Scene 13, Source 3 (index 2) prediction error analysis.

WHY does S13 source 3 have ~29% reconstruction error?  This script
dissects the prediction into spatial, spectral, and amplitude
components to identify the dominant failure mode.

Analyses
--------
    1. |T_norm| distribution (histogram + percentiles)
    2. Spatial error map across receivers (which receivers worst?)
    3. Frequency-resolved error (which frequencies worst?)
    4. Model bias: systematic OVER or UNDER-prediction of |T|?
    5. Error concentration: fraction of total error from top-10% worst receivers
    6. Phase error vs magnitude error decomposition
    7. Receiver-region breakdown (shadow / transition / lit)

Usage
-----
    python scripts/diag_s13_src3.py
"""

import logging
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
from scipy.special import hankel1

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
logger = logging.getLogger("diag_s13_src3")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPEED_OF_SOUND_M_PER_S: float = 343.0
SCENE_ID: int = 13
SOURCE_IDX: int = 2  # 0-indexed: source 3 = index 2
REGION_NAMES: Dict[int, str] = {0: "shadow", 1: "transition", 2: "lit"}

# Paths
H5_PATH = PROJECT_ROOT / "data" / "phase1" / "scene_013.h5"
CKPT_PATH = PROJECT_ROOT / "checkpoints" / "phase2" / "best_v13.pt"
OUT_DIR = PROJECT_ROOT / "results" / "phase2" / "s13_diagnostic"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_scene_data(h5_path: Path) -> Dict:
    """Load S13 source 3 ground truth and geometry.

    Returns
    -------
    dict with keys: freqs_hz, src_pos, rcv_pos, p_total_bem,
                    region_labels, sdf_grid_x, sdf_grid_y, sdf_values
    """
    with h5py.File(h5_path, "r") as f:
        data = {
            "freqs_hz": f["frequencies"][:],                          # (F,)
            "src_pos": f["sources/positions"][:],                     # (S, 2)
            "rcv_pos": f["receivers/positions"][:],                   # (R, 2)
            "p_total_bem": f[f"pressure/src_{SOURCE_IDX:03d}/field"][:],  # (F, R) complex128
            "region_labels": f[f"regions/src_{SOURCE_IDX:03d}/labels"][:],  # (R,)
            "sdf_grid_x": f["sdf/grid_x"][:],
            "sdf_grid_y": f["sdf/grid_y"][:],
            "sdf_values": f["sdf/values"][:],
        }
    logger.info(
        "Loaded S13 src3: F=%d, R=%d, freq_range=[%.0f, %.0f] Hz",
        len(data["freqs_hz"]),
        data["rcv_pos"].shape[0],
        data["freqs_hz"][0],
        data["freqs_hz"][-1],
    )
    return data


def load_model(
    ckpt_path: Path, device: torch.device
) -> Tuple[torch.nn.Module, float, int, Dict]:
    """Load best_v13 model and return (model, scene_scale, scene_0idx, config).

    Returns
    -------
    model : TransferFunctionModel
    scene_scale : float
        Per-scene RMS normalization scale for S13.
    scene_0idx : int
        0-indexed scene embedding ID for S13.
    config : dict
        Full checkpoint config.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = build_transfer_model(
        d_hidden=cfg.get("d_hidden", 768),
        n_blocks=cfg.get("n_blocks", 8),
        n_fourier=cfg.get("n_fourier", 128),
        fourier_sigma=cfg.get("fourier_sigma", 30.0),
        dropout=cfg.get("dropout", 0.0),
        n_scenes=cfg.get("n_scenes", 15),
        scene_emb_dim=cfg.get("scene_emb_dim", 32),
        d_out=cfg.get("d_out", 2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    scene_scales = ckpt["scene_scales"]
    trained_scene_list = cfg.get(
        "trained_scene_list", sorted(scene_scales.keys())
    )
    scene_id_map = {sid: idx for idx, sid in enumerate(trained_scene_list)}

    scene_scale = scene_scales[SCENE_ID]
    scene_0idx = scene_id_map[SCENE_ID]

    logger.info(
        "Model loaded: epoch=%d, val_loss=%.4e, scene_scale=%.4f, "
        "scene_0idx=%d, n_params=%d",
        ckpt["epoch"],
        ckpt["best_val_loss"],
        scene_scale,
        scene_0idx,
        sum(p.numel() for p in model.parameters()),
    )
    return model, scene_scale, scene_0idx, cfg


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_transfer_function(
    model: torch.nn.Module,
    data: Dict,
    scene_scale: float,
    scene_0idx: int,
    device: torch.device,
) -> Dict:
    """Predict T for all (freq, receiver) pairs and reconstruct p_total.

    Returns
    -------
    dict with keys:
        T_pred_complex  : (F, R) complex128  -- denormalized predicted T
        T_norm_pred     : (F*R, 2)           -- normalized (Re, Im) from model
        T_gt_complex    : (F, R) complex128  -- ground-truth T
        p_total_pred    : (F, R) complex128  -- predicted total field
        p_inc           : (F, R) complex128  -- incident field
    """
    freqs_hz = data["freqs_hz"]  # (F,)
    src_pos = data["src_pos"]    # (S, 2)
    rcv_pos = data["rcv_pos"]    # (R, 2)
    p_total_bem = data["p_total_bem"]  # (F, R)

    n_freq = len(freqs_hz)
    n_rcv = rcv_pos.shape[0]
    k_arr = 2.0 * np.pi * freqs_hz / SPEED_OF_SOUND_M_PER_S  # (F,)

    # SDF interpolation at receiver positions
    sdf_interp = RegularGridInterpolator(
        (data["sdf_grid_x"], data["sdf_grid_y"]),
        data["sdf_values"],
        method="linear",
        bounds_error=False,
        fill_value=1.0,
    )
    sdf_at_rcv = sdf_interp(rcv_pos)  # (R,)

    xs_m, ys_m = src_pos[SOURCE_IDX]
    dx_sr = rcv_pos[:, 0] - xs_m  # (R,)
    dy_sr = rcv_pos[:, 1] - ys_m  # (R,)
    dist_sr = np.sqrt(dx_sr ** 2 + dy_sr ** 2)  # (R,)
    dist_sr_safe = np.maximum(dist_sr, 1e-15)

    # Incident field: p_inc = -(i/4) H_0^(1)(kr)
    kr_all = k_arr[:, None] * dist_sr_safe[None, :]  # (F, R)
    p_inc = -0.25j * hankel1(0, kr_all)  # (F, R) complex128

    # Ground-truth transfer function
    p_scat_gt = p_total_bem - p_inc  # (F, R)
    p_inc_safe = np.where(np.abs(p_inc) > 1e-15, p_inc, 1e-15 + 0j)
    T_gt = p_scat_gt / p_inc_safe  # (F, R) complex128

    # Build model inputs and predict in frequency chunks
    all_pred_raw = []  # collect normalized predictions
    chunk_size = 50

    for fi_start in range(0, n_freq, chunk_size):
        fi_end = min(fi_start + chunk_size, n_freq)
        n_f = fi_end - fi_start
        n = n_f * n_rcv

        inputs = np.column_stack([
            np.full(n, xs_m),
            np.full(n, ys_m),
            np.tile(rcv_pos[:, 0], n_f),
            np.tile(rcv_pos[:, 1], n_f),
            np.repeat(k_arr[fi_start:fi_end], n_rcv),
            np.tile(sdf_at_rcv, n_f),
            np.tile(dist_sr, n_f),
            np.tile(dx_sr, n_f),
            np.tile(dy_sr, n_f),
        ]).astype(np.float32)  # (n_f*R, 9)

        inputs_t = torch.from_numpy(inputs).to(device)
        sid_t = torch.full(
            (n,), scene_0idx, dtype=torch.long, device=device
        )
        pred_raw = model(inputs_t, scene_ids=sid_t).cpu().numpy()  # (n_f*R, 2)
        all_pred_raw.append(pred_raw)

    T_norm_pred = np.concatenate(all_pred_raw, axis=0)  # (F*R, 2)

    # Denormalize: T_pred = T_norm * scene_scale
    t_re_denorm = T_norm_pred[:, 0] * scene_scale  # (F*R,)
    t_im_denorm = T_norm_pred[:, 1] * scene_scale  # (F*R,)
    T_pred_complex = (t_re_denorm + 1j * t_im_denorm).reshape(n_freq, n_rcv)  # (F, R)

    # Reconstruct total field
    p_total_pred = p_inc * (1.0 + T_pred_complex)  # (F, R)

    return {
        "T_pred_complex": T_pred_complex,
        "T_norm_pred": T_norm_pred,
        "T_gt_complex": T_gt,
        "p_total_pred": p_total_pred,
        "p_inc": p_inc,
        "dist_sr": dist_sr,
        "sdf_at_rcv": sdf_at_rcv,
    }


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------
def analyze_T_distribution(
    T_gt: np.ndarray,
    T_pred: np.ndarray,
    T_norm_pred: np.ndarray,
    scene_scale: float,
    out_dir: Path,
) -> None:
    """Analysis 1: Distribution of |T| for ground truth and prediction.

    Parameters
    ----------
    T_gt : (F, R) complex128
    T_pred : (F, R) complex128
    T_norm_pred : (F*R, 2) float -- normalized predictions
    """
    abs_T_gt = np.abs(T_gt).ravel()  # (F*R,)
    abs_T_pred = np.abs(T_pred).ravel()  # (F*R,)
    abs_T_norm = np.sqrt(T_norm_pred[:, 0] ** 2 + T_norm_pred[:, 1] ** 2)

    # Print statistics
    print("\n" + "=" * 70)
    print("ANALYSIS 1: |T| Distribution (Scene 13, Source 3)")
    print("=" * 70)

    for name, arr in [
        ("|T_gt| (ground truth)", abs_T_gt),
        ("|T_pred| (denormalized)", abs_T_pred),
        ("|T_norm| (normalized)", abs_T_norm),
    ]:
        pcts = np.percentile(arr, [5, 25, 50, 75, 90, 95, 99, 99.9])
        print(f"\n  {name}:")
        print(f"    mean={arr.mean():.4f}, std={arr.std():.4f}, "
              f"min={arr.min():.4f}, max={arr.max():.4f}")
        print(f"    P5={pcts[0]:.4f}, P25={pcts[1]:.4f}, P50={pcts[2]:.4f}, "
              f"P75={pcts[3]:.4f}")
        print(f"    P90={pcts[4]:.4f}, P95={pcts[5]:.4f}, P99={pcts[6]:.4f}, "
              f"P99.9={pcts[7]:.4f}")
    print(f"\n  scene_scale = {scene_scale:.6f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Histogram of |T_gt| vs |T_pred|
    ax = axes[0]
    bins = np.linspace(0, min(np.percentile(abs_T_gt, 99.5), 10), 80)
    ax.hist(abs_T_gt, bins=bins, alpha=0.6, label="|T| ground truth", density=True)
    ax.hist(abs_T_pred, bins=bins, alpha=0.6, label="|T| predicted", density=True)
    ax.set_xlabel("|T|")
    ax.set_ylabel("Density")
    ax.set_title("(a) |T| Distribution: GT vs Predicted")
    ax.legend(fontsize=8)
    ax.set_xlim(0, bins[-1])

    # (b) |T_norm| histogram
    ax = axes[1]
    bins_norm = np.linspace(0, min(np.percentile(abs_T_norm, 99.5), 10), 80)
    ax.hist(abs_T_norm, bins=bins_norm, alpha=0.7, color="green", density=True)
    ax.axvline(1.0, color="r", linestyle="--", label="RMS=1 reference")
    ax.set_xlabel("|T_norm|")
    ax.set_ylabel("Density")
    ax.set_title("(b) |T_norm| Distribution (Model Output Space)")
    ax.legend(fontsize=8)

    # (c) Q-Q plot: |T_gt| vs |T_pred| (sorted quantiles)
    ax = axes[2]
    n_q = min(1000, len(abs_T_gt))
    quantiles = np.linspace(0, 100, n_q)
    q_gt = np.percentile(abs_T_gt, quantiles)
    q_pred = np.percentile(abs_T_pred, quantiles)
    ax.plot(q_gt, q_pred, "b.", markersize=1)
    lim = max(q_gt[-1], q_pred[-1]) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=0.8, label="y=x")
    ax.set_xlabel("|T| Ground Truth (quantiles)")
    ax.set_ylabel("|T| Predicted (quantiles)")
    ax.set_title("(c) Q-Q Plot: |T| GT vs Pred")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    fig.suptitle("Scene 13 Source 3: Transfer Function Magnitude Distribution", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "01_T_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: 01_T_distribution.png")


def analyze_spatial_error(
    p_total_bem: np.ndarray,
    p_total_pred: np.ndarray,
    rcv_pos: np.ndarray,
    region_labels: np.ndarray,
    out_dir: Path,
) -> None:
    """Analysis 2: Spatial distribution of error across receivers.

    Parameters
    ----------
    p_total_bem : (F, R) complex128
    p_total_pred : (F, R) complex128
    rcv_pos : (R, 2)
    region_labels : (R,)
    """
    n_freq, n_rcv = p_total_bem.shape
    diff = p_total_pred - p_total_bem  # (F, R)

    # Per-receiver L2 error (summed over freq)
    rcv_diff_sq = np.sum(np.abs(diff) ** 2, axis=0)  # (R,)
    rcv_ref_sq = np.sum(np.abs(p_total_bem) ** 2, axis=0)  # (R,)
    rcv_rel_err = np.sqrt(rcv_diff_sq / np.maximum(rcv_ref_sq, 1e-30))  # (R,)

    # Absolute error per receiver
    rcv_abs_err = np.sqrt(rcv_diff_sq)  # (R,)

    # Rankings
    sorted_idx = np.argsort(rcv_rel_err)[::-1]
    top10_n = max(1, int(0.1 * n_rcv))
    top10_idx = sorted_idx[:top10_n]

    print("\n" + "=" * 70)
    print("ANALYSIS 2: Spatial Error Distribution (Per-Receiver)")
    print("=" * 70)
    print(f"\n  N_receivers = {n_rcv}")
    print(f"  Per-receiver relative error:")
    print(f"    mean={rcv_rel_err.mean():.4f}, median={np.median(rcv_rel_err):.4f}, "
          f"max={rcv_rel_err.max():.4f}")
    print(f"    std={rcv_rel_err.std():.4f}")

    print(f"\n  Top-10 worst receivers (by relative error):")
    print(f"    {'Rcv':>5} {'Rel Err%':>10} {'Region':>12} "
          f"{'x [m]':>8} {'y [m]':>8} {'|p_ref|_rms':>12}")
    for rank, ri in enumerate(sorted_idx[:10]):
        reg_name = REGION_NAMES.get(region_labels[ri], f"?{region_labels[ri]}")
        p_ref_rms = np.sqrt(rcv_ref_sq[ri] / n_freq)
        print(f"    {ri:5d} {rcv_rel_err[ri] * 100:9.2f}% {reg_name:>12} "
              f"{rcv_pos[ri, 0]:8.3f} {rcv_pos[ri, 1]:8.3f} "
              f"{p_ref_rms:12.4e}")

    # Error concentration
    total_err_sq = np.sum(rcv_diff_sq)
    top10_err_sq = np.sum(rcv_diff_sq[top10_idx])
    top10_frac = top10_err_sq / max(total_err_sq, 1e-30)
    print(f"\n  Error concentration:")
    print(f"    Top {top10_n} receivers ({top10_n/n_rcv*100:.0f}%) contribute "
          f"{top10_frac*100:.1f}% of total ||delta p||^2")

    # Per-region stats
    print(f"\n  Per-region error:")
    for reg_id, reg_name in REGION_NAMES.items():
        mask = region_labels == reg_id
        if mask.sum() == 0:
            continue
        reg_diff = np.sqrt(np.sum(rcv_diff_sq[mask]))
        reg_ref = np.sqrt(np.sum(rcv_ref_sq[mask]))
        reg_err = reg_diff / max(reg_ref, 1e-30)
        n_in_top10 = np.sum(np.isin(top10_idx, np.where(mask)[0]))
        print(f"    {reg_name:>12}: {reg_err*100:6.2f}% "
              f"(N={mask.sum()}, {n_in_top10} in top-10 worst)")

    # Plot
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # (a) Spatial map of relative error
    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(
        rcv_pos[:, 0], rcv_pos[:, 1],
        c=rcv_rel_err * 100, cmap="hot", s=15, edgecolors="k", linewidths=0.3,
    )
    ax.scatter(
        rcv_pos[top10_idx, 0], rcv_pos[top10_idx, 1],
        facecolors="none", edgecolors="cyan", s=60, linewidths=1.5,
        label=f"Top {top10_n} worst",
    )
    plt.colorbar(sc, ax=ax, label="Relative Error [%]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("(a) Per-Receiver Relative Error")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")

    # (b) Region-colored receiver map
    ax = fig.add_subplot(gs[0, 1])
    colors_map = {0: "red", 1: "orange", 2: "green"}
    for reg_id, reg_name in REGION_NAMES.items():
        mask = region_labels == reg_id
        if mask.sum() > 0:
            ax.scatter(
                rcv_pos[mask, 0], rcv_pos[mask, 1],
                c=colors_map[reg_id], s=15, label=reg_name, alpha=0.7,
            )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("(b) Region Labels")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    # (c) Cumulative error contribution (sorted)
    ax = fig.add_subplot(gs[0, 2])
    sorted_err_sq = np.sort(rcv_diff_sq)[::-1]
    cum_frac = np.cumsum(sorted_err_sq) / max(total_err_sq, 1e-30)
    ax.plot(np.arange(1, n_rcv + 1), cum_frac * 100, "b-", linewidth=1.5)
    ax.axhline(y=80, color="r", linestyle="--", alpha=0.5, label="80%")
    n_for_80 = np.searchsorted(cum_frac, 0.8) + 1
    ax.axvline(x=n_for_80, color="r", linestyle=":", alpha=0.5,
               label=f"{n_for_80} rcvs for 80%")
    ax.set_xlabel("Receivers (sorted by error)")
    ax.set_ylabel("Cumulative Error [%]")
    ax.set_title(f"(c) Error Concentration (80% in {n_for_80} rcvs)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (d) Histogram of per-receiver relative error
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(rcv_rel_err * 100, bins=40, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.axvline(x=np.median(rcv_rel_err) * 100, color="r", linestyle="--",
               label=f"Median={np.median(rcv_rel_err)*100:.1f}%")
    ax.set_xlabel("Relative Error [%]")
    ax.set_ylabel("Count")
    ax.set_title("(d) Error Distribution Across Receivers")
    ax.legend(fontsize=8)

    # (e) Absolute error vs distance from source
    ax = fig.add_subplot(gs[1, 1])
    src_pos_2 = rcv_pos  # receiver positions
    # We need dist from source. Load it via data dict
    xs_m, ys_m = 0, 0  # placeholder, filled below in main
    # This will be filled in the caller -- use a different approach
    ax.scatter(rcv_rel_err * 100, rcv_abs_err, s=10, alpha=0.6,
               c=[colors_map[region_labels[i]] for i in range(n_rcv)])
    ax.set_xlabel("Relative Error [%]")
    ax.set_ylabel("Absolute Error ||delta p||_2")
    ax.set_title("(e) Absolute vs Relative Error (color=region)")

    # (f) Error vs SDF
    ax = fig.add_subplot(gs[1, 2])
    # This will be filled in the caller -- placeholder
    ax.text(0.5, 0.5, "See figure 03", ha="center", va="center",
            transform=ax.transAxes, fontsize=14)
    ax.set_title("(f) See separate SDF analysis")

    fig.suptitle("Scene 13 Source 3: Spatial Error Analysis", fontsize=14)
    fig.savefig(out_dir / "02_spatial_error.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: 02_spatial_error.png")


def analyze_frequency_error(
    p_total_bem: np.ndarray,
    p_total_pred: np.ndarray,
    freqs_hz: np.ndarray,
    region_labels: np.ndarray,
    out_dir: Path,
) -> None:
    """Analysis 3: Frequency-resolved error.

    Parameters
    ----------
    p_total_bem : (F, R) complex128
    p_total_pred : (F, R) complex128
    freqs_hz : (F,)
    """
    n_freq, n_rcv = p_total_bem.shape
    diff = p_total_pred - p_total_bem  # (F, R)

    # Per-frequency L2 error (summed over receivers)
    freq_diff_sq = np.sum(np.abs(diff) ** 2, axis=1)  # (F,)
    freq_ref_sq = np.sum(np.abs(p_total_bem) ** 2, axis=1)  # (F,)
    freq_rel_err = np.sqrt(freq_diff_sq / np.maximum(freq_ref_sq, 1e-30))  # (F,)

    print("\n" + "=" * 70)
    print("ANALYSIS 3: Frequency Distribution of Error")
    print("=" * 70)
    print(f"\n  Per-frequency relative error:")
    print(f"    mean={freq_rel_err.mean()*100:.2f}%, "
          f"std={freq_rel_err.std()*100:.2f}%")
    print(f"    min={freq_rel_err.min()*100:.2f}% at f={freqs_hz[np.argmin(freq_rel_err)]:.0f} Hz")
    print(f"    max={freq_rel_err.max()*100:.2f}% at f={freqs_hz[np.argmax(freq_rel_err)]:.0f} Hz")

    # Top-10 worst frequencies
    sorted_fi = np.argsort(freq_rel_err)[::-1]
    print(f"\n  Top-10 worst frequencies:")
    print(f"    {'Rank':>5} {'f [Hz]':>10} {'Rel Err%':>10} "
          f"{'k [rad/m]':>12} {'lambda [m]':>12}")
    k_arr = 2.0 * np.pi * freqs_hz / SPEED_OF_SOUND_M_PER_S
    for rank, fi in enumerate(sorted_fi[:10]):
        lam = SPEED_OF_SOUND_M_PER_S / freqs_hz[fi]
        print(f"    {rank+1:5d} {freqs_hz[fi]:10.0f} "
              f"{freq_rel_err[fi]*100:9.2f}% "
              f"{k_arr[fi]:12.2f} {lam:12.4f}")

    # Per-region per-frequency error
    region_freq_err = {}
    for reg_id, reg_name in REGION_NAMES.items():
        mask = region_labels == reg_id
        if mask.sum() == 0:
            region_freq_err[reg_id] = np.zeros(n_freq)
            continue
        reg_diff = np.sum(np.abs(diff[:, mask]) ** 2, axis=1)  # (F,)
        reg_ref = np.sum(np.abs(p_total_bem[:, mask]) ** 2, axis=1)  # (F,)
        region_freq_err[reg_id] = np.sqrt(
            reg_diff / np.maximum(reg_ref, 1e-30)
        )

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Per-frequency overall error
    ax = axes[0]
    ax.plot(freqs_hz / 1000, freq_rel_err * 100, "b-", linewidth=1.0)
    ax.axhline(y=5, color="r", linestyle="--", alpha=0.5, label="5% gate")
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel("Relative Error [%]")
    ax.set_title("(a) Per-Frequency Error (All Receivers)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Per-region per-frequency error
    ax = axes[1]
    region_colors = {0: "red", 1: "orange", 2: "green"}
    for reg_id, reg_name in REGION_NAMES.items():
        if np.any(region_freq_err[reg_id] > 0):
            ax.plot(
                freqs_hz / 1000,
                region_freq_err[reg_id] * 100,
                color=region_colors[reg_id],
                linewidth=1.0,
                label=reg_name,
            )
    ax.axhline(y=5, color="r", linestyle="--", alpha=0.3)
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel("Relative Error [%]")
    ax.set_title("(b) Per-Region Per-Frequency Error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Error heatmap: freq x receiver
    ax = axes[2]
    # Sort receivers by region label for grouping
    rcv_order = np.argsort(region_labels)
    err_per_sample = np.abs(diff[:, rcv_order]) / np.maximum(
        np.abs(p_total_bem[:, rcv_order]), 1e-15
    )  # (F, R)
    im = ax.imshow(
        err_per_sample * 100,
        aspect="auto",
        cmap="hot",
        origin="lower",
        extent=[0, n_rcv, freqs_hz[0] / 1000, freqs_hz[-1] / 1000],
        vmin=0,
        vmax=min(100, np.percentile(err_per_sample * 100, 99)),
    )
    plt.colorbar(im, ax=ax, label="Point-wise Error [%]")
    ax.set_xlabel("Receiver (sorted by region)")
    ax.set_ylabel("Frequency [kHz]")
    ax.set_title("(c) Error Heatmap: Freq x Receiver")

    fig.suptitle("Scene 13 Source 3: Frequency Error Analysis", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "03_frequency_error.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: 03_frequency_error.png")


def analyze_model_bias(
    T_gt: np.ndarray,
    T_pred: np.ndarray,
    p_total_bem: np.ndarray,
    p_total_pred: np.ndarray,
    region_labels: np.ndarray,
    freqs_hz: np.ndarray,
    out_dir: Path,
) -> None:
    """Analysis 4: Systematic over/under-prediction of |T| and phase error.

    Parameters
    ----------
    T_gt : (F, R) complex128
    T_pred : (F, R) complex128
    """
    n_freq, n_rcv = T_gt.shape

    abs_T_gt = np.abs(T_gt)  # (F, R)
    abs_T_pred = np.abs(T_pred)  # (F, R)
    phase_T_gt = np.angle(T_gt)  # (F, R)
    phase_T_pred = np.angle(T_pred)  # (F, R)

    # Magnitude ratio: pred / gt
    gt_safe = np.maximum(abs_T_gt, 1e-15)
    mag_ratio = abs_T_pred / gt_safe  # (F, R)

    # Phase difference (wrapped to [-pi, pi])
    phase_diff = np.angle(np.exp(1j * (phase_T_pred - phase_T_gt)))  # (F, R)

    # Bias: average of (pred - gt) / gt
    mag_bias = np.mean(mag_ratio - 1.0)
    phase_bias_rad = np.mean(phase_diff)

    # Decompose error into magnitude and phase components
    # |p_pred - p_bem|^2 can be decomposed using:
    #   p = p_inc * (1 + T)
    #   delta_p = p_inc * delta_T
    #   delta_T = T_pred - T_gt
    # Magnitude-only error: |p_inc| * ||T_pred| - |T_gt||
    # Phase-only error: |p_inc| * |T_gt| * |exp(j*phi_pred) - exp(j*phi_gt)|
    p_inc_abs = np.abs(-0.25j * hankel1(
        0,
        2.0 * np.pi * freqs_hz[:, None] / SPEED_OF_SOUND_M_PER_S
        * np.ones((1, n_rcv))  # placeholder, actual kr needed
    )) if False else np.ones((n_freq, n_rcv))  # skip p_inc for T-space analysis

    mag_err_component = np.abs(abs_T_pred - abs_T_gt)  # (F, R)
    phase_err_component = abs_T_gt * np.abs(
        np.exp(1j * phase_T_pred) - np.exp(1j * phase_T_gt)
    )  # (F, R) -- chord length * |T_gt|

    total_T_err = np.abs(T_pred - T_gt)  # (F, R)

    # Fraction of error from magnitude vs phase
    mag_frac_per_rcv = np.sum(mag_err_component ** 2, axis=0) / np.maximum(
        np.sum(total_T_err ** 2, axis=0), 1e-30
    )  # (R,) -- approximate

    print("\n" + "=" * 70)
    print("ANALYSIS 4: Model Bias (Over/Under-prediction)")
    print("=" * 70)
    print(f"\n  Magnitude ratio (|T_pred| / |T_gt|):")
    print(f"    mean={np.mean(mag_ratio):.4f} (bias={mag_bias:+.4f})")
    print(f"    median={np.median(mag_ratio):.4f}")
    print(f"    std={np.std(mag_ratio):.4f}")
    bias_direction = "OVER-predicts" if mag_bias > 0.01 else (
        "UNDER-predicts" if mag_bias < -0.01 else "Approximately unbiased"
    )
    print(f"    ==> Model {bias_direction} |T| on average")

    print(f"\n  Phase difference (pred - gt):")
    print(f"    mean={np.degrees(phase_bias_rad):.2f} deg")
    print(f"    std={np.degrees(np.std(phase_diff)):.2f} deg")
    print(f"    |phase_diff| > 30 deg: "
          f"{np.sum(np.abs(phase_diff) > np.radians(30)) / phase_diff.size * 100:.1f}%")
    print(f"    |phase_diff| > 90 deg: "
          f"{np.sum(np.abs(phase_diff) > np.radians(90)) / phase_diff.size * 100:.1f}%")

    # Per-region bias
    print(f"\n  Per-region magnitude bias:")
    for reg_id, reg_name in REGION_NAMES.items():
        mask = region_labels == reg_id
        if mask.sum() == 0:
            continue
        reg_ratio = mag_ratio[:, mask]
        print(f"    {reg_name:>12}: mean_ratio={np.mean(reg_ratio):.4f} "
              f"(bias={np.mean(reg_ratio)-1:+.4f}), "
              f"mean_|phase_diff|={np.degrees(np.mean(np.abs(phase_diff[:, mask]))):.1f} deg")

    # Error decomposition
    total_mag_err = np.sqrt(np.sum(mag_err_component ** 2))
    total_phase_err = np.sqrt(np.sum(phase_err_component ** 2))
    total_err = np.sqrt(np.sum(total_T_err ** 2))
    print(f"\n  Error decomposition (in T-space):")
    print(f"    ||T_err||_total  = {total_err:.4f}")
    print(f"    ||T_err||_mag    = {total_mag_err:.4f} "
          f"({total_mag_err/total_err*100:.1f}% of total)")
    print(f"    ||T_err||_phase  = {total_phase_err:.4f} "
          f"({total_phase_err/total_err*100:.1f}% of total)")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (a) Scatter: |T_gt| vs |T_pred|
    ax = axes[0, 0]
    flat_gt = abs_T_gt.ravel()
    flat_pred = abs_T_pred.ravel()
    # Subsample for readability
    n_sub = min(20000, len(flat_gt))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(flat_gt), n_sub, replace=False)
    ax.scatter(flat_gt[idx], flat_pred[idx], s=1, alpha=0.15, c="steelblue")
    lim = np.percentile(np.concatenate([flat_gt, flat_pred]), 99.5)
    ax.plot([0, lim], [0, lim], "r--", linewidth=0.8, label="y=x")
    ax.set_xlabel("|T| Ground Truth")
    ax.set_ylabel("|T| Predicted")
    ax.set_title("(a) |T| Scatter: GT vs Pred")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    # (b) Magnitude ratio histogram
    ax = axes[0, 1]
    ratio_clip = np.clip(mag_ratio.ravel(), 0, 5)
    ax.hist(ratio_clip, bins=80, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.axvline(x=1.0, color="r", linestyle="--", label="Ratio=1 (perfect)")
    ax.axvline(x=np.median(mag_ratio), color="green", linestyle=":",
               label=f"Median={np.median(mag_ratio):.3f}")
    ax.set_xlabel("|T_pred| / |T_gt|")
    ax.set_ylabel("Count")
    ax.set_title("(b) Magnitude Ratio Distribution")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 3)

    # (c) Phase difference histogram
    ax = axes[0, 2]
    ax.hist(
        np.degrees(phase_diff.ravel()), bins=80,
        color="orange", edgecolor="darkorange", alpha=0.8,
    )
    ax.axvline(x=0, color="r", linestyle="--")
    ax.set_xlabel("Phase Difference [deg]")
    ax.set_ylabel("Count")
    ax.set_title("(c) Phase Error Distribution")

    # (d) Magnitude bias vs |T_gt|
    ax = axes[1, 0]
    # Bin by |T_gt| and compute mean ratio per bin
    n_bins = 30
    gt_flat = abs_T_gt.ravel()
    ratio_flat = mag_ratio.ravel()
    bin_edges = np.linspace(0, np.percentile(gt_flat, 99), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_mean_ratio = np.zeros(n_bins)
    bin_count = np.zeros(n_bins)
    for bi in range(n_bins):
        mask = (gt_flat >= bin_edges[bi]) & (gt_flat < bin_edges[bi + 1])
        if mask.sum() > 0:
            bin_mean_ratio[bi] = np.mean(ratio_flat[mask])
            bin_count[bi] = mask.sum()
    valid = bin_count > 10
    ax.bar(bin_centers[valid], bin_mean_ratio[valid] - 1.0, width=np.diff(bin_edges)[0] * 0.8,
           color="steelblue", edgecolor="navy", alpha=0.7)
    ax.axhline(y=0, color="r", linestyle="--")
    ax.set_xlabel("|T| Ground Truth")
    ax.set_ylabel("Mean Bias (ratio - 1)")
    ax.set_title("(d) Magnitude Bias vs |T_gt|")
    ax.grid(True, alpha=0.3)

    # (e) Phase error vs |T_gt|
    ax = axes[1, 1]
    bin_mean_phase = np.zeros(n_bins)
    for bi in range(n_bins):
        mask_b = (gt_flat >= bin_edges[bi]) & (gt_flat < bin_edges[bi + 1])
        if mask_b.sum() > 0:
            bin_mean_phase[bi] = np.degrees(np.mean(np.abs(phase_diff.ravel()[mask_b])))
    ax.bar(bin_centers[valid], bin_mean_phase[valid], width=np.diff(bin_edges)[0] * 0.8,
           color="orange", edgecolor="darkorange", alpha=0.7)
    ax.set_xlabel("|T| Ground Truth")
    ax.set_ylabel("Mean |Phase Error| [deg]")
    ax.set_title("(e) Phase Error vs |T_gt|")
    ax.grid(True, alpha=0.3)

    # (f) Magnitude bias vs frequency
    ax = axes[1, 2]
    freq_mean_ratio = np.mean(mag_ratio, axis=1)  # (F,)
    ax.plot(freqs_hz / 1000, freq_mean_ratio, "b-", linewidth=1.0)
    ax.axhline(y=1.0, color="r", linestyle="--", label="Perfect (ratio=1)")
    ax.fill_between(
        freqs_hz / 1000,
        np.percentile(mag_ratio, 25, axis=1),
        np.percentile(mag_ratio, 75, axis=1),
        alpha=0.2, color="blue", label="IQR",
    )
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel("Mean |T_pred|/|T_gt|")
    ax.set_title("(f) Magnitude Bias vs Frequency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Scene 13 Source 3: Model Bias Analysis", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "04_model_bias.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: 04_model_bias.png")


def analyze_error_concentration(
    p_total_bem: np.ndarray,
    p_total_pred: np.ndarray,
    T_gt: np.ndarray,
    T_pred: np.ndarray,
    rcv_pos: np.ndarray,
    region_labels: np.ndarray,
    dist_sr: np.ndarray,
    sdf_at_rcv: np.ndarray,
    freqs_hz: np.ndarray,
    out_dir: Path,
) -> None:
    """Analysis 5: Error concentration and identifying root cause.

    Combines spatial, spectral, and physical features to pinpoint the
    dominant failure mode.
    """
    n_freq, n_rcv = p_total_bem.shape
    diff = p_total_pred - p_total_bem  # (F, R)

    rcv_diff_sq = np.sum(np.abs(diff) ** 2, axis=0)  # (R,)
    rcv_ref_sq = np.sum(np.abs(p_total_bem) ** 2, axis=0)  # (R,)
    rcv_rel_err = np.sqrt(rcv_diff_sq / np.maximum(rcv_ref_sq, 1e-30))  # (R,)

    abs_T_gt_per_rcv = np.mean(np.abs(T_gt), axis=0)  # (R,) mean |T| per receiver
    abs_T_pred_per_rcv = np.mean(np.abs(T_pred), axis=0)  # (R,)

    print("\n" + "=" * 70)
    print("ANALYSIS 5: Error Concentration & Root Cause")
    print("=" * 70)

    # Correlation between error and physical features
    from scipy.stats import spearmanr

    features = {
        "dist_from_source": dist_sr,
        "SDF_at_receiver": sdf_at_rcv,
        "mean_|T_gt|": abs_T_gt_per_rcv,
        "mean_|T_pred|": abs_T_pred_per_rcv,
    }

    print(f"\n  Spearman correlation with per-receiver relative error:")
    for fname, fvals in features.items():
        rho, pval = spearmanr(fvals, rcv_rel_err)
        print(f"    {fname:>25}: rho={rho:+.4f} (p={pval:.2e})")

    # High-|T| analysis
    t_gt_90 = np.percentile(abs_T_gt_per_rcv, 90)
    high_T_mask = abs_T_gt_per_rcv > t_gt_90
    low_T_mask = ~high_T_mask
    high_err = np.sqrt(np.sum(rcv_diff_sq[high_T_mask])) / max(
        np.sqrt(np.sum(rcv_ref_sq[high_T_mask])), 1e-30
    )
    low_err = np.sqrt(np.sum(rcv_diff_sq[low_T_mask])) / max(
        np.sqrt(np.sum(rcv_ref_sq[low_T_mask])), 1e-30
    )
    print(f"\n  High |T| (>P90={t_gt_90:.2f}) error: {high_err*100:.2f}% "
          f"({high_T_mask.sum()} receivers)")
    print(f"  Low |T| (<P90)             error: {low_err*100:.2f}% "
          f"({low_T_mask.sum()} receivers)")

    # Near-edge analysis (SDF close to 0)
    sdf_thresh = 0.05  # 5cm from surface
    near_edge = np.abs(sdf_at_rcv) < sdf_thresh
    far_from_edge = ~near_edge
    if near_edge.sum() > 0:
        near_err = np.sqrt(np.sum(rcv_diff_sq[near_edge])) / max(
            np.sqrt(np.sum(rcv_ref_sq[near_edge])), 1e-30
        )
        far_err = np.sqrt(np.sum(rcv_diff_sq[far_from_edge])) / max(
            np.sqrt(np.sum(rcv_ref_sq[far_from_edge])), 1e-30
        )
        print(f"\n  Near-edge (|SDF| < {sdf_thresh}m): {near_err*100:.2f}% "
              f"({near_edge.sum()} receivers)")
        print(f"  Far-from-edge:                  {far_err*100:.2f}% "
              f"({far_from_edge.sum()} receivers)")
    else:
        print(f"\n  No receivers with |SDF| < {sdf_thresh}m")

    # Error vs |T_gt|: check if error scales with T magnitude
    total_err_sq = np.sum(rcv_diff_sq)
    top_pct = [5, 10, 20, 50]
    print(f"\n  Error concentration (top-N% receivers by error):")
    for pct in top_pct:
        n_top = max(1, int(pct / 100 * n_rcv))
        top_idx = np.argsort(rcv_diff_sq)[::-1][:n_top]
        frac = np.sum(rcv_diff_sq[top_idx]) / max(total_err_sq, 1e-30)
        mean_T = np.mean(abs_T_gt_per_rcv[top_idx])
        mean_sdf = np.mean(sdf_at_rcv[top_idx])
        print(f"    Top {pct:3d}% ({n_top:3d} rcvs): {frac*100:5.1f}% of error, "
              f"mean|T_gt|={mean_T:.2f}, mean_SDF={mean_sdf:.3f}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (a) Error vs SDF
    ax = axes[0, 0]
    ax.scatter(sdf_at_rcv, rcv_rel_err * 100, s=15, alpha=0.6, c="steelblue")
    ax.axvline(x=0, color="k", linestyle=":", alpha=0.3)
    ax.set_xlabel("SDF at Receiver [m]")
    ax.set_ylabel("Relative Error [%]")
    ax.set_title("(a) Error vs SDF")
    ax.grid(True, alpha=0.3)

    # (b) Error vs distance from source
    ax = axes[0, 1]
    ax.scatter(dist_sr, rcv_rel_err * 100, s=15, alpha=0.6, c="steelblue")
    ax.set_xlabel("Distance from Source [m]")
    ax.set_ylabel("Relative Error [%]")
    ax.set_title("(b) Error vs Distance")
    ax.grid(True, alpha=0.3)

    # (c) Error vs mean |T_gt|
    ax = axes[0, 2]
    ax.scatter(abs_T_gt_per_rcv, rcv_rel_err * 100, s=15, alpha=0.6, c="steelblue")
    ax.set_xlabel("Mean |T| Ground Truth")
    ax.set_ylabel("Relative Error [%]")
    ax.set_title("(c) Error vs Mean |T_gt|")
    ax.grid(True, alpha=0.3)

    # (d) |T_gt| vs |T_pred| per receiver (mean over freq)
    ax = axes[1, 0]
    ax.scatter(abs_T_gt_per_rcv, abs_T_pred_per_rcv, s=15, alpha=0.6,
               c=[{0: "red", 1: "orange", 2: "green"}[region_labels[i]] for i in range(n_rcv)])
    lim = max(abs_T_gt_per_rcv.max(), abs_T_pred_per_rcv.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=0.8)
    ax.set_xlabel("Mean |T_gt| per Receiver")
    ax.set_ylabel("Mean |T_pred| per Receiver")
    ax.set_title("(d) |T| per Receiver: GT vs Pred")
    ax.set_aspect("equal")

    # (e) SDF spatial map with worst receivers highlighted
    ax = axes[1, 1]
    top20_idx = np.argsort(rcv_diff_sq)[::-1][:int(0.1 * n_rcv)]
    sc = ax.scatter(
        rcv_pos[:, 0], rcv_pos[:, 1],
        c=sdf_at_rcv, cmap="RdYlGn", s=15, edgecolors="k", linewidths=0.2,
    )
    ax.scatter(
        rcv_pos[top20_idx, 0], rcv_pos[top20_idx, 1],
        facecolors="none", edgecolors="red", s=60, linewidths=1.5,
        label="Top-10% worst",
    )
    plt.colorbar(sc, ax=ax, label="SDF [m]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("(e) SDF Map + Worst Receivers")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")

    # (f) Error energy per frequency band (low/mid/high)
    ax = axes[1, 2]
    k_arr = 2.0 * np.pi * freqs_hz / SPEED_OF_SOUND_M_PER_S
    freq_diff_sq = np.sum(np.abs(diff) ** 2, axis=1)  # (F,)
    freq_ref_sq = np.sum(np.abs(p_total_bem) ** 2, axis=1)  # (F,)

    # Split into 4 frequency bands
    n_band = n_freq // 4
    bands = []
    band_labels = []
    for bi in range(4):
        fi_s = bi * n_band
        fi_e = (bi + 1) * n_band if bi < 3 else n_freq
        band_err = np.sqrt(np.sum(freq_diff_sq[fi_s:fi_e]) /
                           max(np.sum(freq_ref_sq[fi_s:fi_e]), 1e-30))
        bands.append(band_err * 100)
        band_labels.append(
            f"{freqs_hz[fi_s]/1000:.1f}-{freqs_hz[min(fi_e-1, n_freq-1)]/1000:.1f} kHz"
        )
    ax.bar(range(4), bands, color="steelblue", edgecolor="navy")
    ax.set_xticks(range(4))
    ax.set_xticklabels(band_labels, fontsize=8)
    ax.set_ylabel("Relative Error [%]")
    ax.set_title("(f) Error by Frequency Band")
    ax.axhline(y=5, color="r", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Scene 13 Source 3: Error Concentration & Root Cause", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "05_error_concentration.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: 05_error_concentration.png")


def analyze_worst_receivers_spectra(
    p_total_bem: np.ndarray,
    p_total_pred: np.ndarray,
    T_gt: np.ndarray,
    T_pred: np.ndarray,
    freqs_hz: np.ndarray,
    rcv_pos: np.ndarray,
    region_labels: np.ndarray,
    out_dir: Path,
) -> None:
    """Analysis 6: Deep-dive into the worst 5 receivers.

    Shows their full frequency spectra (BEM vs pred) and T waveforms.
    """
    n_freq, n_rcv = p_total_bem.shape
    diff = p_total_pred - p_total_bem

    rcv_diff_sq = np.sum(np.abs(diff) ** 2, axis=0)  # (R,)
    rcv_ref_sq = np.sum(np.abs(p_total_bem) ** 2, axis=0)
    rcv_rel_err = np.sqrt(rcv_diff_sq / np.maximum(rcv_ref_sq, 1e-30))

    worst_5 = np.argsort(rcv_rel_err)[::-1][:5]

    print("\n" + "=" * 70)
    print("ANALYSIS 6: Worst 5 Receivers -- Spectral Deep-Dive")
    print("=" * 70)

    fig, axes = plt.subplots(5, 4, figsize=(20, 22))

    for row, ri in enumerate(worst_5):
        reg_name = REGION_NAMES.get(region_labels[ri], "?")
        err_pct = rcv_rel_err[ri] * 100

        print(f"\n  Receiver {ri} ({reg_name}): error={err_pct:.1f}%, "
              f"pos=({rcv_pos[ri,0]:.3f}, {rcv_pos[ri,1]:.3f})")

        # (a) |p_total| spectrum
        ax = axes[row, 0]
        ax.plot(freqs_hz / 1000, np.abs(p_total_bem[:, ri]), "b-",
                linewidth=1.0, label="BEM")
        ax.plot(freqs_hz / 1000, np.abs(p_total_pred[:, ri]), "r--",
                linewidth=1.0, label="Pred")
        ax.set_ylabel("|p_total|")
        if row == 0:
            ax.set_title("|p_total| Spectrum")
        ax.legend(fontsize=6)
        ax.text(0.02, 0.95, f"R{ri} ({reg_name}) {err_pct:.0f}%",
                transform=ax.transAxes, fontsize=7, va="top")
        ax.grid(True, alpha=0.3)
        if row == 4:
            ax.set_xlabel("Freq [kHz]")

        # (b) Phase of p_total
        ax = axes[row, 1]
        ax.plot(freqs_hz / 1000,
                np.unwrap(np.angle(p_total_bem[:, ri])), "b-", linewidth=1.0)
        ax.plot(freqs_hz / 1000,
                np.unwrap(np.angle(p_total_pred[:, ri])), "r--", linewidth=1.0)
        ax.set_ylabel("Phase [rad]")
        if row == 0:
            ax.set_title("Phase of p_total (unwrapped)")
        ax.grid(True, alpha=0.3)
        if row == 4:
            ax.set_xlabel("Freq [kHz]")

        # (c) |T| spectrum
        ax = axes[row, 2]
        ax.plot(freqs_hz / 1000, np.abs(T_gt[:, ri]), "b-",
                linewidth=1.0, label="|T| GT")
        ax.plot(freqs_hz / 1000, np.abs(T_pred[:, ri]), "r--",
                linewidth=1.0, label="|T| Pred")
        ax.set_ylabel("|T|")
        if row == 0:
            ax.set_title("|T| Spectrum")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        if row == 4:
            ax.set_xlabel("Freq [kHz]")

        # (d) Re(T) and Im(T) comparison
        ax = axes[row, 3]
        ax.plot(freqs_hz / 1000, T_gt[:, ri].real, "b-",
                linewidth=0.8, label="Re(T) GT")
        ax.plot(freqs_hz / 1000, T_pred[:, ri].real, "r--",
                linewidth=0.8, label="Re(T) Pred")
        ax.plot(freqs_hz / 1000, T_gt[:, ri].imag, "b:",
                linewidth=0.8, label="Im(T) GT")
        ax.plot(freqs_hz / 1000, T_pred[:, ri].imag, "r:",
                linewidth=0.8, label="Im(T) Pred")
        ax.set_ylabel("T (Re/Im)")
        if row == 0:
            ax.set_title("Transfer Function Components")
        ax.legend(fontsize=5, ncol=2)
        ax.grid(True, alpha=0.3)
        if row == 4:
            ax.set_xlabel("Freq [kHz]")

    fig.suptitle("Scene 13 Source 3: Worst 5 Receivers -- Spectral Deep-Dive", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "06_worst_receivers.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: 06_worst_receivers.png")


def summary_table(
    p_total_bem: np.ndarray,
    p_total_pred: np.ndarray,
    region_labels: np.ndarray,
    freqs_hz: np.ndarray,
    dist_sr: np.ndarray,
    sdf_at_rcv: np.ndarray,
    T_gt: np.ndarray,
    T_pred: np.ndarray,
) -> None:
    """Print final summary table with key findings."""
    n_freq, n_rcv = p_total_bem.shape
    diff = p_total_pred - p_total_bem

    # Overall error
    total_diff = np.sqrt(np.sum(np.abs(diff) ** 2))
    total_ref = np.sqrt(np.sum(np.abs(p_total_bem) ** 2))
    overall_err = total_diff / max(total_ref, 1e-30)

    # Per-receiver
    rcv_diff_sq = np.sum(np.abs(diff) ** 2, axis=0)
    rcv_ref_sq = np.sum(np.abs(p_total_bem) ** 2, axis=0)
    rcv_rel_err = np.sqrt(rcv_diff_sq / np.maximum(rcv_ref_sq, 1e-30))

    # Top-10% concentration
    n_top = max(1, int(0.1 * n_rcv))
    top_idx = np.argsort(rcv_diff_sq)[::-1][:n_top]
    top10_frac = np.sum(rcv_diff_sq[top_idx]) / max(np.sum(rcv_diff_sq), 1e-30)

    # Bias
    abs_T_gt = np.abs(T_gt)
    abs_T_pred = np.abs(T_pred)
    gt_safe = np.maximum(abs_T_gt, 1e-15)
    mag_ratio = abs_T_pred / gt_safe
    mag_bias = np.mean(mag_ratio) - 1.0

    # Phase error
    phase_diff = np.angle(np.exp(1j * (np.angle(T_pred) - np.angle(T_gt))))
    mean_abs_phase_err = np.degrees(np.mean(np.abs(phase_diff)))

    print("\n" + "=" * 70)
    print("SUMMARY: Scene 13 Source 3 Diagnostic Results")
    print("=" * 70)
    print(f"""
  +------------------------------------+------------------+
  | Metric                             | Value            |
  +------------------------------------+------------------+
  | Overall relative L2 error          | {overall_err*100:>13.2f}%  |
  | Median per-rcv error               | {np.median(rcv_rel_err)*100:>13.2f}%  |
  | Max per-rcv error                  | {rcv_rel_err.max()*100:>13.2f}%  |
  | Top-10% rcv error concentration    | {top10_frac*100:>13.1f}%  |
  | Magnitude bias (mean ratio - 1)    | {mag_bias:>+13.4f}   |
  | Mean |phase error|                 | {mean_abs_phase_err:>11.1f} deg |
  | |phase| > 30 deg fraction          | {np.sum(np.abs(phase_diff) > np.radians(30)) / phase_diff.size * 100:>13.1f}%  |
  | |phase| > 90 deg fraction          | {np.sum(np.abs(phase_diff) > np.radians(90)) / phase_diff.size * 100:>13.1f}%  |
  +------------------------------------+------------------+
""")

    # Worst region
    region_errs = {}
    for reg_id, reg_name in REGION_NAMES.items():
        mask = region_labels == reg_id
        if mask.sum() == 0:
            continue
        d = np.sqrt(np.sum(rcv_diff_sq[mask]))
        r = np.sqrt(np.sum(rcv_ref_sq[mask]))
        region_errs[reg_name] = d / max(r, 1e-30)
    worst_region = max(region_errs, key=region_errs.get)

    print(f"  Worst region: {worst_region} ({region_errs[worst_region]*100:.2f}%)")
    print(f"  Worst receiver: R{np.argmax(rcv_rel_err)} "
          f"(error={rcv_rel_err.max()*100:.1f}%, "
          f"mean|T_gt|={np.mean(np.abs(T_gt[:, np.argmax(rcv_rel_err)])):.2f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run all diagnostics for Scene 13 Source 3."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load data and model
    data = load_scene_data(H5_PATH)
    model, scene_scale, scene_0idx, config = load_model(CKPT_PATH, device)

    # Predict
    logger.info("Running predictions for S13 source 3 ...")
    pred = predict_transfer_function(model, data, scene_scale, scene_0idx, device)

    p_total_bem = data["p_total_bem"]      # (F, R)
    p_total_pred = pred["p_total_pred"]    # (F, R)
    T_gt = pred["T_gt_complex"]            # (F, R)
    T_pred = pred["T_pred_complex"]        # (F, R)
    T_norm_pred = pred["T_norm_pred"]      # (F*R, 2)
    region_labels = data["region_labels"]  # (R,)
    rcv_pos = data["rcv_pos"]              # (R, 2)
    freqs_hz = data["freqs_hz"]            # (F,)

    # Quick sanity
    overall_diff = np.sqrt(np.sum(np.abs(p_total_pred - p_total_bem) ** 2))
    overall_ref = np.sqrt(np.sum(np.abs(p_total_bem) ** 2))
    overall_err = overall_diff / max(overall_ref, 1e-30)
    logger.info("Overall S13 src3 error: %.2f%%", overall_err * 100)

    # Run all analyses
    analyze_T_distribution(T_gt, T_pred, T_norm_pred, scene_scale, OUT_DIR)
    analyze_spatial_error(p_total_bem, p_total_pred, rcv_pos, region_labels, OUT_DIR)
    analyze_frequency_error(p_total_bem, p_total_pred, freqs_hz, region_labels, OUT_DIR)
    analyze_model_bias(T_gt, T_pred, p_total_bem, p_total_pred,
                       region_labels, freqs_hz, OUT_DIR)
    analyze_error_concentration(
        p_total_bem, p_total_pred, T_gt, T_pred,
        rcv_pos, region_labels, pred["dist_sr"], pred["sdf_at_rcv"],
        freqs_hz, OUT_DIR,
    )
    analyze_worst_receivers_spectra(
        p_total_bem, p_total_pred, T_gt, T_pred,
        freqs_hz, rcv_pos, region_labels, OUT_DIR,
    )
    summary_table(
        p_total_bem, p_total_pred, region_labels, freqs_hz,
        pred["dist_sr"], pred["sdf_at_rcv"], T_gt, T_pred,
    )

    logger.info("All diagnostics saved to: %s", OUT_DIR)
    logger.info("Done.")


if __name__ == "__main__":
    main()
