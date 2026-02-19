"""Phase 4 Gate Evaluation: Cycle-Consistency Correlation.

Full cycle path:
    p_gt(BEM) -> [Inverse] -> SDF(z) -> [Forward Surrogate] -> p_pred

    Pearson r = corr( [Re(p_pred), Im(p_pred)], [Re(p_gt), Im(p_gt)] )

Gate criterion
--------------
    Mean per-scene Pearson r > 0.8

Usage
-----
    python scripts/eval_phase4.py
    python scripts/eval_phase4.py --checkpoint best_phase3_v2 --forward-ckpt best_v11
    python scripts/eval_phase4.py --scenes 1 2 3
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import hankel1

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.forward_model import build_transfer_model
from src.inverse_dataset import InverseSceneData, load_all_scenes
from src.inverse_model import InverseModel, build_inverse_model, compute_sdf_iou

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase4_eval")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "phase1"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase3"
FWD_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase2"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase4"

# ---------------------------------------------------------------------------
# Gate threshold
# ---------------------------------------------------------------------------
CORRELATION_THRESHOLD: float = 0.8

# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
SPEED_OF_SOUND_M_PER_S: float = 343.0


# ---------------------------------------------------------------------------
# Exact incident field (NumPy, no asymptotic approximation)
# ---------------------------------------------------------------------------
def compute_p_inc_exact(
    x_src: np.ndarray,
    x_rcv: np.ndarray,
    k: np.ndarray,
) -> np.ndarray:
    """Exact 2D incident field: p_inc = -(i/4) H_0^{(1)}(kr).

    Parameters
    ----------
    x_src : np.ndarray, shape (B, 2) [m]
        Source positions.
    x_rcv : np.ndarray, shape (B, 2) [m]
        Receiver positions.
    k : np.ndarray, shape (B,) [rad/m]
        Wavenumber.

    Returns
    -------
    p_inc : np.ndarray, shape (B,), complex128
    """
    dx = x_rcv[:, 0] - x_src[:, 0]
    dy = x_rcv[:, 1] - x_src[:, 1]
    r = np.sqrt(dx ** 2 + dy ** 2)  # (B,)

    if not np.all(np.isfinite(r)):
        raise ValueError(f"Non-finite distances: {np.sum(~np.isfinite(r))} values")
    if np.any(r < 1e-10):
        raise ValueError("Source-receiver distance near zero")

    kr = k * r  # (B,)
    p_inc = -0.25j * hankel1(0, kr)  # (B,) complex128
    return p_inc


# ---------------------------------------------------------------------------
# Cycle-consistency evaluation for one scene
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_cycle_consistency(
    inverse_model: InverseModel,
    forward_model: torch.nn.Module,
    scene_data: InverseSceneData,
    scene_idx: int,
    device: torch.device,
    freq_chunk_size: int = 20,
) -> Dict:
    """Evaluate full cycle-consistency for one scene.

    Cycle: z -> SDFDecoder(rcv_pos) -> sdf -> Forward -> T -> p_pred
    Compare: p_pred vs p_gt (BEM)

    Parameters
    ----------
    inverse_model : InverseModel
        Trained Phase 3 inverse model.
    forward_model : torch.nn.Module
        Frozen Phase 2 forward model.
    scene_data : InverseSceneData
        BEM ground truth + scene metadata.
    scene_idx : int
        0-indexed inverse model scene index.
    device : torch.device
    freq_chunk_size : int
        Number of frequencies per GPU batch (controls VRAM usage).

    Returns
    -------
    results : dict
        r_pearson, r_magnitude, rel_l2, per_source_r, per_freq_r, etc.
    """
    sd = scene_data
    S = sd.n_sources        # ~3
    F = sd.n_freqs          # ~200
    R = sd.n_receivers      # ~200

    # 1. Predict SDF at all receiver positions (once per scene, geometry-only)
    rcv_t = torch.from_numpy(sd.rcv_pos).float().to(device)  # (R, 2)
    sdf_rcv = inverse_model.predict_sdf(scene_idx, rcv_t)    # (R, 1)

    # Also compute SDF IoU for reference
    xy_grid = torch.from_numpy(sd.grid_coords).float().to(device)  # (G, 2)
    sdf_gt_grid = torch.from_numpy(sd.sdf_flat).float().to(device)  # (G,)
    sdf_preds = []
    for gi in range(0, len(xy_grid), 8192):
        chunk = xy_grid[gi : gi + 8192]
        sdf_preds.append(inverse_model.predict_sdf(scene_idx, chunk).squeeze(-1))
    sdf_pred_flat = torch.cat(sdf_preds, dim=0)  # (G,)
    iou = compute_sdf_iou(sdf_pred_flat, sdf_gt_grid)

    # 2. Reconstruct pressure for all (source, freq, receiver) tuples
    p_pred_all = np.zeros((S, F, R), dtype=np.complex128)
    p_gt_all = sd.pressure.copy()  # (S, F, R) complex128

    fwd_scene_ids = torch.full(
        (R,), sd.fwd_scene_idx, dtype=torch.long, device=device,
    )

    for si in range(S):
        # Source position repeated for all receivers
        x_src_np = np.tile(sd.src_pos[si], (R, 1))  # (R, 2)
        x_src_t = torch.from_numpy(x_src_np).float().to(device)

        for fi_start in range(0, F, freq_chunk_size):
            fi_end = min(fi_start + freq_chunk_size, F)
            n_f = fi_end - fi_start  # frequencies in this chunk

            # Build batched inputs: (n_f * R, ...)
            x_src_batch = x_src_t.unsqueeze(0).expand(n_f, -1, -1).reshape(
                n_f * R, 2
            )  # (n_f*R, 2)
            x_rcv_batch = rcv_t.unsqueeze(0).expand(n_f, -1, -1).reshape(
                n_f * R, 2
            )  # (n_f*R, 2)
            sdf_batch = sdf_rcv.unsqueeze(0).expand(n_f, -1, -1).reshape(
                n_f * R, 1
            )  # (n_f*R, 1)

            k_chunk = sd.k_arr[fi_start:fi_end]  # (n_f,)
            k_batch = torch.from_numpy(
                np.repeat(k_chunk, R)
            ).float().to(device).unsqueeze(-1)  # (n_f*R, 1)

            fwd_ids_batch = fwd_scene_ids.unsqueeze(0).expand(
                n_f, -1
            ).reshape(n_f * R)  # (n_f*R,)

            # Forward model: predict transfer function T
            T_pred = forward_model.forward_from_coords(
                x_src_batch, x_rcv_batch, k_batch, sdf_batch,
                scene_ids=fwd_ids_batch,
            )  # (n_f*R, 2)

            T_re = T_pred[:, 0].cpu().numpy().reshape(n_f, R)  # (n_f, R)
            T_im = T_pred[:, 1].cpu().numpy().reshape(n_f, R)  # (n_f, R)
            T_complex = (T_re + 1j * T_im) * sd.scene_scale   # (n_f, R)

            # Exact incident field (per frequency)
            for fi_local in range(n_f):
                fi_global = fi_start + fi_local
                k_val = sd.k_arr[fi_global]
                k_arr_r = np.full(R, k_val)  # (R,)

                p_inc = compute_p_inc_exact(
                    x_src_np, sd.rcv_pos, k_arr_r,
                )  # (R,) complex128

                p_pred_all[si, fi_global, :] = p_inc * (1.0 + T_complex[fi_local])

    # 3. Compute metrics
    p_pred_flat = p_pred_all.ravel()  # (S*F*R,) complex
    p_gt_flat = p_gt_all.ravel()      # (S*F*R,) complex

    # Pearson r on stacked [Re, Im] vector
    pred_vec = np.concatenate([p_pred_flat.real, p_pred_flat.imag])
    gt_vec = np.concatenate([p_gt_flat.real, p_gt_flat.imag])
    r_pearson = float(np.corrcoef(pred_vec, gt_vec)[0, 1])

    # Relative L2 error
    rel_l2 = float(
        np.sqrt(np.sum(np.abs(p_pred_flat - p_gt_flat) ** 2))
        / np.sqrt(np.sum(np.abs(p_gt_flat) ** 2))
    )

    # Magnitude correlation
    mag_pred = np.abs(p_pred_flat)
    mag_gt = np.abs(p_gt_flat)
    r_magnitude = float(np.corrcoef(mag_pred, mag_gt)[0, 1])

    # Per-source Pearson r
    per_source_r: Dict[int, float] = {}
    for si in range(S):
        p_s = p_pred_all[si].ravel()
        g_s = p_gt_all[si].ravel()
        sv = np.concatenate([p_s.real, p_s.imag])
        gv = np.concatenate([g_s.real, g_s.imag])
        per_source_r[si] = float(np.corrcoef(sv, gv)[0, 1])

    # Per-frequency Pearson r (vectorized)
    per_freq_r = np.zeros(F, dtype=np.float64)
    for fi in range(F):
        p_f = p_pred_all[:, fi, :].ravel()  # (S*R,)
        g_f = p_gt_all[:, fi, :].ravel()
        fv = np.concatenate([p_f.real, p_f.imag])
        gv = np.concatenate([g_f.real, g_f.imag])
        std_fv = np.std(fv)
        std_gv = np.std(gv)
        if std_fv > 1e-30 and std_gv > 1e-30:
            per_freq_r[fi] = np.corrcoef(fv, gv)[0, 1]

    return {
        "r_pearson": r_pearson,
        "r_magnitude": r_magnitude,
        "rel_l2": rel_l2,
        "iou": iou,
        "per_source_r": per_source_r,
        "per_freq_r": per_freq_r,
        "freqs_hz": sd.freqs_hz.copy(),
        "n_observations": S * F * R,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_per_scene_correlation(
    all_results: Dict[int, Dict],
    output_path: Path,
) -> None:
    """Bar chart of per-scene Pearson r (cycle-consistency)."""
    sids = sorted(all_results.keys())
    r_vals = [all_results[s]["r_pearson"] for s in sids]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(len(sids)), r_vals, color="steelblue", edgecolor="navy")

    for bar, r_val in zip(bars, r_vals):
        if r_val < CORRELATION_THRESHOLD:
            bar.set_color("salmon")
            bar.set_edgecolor("darkred")

    ax.axhline(
        y=CORRELATION_THRESHOLD, color="r", linestyle="--",
        label=f"Gate ({CORRELATION_THRESHOLD})",
    )
    ax.set_xticks(range(len(sids)))
    ax.set_xticklabels([f"S{s}" for s in sids])
    ax.set_xlabel("Scene")
    ax.set_ylabel("Pearson r")
    ax.set_ylim(0, 1.05)
    ax.set_title("Phase 4: Per-Scene Cycle-Consistency Correlation")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Per-scene correlation plot: %s", output_path)


def plot_freq_correlation(
    all_results: Dict[int, Dict],
    output_path: Path,
) -> None:
    """Per-frequency correlation profile (mean across scenes)."""
    sids = sorted(all_results.keys())

    # All scenes should have the same freq array
    freqs_hz = all_results[sids[0]]["freqs_hz"]
    n_freqs = len(freqs_hz)

    # Stack per-frequency r across scenes
    per_freq_stack = np.zeros((len(sids), n_freqs), dtype=np.float64)
    for i, sid in enumerate(sids):
        per_freq_stack[i] = all_results[sid]["per_freq_r"]

    mean_r = per_freq_stack.mean(axis=0)
    std_r = per_freq_stack.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs_hz / 1000.0, mean_r, "b-", linewidth=1.5, label="Mean r")
    ax.fill_between(
        freqs_hz / 1000.0, mean_r - std_r, np.minimum(mean_r + std_r, 1.0),
        alpha=0.2, color="blue",
    )
    ax.axhline(
        y=CORRELATION_THRESHOLD, color="r", linestyle="--",
        label=f"Gate ({CORRELATION_THRESHOLD})",
    )
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel("Pearson r")
    ax.set_ylim(0, 1.05)
    ax.set_title("Phase 4: Cycle-Consistency Correlation vs Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Frequency correlation plot: %s", output_path)


def plot_scatter_summary(
    all_results: Dict[int, Dict],
    inverse_model: InverseModel,
    forward_model: torch.nn.Module,
    all_scenes: Dict[int, InverseSceneData],
    inv_scene_id_map: Dict[int, int],
    device: torch.device,
    output_dir: Path,
    n_samples: int = 5000,
) -> None:
    """Scatter plot: p_pred vs p_gt (Re/Im) for select scenes.

    Uses a random subset of observations for visual clarity.
    """
    sids = sorted(all_results.keys())

    # Select up to 6 scenes for scatter
    display_sids = sids[:6] if len(sids) > 6 else sids
    n_plots = len(display_sids)

    fig, axes = plt.subplots(2, n_plots, figsize=(4 * n_plots, 8))
    if n_plots == 1:
        axes = axes.reshape(2, 1)

    for col, sid in enumerate(display_sids):
        sd = all_scenes[sid]
        scene_idx = inv_scene_id_map[sid]
        r_val = all_results[sid]["r_pearson"]

        # Quick recompute on a random subset for scatter plot
        S, F, R = sd.n_sources, sd.n_freqs, sd.n_receivers
        total = S * F * R
        n_show = min(n_samples, total)
        rng = np.random.RandomState(42)
        idx_flat = rng.choice(total, size=n_show, replace=False)

        si_arr = idx_flat // (F * R)
        rem = idx_flat % (F * R)
        fi_arr = rem // R
        ri_arr = rem % R

        # Ground truth
        p_gt_sub = sd.pressure[si_arr, fi_arr, ri_arr]  # (n_show,) complex

        # Predicted SDF at selected receivers
        unique_ri = np.unique(ri_arr)
        rcv_t = torch.from_numpy(sd.rcv_pos[unique_ri]).float().to(device)
        with torch.no_grad():
            sdf_unique = inverse_model.predict_sdf(scene_idx, rcv_t)  # (U, 1)
        sdf_map = {int(ri): sdf_unique[j].item() for j, ri in enumerate(unique_ri)}
        sdf_arr = np.array([sdf_map[int(ri)] for ri in ri_arr])  # (n_show,)

        # Forward model
        x_src_np = sd.src_pos[si_arr]   # (n_show, 2)
        x_rcv_np = sd.rcv_pos[ri_arr]   # (n_show, 2)
        k_np = sd.k_arr[fi_arr]          # (n_show,)

        x_src_t = torch.from_numpy(x_src_np).float().to(device)
        x_rcv_t = torch.from_numpy(x_rcv_np).float().to(device)
        k_t = torch.from_numpy(k_np).float().to(device).unsqueeze(-1)
        sdf_t = torch.from_numpy(sdf_arr).float().to(device).unsqueeze(-1)
        fwd_ids_t = torch.full(
            (n_show,), sd.fwd_scene_idx, dtype=torch.long, device=device,
        )

        with torch.no_grad():
            T_pred = forward_model.forward_from_coords(
                x_src_t, x_rcv_t, k_t, sdf_t, scene_ids=fwd_ids_t,
            )

        T_re = T_pred[:, 0].cpu().numpy()  # (n_show,)
        T_im = T_pred[:, 1].cpu().numpy()
        T_complex = (T_re + 1j * T_im) * sd.scene_scale

        p_inc = compute_p_inc_exact(x_src_np, x_rcv_np, k_np)
        p_pred_sub = p_inc * (1.0 + T_complex)  # (n_show,) complex

        # Plot Re
        ax_re = axes[0, col]
        ax_re.scatter(
            p_gt_sub.real, p_pred_sub.real,
            s=1, alpha=0.3, c="steelblue", rasterized=True,
        )
        lims_re = [
            min(p_gt_sub.real.min(), p_pred_sub.real.min()),
            max(p_gt_sub.real.max(), p_pred_sub.real.max()),
        ]
        ax_re.plot(lims_re, lims_re, "r-", linewidth=0.8)
        ax_re.set_xlabel("Re(p_gt)")
        ax_re.set_ylabel("Re(p_pred)")
        ax_re.set_title(f"S{sid} Re (r={r_val:.3f})")
        ax_re.set_aspect("equal")
        ax_re.grid(True, alpha=0.2)

        # Plot Im
        ax_im = axes[1, col]
        ax_im.scatter(
            p_gt_sub.imag, p_pred_sub.imag,
            s=1, alpha=0.3, c="darkorange", rasterized=True,
        )
        lims_im = [
            min(p_gt_sub.imag.min(), p_pred_sub.imag.min()),
            max(p_gt_sub.imag.max(), p_pred_sub.imag.max()),
        ]
        ax_im.plot(lims_im, lims_im, "r-", linewidth=0.8)
        ax_im.set_xlabel("Im(p_gt)")
        ax_im.set_ylabel("Im(p_pred)")
        ax_im.set_title(f"S{sid} Im")
        ax_im.set_aspect("equal")
        ax_im.grid(True, alpha=0.2)

    fig.suptitle("Phase 4: Cycle-Consistency Scatter (p_pred vs p_gt)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_summary.png", dpi=150)
    plt.close(fig)
    logger.info("Scatter summary: %s", output_dir / "scatter_summary.png")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def generate_report(
    all_results: Dict[int, Dict],
) -> str:
    """Generate Phase 4 gate report string."""
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("Phase 4 Gate Report: Cycle-Consistency Correlation")
    lines.append("=" * 78)
    lines.append("")

    header = (
        f"{'Scene':>8} {'r_pearson':>10} {'r_mag':>8} "
        f"{'rel_L2%':>9} {'IoU':>8} {'N_obs':>10} {'Pass':>6}"
    )
    lines.append(header)
    lines.append("-" * 78)

    sids = sorted(all_results.keys())
    for sid in sids:
        res = all_results[sid]
        r_p = res["r_pearson"]
        r_m = res["r_magnitude"]
        l2 = res["rel_l2"] * 100.0  # percent
        iou = res["iou"]
        n_obs = res["n_observations"]
        passed = "PASS" if r_p >= CORRELATION_THRESHOLD else "FAIL"
        lines.append(
            f"{sid:>8d} {r_p:>10.4f} {r_m:>8.4f} "
            f"{l2:>8.2f}% {iou:>8.4f} {n_obs:>10d} {passed:>6}"
        )

    lines.append("-" * 78)

    # Mean metrics
    r_vals = [all_results[s]["r_pearson"] for s in sids]
    r_mag_vals = [all_results[s]["r_magnitude"] for s in sids]
    l2_vals = [all_results[s]["rel_l2"] for s in sids]
    iou_vals = [all_results[s]["iou"] for s in sids]
    n_total = sum(all_results[s]["n_observations"] for s in sids)

    mean_r = float(np.mean(r_vals))
    mean_r_mag = float(np.mean(r_mag_vals))
    mean_l2 = float(np.mean(l2_vals)) * 100.0
    mean_iou = float(np.mean(iou_vals))
    overall_pass = mean_r >= CORRELATION_THRESHOLD

    lines.append(
        f"{'Mean':>8} {mean_r:>10.4f} {mean_r_mag:>8.4f} "
        f"{mean_l2:>8.2f}% {mean_iou:>8.4f} {n_total:>10d} "
        f"{'PASS' if overall_pass else 'FAIL':>6}"
    )
    lines.append("")
    lines.append("=" * 78)
    lines.append(f"Gate Criterion: mean Pearson r > {CORRELATION_THRESHOLD}")
    lines.append(f"Result:         {mean_r:.4f}")
    lines.append(
        f"Decision:       "
        f"{'PASS -- Phase 5 UNLOCKED' if overall_pass else 'FAIL -- iterate Phase 4'}"
    )
    lines.append("=" * 78)

    # Per-source breakdown
    lines.append("")
    lines.append("Per-Source Correlation (mean across scenes):")
    n_sources = len(next(iter(all_results.values()))["per_source_r"])
    for si in range(n_sources):
        src_r_vals = [all_results[s]["per_source_r"][si] for s in sids]
        lines.append(f"  Source {si}: r = {np.mean(src_r_vals):.4f} +/- {np.std(src_r_vals):.4f}")

    # Frequency range summary
    lines.append("")
    lines.append("Per-Frequency Correlation (mean across scenes):")
    freqs = all_results[sids[0]]["freqs_hz"]
    per_freq_stack = np.array([all_results[s]["per_freq_r"] for s in sids])
    mean_freq_r = per_freq_stack.mean(axis=0)
    # Low / mid / high frequency bands
    f_lo = freqs < 4000
    f_mid = (freqs >= 4000) & (freqs < 6000)
    f_hi = freqs >= 6000
    if f_lo.any():
        lines.append(f"  Low  (2-4 kHz): r = {mean_freq_r[f_lo].mean():.4f}")
    if f_mid.any():
        lines.append(f"  Mid  (4-6 kHz): r = {mean_freq_r[f_mid].mean():.4f}")
    if f_hi.any():
        lines.append(f"  High (6-8 kHz): r = {mean_freq_r[f_hi].mean():.4f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
def export_csv(
    all_results: Dict[int, Dict],
    output_path: Path,
) -> None:
    """Export per-scene metrics to CSV."""
    sids = sorted(all_results.keys())
    with open(output_path, "w") as fh:
        fh.write("scene_id,r_pearson,r_magnitude,rel_l2,iou,n_observations\n")
        for sid in sids:
            res = all_results[sid]
            fh.write(
                f"{sid},{res['r_pearson']:.6f},{res['r_magnitude']:.6f},"
                f"{res['rel_l2']:.6f},{res['iou']:.6f},{res['n_observations']}\n"
            )
    logger.info("CSV exported: %s", output_path)


# ---------------------------------------------------------------------------
# Model loading (reused from eval_phase3.py)
# ---------------------------------------------------------------------------
def load_models(
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[
    InverseModel, torch.nn.Module,
    Dict[int, int], Dict[int, float], Dict[int, int],
]:
    """Load inverse and forward models from checkpoints.

    Returns
    -------
    inverse_model, forward_model, inv_scene_id_map, scene_scales, fwd_scene_id_map
    """
    # Inverse model
    ckpt_path = CHECKPOINT_DIR / f"{args.checkpoint}.pt"
    if not ckpt_path.exists():
        logger.error("Inverse checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    logger.info(
        "Inverse checkpoint: %s (epoch %d, best IoU=%.4f)",
        ckpt_path.name, ckpt["epoch"], ckpt["best_mean_iou"],
    )

    inv_scene_id_map = {int(k): v for k, v in cfg["inv_scene_id_map"].items()}
    inverse_model = build_inverse_model(
        n_scenes=cfg["n_scenes"],
        d_cond=cfg["d_cond"],
        d_hidden=cfg["d_hidden"],
        n_blocks=cfg["n_blocks"],
        n_fourier=cfg.get("n_fourier", 128),
        fourier_sigma=cfg.get("fourier_sigma", 10.0),
        dropout=cfg.get("dropout", 0.05),
        multi_body_scene_ids=cfg.get("multi_body_scene_ids"),
        inv_scene_id_map=inv_scene_id_map,
    )
    inverse_model.load_state_dict(ckpt["model_state_dict"])
    inverse_model = inverse_model.to(device)
    inverse_model.eval()

    # Forward model (frozen)
    fwd_ckpt_name = args.forward_ckpt or cfg["forward_checkpoint"]
    fwd_ckpt_path = FWD_CHECKPOINT_DIR / f"{fwd_ckpt_name}.pt"
    if not fwd_ckpt_path.exists():
        logger.error("Forward checkpoint not found: %s", fwd_ckpt_path)
        sys.exit(1)

    fwd_ckpt = torch.load(fwd_ckpt_path, map_location=device, weights_only=False)
    fwd_cfg = fwd_ckpt["config"]
    logger.info("Forward checkpoint: %s", fwd_ckpt_path.name)

    forward_model = build_transfer_model(
        d_hidden=fwd_cfg.get("d_hidden", 768),
        n_blocks=fwd_cfg.get("n_blocks", 6),
        n_fourier=fwd_cfg.get("n_fourier", 256),
        fourier_sigma=fwd_cfg.get("fourier_sigma", 30.0),
        dropout=fwd_cfg.get("dropout", 0.0),
        n_scenes=fwd_cfg.get("n_scenes", 0),
        scene_emb_dim=fwd_cfg.get("scene_emb_dim", 32),
        d_out=fwd_cfg.get("d_out", 2),
    )
    forward_model.load_state_dict(fwd_ckpt["model_state_dict"])
    forward_model = forward_model.to(device)
    forward_model.eval()
    for p in forward_model.parameters():
        p.requires_grad = False

    scene_scales = fwd_ckpt["scene_scales"]
    fwd_scene_list = fwd_cfg.get(
        "trained_scene_list", sorted(scene_scales.keys()),
    )
    fwd_scene_id_map = {sid: idx for idx, sid in enumerate(fwd_scene_list)}

    return inverse_model, forward_model, inv_scene_id_map, scene_scales, fwd_scene_id_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 Gate: Cycle-Consistency")
    parser.add_argument(
        "--checkpoint", type=str, default="best_phase3_v2",
        help="Phase 3 inverse model checkpoint name (without .pt)",
    )
    parser.add_argument(
        "--forward-ckpt", type=str, default=None,
        help="Phase 2 forward model checkpoint (overrides config in P3 checkpoint)",
    )
    parser.add_argument("--scenes", nargs="+", type=int, default=None)
    parser.add_argument(
        "--freq-chunk", type=int, default=20,
        help="Frequencies per GPU batch (lower = less VRAM)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    (
        inverse_model, forward_model,
        inv_scene_id_map, scene_scales, fwd_scene_id_map,
    ) = load_models(args, device)

    # ------------------------------------------------------------------
    # Load scene data
    # ------------------------------------------------------------------
    all_scenes = load_all_scenes(
        DATA_DIR, scene_scales, fwd_scene_id_map,
        scene_ids=args.scenes,
    )
    logger.info("Loaded %d scenes", len(all_scenes))

    # ------------------------------------------------------------------
    # Evaluate cycle-consistency per scene
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: Dict[int, Dict] = {}
    t0 = time.time()

    for sid in sorted(all_scenes.keys()):
        sd = all_scenes[sid]
        scene_idx = inv_scene_id_map[sid]
        logger.info("Evaluating scene %d (%d obs) ...", sid, sd.n_observations)

        t_scene = time.time()
        results = evaluate_cycle_consistency(
            inverse_model, forward_model, sd, scene_idx, device,
            freq_chunk_size=args.freq_chunk,
        )
        dt = time.time() - t_scene

        all_results[sid] = results
        logger.info(
            "  Scene %d: r=%.4f, r_mag=%.4f, L2=%.2f%%, IoU=%.4f (%.1fs)",
            sid, results["r_pearson"], results["r_magnitude"],
            results["rel_l2"] * 100, results["iou"], dt,
        )

    elapsed = time.time() - t0
    logger.info("Total evaluation time: %.1f s", elapsed)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    report = generate_report(all_results)
    print(report)

    report_path = RESULTS_DIR / "phase4_gate_report.txt"
    with open(report_path, "w") as fh:
        fh.write(report)
    logger.info("Gate report: %s", report_path)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plot_per_scene_correlation(all_results, RESULTS_DIR / "per_scene_correlation.png")
    plot_freq_correlation(all_results, RESULTS_DIR / "freq_correlation.png")

    # Scatter plots (subsample for speed)
    plot_scatter_summary(
        all_results, inverse_model, forward_model,
        all_scenes, inv_scene_id_map, device, RESULTS_DIR,
    )

    # CSV
    export_csv(all_results, RESULTS_DIR / "cycle_consistency_metrics.csv")

    # ------------------------------------------------------------------
    # Gate decision
    # ------------------------------------------------------------------
    r_vals = [all_results[s]["r_pearson"] for s in sorted(all_results.keys())]
    mean_r = float(np.mean(r_vals))

    if mean_r >= CORRELATION_THRESHOLD:
        logger.info(
            "GATE PASSED: mean r = %.4f (> %.1f). Phase 5 UNLOCKED.",
            mean_r, CORRELATION_THRESHOLD,
        )
    else:
        logger.warning(
            "GATE FAILED: mean r = %.4f (need > %.1f). Iterate Phase 4.",
            mean_r, CORRELATION_THRESHOLD,
        )


if __name__ == "__main__":
    main()
