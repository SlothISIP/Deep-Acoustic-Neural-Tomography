"""Phase 3 Gate Evaluation: SDF IoU + Helmholtz Residual.

Loads the trained inverse model and evaluates:
    1. SDF IoU: predicted vs ground truth SDF on 200x200 grid per scene
    2. Helmholtz residual: PDE satisfaction at random exterior points
    3. SDF contour visualization: predicted vs ground truth boundaries

Gate criterion
--------------
    Mean SDF IoU > 0.8 AND Helmholtz residual < 1e-3

Usage
-----
    python scripts/eval_phase3.py                                # best checkpoint
    python scripts/eval_phase3.py --checkpoint latest_phase3     # specific checkpoint
    python scripts/eval_phase3.py --scenes 1 2 3                 # specific scenes
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.forward_model import build_transfer_model
from src.inverse_dataset import load_all_scenes
from src.inverse_model import (
    InverseModel,
    build_inverse_model,
    compute_sdf_iou,
    helmholtz_residual,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase3_eval")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "phase1"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase3"
FWD_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase2"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase3"

# ---------------------------------------------------------------------------
# Gate thresholds
# ---------------------------------------------------------------------------
IOU_THRESHOLD: float = 0.8
HELMHOLTZ_THRESHOLD: float = 1e-3


# ---------------------------------------------------------------------------
# SDF IoU evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_sdf_iou(
    inverse_model: InverseModel,
    scene_data,
    scene_idx: int,
    device: torch.device,
) -> Dict:
    """Evaluate SDF IoU for one scene on the full grid.

    Returns dict with iou, sdf_pred, sdf_gt arrays for visualization.
    """
    sd = scene_data
    xy_m = torch.from_numpy(sd.grid_coords).float().to(device)  # (G, 2)
    sdf_gt = torch.from_numpy(sd.sdf_flat).float().to(device)   # (G,)

    # Predict in chunks
    chunk_size = 8192
    sdf_preds = []
    for i in range(0, len(xy_m), chunk_size):
        chunk = xy_m[i : i + chunk_size]
        pred = inverse_model.predict_sdf(scene_idx, chunk)
        sdf_preds.append(pred.squeeze(-1))

    sdf_pred_flat = torch.cat(sdf_preds, dim=0)  # (G,)
    iou = compute_sdf_iou(sdf_pred_flat, sdf_gt)

    # Reshape for visualization
    gx, gy = len(sd.grid_x), len(sd.grid_y)
    sdf_pred_grid = sdf_pred_flat.cpu().numpy().reshape(gx, gy)
    sdf_gt_grid = sdf_gt.cpu().numpy().reshape(gx, gy)

    # Additional metrics
    l1_error = (sdf_pred_flat - sdf_gt).abs().mean().item()
    l2_error = ((sdf_pred_flat - sdf_gt) ** 2).mean().sqrt().item()

    return {
        "iou": iou,
        "l1_error": l1_error,
        "l2_error": l2_error,
        "sdf_pred_grid": sdf_pred_grid,
        "sdf_gt_grid": sdf_gt_grid,
    }


# ---------------------------------------------------------------------------
# Helmholtz residual evaluation
# ---------------------------------------------------------------------------
def evaluate_helmholtz(
    inverse_model: InverseModel,
    forward_model,
    scene_data,
    scene_idx: int,
    device: torch.device,
    n_points: int = 1000,
) -> float:
    """Evaluate Helmholtz PDE residual at random exterior points.

    Returns normalized mean residual (dimensionless).
    """
    sd = scene_data

    # Select exterior points (SDF > 0.05 m margin)
    exterior_mask = sd.sdf_flat > 0.05
    exterior_idx = np.where(exterior_mask)[0]

    if len(exterior_idx) < n_points:
        logger.warning(
            "Scene %d: only %d exterior points (need %d)",
            sd.scene_id, len(exterior_idx), n_points,
        )
        n_points = len(exterior_idx)

    if n_points == 0:
        return float("nan")

    # Sample points
    chosen_idx = np.random.choice(exterior_idx, size=n_points, replace=False)
    xy_helm = torch.from_numpy(
        sd.grid_coords[chosen_idx]
    ).float().to(device)  # (N, 2)

    # Random source and frequency
    si_helm = np.random.randint(0, sd.n_sources, size=n_points)
    fi_helm = np.random.randint(0, sd.n_freqs, size=n_points)

    x_src = torch.from_numpy(sd.src_pos[si_helm]).float().to(device)
    k_vals = torch.from_numpy(
        sd.k_arr[fi_helm]
    ).float().to(device).unsqueeze(-1)  # (N, 1)

    fwd_ids = torch.full(
        (n_points,), sd.fwd_scene_idx, dtype=torch.long, device=device,
    )

    z_code = inverse_model.get_code(scene_idx)

    # Compute in chunks to manage VRAM (2nd-order autograd is expensive)
    chunk_size = 128
    residuals = []

    for i in range(0, n_points, chunk_size):
        end = min(i + chunk_size, n_points)
        res_chunk = helmholtz_residual(
            forward_model, inverse_model.sdf_decoder,
            z_code,
            x_src[i:end], xy_helm[i:end], k_vals[i:end],
            sd.scene_scale, fwd_ids[i:end],
        )
        residuals.append(res_chunk.item())

    return float(np.mean(residuals))


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_sdf_contours(
    results: Dict,
    scene_data,
    output_path: Path,
) -> None:
    """Plot predicted vs ground truth SDF contours for one scene."""
    sd = scene_data
    sdf_pred = results["sdf_pred_grid"]
    sdf_gt = results["sdf_gt_grid"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, sdf, title in [
        (axes[0], sdf_gt, "Ground Truth SDF"),
        (axes[1], sdf_pred, "Predicted SDF"),
        (axes[2], sdf_pred - sdf_gt, "Error (Pred - GT)"),
    ]:
        if title == "Error (Pred - GT)":
            vmax = max(abs(sdf.min()), abs(sdf.max()), 0.05)
            im = ax.pcolormesh(
                sd.grid_x, sd.grid_y, sdf.T,
                cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                shading="auto",
            )
        else:
            vmax = max(abs(sdf.min()), abs(sdf.max()))
            im = ax.pcolormesh(
                sd.grid_x, sd.grid_y, sdf.T,
                cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                shading="auto",
            )
            # Zero contour (body boundary)
            ax.contour(
                sd.grid_x, sd.grid_y, sdf.T,
                levels=[0.0], colors="k", linewidths=2,
            )

        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        f"Scene {sd.scene_id}: IoU={results['iou']:.4f}, "
        f"L1={results['l1_error']:.4e}, L2={results['l2_error']:.4e}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_iou_summary(
    all_ious: Dict[int, float],
    output_path: Path,
) -> None:
    """Bar chart of per-scene SDF IoU."""
    sids = sorted(all_ious.keys())
    ious = [all_ious[s] for s in sids]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(
        range(len(sids)), ious, color="steelblue", edgecolor="navy",
    )

    for bar, iou in zip(bars, ious):
        if iou < IOU_THRESHOLD:
            bar.set_color("salmon")
            bar.set_edgecolor("darkred")

    ax.axhline(y=IOU_THRESHOLD, color="r", linestyle="--", label=f"Gate ({IOU_THRESHOLD})")
    ax.set_xticks(range(len(sids)))
    ax.set_xticklabels([f"S{s}" for s in sids])
    ax.set_xlabel("Scene")
    ax.set_ylabel("SDF IoU")
    ax.set_ylim(0, 1.05)
    ax.set_title("Phase 3: Per-Scene SDF IoU")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("IoU summary plot: %s", output_path)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def generate_report(
    all_ious: Dict[int, float],
    all_helmholtz: Dict[int, float],
) -> str:
    """Generate Phase 3 gate report string."""
    lines = []
    lines.append("=" * 70)
    lines.append("Phase 3 Gate Report: SDF IoU + Helmholtz Residual")
    lines.append("=" * 70)
    lines.append("")

    lines.append(
        f"{'Scene':>8} {'IoU':>10} {'Helmholtz':>12} {'IoU Pass':>10} {'Helm Pass':>10}"
    )
    lines.append("-" * 70)

    for sid in sorted(all_ious.keys()):
        iou = all_ious[sid]
        helm = all_helmholtz.get(sid, float("nan"))
        iou_pass = "PASS" if iou >= IOU_THRESHOLD else "FAIL"
        helm_pass = "PASS" if helm < HELMHOLTZ_THRESHOLD else "FAIL"
        lines.append(
            f"{sid:>8d} {iou:>10.4f} {helm:>12.2e} {iou_pass:>10} {helm_pass:>10}"
        )

    lines.append("-" * 70)

    mean_iou = np.mean(list(all_ious.values()))
    mean_helm = np.mean(
        [v for v in all_helmholtz.values() if np.isfinite(v)]
    )
    overall_iou_pass = mean_iou >= IOU_THRESHOLD
    overall_helm_pass = mean_helm < HELMHOLTZ_THRESHOLD

    lines.append(
        f"{'Mean':>8} {mean_iou:>10.4f} {mean_helm:>12.2e} "
        f"{'PASS' if overall_iou_pass else 'FAIL':>10} "
        f"{'PASS' if overall_helm_pass else 'FAIL':>10}"
    )
    lines.append("")

    lines.append("=" * 70)
    lines.append(f"Gate Criteria: IoU > {IOU_THRESHOLD}, Helmholtz < {HELMHOLTZ_THRESHOLD:.0e}")
    lines.append(f"IoU Result:       {mean_iou:.4f}")
    lines.append(f"Helmholtz Result: {mean_helm:.2e}")

    overall_pass = overall_iou_pass and overall_helm_pass
    lines.append(
        f"Decision:         "
        f"{'PASS -- Phase 4 UNLOCKED' if overall_pass else 'FAIL -- iterate Phase 3'}"
    )
    lines.append("=" * 70)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 Gate Evaluation")
    parser.add_argument(
        "--checkpoint", type=str, default="best_phase3",
        help="Phase 3 checkpoint name (without .pt)",
    )
    parser.add_argument(
        "--forward-ckpt", type=str, default=None,
        help="Phase 2 forward model checkpoint (overrides config in checkpoint)",
    )
    parser.add_argument("--scenes", nargs="+", type=int, default=None)
    parser.add_argument(
        "--n-helmholtz", type=int, default=1000,
        help="Number of exterior points for Helmholtz residual evaluation",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Load inverse model checkpoint
    # ------------------------------------------------------------------
    ckpt_path = CHECKPOINT_DIR / f"{args.checkpoint}.pt"
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    logger.info(
        "Loaded checkpoint: %s (epoch %d, best IoU=%.4f)",
        ckpt_path.name,
        ckpt["epoch"],
        ckpt["best_mean_iou"],
    )

    # Build inverse model
    inverse_model = build_inverse_model(
        n_scenes=cfg["n_scenes"],
        d_cond=cfg["d_cond"],
        d_hidden=cfg["d_hidden"],
        n_blocks=cfg["n_blocks"],
        n_fourier=cfg.get("n_fourier", 128),
        fourier_sigma=cfg.get("fourier_sigma", 10.0),
        dropout=cfg.get("dropout", 0.05),
    )
    inverse_model.load_state_dict(ckpt["model_state_dict"])
    inverse_model = inverse_model.to(device)
    inverse_model.eval()

    inv_scene_id_map = cfg["inv_scene_id_map"]
    # Handle JSON key type (may be strings after save/load)
    inv_scene_id_map = {int(k): v for k, v in inv_scene_id_map.items()}

    # ------------------------------------------------------------------
    # Load frozen forward model
    # ------------------------------------------------------------------
    fwd_ckpt_name = args.forward_ckpt or cfg["forward_checkpoint"]
    fwd_ckpt_path = FWD_CHECKPOINT_DIR / f"{fwd_ckpt_name}.pt"
    if not fwd_ckpt_path.exists():
        logger.error("Forward checkpoint not found: %s", fwd_ckpt_path)
        sys.exit(1)

    fwd_ckpt = torch.load(fwd_ckpt_path, map_location=device, weights_only=False)
    fwd_cfg = fwd_ckpt["config"]

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

    # ------------------------------------------------------------------
    # Load scene data
    # ------------------------------------------------------------------
    all_scenes = load_all_scenes(
        DATA_DIR, scene_scales, fwd_scene_id_map,
        scene_ids=args.scenes,
    )

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_ious: Dict[int, float] = {}
    all_helmholtz: Dict[int, float] = {}

    t0 = time.time()

    for sid in sorted(all_scenes.keys()):
        sd = all_scenes[sid]
        scene_idx = inv_scene_id_map[sid]

        logger.info("Evaluating scene %d ...", sid)

        # SDF IoU
        sdf_results = evaluate_sdf_iou(
            inverse_model, sd, scene_idx, device,
        )
        all_ious[sid] = sdf_results["iou"]

        # Helmholtz residual
        helm_res = evaluate_helmholtz(
            inverse_model, forward_model, sd, scene_idx, device,
            n_points=args.n_helmholtz,
        )
        all_helmholtz[sid] = helm_res

        logger.info(
            "  Scene %d: IoU=%.4f, L1=%.4e, L2=%.4e, Helmholtz=%.2e",
            sid,
            sdf_results["iou"],
            sdf_results["l1_error"],
            sdf_results["l2_error"],
            helm_res,
        )

        # SDF contour plot
        plot_path = RESULTS_DIR / f"sdf_contour_scene_{sid:03d}.png"
        plot_sdf_contours(sdf_results, sd, plot_path)
        logger.info("  Contour plot: %s", plot_path)

    elapsed = time.time() - t0
    logger.info("Evaluation complete in %.1f s", elapsed)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    report = generate_report(all_ious, all_helmholtz)
    print(report)

    report_path = RESULTS_DIR / "phase3_gate_report.txt"
    with open(report_path, "w") as fh:
        fh.write(report)
    logger.info("Gate report: %s", report_path)

    # Summary plot
    plot_iou_summary(all_ious, RESULTS_DIR / "per_scene_iou.png")

    # ------------------------------------------------------------------
    # Gate decision
    # ------------------------------------------------------------------
    mean_iou = np.mean(list(all_ious.values()))
    mean_helm = np.mean(
        [v for v in all_helmholtz.values() if np.isfinite(v)]
    )

    if mean_iou >= IOU_THRESHOLD and mean_helm < HELMHOLTZ_THRESHOLD:
        logger.info(
            "GATE PASSED: IoU=%.4f (>%.1f), Helmholtz=%.2e (<%.0e)",
            mean_iou, IOU_THRESHOLD, mean_helm, HELMHOLTZ_THRESHOLD,
        )
        logger.info("Phase 4 UNLOCKED")
    else:
        logger.warning(
            "GATE FAILED: IoU=%.4f (need >%.1f), Helmholtz=%.2e (need <%.0e)",
            mean_iou, IOU_THRESHOLD, mean_helm, HELMHOLTZ_THRESHOLD,
        )


if __name__ == "__main__":
    main()
