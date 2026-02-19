#!/usr/bin/env python3
"""P1: Extended SDF Quality Metrics — Chamfer, Hausdorff, Boundary L1.

Loads the trained inverse model and evaluates SDF reconstruction quality
using metrics beyond binary IoU:

    1. Chamfer Distance (CD) — mean bidirectional boundary error [m]
    2. Hausdorff Distance (HD) — worst-case boundary deviation [m]
    3. SDF L1 near boundary (|s_gt| < 0.1m) — geometry-critical region
    4. SDF L1 far from boundary (|s_gt| > 0.5m) — far-field region
    5. IoU (recomputed for consistency)

Output: results/experiments/sdf_metrics_extended.csv

Usage
-----
    python scripts/eval_sdf_metrics.py
    python scripts/eval_sdf_metrics.py --device cpu     # force CPU (GPU busy)
    python scripts/eval_sdf_metrics.py --scenes 1 5 12  # specific scenes
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

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
    compute_chamfer_hausdorff,
    compute_sdf_boundary_errors,
    compute_sdf_iou,
    extract_zero_contour,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sdf_metrics")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "phase1"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase3"
FWD_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase2"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"


# ---------------------------------------------------------------------------
# Evaluate one scene
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_scene_metrics(
    inverse_model: InverseModel,
    scene_data,
    scene_idx: int,
    device: torch.device,
) -> Dict:
    """Evaluate all SDF metrics for one scene.

    Returns dict with IoU, Chamfer, Hausdorff, boundary L1 errors.
    """
    sd = scene_data
    xy_m = torch.from_numpy(sd.grid_coords).float().to(device)  # (G, 2)
    sdf_gt = torch.from_numpy(sd.sdf_flat).float().to(device)   # (G,)

    # Predict SDF in chunks
    chunk_size = 8192
    sdf_preds = []
    for i in range(0, len(xy_m), chunk_size):
        chunk = xy_m[i : i + chunk_size]
        pred = inverse_model.predict_sdf(scene_idx, chunk)
        sdf_preds.append(pred.squeeze(-1))

    sdf_pred_flat = torch.cat(sdf_preds, dim=0)  # (G,)

    # IoU
    iou = compute_sdf_iou(sdf_pred_flat, sdf_gt)

    # Convert to numpy for contour extraction
    gx, gy = len(sd.grid_x), len(sd.grid_y)
    sdf_pred_np = sdf_pred_flat.cpu().numpy().reshape(gx, gy)
    sdf_gt_np = sdf_gt.cpu().numpy().reshape(gx, gy)

    # Extract zero-level contours
    contour_pred = extract_zero_contour(sdf_pred_np, sd.grid_x, sd.grid_y)
    contour_gt = extract_zero_contour(sdf_gt_np, sd.grid_x, sd.grid_y)

    # Chamfer & Hausdorff
    chamfer_m, hausdorff_m = compute_chamfer_hausdorff(contour_pred, contour_gt)

    # Boundary-stratified L1 errors
    bdy_errors = compute_sdf_boundary_errors(
        sdf_pred_flat.cpu().numpy(),
        sdf_gt.cpu().numpy(),
        near_threshold_m=0.1,
        far_threshold_m=0.5,
    )

    return {
        "iou": iou,
        "chamfer_m": chamfer_m,
        "hausdorff_m": hausdorff_m,
        "l1_near": bdy_errors["l1_near"],
        "l1_far": bdy_errors["l1_far"],
        "l1_overall": bdy_errors["l1_overall"],
        "n_contour_pred": len(contour_pred),
        "n_contour_gt": len(contour_gt),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="P1: Extended SDF Quality Metrics"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="best_phase3",
        help="Phase 3 inverse model checkpoint name (without .pt)",
    )
    parser.add_argument(
        "--forward-ckpt", type=str, default=None,
        help="Phase 2 forward model checkpoint name",
    )
    parser.add_argument("--scenes", nargs="+", type=int, default=None)
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cpu' or 'cuda' (default: auto-detect)",
    )
    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
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
        "Loaded inverse checkpoint: %s (epoch %d)",
        ckpt_path.name, ckpt["epoch"],
    )

    inv_scene_id_map_raw = {int(k): v for k, v in cfg["inv_scene_id_map"].items()}
    inverse_model = build_inverse_model(
        n_scenes=cfg["n_scenes"],
        d_cond=cfg["d_cond"],
        d_hidden=cfg["d_hidden"],
        n_blocks=cfg["n_blocks"],
        n_fourier=cfg.get("n_fourier", 128),
        fourier_sigma=cfg.get("fourier_sigma", 10.0),
        dropout=cfg.get("dropout", 0.05),
        multi_body_scene_ids=cfg.get("multi_body_scene_ids"),
        inv_scene_id_map=inv_scene_id_map_raw,
    )
    inverse_model.load_state_dict(ckpt["model_state_dict"])
    inverse_model = inverse_model.to(device)
    inverse_model.eval()

    # ------------------------------------------------------------------
    # Load forward model (needed for scene_scales and fwd_scene_id_map)
    # ------------------------------------------------------------------
    fwd_ckpt_name = args.forward_ckpt or cfg["forward_checkpoint"]
    fwd_ckpt_path = FWD_CHECKPOINT_DIR / f"{fwd_ckpt_name}.pt"
    if not fwd_ckpt_path.exists():
        logger.error("Forward checkpoint not found: %s", fwd_ckpt_path)
        sys.exit(1)

    fwd_ckpt = torch.load(fwd_ckpt_path, map_location=device, weights_only=False)
    fwd_cfg = fwd_ckpt["config"]
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
    # Evaluate all scenes
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("P1: Extended SDF Metrics (IoU, Chamfer, Hausdorff, L1)")
    logger.info("=" * 70)

    results_rows: List[Dict] = []
    t0 = time.time()

    for sid in sorted(all_scenes.keys()):
        sd = all_scenes[sid]
        scene_idx = inv_scene_id_map_raw[sid]

        metrics = evaluate_scene_metrics(
            inverse_model, sd, scene_idx, device,
        )

        logger.info(
            "  S%2d: IoU=%.4f  CD=%.4f m  HD=%.4f m  L1_near=%.4f  L1_far=%.4f  "
            "contour: %d/%d pts",
            sid,
            metrics["iou"],
            metrics["chamfer_m"],
            metrics["hausdorff_m"],
            metrics["l1_near"],
            metrics["l1_far"],
            metrics["n_contour_pred"],
            metrics["n_contour_gt"],
        )

        results_rows.append({
            "scene": sid,
            "iou": f"{metrics['iou']:.4f}",
            "chamfer_m": f"{metrics['chamfer_m']:.6f}",
            "hausdorff_m": f"{metrics['hausdorff_m']:.6f}",
            "l1_near_bdy": f"{metrics['l1_near']:.6f}",
            "l1_far_bdy": f"{metrics['l1_far']:.6f}",
            "l1_overall": f"{metrics['l1_overall']:.6f}",
        })

    elapsed = time.time() - t0
    logger.info("Evaluation complete in %.1f s", elapsed)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    ious = [float(r["iou"]) for r in results_rows]
    chamfers = [float(r["chamfer_m"]) for r in results_rows]
    hausdorffs = [float(r["hausdorff_m"]) for r in results_rows]
    l1_nears = [float(r["l1_near_bdy"]) for r in results_rows]
    l1_fars = [float(r["l1_far_bdy"]) for r in results_rows]

    # Filter out inf values for mean computation
    chamfers_finite = [c for c in chamfers if np.isfinite(c)]
    hausdorffs_finite = [h for h in hausdorffs if np.isfinite(h)]

    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("  Mean IoU:      %.4f ± %.4f", np.mean(ious), np.std(ious))
    if chamfers_finite:
        logger.info(
            "  Mean CD:       %.4f ± %.4f m",
            np.mean(chamfers_finite), np.std(chamfers_finite),
        )
    if hausdorffs_finite:
        logger.info(
            "  Mean HD:       %.4f ± %.4f m",
            np.mean(hausdorffs_finite), np.std(hausdorffs_finite),
        )
    logger.info(
        "  Mean L1 near:  %.4f ± %.4f",
        np.mean([l for l in l1_nears if np.isfinite(l)]),
        np.std([l for l in l1_nears if np.isfinite(l)]),
    )
    logger.info(
        "  Mean L1 far:   %.4f ± %.4f",
        np.mean([l for l in l1_fars if np.isfinite(l)]),
        np.std([l for l in l1_fars if np.isfinite(l)]),
    )
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "sdf_metrics_extended.csv"

    fieldnames = [
        "scene", "iou", "chamfer_m", "hausdorff_m",
        "l1_near_bdy", "l1_far_bdy", "l1_overall",
    ]

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_rows:
            writer.writerow(row)
        # Mean row
        writer.writerow({
            "scene": "mean",
            "iou": f"{np.mean(ious):.4f}",
            "chamfer_m": f"{np.mean(chamfers_finite):.6f}" if chamfers_finite else "nan",
            "hausdorff_m": f"{np.mean(hausdorffs_finite):.6f}" if hausdorffs_finite else "nan",
            "l1_near_bdy": f"{np.mean([l for l in l1_nears if np.isfinite(l)]):.6f}",
            "l1_far_bdy": f"{np.mean([l for l in l1_fars if np.isfinite(l)]):.6f}",
            "l1_overall": f"{np.mean([float(r['l1_overall']) for r in results_rows]):.6f}",
        })

    logger.info("Results saved: %s", csv_path)


if __name__ == "__main__":
    main()
