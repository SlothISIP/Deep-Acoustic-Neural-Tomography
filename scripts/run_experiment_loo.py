"""Experiment B: Leave-One-Out (LOO) Code Optimization.

Tests whether the frozen SDF decoder has learned general geometric
primitives by optimizing a fresh random code for held-out scenes.

Method
------
    For each fold scene:
    1. Load best_phase3_v3.pt (full model)
    2. Freeze entire decoder (sdf_decoder parameters)
    3. Re-initialize the code for the target scene to random
    4. Optimize code-only via SDF + Eikonal loss
    5. Evaluate IoU vs ground truth SDF

Folds
-----
    S1  (wedge 60deg)
    S5  (barrier)
    S7  (cylinder)
    S10 (triangle)
    S14 (wedge + cylinder)

Output
------
    results/experiments/loo_generalization.csv

Usage
-----
    python scripts/run_experiment_loo.py
    python scripts/run_experiment_loo.py --epochs 500 --lr 1e-3
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.optim import Adam

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
    eikonal_loss,
    sdf_loss,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiment_loo")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "phase1"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase3"
FWD_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase2"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"

# ---------------------------------------------------------------------------
# Fold scenes
# ---------------------------------------------------------------------------
FOLD_SCENES: List[int] = [1, 5, 7, 10, 14]
SEED: int = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_model_and_config(
    checkpoint_name: str,
    device: torch.device,
) -> Tuple[InverseModel, dict, Dict[int, int]]:
    """Load inverse model from checkpoint.

    Returns model, config, inv_scene_id_map.
    """
    ckpt_path = CHECKPOINT_DIR / f"{checkpoint_name}.pt"
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    inv_scene_id_map = {int(k): v for k, v in cfg["inv_scene_id_map"].items()}

    model = build_inverse_model(
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
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    logger.info(
        "Loaded checkpoint: %s (epoch %d, IoU=%.4f)",
        ckpt_path.name, ckpt["epoch"], ckpt["best_mean_iou"],
    )
    return model, cfg, inv_scene_id_map


@torch.no_grad()
def compute_scene_iou(
    model: InverseModel,
    scene_data,
    scene_idx: int,
    device: torch.device,
) -> float:
    """Compute SDF IoU for one scene."""
    model.eval()
    xy_m = torch.from_numpy(scene_data.grid_coords).float().to(device)
    sdf_gt = torch.from_numpy(scene_data.sdf_flat).float().to(device)

    preds = []
    for i in range(0, len(xy_m), 8192):
        chunk = xy_m[i : i + 8192]
        pred = model.predict_sdf(scene_idx, chunk)
        preds.append(pred.squeeze(-1))
    sdf_pred = torch.cat(preds, dim=0)
    return compute_sdf_iou(sdf_pred, sdf_gt)


@torch.no_grad()
def compute_scene_l1(
    model: InverseModel,
    scene_data,
    scene_idx: int,
    device: torch.device,
) -> float:
    """Compute mean |SDF_pred - SDF_gt| for one scene."""
    model.eval()
    xy_m = torch.from_numpy(scene_data.grid_coords).float().to(device)
    sdf_gt = torch.from_numpy(scene_data.sdf_flat).float().to(device)

    preds = []
    for i in range(0, len(xy_m), 8192):
        chunk = xy_m[i : i + 8192]
        pred = model.predict_sdf(scene_idx, chunk)
        preds.append(pred.squeeze(-1))
    sdf_pred = torch.cat(preds, dim=0)
    return float(torch.abs(sdf_pred - sdf_gt).mean().item())


def run_loo_fold(
    fold_sid: int,
    model: InverseModel,
    inv_scene_id_map: Dict[int, int],
    all_scenes: Dict,
    device: torch.device,
    epochs: int = 500,
    lr: float = 1e-3,
) -> Dict:
    """Run one LOO fold: re-init code, freeze decoder, optimize code-only.

    Returns result dict.
    """
    scene_idx = inv_scene_id_map[fold_sid]
    sd = all_scenes[fold_sid]

    logger.info("-" * 50)
    logger.info("LOO Fold: S%d (idx=%d)", fold_sid, scene_idx)
    logger.info("-" * 50)

    # 1. Record pre-reset IoU (trained model)
    pre_iou = compute_scene_iou(model, sd, scene_idx, device)
    logger.info("  Pre-reset IoU (trained): %.4f", pre_iou)

    # 2. Re-initialize code for this scene to random
    start, end = model._scene_code_ranges[scene_idx]
    with torch.no_grad():
        model.auto_decoder_codes.weight[start:end].normal_(std=0.01)

    # Verify IoU dropped
    post_reset_iou = compute_scene_iou(model, sd, scene_idx, device)
    logger.info("  Post-reset IoU (random code): %.4f", post_reset_iou)

    # 3. Freeze entire decoder
    for param in model.sdf_decoder.parameters():
        param.requires_grad = False

    # Only the code for this scene is trainable
    model.auto_decoder_codes.weight.requires_grad = True

    # Create optimizer for the embedding weight only
    optimizer = Adam([model.auto_decoder_codes.weight], lr=lr)

    # Gradient hook: zero out non-target code gradients
    def _code_grad_hook(grad: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(grad)
        mask[start:end] = 1.0
        return grad * mask

    hook_handle = model.auto_decoder_codes.weight.register_hook(_code_grad_hook)

    n_sdf_batch = 4096
    t0 = time.time()
    iou_history: List[float] = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Sample SDF points with boundary oversampling
        bdy_mask = np.abs(sd.sdf_flat) < 0.1
        bdy_idx = np.where(bdy_mask)[0]
        far_idx = np.where(~bdy_mask)[0]
        n_bdy = min(int(n_sdf_batch * 0.75), len(bdy_idx))
        n_far = min(n_sdf_batch - n_bdy, len(far_idx))

        if n_bdy > 0 and n_far > 0:
            chosen = np.concatenate([
                np.random.choice(bdy_idx, n_bdy, replace=n_bdy > len(bdy_idx)),
                np.random.choice(far_idx, n_far, replace=n_far > len(far_idx)),
            ])
        else:
            chosen = np.random.choice(len(sd.sdf_flat), n_sdf_batch, replace=False)
        sdf_idx = torch.from_numpy(chosen).long().to(device)

        xy_m = torch.from_numpy(sd.grid_coords).float().to(device)
        sdf_gt_all = torch.from_numpy(sd.sdf_flat).float().to(device)

        xy_batch = xy_m[sdf_idx].clone().requires_grad_(True)  # (B, 2)
        sdf_gt_batch = sdf_gt_all[sdf_idx].unsqueeze(-1)  # (B, 1)

        sdf_pred = model.predict_sdf(scene_idx, xy_batch)  # (B, 1)
        l_sdf = sdf_loss(sdf_pred, sdf_gt_batch)
        l_eik = eikonal_loss(sdf_pred, xy_batch)
        z = model.get_code(scene_idx)
        l_z = (z ** 2).mean()

        total = l_sdf + 0.1 * l_eik + 1e-3 * l_z

        if torch.isfinite(total):
            total.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            iou = compute_scene_iou(model, sd, scene_idx, device)
            iou_history.append(iou)
            logger.info(
                "  Ep %3d: loss=%.4e sdf=%.4e eik=%.4e IoU=%.4f",
                epoch + 1, total.item(), l_sdf.item(), l_eik.item(), iou,
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    hook_handle.remove()

    # Final evaluation
    final_iou = compute_scene_iou(model, sd, scene_idx, device)
    final_l1 = compute_scene_l1(model, sd, scene_idx, device)

    # Unfreeze decoder for next fold
    for param in model.sdf_decoder.parameters():
        param.requires_grad = True

    logger.info(
        "  S%d LOO result: IoU %.4f -> %.4f -> %.4f (pre/reset/final), L1=%.4f, %.1fs",
        fold_sid, pre_iou, post_reset_iou, final_iou, final_l1, elapsed,
    )

    return {
        "fold_scene": fold_sid,
        "pre_iou": pre_iou,
        "post_reset_iou": post_reset_iou,
        "final_iou": final_iou,
        "l1_error": final_l1,
        "epochs": epochs,
        "time_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment B: LOO Code Optimization"
    )
    parser.add_argument("--checkpoint", type=str, default="best_phase3_v3")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load model
    model, cfg, inv_scene_id_map = load_model_and_config(args.checkpoint, device)

    # Load forward model metadata for scene_scales
    fwd_ckpt_name = cfg["forward_checkpoint"]
    fwd_ckpt_path = FWD_CHECKPOINT_DIR / f"{fwd_ckpt_name}.pt"
    fwd_ckpt = torch.load(fwd_ckpt_path, map_location=device, weights_only=False)
    scene_scales = fwd_ckpt["scene_scales"]
    fwd_cfg = fwd_ckpt["config"]
    fwd_scene_list = fwd_cfg.get("trained_scene_list", sorted(scene_scales.keys()))
    fwd_scene_id_map = {sid: idx for idx, sid in enumerate(fwd_scene_list)}

    # Load scene data
    all_scenes = load_all_scenes(DATA_DIR, scene_scales, fwd_scene_id_map)

    fold_scenes = args.folds if args.folds else FOLD_SCENES
    results: List[Dict] = []

    for fold_sid in fold_scenes:
        if fold_sid not in all_scenes:
            logger.warning("Fold scene %d not found, skipping", fold_sid)
            continue
        if fold_sid not in inv_scene_id_map:
            logger.warning("Fold scene %d not in inverse model, skipping", fold_sid)
            continue

        # Reload model fresh for each fold (important: independent experiments)
        model, _, inv_scene_id_map = load_model_and_config(args.checkpoint, device)

        res = run_loo_fold(
            fold_sid, model, inv_scene_id_map, all_scenes,
            device, epochs=args.epochs, lr=args.lr,
        )
        results.append(res)

    # Write CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "loo_generalization.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold_scene", "pre_iou", "post_reset_iou",
            "final_iou", "l1_error", "epochs", "time_s",
        ])
        for res in results:
            writer.writerow([
                res["fold_scene"],
                f"{res['pre_iou']:.4f}",
                f"{res['post_reset_iou']:.4f}",
                f"{res['final_iou']:.4f}",
                f"{res['l1_error']:.6f}",
                res["epochs"],
                f"{res['time_s']:.1f}",
            ])

    logger.info("Results CSV: %s", csv_path)

    # Print summary
    print("\n" + "=" * 70)
    print("Experiment B: LOO Code Optimization (Generalization)")
    print("=" * 70)
    print(
        f"{'Scene':<8} {'Pre IoU':>10} {'Reset IoU':>10} "
        f"{'Final IoU':>10} {'L1':>10} {'Time':>8}"
    )
    print("-" * 60)
    for res in results:
        print(
            f"S{res['fold_scene']:<7d} {res['pre_iou']:>10.4f} "
            f"{res['post_reset_iou']:>10.4f} {res['final_iou']:>10.4f} "
            f"{res['l1_error']:>10.6f} {res['time_s']:>7.1f}s"
        )

    if results:
        mean_final = np.mean([r["final_iou"] for r in results])
        mean_pre = np.mean([r["pre_iou"] for r in results])
        print("-" * 60)
        print(f"{'Mean':<8} {mean_pre:>10.4f} {'':>10} {mean_final:>10.4f}")
        print(f"Recovery: {mean_final / mean_pre * 100:.1f}% of trained IoU")
    print("=" * 70)


if __name__ == "__main__":
    main()
