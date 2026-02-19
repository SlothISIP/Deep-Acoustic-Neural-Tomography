"""Experiment A2: S12 Frozen-Decoder Code-Only Optimization.

Tests whether freezing the SDF decoder and optimizing only S12 codes
can improve S12 IoU without degrading other scenes.

Rationale
---------
    Experiment A showed that co-training decoder with S12 causes
    catastrophic forgetting on other scenes (0.95 -> 0.60 IoU).
    By freezing the decoder, we guarantee other scenes remain
    unchanged while giving S12 codes maximum freedom to adapt.

Method
------
    1. Load best_phase3_v3.pt (K=2 for S12, alpha=50)
    2. Rebuild model with new alpha (default: 100)
    3. Load weights via compat remapping
    4. Freeze entire sdf_decoder (requires_grad=False)
    5. Only optimize S12 codes via gradient hook
    6. SDF + Eikonal + code regularization loss
    7. Evaluate all 15 scenes to confirm others unchanged

Configs
-------
    alpha50-frozen:   K=2, alpha=50,  frozen decoder (control)
    alpha100-frozen:  K=2, alpha=100, frozen decoder (primary)
    alpha200-frozen:  K=2, alpha=200, frozen decoder (aggressive)

Output
------
    results/experiments/s12_frozen_decoder.csv

Usage
-----
    python scripts/run_experiment_s12_frozen.py
    python scripts/run_experiment_s12_frozen.py --epochs 500 --lr 1e-3
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
logger = logging.getLogger("experiment_s12_frozen")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "phase1"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase3"
FWD_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase2"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
S12_SCENE_ID: int = 12
SEED: int = 42

CONFIGS: List[Dict] = [
    {"name": "alpha50-frozen", "K": 2, "alpha": 50.0},
    {"name": "alpha100-frozen", "K": 2, "alpha": 100.0},
    {"name": "alpha200-frozen", "K": 2, "alpha": 200.0},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_base_checkpoint(
    device: torch.device,
    checkpoint_name: str = "best_phase3_v3",
) -> Tuple[dict, dict]:
    """Load base checkpoint state dict and config.

    Returns
    -------
    state_dict : dict
    config : dict
    """
    ckpt_path = CHECKPOINT_DIR / f"{checkpoint_name}.pt"
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    logger.info(
        "Base checkpoint: %s (epoch %d, IoU=%.4f)",
        ckpt_path.name, ckpt["epoch"], ckpt["best_mean_iou"],
    )
    return ckpt["model_state_dict"], ckpt["config"]


def build_model_for_config(
    cfg: Dict,
    base_config: dict,
    base_state_dict: dict,
    device: torch.device,
) -> Tuple[InverseModel, Dict[int, int]]:
    """Build inverse model with specific alpha, load weights, freeze decoder.

    Returns model and inv_scene_id_map.
    """
    inv_scene_id_map = {
        int(k): v for k, v in base_config["inv_scene_id_map"].items()
    }
    multi_body = {S12_SCENE_ID: cfg["K"]}

    model = build_inverse_model(
        n_scenes=base_config["n_scenes"],
        d_cond=base_config["d_cond"],
        d_hidden=base_config["d_hidden"],
        n_blocks=base_config["n_blocks"],
        n_fourier=base_config.get("n_fourier", 128),
        fourier_sigma=base_config.get("fourier_sigma", 10.0),
        dropout=base_config.get("dropout", 0.05),
        multi_body_scene_ids=multi_body,
        inv_scene_id_map=inv_scene_id_map,
        smooth_min_alpha=cfg["alpha"],
    )

    # Load base weights with compat remapping (handles K changes)
    model.load_state_dict_compat(base_state_dict)
    model = model.to(device)
    return model, inv_scene_id_map


@torch.no_grad()
def evaluate_all_ious(
    model: InverseModel,
    all_scenes: Dict,
    inv_scene_id_map: Dict[int, int],
    device: torch.device,
) -> Dict[int, float]:
    """Compute per-scene IoU for all scenes."""
    model.eval()
    ious: Dict[int, float] = {}

    for sid, sd in sorted(all_scenes.items()):
        scene_idx = inv_scene_id_map[sid]
        xy_m = torch.from_numpy(sd.grid_coords).float().to(device)
        sdf_gt = torch.from_numpy(sd.sdf_flat).float().to(device)

        preds = []
        for i in range(0, len(xy_m), 8192):
            chunk = xy_m[i : i + 8192]
            pred = model.predict_sdf(scene_idx, chunk)
            preds.append(pred.squeeze(-1))
        sdf_pred = torch.cat(preds, dim=0)
        ious[sid] = compute_sdf_iou(sdf_pred, sdf_gt)

    return ious


def train_s12_frozen_decoder(
    cfg: Dict,
    base_state_dict: dict,
    base_config: dict,
    all_scenes: Dict,
    device: torch.device,
    epochs: int = 500,
    lr: float = 1e-3,
) -> Dict:
    """Train S12 codes with frozen decoder. Returns results dict."""
    logger.info("=" * 60)
    logger.info(
        "Config: %s (K=%d, alpha=%.0f, decoder=FROZEN)",
        cfg["name"], cfg["K"], cfg["alpha"],
    )
    logger.info("=" * 60)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Build model and load weights
    model, inv_scene_id_map = build_model_for_config(
        cfg, base_config, base_state_dict, device,
    )

    # Record pre-training IoU (with new alpha, before code optimization)
    pre_ious = evaluate_all_ious(model, all_scenes, inv_scene_id_map, device)
    pre_s12 = pre_ious.get(S12_SCENE_ID, 0.0)
    pre_others = float(np.mean([v for k, v in pre_ious.items() if k != S12_SCENE_ID]))
    logger.info(
        "Pre-training: S12 IoU=%.4f, Others=%.4f (with new alpha=%.0f)",
        pre_s12, pre_others, cfg["alpha"],
    )

    # ---- Freeze decoder completely ----
    for param in model.sdf_decoder.parameters():
        param.requires_grad = False

    # Only auto_decoder_codes is trainable
    model.auto_decoder_codes.weight.requires_grad = True

    # Determine S12 code range
    s12_idx = inv_scene_id_map[S12_SCENE_ID]
    s12_start, s12_end = model._scene_code_ranges[s12_idx]
    n_s12_codes = s12_end - s12_start

    # Gradient hook: zero out all non-S12 code gradients
    def _s12_code_grad_hook(grad: torch.Tensor) -> torch.Tensor:
        """Zero gradients for all codes except S12 range."""
        mask = torch.zeros_like(grad)
        mask[s12_start:s12_end] = 1.0
        return grad * mask

    hook_handle = model.auto_decoder_codes.weight.register_hook(_s12_code_grad_hook)

    # Count trainable params (should be only S12 codes)
    trainable = n_s12_codes * base_config["d_cond"]
    logger.info(
        "Trainable: %d parameters (S12: %d codes x %d dim, decoder: FROZEN)",
        trainable, n_s12_codes, base_config["d_cond"],
    )

    # Optimizer: only codes (decoder frozen, won't get gradients)
    optimizer = Adam([model.auto_decoder_codes.weight], lr=lr)

    # Training data: S12 scene
    sd = all_scenes[S12_SCENE_ID]
    n_sdf_batch: int = 4096

    t0 = time.time()
    best_s12_iou: float = pre_s12
    best_epoch: int = -1

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Sample SDF supervision points (boundary oversampled)
        bdy_mask = np.abs(sd.sdf_flat) < 0.1
        bdy_idx = np.where(bdy_mask)[0]
        far_idx = np.where(~bdy_mask)[0]
        n_bdy = min(int(n_sdf_batch * 0.75), len(bdy_idx))
        n_far = min(n_sdf_batch - n_bdy, len(far_idx))

        chosen = np.concatenate([
            np.random.choice(bdy_idx, n_bdy, replace=n_bdy > len(bdy_idx)),
            np.random.choice(far_idx, n_far, replace=n_far > len(far_idx)),
        ])
        sdf_idx = torch.from_numpy(chosen).long().to(device)

        xy_m = torch.from_numpy(sd.grid_coords).float().to(device)
        sdf_gt_all = torch.from_numpy(sd.sdf_flat).float().to(device)

        xy_batch = xy_m[sdf_idx].clone().requires_grad_(True)  # (B, 2)
        sdf_gt_batch = sdf_gt_all[sdf_idx].unsqueeze(-1)  # (B, 1)

        sdf_pred = model.predict_sdf(s12_idx, xy_batch)  # (B, 1)
        l_sdf = sdf_loss(sdf_pred, sdf_gt_batch)
        l_eik = eikonal_loss(sdf_pred, xy_batch)
        z = model.get_code(s12_idx)
        l_z = (z ** 2).mean()

        total = l_sdf + 0.1 * l_eik + 1e-3 * l_z

        if torch.isfinite(total):
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                [model.auto_decoder_codes.weight], max_norm=1.0,
            )
            optimizer.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            s12_iou = evaluate_all_ious(
                model, {S12_SCENE_ID: sd}, inv_scene_id_map, device,
            )[S12_SCENE_ID]
            if s12_iou > best_s12_iou:
                best_s12_iou = s12_iou
                best_epoch = epoch + 1
            logger.info(
                "  Ep %3d: loss=%.4e sdf=%.4e eik=%.4e S12_IoU=%.4f (best=%.4f@%d)",
                epoch + 1, total.item(), l_sdf.item(), l_eik.item(),
                s12_iou, best_s12_iou, best_epoch,
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    hook_handle.remove()

    # Final evaluation: ALL scenes
    final_ious = evaluate_all_ious(model, all_scenes, inv_scene_id_map, device)
    s12_iou = final_ious.get(S12_SCENE_ID, 0.0)
    others_ious = {k: v for k, v in final_ious.items() if k != S12_SCENE_ID}
    mean_others = float(np.mean(list(others_ious.values())))
    mean_all = float(np.mean(list(final_ious.values())))

    # Verify decoder frozen: other scenes should be IDENTICAL to pre-training
    others_drift = {
        k: abs(final_ious[k] - pre_ious[k]) for k in others_ious
    }
    max_drift = max(others_drift.values()) if others_drift else 0.0

    logger.info("-" * 60)
    logger.info("Config %s done in %.1fs", cfg["name"], elapsed)
    logger.info(
        "  S12: %.4f -> %.4f (delta=%.4f)",
        pre_s12, s12_iou, s12_iou - pre_s12,
    )
    logger.info("  Others: %.4f (max drift: %.6f)", mean_others, max_drift)
    logger.info("  Overall: %.4f", mean_all)

    return {
        "config": cfg["name"],
        "K": cfg["K"],
        "alpha": cfg["alpha"],
        "s12_iou_pre": pre_s12,
        "s12_iou_final": s12_iou,
        "mean_iou_others": mean_others,
        "mean_iou_all": mean_all,
        "max_others_drift": max_drift,
        "best_s12_iou": best_s12_iou,
        "best_epoch": best_epoch,
        "per_scene_iou": final_ious,
        "time_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment A2: S12 Frozen-Decoder Code-Only Optimization",
    )
    parser.add_argument("--checkpoint", type=str, default="best_phase3_v3")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--configs", nargs="+", type=str, default=None,
        help="Subset of config names to run (default: all)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load base checkpoint
    base_state_dict, base_config = load_base_checkpoint(device, args.checkpoint)

    # Load forward model metadata for scene_scales
    fwd_ckpt_name = base_config["forward_checkpoint"]
    fwd_ckpt_path = FWD_CHECKPOINT_DIR / f"{fwd_ckpt_name}.pt"
    fwd_ckpt = torch.load(fwd_ckpt_path, map_location=device, weights_only=False)
    scene_scales = fwd_ckpt["scene_scales"]
    fwd_cfg = fwd_ckpt["config"]
    fwd_scene_list = fwd_cfg.get("trained_scene_list", sorted(scene_scales.keys()))
    fwd_scene_id_map = {sid: idx for idx, sid in enumerate(fwd_scene_list)}

    # Load all scenes
    all_scenes = load_all_scenes(DATA_DIR, scene_scales, fwd_scene_id_map)
    logger.info("Loaded %d scenes", len(all_scenes))

    # Filter configs if specified
    configs_to_run = CONFIGS
    if args.configs:
        configs_to_run = [c for c in CONFIGS if c["name"] in args.configs]

    # Run all configs
    results: List[Dict] = []
    for cfg in configs_to_run:
        res = train_s12_frozen_decoder(
            cfg, base_state_dict, base_config, all_scenes,
            device, epochs=args.epochs, lr=args.lr,
        )
        results.append(res)

    # Write CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "s12_frozen_decoder.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "config", "K", "alpha",
            "s12_iou_pre", "s12_iou_final", "s12_delta",
            "mean_iou_others", "max_others_drift",
            "mean_iou_all", "best_s12_iou", "best_epoch", "time_s",
        ]
        writer.writerow(header)

        # Baseline row
        writer.writerow([
            "baseline (v3)", 2, 50.0,
            0.4928, 0.4928, 0.0,
            "0.9530", 0.0,
            0.9491, 0.4928, "---", "---",
        ])

        for res in results:
            writer.writerow([
                res["config"],
                res["K"],
                res["alpha"],
                f"{res['s12_iou_pre']:.4f}",
                f"{res['s12_iou_final']:.4f}",
                f"{res['s12_iou_final'] - res['s12_iou_pre']:.4f}",
                f"{res['mean_iou_others']:.4f}",
                f"{res['max_others_drift']:.6f}",
                f"{res['mean_iou_all']:.4f}",
                f"{res['best_s12_iou']:.4f}",
                res["best_epoch"],
                f"{res['time_s']:.1f}",
            ])

    logger.info("Results CSV: %s", csv_path)

    # Print summary table
    print("\n" + "=" * 80)
    print("Experiment A2: S12 Frozen-Decoder Code-Only Optimization")
    print("=" * 80)
    print(
        f"{'Config':<22} {'K':>2} {'alpha':>5} "
        f"{'S12 pre':>8} {'S12 post':>9} {'delta':>7} "
        f"{'Others':>7} {'Drift':>8} {'All':>7}"
    )
    print("-" * 80)
    print(
        f"{'baseline (v3)':<22} {'2':>2} {'50':>5} "
        f"{'0.4928':>8} {'0.4928':>9} {'0.0':>7} "
        f"{'0.9530':>7} {'0.0':>8} {'0.9491':>7}"
    )
    for res in results:
        delta = res["s12_iou_final"] - res["s12_iou_pre"]
        print(
            f"{res['config']:<22} {res['K']:>2} {res['alpha']:>5.0f} "
            f"{res['s12_iou_pre']:>8.4f} {res['s12_iou_final']:>9.4f} "
            f"{delta:>+7.4f} {res['mean_iou_others']:>7.4f} "
            f"{res['max_others_drift']:>8.6f} {res['mean_iou_all']:>7.4f}"
        )
    print("=" * 80)

    # Per-scene breakdown for best config
    if results:
        best = max(results, key=lambda r: r["s12_iou_final"])
        print(f"\nBest config: {best['config']} (S12 IoU={best['s12_iou_final']:.4f})")
        print(f"{'Scene':>6} {'IoU':>8}")
        print("-" * 16)
        for sid, iou in sorted(best["per_scene_iou"].items()):
            marker = " *" if sid == S12_SCENE_ID else ""
            print(f"S{sid:>4d} {iou:>8.4f}{marker}")


if __name__ == "__main__":
    main()
