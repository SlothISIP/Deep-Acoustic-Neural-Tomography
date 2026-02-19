"""Experiment A: S12 Multi-Body Architecture Sweep.

Tests whether increasing number of latent codes (K) or sharpening
smooth-min alpha improves S12 (dual parallel bars) IoU.

Configs
-------
    S12-K3:       K=3, alpha=50  (3 codes, default alpha)
    S12-K4:       K=4, alpha=50  (4 codes, default alpha)
    S12-alpha100: K=2, alpha=100 (original K, sharper smooth-min)

Method
------
    Load best_phase3_v3.pt -> Rebuild model with new K/alpha.
    Freeze decoder except codes: gradient hook on S12 codes only.
    Actually, freeze ALL except S12 codes + decoder (decoder must adapt
    to new composition). 200 epochs, LR=1e-4.

Output
------
    results/experiments/s12_sweep.csv

Usage
-----
    python scripts/run_experiment_s12.py
    python scripts/run_experiment_s12.py --epochs 200 --lr 1e-4
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
from torch.optim import AdamW

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
logger = logging.getLogger("experiment_s12")

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


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
CONFIGS: List[Dict] = [
    {"name": "S12-K3", "K": 3, "alpha": 50.0},
    {"name": "S12-K4", "K": 4, "alpha": 50.0},
    {"name": "S12-alpha100", "K": 2, "alpha": 100.0},
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
    device: torch.device,
) -> Tuple[InverseModel, Dict[int, int]]:
    """Build inverse model with specific K and alpha for S12.

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
    model = model.to(device)
    return model, inv_scene_id_map


@torch.no_grad()
def evaluate_iou(
    model: InverseModel,
    all_scenes: Dict,
    inv_scene_id_map: Dict[int, int],
    device: torch.device,
) -> Dict[int, float]:
    """Compute per-scene IoU."""
    model.eval()
    ious = {}

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

    model.train()
    return ious


def train_s12_config(
    cfg: Dict,
    base_state_dict: dict,
    base_config: dict,
    all_scenes: Dict,
    device: torch.device,
    epochs: int = 200,
    lr: float = 1e-4,
) -> Dict:
    """Train one S12 config, return results dict."""
    logger.info("=" * 60)
    logger.info("Config: %s (K=%d, alpha=%.0f)", cfg["name"], cfg["K"], cfg["alpha"])
    logger.info("=" * 60)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model, inv_scene_id_map = build_model_for_config(cfg, base_config, device)

    # Load base weights with compat remapping
    model.load_state_dict_compat(base_state_dict)

    # Determine S12 scene_idx
    s12_idx = inv_scene_id_map[S12_SCENE_ID]
    s12_start, s12_end = model._scene_code_ranges[s12_idx]

    # Freeze everything except S12 codes + decoder
    for name, param in model.named_parameters():
        if "sdf_decoder" in name:
            param.requires_grad = True
        elif "auto_decoder_codes" in name:
            # Only S12 codes get gradient
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Register gradient hook: zero out non-S12 code gradients
    def _s12_code_grad_hook(grad: torch.Tensor) -> torch.Tensor:
        """Zero gradients for all codes except S12 range."""
        mask = torch.zeros_like(grad)
        mask[s12_start:s12_end] = 1.0
        return grad * mask

    model.auto_decoder_codes.weight.register_hook(_s12_code_grad_hook)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %d", trainable)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    # Only need S12 scene data for training; others for IoU eval
    sd = all_scenes[S12_SCENE_ID]
    n_sdf_batch = 4096

    t0 = time.time()
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            ious = evaluate_iou(model, all_scenes, inv_scene_id_map, device)
            s12_iou = ious.get(S12_SCENE_ID, 0.0)
            mean_others = np.mean([v for k, v in ious.items() if k != S12_SCENE_ID])
            logger.info(
                "  Ep %3d: loss=%.4e sdf=%.4e eik=%.4e S12_IoU=%.4f others=%.4f",
                epoch + 1, total.item(), l_sdf.item(), l_eik.item(),
                s12_iou, mean_others,
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    logger.info("Config %s done in %.1fs", cfg["name"], elapsed)

    # Final evaluation
    final_ious = evaluate_iou(model, all_scenes, inv_scene_id_map, device)
    s12_iou = final_ious.get(S12_SCENE_ID, 0.0)
    mean_others = float(np.mean([v for k, v in final_ious.items() if k != S12_SCENE_ID]))
    mean_all = float(np.mean(list(final_ious.values())))

    return {
        "config": cfg["name"],
        "K": cfg["K"],
        "alpha": cfg["alpha"],
        "s12_iou": s12_iou,
        "mean_iou_others": mean_others,
        "mean_iou_all": mean_all,
        "per_scene_iou": final_ious,
        "time_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment A: S12 Multi-Body Sweep")
    parser.add_argument("--checkpoint", type=str, default="best_phase3_v3")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load base checkpoint
    base_state_dict, base_config = load_base_checkpoint(device, args.checkpoint)

    # Load forward model for scene_scales
    fwd_ckpt_name = base_config["forward_checkpoint"]
    fwd_ckpt_path = FWD_CHECKPOINT_DIR / f"{fwd_ckpt_name}.pt"
    fwd_ckpt = torch.load(fwd_ckpt_path, map_location=device, weights_only=False)
    scene_scales = fwd_ckpt["scene_scales"]
    fwd_cfg = fwd_ckpt["config"]
    fwd_scene_list = fwd_cfg.get("trained_scene_list", sorted(scene_scales.keys()))
    fwd_scene_id_map = {sid: idx for idx, sid in enumerate(fwd_scene_list)}

    # Load scene data
    all_scenes = load_all_scenes(DATA_DIR, scene_scales, fwd_scene_id_map)
    logger.info("Loaded %d scenes", len(all_scenes))

    # Run all configs
    results: List[Dict] = []
    for cfg in CONFIGS:
        res = train_s12_config(
            cfg, base_state_dict, base_config, all_scenes,
            device, epochs=args.epochs, lr=args.lr,
        )
        results.append(res)

    # Write CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "s12_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["config", "K", "alpha", "s12_iou", "mean_iou_others", "mean_iou_all", "time_s"]
        writer.writerow(header)

        # Baseline row from MEMORY
        writer.writerow(["baseline (v3)", 2, 50.0, 0.4928, "---", 0.9491, "---"])

        for res in results:
            writer.writerow([
                res["config"],
                res["K"],
                res["alpha"],
                f"{res['s12_iou']:.4f}",
                f"{res['mean_iou_others']:.4f}",
                f"{res['mean_iou_all']:.4f}",
                f"{res['time_s']:.1f}",
            ])

    logger.info("Results CSV: %s", csv_path)

    # Print summary
    print("\n" + "=" * 70)
    print("Experiment A: S12 Multi-Body Architecture Sweep")
    print("=" * 70)
    print(f"{'Config':<20} {'K':>3} {'alpha':>6} {'S12 IoU':>10} {'Others':>10} {'All':>10}")
    print("-" * 65)
    print(f"{'baseline (v3)':<20} {'2':>3} {'50':>6} {'0.4928':>10} {'---':>10} {'0.9491':>10}")
    for res in results:
        print(
            f"{res['config']:<20} {res['K']:>3} {res['alpha']:>6.0f} "
            f"{res['s12_iou']:>10.4f} {res['mean_iou_others']:>10.4f} "
            f"{res['mean_iou_all']:>10.4f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
