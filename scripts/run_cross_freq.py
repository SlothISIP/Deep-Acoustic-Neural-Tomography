#!/usr/bin/env python3
"""P3: Cross-Frequency Generalization Test.

Tests whether the forward model can generalize to unseen frequencies.
Two experiments:

    1. Extrapolation: Train on 2-6 kHz, test on 6-8 kHz
    2. Interpolation: Train on even-index frequencies, test on odd-index

Uses the Transfer Function formulation and same architecture as production.

Output: results/experiments/cross_freq_generalization.csv
        checkpoints/baseline/best_cross_freq_{split}.pt

Usage
-----
    python scripts/run_cross_freq.py --split extrapolation
    python scripts/run_cross_freq.py --split interpolation
    python scripts/run_cross_freq.py --split both  # run both sequentially
"""

import argparse
import csv
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from scipy.special import hankel1

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import Phase1Dataset, SPEED_OF_SOUND_M_PER_S
from src.forward_model import TransferFunctionModel, build_transfer_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cross_freq")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "phase1"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "baseline"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"


# ---------------------------------------------------------------------------
# Frequency-split dataset wrapper
# ---------------------------------------------------------------------------
class FreqSplitDataset:
    """Wrapper that filters Phase1Dataset samples by frequency range.

    Parameters
    ----------
    dataset : Phase1Dataset
        Full dataset (all frequencies).
    freq_mask : np.ndarray, shape (N,), bool
        True for samples to include.
    """

    def __init__(self, dataset: Phase1Dataset, freq_mask: np.ndarray) -> None:
        self.inputs = dataset.inputs[freq_mask]
        self.targets = dataset.targets[freq_mask]
        self.scene_ids = dataset.scene_ids[freq_mask]
        self.scales = dataset.scales[freq_mask]
        self.scene_scales = dataset.scene_scales

    def __len__(self) -> int:
        return self.inputs.shape[0]


def get_freq_masks(
    dataset: Phase1Dataset,
    split_mode: str,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Create train/test frequency masks.

    Parameters
    ----------
    dataset : Phase1Dataset
        Full dataset with all frequencies.
    split_mode : str
        'extrapolation': train 2-6 kHz, test 6-8 kHz
        'interpolation': train even-index freqs, test odd-index freqs

    Returns
    -------
    train_mask : (N,) bool
    test_mask : (N,) bool
    info : dict with frequency range details
    """
    # Feature index 4 = wavenumber k (rad/m)
    k_values = dataset.inputs[:, 4].numpy()  # (N,)

    if split_mode == "extrapolation":
        # 6 kHz boundary: k = 2*pi*6000/343 = 109.9 rad/m
        k_boundary = 2.0 * np.pi * 6000.0 / SPEED_OF_SOUND_M_PER_S
        train_mask = k_values < k_boundary
        test_mask = k_values >= k_boundary
        info = {
            "train_freq_range": "2-6 kHz",
            "test_freq_range": "6-8 kHz",
            "k_boundary": k_boundary,
        }
    elif split_mode == "interpolation":
        # Even/odd by quantizing k to nearest frequency index
        # Frequencies are spaced 30 Hz apart: 2000, 2030, ..., 8000
        # Wavenumber = 2*pi*f/c
        freq_hz = k_values * SPEED_OF_SOUND_M_PER_S / (2.0 * np.pi)
        freq_idx = np.round((freq_hz - 2000.0) / 30.0).astype(int)
        train_mask = (freq_idx % 2 == 0)  # even indices
        test_mask = (freq_idx % 2 == 1)   # odd indices
        info = {
            "train_freq_range": "even-index (100 freqs)",
            "test_freq_range": "odd-index (100 freqs)",
        }
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    info["n_train"] = int(train_mask.sum())
    info["n_test"] = int(test_mask.sum())
    info["train_frac"] = info["n_train"] / len(k_values)
    info["test_frac"] = info["n_test"] / len(k_values)

    return train_mask, test_mask, info


# ---------------------------------------------------------------------------
# GPU data from split
# ---------------------------------------------------------------------------
def prepare_split_gpu_data(
    dataset: Phase1Dataset,
    freq_mask: np.ndarray,
    val_fraction: float,
    seed: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Prepare GPU-resident train/val data from frequency-filtered samples."""
    inputs = dataset.inputs[freq_mask]
    targets = dataset.targets[freq_mask]
    scene_ids_raw = dataset.scene_ids[freq_mask]

    n_total = inputs.shape[0]
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=gen)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    trained_scene_list = sorted(dataset.scene_scales.keys())
    scene_id_map = {sid: idx for idx, sid in enumerate(trained_scene_list)}
    scene_ids_0idx = torch.tensor(
        [scene_id_map[int(sid)] for sid in scene_ids_raw],
        dtype=torch.long,
    )

    data = {
        "train_inputs": inputs[train_idx].to(device),
        "train_targets": targets[train_idx].to(device),
        "train_scene_ids": scene_ids_0idx[train_idx].to(device),
        "val_inputs": inputs[val_idx].to(device),
        "val_targets": targets[val_idx].to(device),
        "val_scene_ids": scene_ids_0idx[val_idx].to(device),
        "n_train": n_train,
        "n_val": n_val,
    }

    mem_mb = sum(
        v.nbytes for v in data.values() if isinstance(v, torch.Tensor)
    ) / 1e6
    logger.info("GPU data: train=%d, val=%d (%.1f MB)", n_train, n_val, mem_mb)
    return data


# ---------------------------------------------------------------------------
# Training / Validation (same as other baseline scripts)
# ---------------------------------------------------------------------------
def train_one_epoch(model, data, batch_size, optimizer):
    """Train one epoch."""
    model.train()
    inputs_all = data["train_inputs"]
    targets_all = data["train_targets"]
    sids_all = data["train_scene_ids"]
    n_train = data["n_train"]

    perm = torch.randperm(n_train, device=inputs_all.device)
    total_diff_sq = 0.0
    total_tgt_sq = 0.0
    total_loss = 0.0
    n_batches = 0

    for i in range(0, n_train - batch_size + 1, batch_size):
        idx = perm[i : i + batch_size]
        optimizer.zero_grad(set_to_none=True)
        pred = model(inputs_all[idx], scene_ids=sids_all[idx])
        loss = nn.functional.mse_loss(pred, targets_all[idx])

        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss: {loss.item():.4e}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_diff_sq += ((pred.detach() - targets_all[idx]) ** 2).sum().item()
        total_tgt_sq += (targets_all[idx] ** 2).sum().item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "var_explained": 1.0 - total_diff_sq / max(total_tgt_sq, 1e-30),
    }


@torch.no_grad()
def validate(model, data, batch_size):
    """Validate."""
    model.eval()
    inputs_all = data["val_inputs"]
    targets_all = data["val_targets"]
    sids_all = data["val_scene_ids"]
    n_val = data["n_val"]

    total_diff_sq = 0.0
    total_tgt_sq = 0.0
    total_loss = 0.0
    n_batches = 0

    for i in range(0, n_val, batch_size):
        end = min(i + batch_size, n_val)
        pred = model(inputs_all[i:end], scene_ids=sids_all[i:end])
        loss = nn.functional.mse_loss(pred, targets_all[i:end])
        total_loss += loss.item()
        total_diff_sq += ((pred - targets_all[i:end]) ** 2).sum().item()
        total_tgt_sq += (targets_all[i:end] ** 2).sum().item()
        n_batches += 1

    return {
        "val_loss": total_loss / max(n_batches, 1),
        "val_var_explained": 1.0 - total_diff_sq / max(total_tgt_sq, 1e-30),
    }


# ---------------------------------------------------------------------------
# Evaluation on test frequencies (per-scene L2 error)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_test_freqs(
    model: TransferFunctionModel,
    dataset: Phase1Dataset,
    test_mask: np.ndarray,
    device: torch.device,
) -> Tuple[float, List[Dict]]:
    """Evaluate model on held-out frequencies using BEM ground truth.

    Returns overall error and per-scene errors.
    """
    model.eval()
    from scipy.interpolate import RegularGridInterpolator

    # Get test frequency k-values
    test_k = dataset.inputs[test_mask, 4].numpy()
    unique_k = np.unique(test_k)

    trained_scene_list = sorted(dataset.scene_scales.keys())
    scene_id_map = {sid: idx for idx, sid in enumerate(trained_scene_list)}

    total_diff_sq = 0.0
    total_ref_sq = 0.0
    per_scene = []

    for sid in trained_scene_list:
        h5_path = DATA_DIR / f"scene_{sid:03d}.h5"
        if not h5_path.exists():
            continue

        scale = dataset.scene_scales[sid]
        sid_0idx = scene_id_map[sid]

        with h5py.File(h5_path, "r") as f:
            freqs_hz = f["frequencies"][:]
            k_arr = 2.0 * np.pi * freqs_hz / SPEED_OF_SOUND_M_PER_S

            # Filter to test frequencies only
            test_freq_mask = np.isin(k_arr, unique_k, )
            # Use approximate matching due to float precision
            test_freq_indices = []
            for fi, k in enumerate(k_arr):
                if np.any(np.abs(unique_k - k) < 1e-6):
                    test_freq_indices.append(fi)
            test_freq_indices = np.array(test_freq_indices)

            if len(test_freq_indices) == 0:
                continue

            src_pos = f["sources/positions"][:]
            rcv_pos = f["receivers/positions"][:]
            sdf_grid_x = f["sdf/grid_x"][:]
            sdf_grid_y = f["sdf/grid_y"][:]
            sdf_values = f["sdf/values"][:]

            n_rcv = rcv_pos.shape[0]

            sdf_interp = RegularGridInterpolator(
                (sdf_grid_x, sdf_grid_y), sdf_values,
                method="linear", bounds_error=False, fill_value=1.0,
            )
            sdf_at_rcv = sdf_interp(rcv_pos)

            scene_diff_sq = 0.0
            scene_ref_sq = 0.0

            for si in range(src_pos.shape[0]):
                p_total_bem = f[f"pressure/src_{si:03d}/field"][:]  # (F, R)
                xs_m, ys_m = src_pos[si]
                dx_sr = rcv_pos[:, 0] - xs_m
                dy_sr = rcv_pos[:, 1] - ys_m
                dist_sr = np.sqrt(dx_sr ** 2 + dy_sr ** 2)
                dist_sr_safe = np.maximum(dist_sr, 1e-15)

                for fi in test_freq_indices:
                    k = k_arr[fi]
                    n = n_rcv

                    inputs = np.column_stack([
                        np.full(n, xs_m),
                        np.full(n, ys_m),
                        rcv_pos[:, 0],
                        rcv_pos[:, 1],
                        np.full(n, k),
                        sdf_at_rcv,
                        dist_sr,
                        dx_sr,
                        dy_sr,
                    ]).astype(np.float32)

                    inputs_t = torch.from_numpy(inputs).to(device)
                    sid_t = torch.full(
                        (n,), sid_0idx, dtype=torch.long, device=device,
                    )
                    pred = model(inputs_t, scene_ids=sid_t).detach().cpu().numpy()

                    kr = k * dist_sr_safe
                    p_inc = -0.25j * hankel1(0, kr)

                    t_re = pred[:, 0] * scale
                    t_im = pred[:, 1] * scale
                    t_c = t_re + 1j * t_im
                    p_pred = p_inc * (1.0 + t_c)

                    p_bem_fi = p_total_bem[fi, :]

                    diff = p_pred - p_bem_fi
                    scene_diff_sq += np.sum(np.abs(diff) ** 2)
                    scene_ref_sq += np.sum(np.abs(p_bem_fi) ** 2)

            scene_error = np.sqrt(scene_diff_sq / max(scene_ref_sq, 1e-30))
            per_scene.append({"scene": sid, "error_pct": scene_error * 100})

            total_diff_sq += scene_diff_sq
            total_ref_sq += scene_ref_sq

    overall = np.sqrt(total_diff_sq / max(total_ref_sq, 1e-30))
    return overall, per_scene


# ---------------------------------------------------------------------------
# Run one split experiment
# ---------------------------------------------------------------------------
def run_experiment(
    split_mode: str,
    dataset: Phase1Dataset,
    args,
    device: torch.device,
) -> Dict:
    """Run one cross-frequency experiment (train + evaluate)."""
    logger.info("=" * 60)
    logger.info("Cross-Frequency: %s", split_mode)
    logger.info("=" * 60)

    train_mask, test_mask, info = get_freq_masks(dataset, split_mode)
    logger.info(
        "Train: %s (%d samples, %.0f%%)",
        info["train_freq_range"], info["n_train"], info["train_frac"] * 100,
    )
    logger.info(
        "Test: %s (%d samples, %.0f%%)",
        info["test_freq_range"], info["n_test"], info["test_frac"] * 100,
    )

    # Prepare GPU data (train split only)
    gpu_data = prepare_split_gpu_data(
        dataset, train_mask, args.val_fraction, args.seed, device,
    )

    # Model
    trained_scene_list = sorted(dataset.scene_scales.keys())
    n_scenes = len(trained_scene_list)
    input_mean, input_std = dataset.get_input_stats()

    model = build_transfer_model(
        d_hidden=args.d_hidden,
        n_blocks=args.n_blocks,
        n_fourier=args.n_fourier,
        fourier_sigma=args.fourier_sigma,
        n_scenes=n_scenes,
        scene_emb_dim=32,
        d_out=2,
    )
    model.set_normalization(input_mean, input_std)
    model = model.to(device)

    # Train
    tag = split_mode[:6]  # "extrap" or "interp"
    best_ckpt = CHECKPOINT_DIR / f"best_cross_freq_{tag}.pt"

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    lr_warmup = args.lr_warmup_epochs

    def lr_lambda(epoch):
        if epoch < lr_warmup:
            return (epoch + 1) / lr_warmup
        progress = (epoch - lr_warmup) / max(args.epochs - lr_warmup, 1)
        return max(1e-6 / args.lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")
    patience_counter = 0
    log_epochs = set(
        list(range(5)) + list(range(49, args.epochs, 50)) + [args.epochs - 1]
    )

    t0 = time.time()
    for epoch in range(args.epochs):
        train_m = train_one_epoch(model, gpu_data, args.batch_size, optimizer)
        val_m = validate(model, gpu_data, args.batch_size)
        scheduler.step()

        if epoch in log_epochs:
            logger.info(
                "Ep %4d: train=%.4e val=%.4e var=%.1f%% [%.0fs]",
                epoch + 1,
                train_m["loss"],
                val_m["val_loss"],
                val_m["val_var_explained"] * 100,
                time.time() - t0,
            )

        if val_m["val_loss"] < best_val_loss:
            best_val_loss = val_m["val_loss"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "scene_scales": dict(dataset.scene_scales),
                "config": {
                    "d_hidden": args.d_hidden,
                    "n_blocks": args.n_blocks,
                    "n_fourier": args.n_fourier,
                    "fourier_sigma": args.fourier_sigma,
                    "n_scenes": n_scenes,
                    "scene_emb_dim": 32,
                    "d_out": 2,
                    "trained_scene_list": trained_scene_list,
                    "split_mode": split_mode,
                },
            }, best_ckpt)
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    logger.info("Training: %.1f min", (time.time() - t0) / 60)

    # Load best and evaluate on TEST frequencies
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    overall_test, per_scene = evaluate_test_freqs(
        model, dataset, test_mask, device,
    )

    # Also evaluate on TRAIN frequencies for reference
    overall_train, _ = evaluate_test_freqs(
        model, dataset, train_mask, device,
    )

    logger.info("Train freq error: %.2f%%", overall_train * 100)
    logger.info("Test freq error:  %.2f%%", overall_test * 100)

    return {
        "split": split_mode,
        "train_error_pct": overall_train * 100,
        "test_error_pct": overall_test * 100,
        "per_scene": per_scene,
        "info": info,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="P3: Cross-Frequency Generalization"
    )
    parser.add_argument(
        "--split", type=str, default="both",
        choices=["extrapolation", "interpolation", "both"],
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lr-warmup-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-hidden", type=int, default=768)
    parser.add_argument("--n-blocks", type=int, default=8)
    parser.add_argument("--n-fourier", type=int, default=128)
    parser.add_argument("--fourier-sigma", type=float, default=30.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # Load full dataset
    logger.info("Loading full dataset...")
    dataset = Phase1Dataset(DATA_DIR, scene_ids=None, target_mode="cartesian")
    logger.info("Loaded %d samples", len(dataset))

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run experiment(s)
    splits = ["extrapolation", "interpolation"] if args.split == "both" else [args.split]
    all_results = []

    for split_mode in splits:
        result = run_experiment(split_mode, dataset, args, device)
        all_results.append(result)

    # Save CSV
    csv_path = RESULTS_DIR / "cross_freq_generalization.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "split", "train_freq_range", "test_freq_range",
            "train_error_pct", "test_error_pct",
        ])
        for r in all_results:
            writer.writerow([
                r["split"],
                r["info"]["train_freq_range"],
                r["info"]["test_freq_range"],
                f"{r['train_error_pct']:.2f}",
                f"{r['test_error_pct']:.2f}",
            ])

    # Per-scene detail CSV
    csv_detail = RESULTS_DIR / "cross_freq_per_scene.csv"
    with open(csv_detail, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["split", "scene", "test_error_pct"])
        for r in all_results:
            for ps in r["per_scene"]:
                writer.writerow([
                    r["split"], ps["scene"], f"{ps['error_pct']:.2f}",
                ])

    logger.info("Results: %s", csv_path)
    logger.info("Per-scene: %s", csv_detail)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("CROSS-FREQUENCY GENERALIZATION SUMMARY")
    logger.info("=" * 60)
    logger.info(
        "%-15s %15s %15s %10s",
        "Split", "Train Error%", "Test Error%", "Gap%",
    )
    logger.info("-" * 60)
    for r in all_results:
        gap = r["test_error_pct"] - r["train_error_pct"]
        logger.info(
            "%-15s %14.2f%% %14.2f%% %9.2f%%",
            r["split"], r["train_error_pct"], r["test_error_pct"], gap,
        )
    logger.info("=" * 60)
    logger.info("Reference: Full freq training (Config A) = 11.54%%")


if __name__ == "__main__":
    main()
