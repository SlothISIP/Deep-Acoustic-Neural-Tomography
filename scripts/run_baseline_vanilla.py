#!/usr/bin/env python3
"""P0 + P2a: Vanilla MLP Baseline & No-Scatterer Trivial Baseline.

Trains a forward model on raw p_scat (scattered pressure) instead of
the transfer function T = p_scat / p_inc, using the SAME architecture
and hyperparameters. Demonstrates the necessity of the Transfer Function
formulation (core contribution of the paper).

Also computes the trivial no-scatterer baseline (p_total = p_inc, T=0).

Reconstruction formulae
-----------------------
    Transfer fn:   p_total = p_inc * (1 + T_pred * scale)
    Vanilla (P0):  p_total = p_inc + p_scat_pred * scale
    No-scatter:    p_total = p_inc   (P2a trivial baseline)

Output
------
    results/experiments/baseline_comparison.csv
    checkpoints/baseline/best_vanilla.pt

Hardware optimization
---------------------
    - All data pre-loaded to GPU (zero DataLoader overhead)
    - batch_size=32768 for maximum GPU utilization (RTX 2080S 8GB)
    - cudnn.benchmark enabled
    - Early stopping with patience=100

Usage
-----
    python scripts/run_baseline_vanilla.py
    python scripts/run_baseline_vanilla.py --epochs 500 --d-hidden 768 --n-blocks 8
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

from src.dataset import Phase1Dataset
from src.forward_model import TransferFunctionModel, build_transfer_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("baseline_vanilla")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "phase1"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "baseline"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPEED_OF_SOUND_M_PER_S: float = 343.0


# ---------------------------------------------------------------------------
# GPU-resident data preparation
# ---------------------------------------------------------------------------
def prepare_gpu_data(
    dataset: Phase1Dataset,
    val_fraction: float,
    seed: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Split and move all data to GPU for zero-overhead batching.

    Parameters
    ----------
    dataset : Phase1Dataset
        Dataset with pressure or transfer function targets.
    val_fraction : float
        Fraction held out for validation.
    seed : int
        Random seed for reproducible split.
    device : torch.device
        Target device (cuda or cpu).

    Returns
    -------
    dict with train/val tensors on device.
    """
    n_total = len(dataset)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=gen)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    # Scene IDs: 0-indexed
    trained_scene_list = sorted(dataset.scene_scales.keys())
    scene_id_map = {sid: idx for idx, sid in enumerate(trained_scene_list)}
    scene_ids_0idx = torch.tensor(
        [scene_id_map[int(sid)] for sid in dataset.scene_ids],
        dtype=torch.long,
    )  # (N,)

    data = {
        "train_inputs": dataset.inputs[train_idx].to(device),
        "train_targets": dataset.targets[train_idx].to(device),
        "train_scene_ids": scene_ids_0idx[train_idx].to(device),
        "val_inputs": dataset.inputs[val_idx].to(device),
        "val_targets": dataset.targets[val_idx].to(device),
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
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: TransferFunctionModel,
    data: Dict[str, torch.Tensor],
    batch_size: int,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    """Train for one epoch with manual GPU batching.

    Returns dict with 'loss', 'var_explained'.
    """
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
        inputs = inputs_all[idx]  # (B, 9)
        targets = targets_all[idx]  # (B, 2)
        sids = sids_all[idx]  # (B,)

        optimizer.zero_grad(set_to_none=True)
        pred = model(inputs, scene_ids=sids)  # (B, 2)
        loss = nn.functional.mse_loss(pred, targets)

        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss: {loss.item():.4e}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_diff_sq += ((pred.detach() - targets) ** 2).sum().item()
        total_tgt_sq += (targets ** 2).sum().item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    var_exp = 1.0 - total_diff_sq / max(total_tgt_sq, 1e-30)
    return {"loss": avg_loss, "var_explained": var_exp}


@torch.no_grad()
def validate(
    model: TransferFunctionModel,
    data: Dict[str, torch.Tensor],
    batch_size: int,
) -> Dict[str, float]:
    """Validate with manual GPU batching."""
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
        inputs = inputs_all[i : i + batch_size]
        targets = targets_all[i : i + batch_size]
        sids = sids_all[i : i + batch_size]

        pred = model(inputs, scene_ids=sids)
        loss = nn.functional.mse_loss(pred, targets)

        total_loss += loss.item()
        total_diff_sq += ((pred - targets) ** 2).sum().item()
        total_tgt_sq += (targets ** 2).sum().item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    var_exp = 1.0 - total_diff_sq / max(total_tgt_sq, 1e-30)
    return {"val_loss": avg_loss, "val_var_explained": var_exp}


# ---------------------------------------------------------------------------
# Evaluation: per-scene reconstruction error
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_scene_pressure(
    model: TransferFunctionModel,
    h5_path: Path,
    scene_scale: float,
    scene_id_0idx: int,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate vanilla baseline (p_total = p_inc + p_scat_pred) for one scene.

    Also computes no-scatterer baseline (p_total = p_inc).

    Returns
    -------
    dict with 'vanilla_error', 'no_scatter_error', per-source errors.
    """
    model.eval()

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

        from scipy.interpolate import RegularGridInterpolator
        sdf_interp = RegularGridInterpolator(
            (sdf_grid_x, sdf_grid_y),
            sdf_values,
            method="linear",
            bounds_error=False,
            fill_value=1.0,
        )
        sdf_at_rcv = sdf_interp(rcv_pos)  # (R,)

        # Accumulators for energy-weighted errors
        vanilla_diff_sq = 0.0
        vanilla_ref_sq = 0.0
        no_scatter_diff_sq = 0.0
        no_scatter_ref_sq = 0.0
        vanilla_per_src = []

        for si in range(n_src):
            p_total_bem = f[f"pressure/src_{si:03d}/field"][:]  # (F, R) complex128

            xs_m, ys_m = src_pos[si]
            dx_sr = rcv_pos[:, 0] - xs_m  # (R,)
            dy_sr = rcv_pos[:, 1] - ys_m  # (R,)
            dist_sr = np.sqrt(dx_sr ** 2 + dy_sr ** 2)  # (R,)
            dist_sr_safe = np.maximum(dist_sr, 1e-15)

            p_total_vanilla = np.zeros((n_freq, n_rcv), dtype=np.complex128)
            p_inc_all = np.zeros((n_freq, n_rcv), dtype=np.complex128)

            chunk_size = 50
            for fi_start in range(0, n_freq, chunk_size):
                fi_end = min(fi_start + chunk_size, n_freq)
                n_f = fi_end - fi_start
                n = n_f * n_rcv

                # Build input features: (n_f*R, 9)
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
                    (n,), scene_id_0idx, dtype=torch.long, device=device
                )
                pred_raw = model(inputs_t, scene_ids=sid_t).cpu().numpy()  # (n, 2)

                # Incident field
                kr = k_arr[fi_start:fi_end, None] * dist_sr_safe[None, :]  # (n_f, R)
                p_inc = -0.25j * hankel1(0, kr)  # (n_f, R)
                p_inc_all[fi_start:fi_end] = p_inc

                # Reconstruct: p_total = p_inc + p_scat_pred * scale
                p_scat_re = pred_raw[:, 0] * scene_scale  # (n,) denormalized
                p_scat_im = pred_raw[:, 1] * scene_scale  # (n,)
                p_scat_complex = (p_scat_re + 1j * p_scat_im).reshape(n_f, n_rcv)
                p_total_vanilla[fi_start:fi_end] = p_inc + p_scat_complex

            # Vanilla baseline errors
            diff_v = p_total_vanilla - p_total_bem
            diff_sq_v = np.abs(diff_v) ** 2
            ref_sq = np.abs(p_total_bem) ** 2

            src_err_v = np.sqrt(np.sum(diff_sq_v)) / max(np.sqrt(np.sum(ref_sq)), 1e-30)
            vanilla_per_src.append(src_err_v)
            vanilla_diff_sq += np.sum(diff_sq_v)
            vanilla_ref_sq += np.sum(ref_sq)

            # No-scatterer baseline: p_total = p_inc
            diff_ns = p_inc_all - p_total_bem
            no_scatter_diff_sq += np.sum(np.abs(diff_ns) ** 2)
            no_scatter_ref_sq += np.sum(ref_sq)

    vanilla_error = np.sqrt(vanilla_diff_sq) / max(np.sqrt(vanilla_ref_sq), 1e-30)
    no_scatter_error = np.sqrt(no_scatter_diff_sq) / max(np.sqrt(no_scatter_ref_sq), 1e-30)

    return {
        "vanilla_error": vanilla_error,
        "no_scatter_error": no_scatter_error,
        "vanilla_per_src": vanilla_per_src,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="P0: Vanilla MLP Baseline + P2a: No-Scatterer Baseline"
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
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training, load checkpoint and evaluate only",
    )
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # GPU optimization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info(
            "GPU: %s (%.1f GB VRAM), cudnn.benchmark=True",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    # ---------------------------------------------------------------
    # Data: load with pressure targets (raw p_scat)
    # ---------------------------------------------------------------
    logger.info("Loading data with target_mode='pressure' ...")
    t_load = time.time()
    dataset = Phase1Dataset(
        DATA_DIR,
        scene_ids=None,  # all 15 scenes
        target_mode="pressure",
    )
    input_mean, input_std = dataset.get_input_stats()
    logger.info(
        "Loaded %d samples in %.1fs (pressure targets)",
        len(dataset), time.time() - t_load,
    )

    trained_scene_list = sorted(dataset.scene_scales.keys())
    scene_id_map = {sid: idx for idx, sid in enumerate(trained_scene_list)}
    n_scenes = len(trained_scene_list)

    # ---------------------------------------------------------------
    # Model: same architecture as production forward model
    # ---------------------------------------------------------------
    model = build_transfer_model(
        d_hidden=args.d_hidden,
        n_blocks=args.n_blocks,
        n_fourier=args.n_fourier,
        fourier_sigma=args.fourier_sigma,
        dropout=0.0,
        n_scenes=n_scenes,
        scene_emb_dim=32,
        d_out=2,
    )
    model.set_normalization(input_mean, input_std)
    model = model.to(device)
    n_params = model.count_parameters()
    logger.info("Model: %d params (%.2f MB)", n_params, n_params * 4 / 1e6)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_ckpt = CHECKPOINT_DIR / "best_vanilla.pt"

    if not args.eval_only:
        # ---------------------------------------------------------------
        # GPU data
        # ---------------------------------------------------------------
        gpu_data = prepare_gpu_data(
            dataset, args.val_fraction, args.seed, device,
        )

        # ---------------------------------------------------------------
        # Optimizer & scheduler
        # ---------------------------------------------------------------
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )
        lr_warmup = args.lr_warmup_epochs

        def lr_lambda(epoch: int) -> float:
            if epoch < lr_warmup:
                return (epoch + 1) / lr_warmup
            progress = (epoch - lr_warmup) / max(args.epochs - lr_warmup, 1)
            return max(1e-6 / args.lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # ---------------------------------------------------------------
        # Training
        # ---------------------------------------------------------------
        best_val_loss = float("inf")
        patience_counter = 0
        log_epochs = set(
            list(range(5))
            + list(range(49, args.epochs, 50))
            + [args.epochs - 1]
        )

        logger.info("Starting vanilla baseline training (%d epochs)...", args.epochs)
        t0_total = time.time()

        for epoch in range(args.epochs):
            t0 = time.time()
            train_m = train_one_epoch(model, gpu_data, args.batch_size, optimizer)
            val_m = validate(model, gpu_data, args.batch_size)
            scheduler.step()

            lr_now = optimizer.param_groups[0]["lr"]

            if epoch in log_epochs:
                logger.info(
                    "Ep %4d: train=%.4e val=%.4e var=%.1f%% lr=%.2e [%.0fs]",
                    epoch + 1,
                    train_m["loss"],
                    val_m["val_loss"],
                    val_m["val_var_explained"] * 100,
                    lr_now,
                    time.time() - t0_total,
                )

            # Best model
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
                        "target_mode": "pressure",
                        "trained_scene_list": trained_scene_list,
                    },
                }, best_ckpt)
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d)",
                    epoch + 1, args.patience,
                )
                break

        total_time = time.time() - t0_total
        logger.info(
            "Training done in %.1f min. Best val loss: %.4e",
            total_time / 60, best_val_loss,
        )

    # ---------------------------------------------------------------
    # Load best checkpoint for evaluation
    # ---------------------------------------------------------------
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            "Loaded best checkpoint (epoch %d, val_loss=%.4e)",
            ckpt["epoch"], ckpt["best_val_loss"],
        )
        scene_scales = ckpt["scene_scales"]
    else:
        logger.error("No checkpoint found at %s", best_ckpt)
        sys.exit(1)

    model.eval()

    # ---------------------------------------------------------------
    # Evaluate: Vanilla (P0) + No-Scatterer (P2a)
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Evaluating baselines: P0 (Vanilla MLP) + P2a (No-Scatterer)")
    logger.info("=" * 60)

    scene_ids = list(range(1, 16))
    results_rows: List[Dict] = []

    total_vanilla_diff = 0.0
    total_vanilla_ref = 0.0
    total_noscatter_diff = 0.0
    total_noscatter_ref = 0.0

    for sid in scene_ids:
        h5_path = DATA_DIR / f"scene_{sid:03d}.h5"
        if not h5_path.exists():
            logger.warning("Scene %d not found, skipping", sid)
            continue

        scale = scene_scales.get(sid)
        sid_0idx = scene_id_map.get(sid)
        if scale is None or sid_0idx is None:
            logger.warning("Scene %d not in checkpoint, skipping", sid)
            continue

        res = evaluate_scene_pressure(model, h5_path, scale, sid_0idx, device)

        logger.info(
            "  S%2d: vanilla=%.2f%%, no_scatter=%.2f%%",
            sid,
            res["vanilla_error"] * 100,
            res["no_scatter_error"] * 100,
        )

        results_rows.append({
            "scene": sid,
            "vanilla_error_pct": res["vanilla_error"] * 100,
            "no_scatter_error_pct": res["no_scatter_error"] * 100,
        })

        # Accumulate for overall (need to recompute from per-scene data)
        # For overall, we evaluate in evaluate_scene_pressure already
        # but we don't have the raw sq values here, so re-evaluate...

    # ---------------------------------------------------------------
    # Overall errors (energy-weighted across all scenes)
    # ---------------------------------------------------------------
    # Re-evaluate with accumulation for proper overall metric
    overall_v_diff_sq = 0.0
    overall_v_ref_sq = 0.0
    overall_ns_diff_sq = 0.0
    overall_ns_ref_sq = 0.0

    for sid in scene_ids:
        h5_path = DATA_DIR / f"scene_{sid:03d}.h5"
        if not h5_path.exists():
            continue
        scale = scene_scales.get(sid)
        sid_0idx = scene_id_map.get(sid)
        if scale is None or sid_0idx is None:
            continue

        with h5py.File(h5_path, "r") as f:
            freqs_hz = f["frequencies"][:]
            src_pos = f["sources/positions"][:]
            rcv_pos = f["receivers/positions"][:]
            sdf_grid_x = f["sdf/grid_x"][:]
            sdf_grid_y = f["sdf/grid_y"][:]
            sdf_values = f["sdf/values"][:]

            n_freq = len(freqs_hz)
            n_rcv = rcv_pos.shape[0]
            k_arr = 2.0 * np.pi * freqs_hz / SPEED_OF_SOUND_M_PER_S

            from scipy.interpolate import RegularGridInterpolator
            sdf_interp = RegularGridInterpolator(
                (sdf_grid_x, sdf_grid_y), sdf_values,
                method="linear", bounds_error=False, fill_value=1.0,
            )
            sdf_at_rcv = sdf_interp(rcv_pos)

            for si in range(src_pos.shape[0]):
                p_total_bem = f[f"pressure/src_{si:03d}/field"][:]
                xs_m, ys_m = src_pos[si]
                dx_sr = rcv_pos[:, 0] - xs_m
                dy_sr = rcv_pos[:, 1] - ys_m
                dist_sr = np.sqrt(dx_sr ** 2 + dy_sr ** 2)
                dist_sr_safe = np.maximum(dist_sr, 1e-15)

                ref_sq = np.sum(np.abs(p_total_bem) ** 2)
                overall_v_ref_sq += ref_sq
                overall_ns_ref_sq += ref_sq

                p_total_vanilla = np.zeros((n_freq, n_rcv), dtype=np.complex128)

                chunk_size = 50
                for fi_s in range(0, n_freq, chunk_size):
                    fi_e = min(fi_s + chunk_size, n_freq)
                    n_f = fi_e - fi_s
                    n = n_f * n_rcv

                    inputs = np.column_stack([
                        np.full(n, xs_m),
                        np.full(n, ys_m),
                        np.tile(rcv_pos[:, 0], n_f),
                        np.tile(rcv_pos[:, 1], n_f),
                        np.repeat(k_arr[fi_s:fi_e], n_rcv),
                        np.tile(sdf_at_rcv, n_f),
                        np.tile(dist_sr, n_f),
                        np.tile(dx_sr, n_f),
                        np.tile(dy_sr, n_f),
                    ]).astype(np.float32)

                    inputs_t = torch.from_numpy(inputs).to(device)
                    sid_t = torch.full(
                        (n,), sid_0idx, dtype=torch.long, device=device,
                    )
                    pred_raw = model(inputs_t, scene_ids=sid_t).detach().cpu().numpy()

                    kr = k_arr[fi_s:fi_e, None] * dist_sr_safe[None, :]
                    p_inc = -0.25j * hankel1(0, kr)

                    p_scat_re = pred_raw[:, 0] * scale
                    p_scat_im = pred_raw[:, 1] * scale
                    p_scat_c = (p_scat_re + 1j * p_scat_im).reshape(n_f, n_rcv)
                    p_total_vanilla[fi_s:fi_e] = p_inc + p_scat_c

                    # No-scatterer
                    diff_ns = p_inc - p_total_bem[fi_s:fi_e]
                    overall_ns_diff_sq += np.sum(np.abs(diff_ns) ** 2)

                diff_v = p_total_vanilla - p_total_bem
                overall_v_diff_sq += np.sum(np.abs(diff_v) ** 2)

    overall_vanilla = np.sqrt(overall_v_diff_sq / max(overall_v_ref_sq, 1e-30))
    overall_noscatter = np.sqrt(overall_ns_diff_sq / max(overall_ns_ref_sq, 1e-30))

    # ---------------------------------------------------------------
    # Print summary table
    # ---------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("BASELINE COMPARISON RESULTS")
    logger.info("=" * 60)
    logger.info(
        "%8s %15s %15s",
        "Scene", "Vanilla(P0)%", "NoScatter(P2a)%",
    )
    logger.info("-" * 40)
    for row in results_rows:
        logger.info(
            "%8d %14.2f%% %14.2f%%",
            row["scene"],
            row["vanilla_error_pct"],
            row["no_scatter_error_pct"],
        )
    logger.info("-" * 40)
    logger.info(
        "%8s %14.2f%% %14.2f%%",
        "Overall", overall_vanilla * 100, overall_noscatter * 100,
    )
    logger.info("=" * 60)
    logger.info(
        "Reference: Transfer Function (Config A, single model) = 11.54%%"
    )
    logger.info("=" * 60)

    # ---------------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "baseline_comparison.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "scene", "vanilla_error_pct", "no_scatter_error_pct",
        ])
        writer.writeheader()
        for row in results_rows:
            writer.writerow(row)
        # Overall row
        writer.writerow({
            "scene": "overall",
            "vanilla_error_pct": f"{overall_vanilla * 100:.2f}",
            "no_scatter_error_pct": f"{overall_noscatter * 100:.2f}",
        })
    logger.info("Results saved: %s", csv_path)


if __name__ == "__main__":
    main()
