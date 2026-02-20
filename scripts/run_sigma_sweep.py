"""Experiment C: Fourier Feature sigma Sweep.

Trains forward models at sigma in {1, 5, 10} and measures:
    1. Forward reconstruction error (%)
    2. Helmholtz residual magnitude
    3. Generates accuracy-physics tradeoff curve

sigma=30 uses existing best_v7 checkpoint.

Usage
-----
    python scripts/run_sigma_sweep.py
    python scripts/run_sigma_sweep.py --epochs 150 --sigmas 1 5 10
"""

import argparse
import csv
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
import torch.nn as nn
from torch.optim import AdamW

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
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
logger = logging.getLogger("sigma_sweep")

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
# Data preparation (simplified from run_phase2.py)
# ---------------------------------------------------------------------------
def prepare_gpu_data(
    dataset: Phase1Dataset,
    val_fraction: float,
    seed: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Split and move all data to GPU."""
    n_total = len(dataset)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=gen)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    # Uniform weighting for fair comparison
    sample_weights = torch.ones(n_total)

    trained_scene_list = sorted(dataset.scene_scales.keys())
    scene_id_map = {sid: idx for idx, sid in enumerate(trained_scene_list)}
    scene_ids_0idx = torch.tensor(
        [scene_id_map[int(sid)] for sid in dataset.scene_ids],
        dtype=torch.long,
    )

    data = {
        "train_inputs": dataset.inputs[train_idx].to(device),
        "train_targets": dataset.targets[train_idx].to(device),
        "train_weights": sample_weights[train_idx].to(device),
        "train_scene_ids": scene_ids_0idx[train_idx].to(device),
        "val_inputs": dataset.inputs[val_idx].to(device),
        "val_targets": dataset.targets[val_idx].to(device),
        "val_weights": sample_weights[val_idx].to(device),
        "val_scene_ids": scene_ids_0idx[val_idx].to(device),
        "n_train": n_train,
        "n_val": n_val,
    }
    logger.info("GPU data: train=%d, val=%d", n_train, n_val)
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
    """Train for one epoch."""
    model.train()
    inputs_all = data["train_inputs"]
    targets_all = data["train_targets"]
    sids_all = data["train_scene_ids"]
    n_train = data["n_train"]

    perm = torch.randperm(n_train, device=inputs_all.device)
    total_loss = 0.0
    n_batches = 0

    for i in range(0, n_train - batch_size + 1, batch_size):
        idx = perm[i : i + batch_size]
        inputs = inputs_all[idx]
        targets = targets_all[idx]
        sids = sids_all[idx]

        optimizer.zero_grad(set_to_none=True)
        t_pred = model(inputs, scene_ids=sids)
        loss = nn.functional.mse_loss(t_pred, targets)

        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss: {loss.item():.4e}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def validate(
    model: TransferFunctionModel,
    data: Dict[str, torch.Tensor],
    batch_size: int,
) -> Dict[str, float]:
    """Validate."""
    model.eval()
    inputs_all = data["val_inputs"]
    targets_all = data["val_targets"]
    sids_all = data["val_scene_ids"]
    n_val = data["n_val"]

    total_loss = 0.0
    n_batches = 0

    for i in range(0, n_val, batch_size):
        inputs = inputs_all[i : i + batch_size]
        targets = targets_all[i : i + batch_size]
        sids = sids_all[i : i + batch_size]
        t_pred = model(inputs, scene_ids=sids)
        loss = nn.functional.mse_loss(t_pred, targets)
        total_loss += loss.item()
        n_batches += 1

    return {"val_loss": total_loss / max(n_batches, 1)}


# ---------------------------------------------------------------------------
# Helmholtz residual measurement
# ---------------------------------------------------------------------------
def measure_helmholtz_residual(
    model: TransferFunctionModel,
    scene_scales: Dict[int, float],
    scene_id_map: Dict[int, int],
    device: torch.device,
    n_samples: int = 2000,
    scenes: List[int] = None,
) -> Dict:
    """Measure Helmholtz residual for trained model."""
    if scenes is None:
        scenes = [1, 5, 8, 10, 14]

    all_residuals = []
    all_k_vals = []

    for sid in scenes:
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
            n_src = src_pos.shape[0]
            n_rcv = rcv_pos.shape[0]
            k_arr = 2.0 * np.pi * freqs_hz / SPEED_OF_SOUND_M_PER_S

            sdf_interp = RegularGridInterpolator(
                (sdf_grid_x, sdf_grid_y), sdf_values,
                method="linear", bounds_error=False, fill_value=1.0,
            )
            sdf_at_rcv = sdf_interp(rcv_pos)

            rng = np.random.RandomState(42 + sid)
            total_combos = n_src * n_freq * n_rcv
            n_draw = min(n_samples, total_combos)
            indices = rng.choice(total_combos, size=n_draw, replace=False)

            si_arr = indices // (n_freq * n_rcv)
            remainder = indices % (n_freq * n_rcv)
            fi_arr = remainder // n_rcv
            ri_arr = remainder % n_rcv

            x_src_np = src_pos[si_arr]
            x_rcv_np = rcv_pos[ri_arr]
            k_np = k_arr[fi_arr]
            sdf_np = sdf_at_rcv[ri_arr]

        # Neural Laplacian via 2nd-order autograd
        chunk_size = 256
        for ci in range(0, n_draw, chunk_size):
            ce = min(ci + chunk_size, n_draw)
            x_src_t = torch.tensor(x_src_np[ci:ce], dtype=torch.float32, device=device)
            x_rcv_t = torch.tensor(x_rcv_np[ci:ce], dtype=torch.float32, device=device)
            k_t = torch.tensor(k_np[ci:ce], dtype=torch.float32, device=device)
            sdf_t = torch.tensor(sdf_np[ci:ce], dtype=torch.float32, device=device)
            sid_t = torch.full((ce - ci,), sid_0idx, dtype=torch.long, device=device)

            x_eval = x_rcv_t.clone().requires_grad_(True)
            dx = x_eval[:, 0:1] - x_src_t[:, 0:1]
            dy = x_eval[:, 1:2] - x_src_t[:, 1:2]
            dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-30)
            inputs = torch.cat(
                [x_src_t, x_eval, k_t.unsqueeze(-1), sdf_t.unsqueeze(-1), dist, dx, dy],
                dim=-1,
            )

            t_pred = model(inputs, scene_ids=sid_t)
            t_re = t_pred[:, 0] * scale
            t_im = t_pred[:, 1] * scale

            r = dist.squeeze(-1)
            kr = k_t * r
            amp = 0.25 * torch.sqrt(2.0 / (math.pi * kr.clamp(min=1.0)))
            phase = kr - math.pi / 4.0
            p_inc_re = amp * torch.sin(phase)
            p_inc_im = -amp * torch.cos(phase)

            p_re = p_inc_re * (1.0 + t_re) - p_inc_im * t_im
            p_im = p_inc_im * (1.0 + t_re) + p_inc_re * t_im

            # 2nd-order Laplacian
            grad_p_re = torch.autograd.grad(
                p_re.sum(), x_eval, create_graph=True, retain_graph=True,
            )[0]
            d2_re_dx2 = torch.autograd.grad(
                grad_p_re[:, 0].sum(), x_eval, create_graph=False, retain_graph=True,
            )[0][:, 0]
            d2_re_dy2 = torch.autograd.grad(
                grad_p_re[:, 1].sum(), x_eval, create_graph=False, retain_graph=True,
            )[0][:, 1]
            lap_re = d2_re_dx2 + d2_re_dy2

            grad_p_im = torch.autograd.grad(
                p_im.sum(), x_eval, create_graph=True, retain_graph=True,
            )[0]
            d2_im_dx2 = torch.autograd.grad(
                grad_p_im[:, 0].sum(), x_eval, create_graph=False, retain_graph=True,
            )[0][:, 0]
            d2_im_dy2 = torch.autograd.grad(
                grad_p_im[:, 1].sum(), x_eval, create_graph=False, retain_graph=True,
            )[0][:, 1]
            lap_im = d2_im_dx2 + d2_im_dy2

            k_sq = k_t ** 2
            res_re = lap_re + k_sq * p_re
            res_im = lap_im + k_sq * p_im
            residual_mag = torch.sqrt(res_re ** 2 + res_im ** 2)

            all_residuals.append(residual_mag.detach().cpu().numpy())
            all_k_vals.append(k_np[ci:ce])

    residuals = np.concatenate(all_residuals)
    k_all = np.concatenate(all_k_vals)

    return {
        "median_residual": float(np.median(residuals)),
        "mean_residual": float(np.mean(residuals)),
        "p90_residual": float(np.percentile(residuals, 90)),
        "log10_median": float(np.log10(np.median(residuals))),
    }


# ---------------------------------------------------------------------------
# Forward error evaluation (simplified)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_forward_error(
    model: TransferFunctionModel,
    scene_scales: Dict[int, float],
    scene_id_map: Dict[int, int],
    device: torch.device,
    scenes: List[int] = None,
) -> float:
    """Evaluate overall forward reconstruction error."""
    if scenes is None:
        scenes = list(range(1, 16))

    total_diff_sq = 0.0
    total_ref_sq = 0.0

    for sid in scenes:
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
            n_src = src_pos.shape[0]
            n_rcv = rcv_pos.shape[0]
            k_arr = 2.0 * np.pi * freqs_hz / SPEED_OF_SOUND_M_PER_S

            sdf_interp = RegularGridInterpolator(
                (sdf_grid_x, sdf_grid_y), sdf_values,
                method="linear", bounds_error=False, fill_value=1.0,
            )
            sdf_at_rcv = sdf_interp(rcv_pos)

            for si in range(n_src):
                p_total_bem = f[f"pressure/src_{si:03d}/field"][:]  # (F, R)

                xs, ys = src_pos[si]
                dx_sr = rcv_pos[:, 0] - xs
                dy_sr = rcv_pos[:, 1] - ys
                dist_sr = np.sqrt(dx_sr ** 2 + dy_sr ** 2)
                dist_safe = np.maximum(dist_sr, 1e-15)

                # Process in chunks
                chunk_size = 50
                p_pred = np.zeros((n_freq, n_rcv), dtype=np.complex128)

                for fi in range(0, n_freq, chunk_size):
                    fi_end = min(fi + chunk_size, n_freq)
                    n_f = fi_end - fi
                    n = n_f * n_rcv

                    inputs = np.column_stack([
                        np.full(n, xs),
                        np.full(n, ys),
                        np.tile(rcv_pos[:, 0], n_f),
                        np.tile(rcv_pos[:, 1], n_f),
                        np.repeat(k_arr[fi:fi_end], n_rcv),
                        np.tile(sdf_at_rcv, n_f),
                        np.tile(dist_sr, n_f),
                        np.tile(dx_sr, n_f),
                        np.tile(dy_sr, n_f),
                    ]).astype(np.float32)

                    inputs_t = torch.from_numpy(inputs).to(device)
                    sid_t = torch.full((n,), sid_0idx, dtype=torch.long, device=device)
                    pred_raw = model(inputs_t, scene_ids=sid_t).cpu().numpy()

                    kr = k_arr[fi:fi_end, None] * dist_safe[None, :]
                    p_inc = -0.25j * hankel1(0, kr)

                    t_complex = (pred_raw[:, 0] + 1j * pred_raw[:, 1]).reshape(n_f, n_rcv) * scale
                    p_pred[fi:fi_end] = p_inc * (1.0 + t_complex)

                diff = p_pred - p_total_bem
                total_diff_sq += np.sum(np.abs(diff) ** 2)
                total_ref_sq += np.sum(np.abs(p_total_bem) ** 2)

    return float(np.sqrt(total_diff_sq / max(total_ref_sq, 1e-30)))


# ---------------------------------------------------------------------------
# Main: Train and evaluate at each sigma
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Sigma Sweep Experiment")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--sigmas", nargs="+", type=float, default=[1.0, 5.0, 10.0])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset once
    logger.info("Loading Phase 1 dataset...")
    dataset = Phase1Dataset(
        DATA_DIR, scene_ids=list(range(1, 16)),
        target_mode="cartesian",
    )
    data = prepare_gpu_data(dataset, val_fraction=0.2, seed=args.seed, device=device)
    scene_scales = dataset.scene_scales

    # Results storage
    all_results = []

    # First: evaluate existing sigma=30 model
    logger.info("=" * 60)
    logger.info("Evaluating existing sigma=30.0 model (best_v7)")
    logger.info("=" * 60)
    ckpt30 = torch.load(CHECKPOINT_DIR / "best_v7.pt", map_location=device, weights_only=False)
    cfg30 = ckpt30["config"]
    model30 = build_transfer_model(
        d_hidden=cfg30.get("d_hidden", 768),
        n_blocks=cfg30.get("n_blocks", 6),
        n_fourier=cfg30.get("n_fourier", 256),
        fourier_sigma=cfg30.get("fourier_sigma", 30.0),
        n_scenes=cfg30.get("n_scenes", 15),
        scene_emb_dim=cfg30.get("scene_emb_dim", 32),
        d_out=cfg30.get("d_out", 2),
    )
    model30.load_state_dict(ckpt30["model_state_dict"])
    model30 = model30.to(device).eval()
    tsl30 = cfg30.get("trained_scene_list", sorted(scene_scales.keys()))
    sid_map30 = {sid: idx for idx, sid in enumerate(tsl30)}

    fwd_err_30 = evaluate_forward_error(model30, scene_scales, sid_map30, device)
    helm_30 = measure_helmholtz_residual(model30, scene_scales, sid_map30, device, n_samples=1000)
    logger.info("sigma=30: fwd_err=%.2f%%, median_residual=%.2e", fwd_err_30 * 100, helm_30["median_residual"])
    all_results.append({
        "sigma": 30.0,
        "fwd_error_pct": fwd_err_30 * 100,
        "median_residual": helm_30["median_residual"],
        "mean_residual": helm_30["mean_residual"],
        "log10_median": helm_30["log10_median"],
        "epochs": cfg30.get("epochs", "pretrained"),
    })
    del model30
    torch.cuda.empty_cache()

    # Train new models at each sigma
    for sigma in args.sigmas:
        logger.info("=" * 60)
        logger.info("Training sigma=%.1f model (%d epochs)", sigma, args.epochs)
        logger.info("=" * 60)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # Use same architecture as best_v7 except sigma
        model = build_transfer_model(
            d_hidden=cfg30.get("d_hidden", 768),
            n_blocks=cfg30.get("n_blocks", 6),
            n_fourier=cfg30.get("n_fourier", 256),
            fourier_sigma=sigma,
            n_scenes=cfg30.get("n_scenes", 15),
            scene_emb_dim=cfg30.get("scene_emb_dim", 32),
            d_out=cfg30.get("d_out", 2),
        )
        model = model.to(device)

        # Set input normalization from data
        input_mean, input_std = dataset.get_input_stats()
        model.set_normalization(input_mean, input_std)

        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6,
        )

        trained_scene_list = sorted(scene_scales.keys())
        scene_id_map = {sid: idx for idx, sid in enumerate(trained_scene_list)}

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        patience = 50

        t0 = time.time()
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(model, data, args.batch_size, optimizer)
            val_metrics = validate(model, data, args.batch_size)
            scheduler.step()

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 50 == 0 or epoch == 1:
                logger.info(
                    "  Epoch %d: train_loss=%.4e, val_loss=%.4e, best=%.4e",
                    epoch, train_metrics["loss"], val_metrics["val_loss"], best_val_loss,
                )

            if patience_counter >= patience:
                logger.info("  Early stopping at epoch %d", epoch)
                break

        train_time = time.time() - t0
        final_epoch = epoch

        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)
        model = model.to(device).eval()

        # Save checkpoint
        ckpt_path = CHECKPOINT_DIR / f"sigma_sweep_{sigma:.0f}.pt"
        torch.save({
            "epoch": final_epoch,
            "model_state_dict": model.state_dict(),
            "best_val_loss": best_val_loss,
            "scene_scales": dict(scene_scales),
            "config": {
                "d_hidden": cfg30.get("d_hidden", 768),
                "n_blocks": cfg30.get("n_blocks", 6),
                "n_fourier": cfg30.get("n_fourier", 256),
                "fourier_sigma": sigma,
                "n_scenes": cfg30.get("n_scenes", 15),
                "scene_emb_dim": cfg30.get("scene_emb_dim", 32),
                "d_out": cfg30.get("d_out", 2),
                "trained_scene_list": trained_scene_list,
            },
        }, ckpt_path)
        logger.info("  Saved: %s", ckpt_path.name)

        # Evaluate
        fwd_err = evaluate_forward_error(model, scene_scales, scene_id_map, device)
        helm = measure_helmholtz_residual(model, scene_scales, scene_id_map, device, n_samples=1000)

        logger.info(
            "  sigma=%.1f: fwd_err=%.2f%%, median_residual=%.2e, time=%.1fs, epochs=%d",
            sigma, fwd_err * 100, helm["median_residual"], train_time, final_epoch,
        )

        all_results.append({
            "sigma": sigma,
            "fwd_error_pct": fwd_err * 100,
            "median_residual": helm["median_residual"],
            "mean_residual": helm["mean_residual"],
            "log10_median": helm["log10_median"],
            "epochs": final_epoch,
            "train_time_s": train_time,
        })

        del model, best_state
        torch.cuda.empty_cache()

    # Sort by sigma
    all_results.sort(key=lambda r: r["sigma"])

    # Save CSV
    csv_path = RESULTS_DIR / "sigma_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sigma", "fwd_error_pct", "median_residual", "mean_residual",
            "log10_median", "epochs",
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})
    logger.info("Saved: %s", csv_path)

    # Plot: Accuracy-Physics tradeoff
    fig, ax1 = plt.subplots(figsize=(8, 5))

    sigmas = [r["sigma"] for r in all_results]
    fwd_errors = [r["fwd_error_pct"] for r in all_results]
    med_residuals = [r["median_residual"] for r in all_results]

    color1 = "steelblue"
    ax1.set_xlabel(r"Fourier feature $\sigma$ [m$^{-1}$]", fontsize=12)
    ax1.set_ylabel("Forward error [%]", color=color1, fontsize=12)
    line1, = ax1.plot(sigmas, fwd_errors, "o-", color=color1, lw=2, ms=8, label="Forward error")
    ax1.tick_params(axis="y", labelcolor=color1)

    color2 = "coral"
    ax2 = ax1.twinx()
    ax2.set_ylabel(r"Median Helmholtz residual $|\nabla^2 p + k^2 p|$", color=color2, fontsize=12)
    line2, = ax2.plot(sigmas, med_residuals, "s--", color=color2, lw=2, ms=8, label="Helmholtz residual")
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor=color2)

    # Combined legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=10)

    ax1.set_title(r"Accuracy--Physics Tradeoff: Forward Error vs Helmholtz Residual", fontsize=13)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.savefig(RESULTS_DIR / "sigma_sweep_tradeoff.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "sigma_sweep_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: sigma_sweep_tradeoff.pdf")

    # Final summary
    print("\n" + "=" * 60)
    print("SIGMA SWEEP RESULTS")
    print("=" * 60)
    print(f"{'sigma':>8} {'Error%':>10} {'Med Residual':>15} {'log10(res)':>12} {'Epochs':>8}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['sigma']:>8.0f} {r['fwd_error_pct']:>9.2f}% {r['median_residual']:>15.2e} "
              f"{r['log10_median']:>12.2f} {r['epochs']:>8}")
    print("=" * 60)


if __name__ == "__main__":
    main()
