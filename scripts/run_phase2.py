"""Phase 2 Training: Transfer Function Surrogate.

Trains a Fourier-feature + Residual MLP to predict the transfer function
T = p_scat / p_inc from Phase 1 BEM data.

Architecture
------------
    T_pred = FourierMLP(x_s, y_s, x_r, y_r, k, sdf, dist, dx, dy)
    p_total = p_inc * (1 + T_pred * scene_scale)

Training strategy
-----------------
    1. Data loss: MSE on normalized (Re(T), Im(T))
    2. Variance explained metric: 1 - MSE / Var(target)
    3. AdamW optimizer with linear warmup + cosine decay
    4. FP32 throughout
    5. All data pre-loaded to GPU for zero DataLoader overhead

Gate criterion
--------------
    BEM reconstruction error < 5% (evaluated by scripts/eval_phase2.py)

Usage
-----
    python scripts/run_phase2.py                     # default settings
    python scripts/run_phase2.py --epochs 2000        # more epochs
    python scripts/run_phase2.py --resume              # resume from checkpoint
    python scripts/run_phase2.py --scenes 1 2 3        # subset of scenes
"""

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

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
logger = logging.getLogger("phase2_train")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "phase1"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase2"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase2"

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
DEFAULTS = {
    "epochs": 1000,
    "batch_size": 8192,
    "lr": 5e-4,
    "weight_decay": 1e-5,
    "lr_warmup_epochs": 10,
    "val_fraction": 0.2,
    "seed": 42,
    "checkpoint_every": 50,
    "patience": 100,
    # Architecture
    "fourier_sigma": 30.0,
    "n_fourier": 128,
    "d_hidden": 512,
    "n_blocks": 5,
}


# ---------------------------------------------------------------------------
# GPU-resident data splits
# ---------------------------------------------------------------------------
def compute_importance_weights(
    dataset: Phase1Dataset,
    cap_percentile: float = 99.9,
) -> torch.Tensor:
    """Compute per-sample importance weights |R|^2 = |1 + T|^2.

    Aligns training loss with gate metric (energy-weighted L2 on pressure).
    High-|R| samples (e.g. S13 rcv49 with |T|=35) dominate the gate metric
    and receive proportionally more weight in the loss.

    Parameters
    ----------
    dataset : Phase1Dataset
        Dataset with targets and scales.
    cap_percentile : float
        Cap weights at this percentile to prevent extreme gradients.

    Returns
    -------
    weights : torch.Tensor, shape (N,)
        Per-sample weights normalized so mean = 1.0.
    """
    targets_np = dataset.targets.numpy()  # (N, 2 or 3)
    scales_np = dataset.scales.numpy()  # (N,)

    if dataset.target_mode == "log_polar":
        # Denormalize log|R| from z-scored log-polar targets
        ts = dataset.target_stats
        log_abs_R = targets_np[:, 0] * ts["std"][0] + ts["mean"][0]
        R_sq = np.exp(2.0 * log_abs_R)  # |R|^2 = exp(2 * log|R|)
    else:
        # Cartesian: denormalize and optionally decompress
        t_re_norm = targets_np[:, 0]  # (N,)
        t_im_norm = targets_np[:, 1]  # (N,)

        if dataset.log_compress:
            # target_norm = sign(T_logc) * |T_logc| / rms, where T_logc = sign(T)*log(1+|T|)
            t_re_logc = t_re_norm * scales_np
            t_im_logc = t_im_norm * scales_np
            # Decompress: T = sign(T_logc) * expm1(|T_logc|)
            t_re = np.sign(t_re_logc) * np.expm1(np.abs(t_re_logc))
            t_im = np.sign(t_im_logc) * np.expm1(np.abs(t_im_logc))
        else:
            t_re = t_re_norm * scales_np
            t_im = t_im_norm * scales_np

        R_sq = (1.0 + t_re) ** 2 + t_im ** 2  # |1 + T|^2

    # Cap at percentile to prevent extreme weights
    cap = float(np.percentile(R_sq, cap_percentile))
    R_sq_capped = np.minimum(R_sq, cap)

    # Normalize so mean weight = 1.0
    mean_w = R_sq_capped.mean()
    if mean_w < 1e-10:
        mean_w = 1.0
    w = R_sq_capped / mean_w

    logger.info(
        "Importance weights: min=%.2f, median=%.2f, max=%.2f, cap=%.1f (p%.1f)",
        w.min(), np.median(w), w.max(), cap, cap_percentile,
    )

    return torch.from_numpy(w.astype(np.float32))


def prepare_gpu_data(
    dataset: Phase1Dataset,
    val_fraction: float,
    seed: int,
    device: torch.device,
    scene_id_map_override: Optional[Dict[int, int]] = None,
    weight_mode: str = "scale",
    scene_boost: Optional[Dict[int, float]] = None,
) -> Dict[str, torch.Tensor]:
    """Split and move all data to GPU for zero-overhead batching.

    Memory: ~85 MB for 1.77M samples (trivial for 8GB VRAM).

    Parameters
    ----------
    scene_id_map_override : dict, optional
        If provided, maps scene_id -> 0-indexed embedding index.
        Used for fine-tuning to preserve source model's scene mapping.
    weight_mode : str
        "scale" = per-scene RMS weighting (default),
        "importance" = per-sample |R|^2 weighting (aligns with gate metric),
        "none" = uniform weighting.
    scene_boost : dict, optional
        Per-scene weight multiplier. E.g., {13: 5.0} gives S13 samples 5x weight.
        Applied after weight_mode weighting, then re-normalized to mean=1.

    Returns
    -------
    dict with train/val splits on device.
    """
    n_total = len(dataset)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val

    # Deterministic shuffle
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=gen)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    # Sample weights
    if weight_mode == "importance":
        sample_weights = compute_importance_weights(dataset)  # (N,)
    elif weight_mode == "gate_aligned":
        # Gate-aligned weighting: w_i = |p_inc(k_i, r_i)|^2 * scale_i^2
        # This makes weighted MSE(T_norm) ∝ gate metric numerator Σ|Δp_total|^2.
        #
        # Derivation:
        #   Δp = p_inc · ΔT_denorm = p_inc · (ΔT_norm · scale)
        #   |Δp|^2 = |p_inc|^2 · scale^2 · |ΔT_norm|^2
        #   gate^2 = Σ|Δp|^2 / Σ|p_ref|^2  (denominator is constant)
        #   ∴ minimize Σ w_i · |ΔT_norm,i|^2 where w_i = |p_inc_i|^2 · scale_i^2
        #
        # For kr >> 1: |p_inc|^2 = (1/16)|H0(kr)|^2 ≈ 1/(8π·k·r)
        # Our min kr ≈ 36*0.2 = 7.2, so the asymptotic approx is accurate (<1% error).
        k_arr = dataset.inputs[:, 4].numpy()     # (N,) wavenumber [rad/m]
        dist_arr = dataset.inputs[:, 6].numpy()  # (N,) source-receiver distance [m]
        scale_arr = dataset.scales.numpy()        # (N,) per-scene RMS scale

        kr = k_arr * dist_arr  # (N,)
        kr_safe = np.maximum(kr, 1.0)  # safety floor

        # |p_inc|^2 ≈ 1 / (8π kr) for 2D free-space Green's function
        p_inc_sq = 1.0 / (8.0 * np.pi * kr_safe)  # (N,)

        w = p_inc_sq * (scale_arr ** 2)  # (N,) gate-aligned weight

        # Normalize to mean=1
        w_mean = w.mean()
        if w_mean < 1e-30:
            w_mean = 1.0
        w = w / w_mean

        logger.info(
            "Gate-aligned weights: min=%.3f, median=%.3f, max=%.3f, "
            "kr_range=[%.1f, %.1f]",
            w.min(), np.median(w), w.max(),
            kr.min(), kr.max(),
        )
        sample_weights = torch.from_numpy(w.astype(np.float32))
    elif weight_mode == "scat_energy":
        # Scattered energy weighting: w_i = |T_gt_i|^2 * |p_inc_i|^2 * scale_i^2
        # = |p_scat_gt_i|^2.  Gives HIGH weight to shadow zone receivers where
        # |T| is large, forcing the model to learn extreme diffraction patterns.
        #
        # Compared to gate_aligned (w ∝ |p_inc|^2 * scale^2), this adds |T_gt|^2
        # which up-weights shadow zone samples by up to 1000x.
        k_arr = dataset.inputs[:, 4].numpy()     # (N,) wavenumber
        dist_arr = dataset.inputs[:, 6].numpy()  # (N,) distance
        scale_arr = dataset.scales.numpy()        # (N,) scene scale

        kr = k_arr * dist_arr
        kr_safe = np.maximum(kr, 1.0)
        p_inc_sq = 1.0 / (8.0 * np.pi * kr_safe)  # |p_inc|^2 approx

        # |T_gt|^2 from targets (already normalized by scale)
        t_re = dataset.targets[:, 0].numpy()  # (N,)
        t_im = dataset.targets[:, 1].numpy()  # (N,)
        t_mag_sq = t_re**2 + t_im**2  # (N,) |T_norm|^2

        # w = |T_norm|^2 * scale^2 * |p_inc|^2 = |p_scat|^2
        w = t_mag_sq * (scale_arr ** 2) * p_inc_sq
        # Clip extreme weights (top 1%) to prevent instability
        w_clip = np.percentile(w, 99)
        w = np.minimum(w, w_clip)

        w_mean = w.mean()
        if w_mean < 1e-30:
            w_mean = 1.0
        w = w / w_mean

        logger.info(
            "Scat-energy weights: min=%.3f, median=%.3f, max=%.3f, "
            "w_clip=%.3e",
            w.min(), np.median(w), w.max(), w_clip,
        )
        sample_weights = torch.from_numpy(w.astype(np.float32))
    elif weight_mode == "scale":
        # Scale-based sample weights: weight = scale / mean(scale)
        # Gives proportionally more weight to scenes with larger T magnitude
        scales = dataset.scales
        scale_mean = scales.mean().clamp(min=1e-10)
        sample_weights = scales / scale_mean  # (N,) centered around 1.0
    else:
        sample_weights = torch.ones(n_total)  # uniform

    # Apply per-scene boost multiplier
    if scene_boost is not None:
        scene_ids_np = dataset.scene_ids.numpy()
        boost_factors = np.ones(n_total, dtype=np.float32)
        for sid, factor in scene_boost.items():
            mask = scene_ids_np == sid
            boost_factors[mask] = factor
            n_boosted = mask.sum()
            logger.info(
                "Scene boost: S%d x%.1f (%d samples)", sid, factor, n_boosted,
            )
        sample_weights = sample_weights * torch.from_numpy(boost_factors)
        # Re-normalize to mean=1
        sw_mean = sample_weights.mean().clamp(min=1e-10)
        sample_weights = sample_weights / sw_mean

    # Scene IDs: 0-indexed within the trained scene list
    # For full 15-scene training: scene 1 -> 0, scene 15 -> 14
    # For subset training (e.g. --scenes 13): scene 13 -> 0
    # For fine-tuning: use source model's mapping (e.g. scene 13 -> 12)
    if scene_id_map_override is not None:
        scene_id_map = scene_id_map_override
    else:
        trained_scene_list = sorted(dataset.scene_scales.keys())
        scene_id_map = {sid: idx for idx, sid in enumerate(trained_scene_list)}
    scene_ids_0idx = torch.tensor(
        [scene_id_map[int(sid)] for sid in dataset.scene_ids],
        dtype=torch.long,
    )  # (N,) 0-indexed within trained scene list

    data = {
        "train_inputs": dataset.inputs[train_idx].to(device),  # (N_train, 9)
        "train_targets": dataset.targets[train_idx].to(device),  # (N_train, 2)
        "train_weights": sample_weights[train_idx].to(device),  # (N_train,)
        "train_scene_ids": scene_ids_0idx[train_idx].to(device),  # (N_train,)
        "val_inputs": dataset.inputs[val_idx].to(device),  # (N_val, 9)
        "val_targets": dataset.targets[val_idx].to(device),  # (N_val, 2)
        "val_weights": sample_weights[val_idx].to(device),  # (N_val,)
        "val_scene_ids": scene_ids_0idx[val_idx].to(device),  # (N_val,)
        "n_train": n_train,
        "n_val": n_val,
    }

    logger.info(
        "GPU data: train=%d, val=%d (%.1f MB VRAM)",
        n_train,
        n_val,
        (data["train_inputs"].nbytes + data["train_targets"].nbytes
         + data["val_inputs"].nbytes + data["val_targets"].nbytes) / 1e6,
    )

    return data


# ---------------------------------------------------------------------------
# Training loop (manual batching, zero DataLoader overhead)
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: TransferFunctionModel,
    data: Dict[str, torch.Tensor],
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    use_scale_weight: bool = True,
    noise_std: float = 0.0,
) -> Dict[str, float]:
    """Train for one epoch with manual GPU batching.

    Parameters
    ----------
    noise_std : float
        Input noise augmentation factor.  If > 0, Gaussian noise with
        std = noise_std * per-feature-std is added to inputs each batch.
        Regularizes against overfitting when training on small datasets.

    Returns dict with 'loss', 'var_explained'.
    """
    model.train()
    inputs_all = data["train_inputs"]
    targets_all = data["train_targets"]
    weights_all = data["train_weights"]
    sids_all = data["train_scene_ids"]
    n_train = data["n_train"]

    # Pre-compute per-feature noise scale (once per epoch)
    if noise_std > 0.0:
        feat_std = data.get("_feat_std")
        if feat_std is None:
            feat_std = inputs_all.std(dim=0).clamp(min=1e-6)  # (9,)
            data["_feat_std"] = feat_std

    # Shuffle indices on GPU
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

        # Input noise augmentation
        if noise_std > 0.0:
            noise = torch.randn_like(inputs) * (noise_std * feat_std)
            inputs = inputs + noise

        optimizer.zero_grad(set_to_none=True)

        t_pred = model(inputs, scene_ids=sids)  # (B, 2)

        if use_scale_weight:
            w = weights_all[idx].unsqueeze(-1)  # (B, 1)
            loss = (w * (t_pred - targets) ** 2).mean()
        else:
            loss = nn.functional.mse_loss(t_pred, targets)

        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss: {loss.item():.4e}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_diff_sq += ((t_pred.detach() - targets) ** 2).sum().item()
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
    use_scale_weight: bool = True,
) -> Dict[str, float]:
    """Validate with manual GPU batching.

    Returns dict with 'val_loss', 'val_var_explained'.
    """
    model.eval()
    inputs_all = data["val_inputs"]
    targets_all = data["val_targets"]
    weights_all = data["val_weights"]
    sids_all = data["val_scene_ids"]
    n_val = data["n_val"]

    total_diff_sq = 0.0
    total_tgt_sq = 0.0
    total_loss = 0.0
    n_batches = 0

    for i in range(0, n_val, batch_size):
        inputs = inputs_all[i : i + batch_size]  # (B, 9)
        targets = targets_all[i : i + batch_size]  # (B, 2)
        sids = sids_all[i : i + batch_size]  # (B,)

        t_pred = model(inputs, scene_ids=sids)  # (B, 2)

        if use_scale_weight:
            w = weights_all[i : i + batch_size].unsqueeze(-1)  # (B, 1)
            loss = (w * (t_pred - targets) ** 2).mean()
        else:
            loss = nn.functional.mse_loss(t_pred, targets)

        total_loss += loss.item()
        total_diff_sq += ((t_pred - targets) ** 2).sum().item()
        total_tgt_sq += (targets ** 2).sum().item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    var_exp = 1.0 - total_diff_sq / max(total_tgt_sq, 1e-30)

    return {"val_loss": avg_loss, "val_var_explained": var_exp}


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: Path,
    model: TransferFunctionModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_val_loss: float,
    scene_scales: Dict[int, float],
    config: dict,
) -> None:
    """Save training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "scene_scales": scene_scales,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: TransferFunctionModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple:
    """Load checkpoint. Returns (epoch, best_val_loss, scene_scales)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    logger.info(
        "Checkpoint loaded: %s (epoch %d)", path.name, ckpt["epoch"]
    )
    return ckpt["epoch"], ckpt["best_val_loss"], ckpt["scene_scales"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 Transfer Function Training"
    )
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--lr-warmup-epochs", type=int, default=DEFAULTS["lr_warmup_epochs"])
    parser.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULTS["checkpoint_every"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--scenes", nargs="+", type=int, default=None)
    parser.add_argument(
        "--resume", action="store_true", help="Resume from latest checkpoint"
    )
    parser.add_argument("--d-hidden", type=int, default=DEFAULTS["d_hidden"])
    parser.add_argument("--n-blocks", type=int, default=DEFAULTS["n_blocks"])
    parser.add_argument("--n-fourier", type=int, default=DEFAULTS["n_fourier"])
    parser.add_argument("--dropout", type=float, default=DEFAULTS.get("dropout", 0.0))
    parser.add_argument(
        "--fourier-sigma", type=str, default=str(DEFAULTS["fourier_sigma"]),
        help="Fourier feature sigma [m^-1]. Comma-separated for multi-scale, "
             "e.g., '10,30,90'. Single value for single-scale.",
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="Tag for checkpoint naming: best_{tag}.pt (e.g., --tag scene13)",
    )
    parser.add_argument(
        "--log-compress", action="store_true",
        help="Use sign(x)*log(1+|x|) compression on transfer function targets",
    )
    parser.add_argument(
        "--target-mode", type=str, default="cartesian",
        choices=["cartesian", "log_polar"],
        help="Target representation: 'cartesian' (Re,Im) or 'log_polar' (log|R|,cos,sin)",
    )
    parser.add_argument(
        "--weight-mode", type=str, default="scale",
        choices=["scale", "importance", "gate_aligned", "scat_energy", "none"],
        help="Sample weighting: 'scale' (per-scene RMS), 'importance' (per-sample |R|^2), "
             "'gate_aligned' (|p_inc|^2 * scale^2), 'scat_energy' (|T_gt|^2 * |p_inc|^2 * scale^2, "
             "targets shadow zone accuracy), 'none' (uniform)",
    )
    parser.add_argument(
        "--scene-boost", type=str, default=None,
        help="Per-scene weight multiplier. Format: 'SCENE_ID:FACTOR,...' "
             "e.g., '13:5.0' gives scene 13 samples 5x weight. "
             "Applied on top of weight_mode weighting.",
    )
    parser.add_argument(
        "--finetune-from", type=str, default=None,
        help="Checkpoint to fine-tune from (e.g., 'best_logc'). "
             "Loads model weights, uses checkpoint's log_compress setting.",
    )
    parser.add_argument(
        "--freeze-blocks", type=int, default=0,
        help="Freeze first N residual blocks during training (for fine-tuning)",
    )
    parser.add_argument(
        "--noise-std", type=float, default=0.0,
        help="Input noise augmentation factor. Gaussian noise with "
             "std=noise_std*feature_std is added per batch. "
             "Regularizes fine-tuning on small datasets (e.g., 0.02).",
    )
    args = parser.parse_args()

    # Parse fourier_sigma: "30.0" -> 30.0, "10,30,90" -> [10.0, 30.0, 90.0]
    sigma_parts = [float(s.strip()) for s in args.fourier_sigma.split(",")]
    fourier_sigma = sigma_parts if len(sigma_parts) > 1 else sigma_parts[0]

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info(
            "VRAM: %.1f GB",
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    # ---------------------------------------------------------------
    # Fine-tuning: load source checkpoint BEFORE data loading
    # ---------------------------------------------------------------
    finetune_mode = args.finetune_from is not None
    src_ckpt = None
    src_config = None
    src_scene_id_map = None

    if finetune_mode:
        src_ckpt_path = CHECKPOINT_DIR / f"{args.finetune_from}.pt"
        if not src_ckpt_path.exists():
            logger.error("Fine-tune source not found: %s", src_ckpt_path)
            sys.exit(1)
        src_ckpt = torch.load(
            src_ckpt_path, map_location=device, weights_only=False
        )
        src_config = src_ckpt["config"]
        # Inherit log_compress from source model
        ft_log_compress = src_config.get("log_compress", False)
        # Build source scene_id_map
        src_tsl = src_config.get(
            "trained_scene_list",
            sorted(src_ckpt["scene_scales"].keys()),
        )
        src_scene_id_map = {sid: idx for idx, sid in enumerate(src_tsl)}
        logger.info(
            "Fine-tuning from: %s (epoch %d, log_compress=%s)",
            src_ckpt_path.name,
            src_ckpt["epoch"],
            ft_log_compress,
        )
    else:
        ft_log_compress = args.log_compress

    # ---------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------
    # Determine target_mode: fine-tune inherits from source; otherwise use arg
    if finetune_mode:
        target_mode = src_config.get("target_mode", "cartesian")
    else:
        target_mode = args.target_mode

    # d_out depends on target mode
    d_out = 3 if target_mode == "log_polar" else 2

    logger.info("Loading data (target_mode=%s, d_out=%d)...", target_mode, d_out)
    t_load = time.time()
    dataset = Phase1Dataset(
        DATA_DIR,
        scene_ids=args.scenes,
        log_compress=ft_log_compress,
        target_mode=target_mode,
    )
    input_mean, input_std = dataset.get_input_stats()

    # Parse scene-boost: "13:5.0,12:2.0" -> {13: 5.0, 12: 2.0}
    scene_boost_dict: Optional[Dict[int, float]] = None
    if args.scene_boost is not None:
        scene_boost_dict = {}
        for part in args.scene_boost.split(","):
            sid_str, factor_str = part.strip().split(":")
            scene_boost_dict[int(sid_str)] = float(factor_str)

    gpu_data = prepare_gpu_data(
        dataset,
        val_fraction=DEFAULTS["val_fraction"],
        seed=args.seed,
        device=device,
        scene_id_map_override=src_scene_id_map,
        weight_mode=args.weight_mode,
        scene_boost=scene_boost_dict,
    )
    logger.info("Loaded %d samples in %.1fs", len(dataset), time.time() - t_load)

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------
    if finetune_mode:
        # Build model with SOURCE architecture (must match checkpoint)
        scene_emb_dim = src_config.get("scene_emb_dim", 32)
        n_scenes = src_config.get("n_scenes", 15)
        trained_scene_list = src_tsl

        model = build_transfer_model(
            d_hidden=src_config.get("d_hidden", 768),
            n_blocks=src_config.get("n_blocks", 8),
            n_fourier=src_config.get("n_fourier", 128),
            fourier_sigma=src_config.get("fourier_sigma", 30.0),
            dropout=src_config.get("dropout", 0.0),
            n_scenes=n_scenes,
            scene_emb_dim=scene_emb_dim,
            d_out=src_config.get("d_out", 2),
        )
        # Load weights (includes normalization buffers + Fourier B matrix)
        model.load_state_dict(src_ckpt["model_state_dict"])
        model = model.to(device)

        # Freeze specified layers
        n_freeze = args.freeze_blocks
        if n_freeze > 0:
            # Freeze Fourier encoder (B matrix is buffer, but freeze any params)
            for p in model.encoder.parameters():
                p.requires_grad = False
            # Freeze input projection
            for p in model.input_proj.parameters():
                p.requires_grad = False
            # Freeze scene embedding
            if model.n_scenes > 0:
                model.scene_embedding.weight.requires_grad = False
            # Freeze first N residual blocks
            for i, block in enumerate(model.blocks):
                if i < n_freeze:
                    for p in block.parameters():
                        p.requires_grad = False

        n_trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        n_total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            "Fine-tune: %d/%d trainable params (%.1f%%), freeze_blocks=%d",
            n_trainable,
            n_total_params,
            n_trainable / n_total_params * 100,
            n_freeze,
        )
    else:
        # Standard model creation
        trained_scene_list = sorted(dataset.scene_scales.keys())
        n_scenes = len(trained_scene_list)
        scene_emb_dim = 32

        model = build_transfer_model(
            d_hidden=args.d_hidden,
            n_blocks=args.n_blocks,
            n_fourier=args.n_fourier,
            fourier_sigma=fourier_sigma,
            dropout=args.dropout,
            n_scenes=n_scenes,
            scene_emb_dim=scene_emb_dim,
            d_out=d_out,
        )
        model.set_normalization(input_mean, input_std)
        model = model.to(device)

    n_params = model.count_parameters()
    logger.info(
        "Model: %d params, Train: %d, Val: %d",
        n_params,
        gpu_data["n_train"],
        gpu_data["n_val"],
    )

    # ---------------------------------------------------------------
    # Optimizer & scheduler
    # ---------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )

    lr_warmup = args.lr_warmup_epochs

    def lr_lambda(epoch: int) -> float:
        if epoch < lr_warmup:
            return (epoch + 1) / lr_warmup
        progress = (epoch - lr_warmup) / max(args.epochs - lr_warmup, 1)
        return max(1e-6 / args.lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---------------------------------------------------------------
    # Resume
    # ---------------------------------------------------------------
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    # Checkpoint naming with optional tag
    tag_suffix = f"_{args.tag}" if args.tag else ""
    best_ckpt_name = f"best{tag_suffix}.pt"
    latest_ckpt_name = f"latest{tag_suffix}.pt"

    if args.resume:
        ckpt_path = CHECKPOINT_DIR / latest_ckpt_name
        if ckpt_path.exists():
            start_epoch, best_val_loss, _ = load_checkpoint(
                ckpt_path, model, optimizer, scheduler, device
            )
            start_epoch += 1
        else:
            logger.warning("No checkpoint at %s -- starting fresh", ckpt_path)

    # ---------------------------------------------------------------
    # Config
    # ---------------------------------------------------------------
    if finetune_mode:
        # Inherit architecture config from source, add fine-tune metadata
        config = dict(src_config)
        config.update({
            "finetune_from": args.finetune_from,
            "finetune_scenes": args.scenes,
            "freeze_blocks": args.freeze_blocks,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lr_warmup_epochs": args.lr_warmup_epochs,
            "n_train": gpu_data["n_train"],
            "n_val": gpu_data["n_val"],
        })
    else:
        config = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lr_warmup_epochs": args.lr_warmup_epochs,
            "d_hidden": args.d_hidden,
            "n_blocks": args.n_blocks,
            "n_fourier": args.n_fourier,
            "fourier_sigma": fourier_sigma,
            "dropout": args.dropout,
            "d_out": d_out,
            "target_mode": target_mode,
            "n_train": gpu_data["n_train"],
            "n_val": gpu_data["n_val"],
            "n_params": n_params,
            "scenes": args.scenes or list(range(1, 16)),
            "trained_scene_list": trained_scene_list,
            "n_scenes": n_scenes,
            "scene_emb_dim": scene_emb_dim,
            "log_compress": ft_log_compress,
            "weight_mode": args.weight_mode,
            "scene_boost": args.scene_boost,
        }
        # Store target normalization stats for log_polar mode
        if dataset.target_stats is not None:
            config["target_stats"] = dataset.target_stats

    # ---------------------------------------------------------------
    # Metrics CSV
    # ---------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / "training_metrics.csv"
    if not metrics_path.exists() or not args.resume:
        with open(metrics_path, "w") as fh:
            fh.write("epoch,train_loss,val_loss,train_var_exp,val_var_exp,lr,time_s\n")

    # ---------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------
    logger.info("Starting training from epoch %d ...", start_epoch)
    t0_total = time.time()

    # Milestone epochs for compact logging
    log_epochs = set(
        list(range(5))
        + list(range(49, args.epochs, 50))
        + [args.epochs - 1]
    )

    val_var_exp = 0.0  # track for final reporting

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Use sample weights unless weight_mode="none"
        use_sw = args.weight_mode != "none"
        train_m = train_one_epoch(
            model, gpu_data, args.batch_size, optimizer,
            use_scale_weight=use_sw,
            noise_std=args.noise_std,
        )
        val_m = validate(model, gpu_data, args.batch_size, use_scale_weight=use_sw)

        scheduler.step()
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        val_var_exp = val_m["val_var_explained"]

        if epoch in log_epochs:
            logger.info(
                "Ep %4d: train=%.4e val=%.4e var_exp=%.1f%% lr=%.2e %.0fs",
                epoch + 1,
                train_m["loss"],
                val_m["val_loss"],
                val_var_exp * 100,
                lr_now,
                time.time() - t0_total,
            )

        # Save metrics
        with open(metrics_path, "a") as fh:
            fh.write(
                f"{epoch},"
                f"{train_m['loss']:.6e},"
                f"{val_m['val_loss']:.6e},"
                f"{train_m['var_explained']:.6f},"
                f"{val_m['val_var_explained']:.6f},"
                f"{lr_now:.6e},"
                f"{elapsed:.1f}\n"
            )

        # Periodic checkpoint
        if (epoch + 1) % args.checkpoint_every == 0:
            save_checkpoint(
                CHECKPOINT_DIR / latest_ckpt_name,
                model, optimizer, scheduler, epoch,
                best_val_loss, dataset.scene_scales, config,
            )

        # Best model
        if val_m["val_loss"] < best_val_loss:
            best_val_loss = val_m["val_loss"]
            patience_counter = 0
            save_checkpoint(
                CHECKPOINT_DIR / best_ckpt_name,
                model, optimizer, scheduler, epoch,
                best_val_loss, dataset.scene_scales, config,
            )
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            logger.info(
                "Early stopping at epoch %d (patience=%d)",
                epoch + 1,
                args.patience,
            )
            break

    total_time = time.time() - t0_total
    logger.info("")
    logger.info("Done in %.1f min", total_time / 60)
    logger.info(
        "Best val: %.4e (%.1f%% variance explained)",
        best_val_loss,
        val_var_exp * 100,
    )
    logger.info("Saved: %s", CHECKPOINT_DIR / best_ckpt_name)


if __name__ == "__main__":
    main()
