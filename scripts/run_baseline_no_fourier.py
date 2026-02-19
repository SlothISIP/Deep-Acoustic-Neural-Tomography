#!/usr/bin/env python3
"""P2b: No-Fourier-Feature Ablation.

Trains a forward model with raw coordinate input (no Fourier feature encoding)
using the Transfer Function formulation. Tests whether random Fourier features
are essential for learning high-frequency scattering patterns.

Architecture comparison
-----------------------
    Full (production):  FourierFeatures(128, sigma=30) + 8xResBlock(768) + SceneEmbed(32)
    No-Fourier (P2b):   Linear(9 -> d_hidden) + 8xResBlock(768) + SceneEmbed(32)

The model replaces FourierFeatureEncoder with a direct linear projection,
removing the cos(2pi*B*v), sin(2pi*B*v) encoding. This isolates the
contribution of spectral bias mitigation via Fourier features.

Output: results/experiments/no_fourier_ablation.csv
        checkpoints/baseline/best_no_fourier.pt

Usage
-----
    python scripts/run_baseline_no_fourier.py
    python scripts/run_baseline_no_fourier.py --epochs 1000 --batch-size 32768
"""

import argparse
import csv
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

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
from src.forward_model import ResidualBlock

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("no_fourier_ablation")

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
N_RAW_FEATURES: int = 9


# ---------------------------------------------------------------------------
# No-Fourier Model: Linear projection instead of Fourier features
# ---------------------------------------------------------------------------
class NoFourierModel(nn.Module):
    """Forward model without Fourier feature encoding.

    Replaces FourierFeatureEncoder with direct linear projection.
    Otherwise identical architecture to TransferFunctionModel.

    Parameters
    ----------
    d_in : int
        Raw input dimension (9).
    d_hidden : int
        Hidden width.
    d_out : int
        Output dimension (2 for Re/Im).
    n_blocks : int
        Number of residual blocks.
    n_scenes : int
        Number of scenes for embedding.
    scene_emb_dim : int
        Scene embedding dimension.
    """

    def __init__(
        self,
        d_in: int = N_RAW_FEATURES,
        d_hidden: int = 768,
        d_out: int = 2,
        n_blocks: int = 8,
        n_scenes: int = 15,
        scene_emb_dim: int = 32,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.n_scenes = n_scenes

        # Scene embedding
        self.scene_emb_dim = scene_emb_dim if n_scenes > 0 else 0
        if n_scenes > 0:
            self.scene_embedding = nn.Embedding(n_scenes, scene_emb_dim)

        feat_dim = d_in + self.scene_emb_dim

        # Input projection: raw features -> d_hidden (NO Fourier encoding)
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_hidden),
            nn.GELU(),
        )

        # Residual blocks (same as TransferFunctionModel)
        self.blocks = nn.ModuleList(
            [ResidualBlock(d_hidden) for _ in range(n_blocks)]
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_out),
        )

        # Input normalization buffers
        self.register_buffer("input_mean", torch.zeros(d_in))
        self.register_buffer("input_std", torch.ones(d_in))

    def set_normalization(
        self, mean: torch.Tensor, std: torch.Tensor
    ) -> None:
        """Set z-score normalization statistics."""
        self.input_mean.copy_(mean)
        self.input_std.copy_(std)

    def forward(
        self,
        inputs: torch.Tensor,
        scene_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict normalized transfer function (Re(T), Im(T)).

        Parameters
        ----------
        inputs : (B, d_in)
        scene_ids : (B,) long, optional

        Returns
        -------
        pred : (B, 2)
        """
        x = (inputs - self.input_mean) / self.input_std  # (B, d_in)

        if self.n_scenes > 0 and scene_ids is not None:
            s_emb = self.scene_embedding(scene_ids)  # (B, emb_dim)
            x = torch.cat([x, s_emb], dim=-1)  # (B, d_in + emb_dim)

        h = self.input_proj(x)  # (B, d_hidden)
        for block in self.blocks:
            h = block(h)
        return self.output_head(h)  # (B, 2)

    @torch.no_grad()
    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# GPU data preparation (same as run_baseline_vanilla.py)
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

    trained_scene_list = sorted(dataset.scene_scales.keys())
    scene_id_map = {sid: idx for idx, sid in enumerate(trained_scene_list)}
    scene_ids_0idx = torch.tensor(
        [scene_id_map[int(sid)] for sid in dataset.scene_ids],
        dtype=torch.long,
    )

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
    return data


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(model, data, batch_size, optimizer):
    """Train for one epoch."""
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
        inputs = inputs_all[idx]
        targets = targets_all[idx]
        sids = sids_all[idx]

        optimizer.zero_grad(set_to_none=True)
        pred = model(inputs, scene_ids=sids)
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
# Evaluation (same metric as Phase 2 gate: relative L2 on p_total)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_scene(model, h5_path, scene_scale, scene_id_0idx, device):
    """Evaluate reconstruction error for one scene.

    Uses Transfer Function reconstruction: p_total = p_inc * (1 + T * scale)
    """
    model.eval()
    from scipy.interpolate import RegularGridInterpolator

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

        sdf_interp = RegularGridInterpolator(
            (sdf_grid_x, sdf_grid_y), sdf_values,
            method="linear", bounds_error=False, fill_value=1.0,
        )
        sdf_at_rcv = sdf_interp(rcv_pos)

        total_diff_sq = 0.0
        total_ref_sq = 0.0

        for si in range(src_pos.shape[0]):
            p_total_bem = f[f"pressure/src_{si:03d}/field"][:]
            xs_m, ys_m = src_pos[si]
            dx_sr = rcv_pos[:, 0] - xs_m
            dy_sr = rcv_pos[:, 1] - ys_m
            dist_sr = np.sqrt(dx_sr ** 2 + dy_sr ** 2)
            dist_sr_safe = np.maximum(dist_sr, 1e-15)

            p_total_pred = np.zeros((n_freq, n_rcv), dtype=np.complex128)

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
                sid_t = torch.full((n,), scene_id_0idx, dtype=torch.long, device=device)
                pred_raw = model(inputs_t, scene_ids=sid_t).cpu().numpy()

                kr = k_arr[fi_s:fi_e, None] * dist_sr_safe[None, :]
                p_inc = -0.25j * hankel1(0, kr)

                t_re = pred_raw[:, 0] * scene_scale
                t_im = pred_raw[:, 1] * scene_scale
                t_complex = (t_re + 1j * t_im).reshape(n_f, n_rcv)
                p_total_pred[fi_s:fi_e] = p_inc * (1.0 + t_complex)

            diff = p_total_pred - p_total_bem
            total_diff_sq += np.sum(np.abs(diff) ** 2)
            total_ref_sq += np.sum(np.abs(p_total_bem) ** 2)

    return np.sqrt(total_diff_sq / max(total_ref_sq, 1e-30))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="P2b: No-Fourier-Feature Ablation"
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
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training, evaluate from checkpoint",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info(
            "GPU: %s (%.1f GB VRAM), cudnn.benchmark=True",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    # ---------------------------------------------------------------
    # Data: T formulation (same as production, NOT pressure mode)
    # ---------------------------------------------------------------
    logger.info("Loading data (target_mode='cartesian', T formulation)...")
    t_load = time.time()
    dataset = Phase1Dataset(DATA_DIR, scene_ids=None, target_mode="cartesian")
    input_mean, input_std = dataset.get_input_stats()
    logger.info("Loaded %d samples in %.1fs", len(dataset), time.time() - t_load)

    trained_scene_list = sorted(dataset.scene_scales.keys())
    scene_id_map = {sid: idx for idx, sid in enumerate(trained_scene_list)}
    n_scenes = len(trained_scene_list)

    # ---------------------------------------------------------------
    # Model: NoFourierModel (raw coordinates, no encoding)
    # ---------------------------------------------------------------
    model = NoFourierModel(
        d_in=N_RAW_FEATURES,
        d_hidden=args.d_hidden,
        d_out=2,
        n_blocks=args.n_blocks,
        n_scenes=n_scenes,
        scene_emb_dim=32,
    )
    model.set_normalization(input_mean, input_std)
    model = model.to(device)
    n_params = model.count_parameters()
    logger.info("NoFourierModel: %d params (%.2f MB)", n_params, n_params * 4 / 1e6)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_ckpt = CHECKPOINT_DIR / "best_no_fourier.pt"

    if not args.eval_only:
        gpu_data = prepare_gpu_data(dataset, args.val_fraction, args.seed, device)

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

        logger.info("Starting No-Fourier training (%d epochs)...", args.epochs)
        t0_total = time.time()

        for epoch in range(args.epochs):
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
                        "n_scenes": n_scenes,
                        "scene_emb_dim": 32,
                        "d_out": 2,
                        "target_mode": "cartesian",
                        "trained_scene_list": trained_scene_list,
                        "fourier_features": False,
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

        logger.info(
            "Training done in %.1f min. Best val loss: %.4e",
            (time.time() - t0_total) / 60, best_val_loss,
        )

    # ---------------------------------------------------------------
    # Load best checkpoint
    # ---------------------------------------------------------------
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        scene_scales = ckpt["scene_scales"]
        logger.info("Loaded best checkpoint (epoch %d)", ckpt["epoch"])
    else:
        logger.error("No checkpoint: %s", best_ckpt)
        sys.exit(1)

    model.eval()

    # ---------------------------------------------------------------
    # Evaluate per scene
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("P2b: No-Fourier Ablation Results")
    logger.info("=" * 60)

    scene_ids = list(range(1, 16))
    results_rows: List[Dict] = []

    total_diff_sq = 0.0
    total_ref_sq = 0.0

    for sid in scene_ids:
        h5_path = DATA_DIR / f"scene_{sid:03d}.h5"
        if not h5_path.exists():
            continue
        scale = scene_scales.get(sid)
        sid_0idx = scene_id_map.get(sid)
        if scale is None or sid_0idx is None:
            continue

        error = evaluate_scene(model, h5_path, scale, sid_0idx, device)
        logger.info("  S%2d: error=%.2f%%", sid, error * 100)
        results_rows.append({
            "scene": sid,
            "no_fourier_error_pct": f"{error * 100:.2f}",
        })

    # Overall error (re-evaluate with accumulation)
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

                total_ref_sq += np.sum(np.abs(p_total_bem) ** 2)

                p_total_pred = np.zeros((n_freq, n_rcv), dtype=np.complex128)
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
                    sid_t = torch.full((n,), sid_0idx, dtype=torch.long, device=device)
                    pred_raw = model(inputs_t, scene_ids=sid_t).detach().cpu().numpy()

                    kr = k_arr[fi_s:fi_e, None] * dist_sr_safe[None, :]
                    p_inc = -0.25j * hankel1(0, kr)
                    t_re = pred_raw[:, 0] * scale
                    t_im = pred_raw[:, 1] * scale
                    t_c = (t_re + 1j * t_im).reshape(n_f, n_rcv)
                    p_total_pred[fi_s:fi_e] = p_inc * (1.0 + t_c)

                diff = p_total_pred - p_total_bem
                total_diff_sq += np.sum(np.abs(diff) ** 2)

    overall_error = np.sqrt(total_diff_sq / max(total_ref_sq, 1e-30))

    logger.info("-" * 40)
    logger.info("Overall: %.2f%%", overall_error * 100)
    logger.info("Reference: With Fourier (Config A) = 11.54%%")
    logger.info("=" * 60)

    # Save CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "no_fourier_ablation.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scene", "no_fourier_error_pct"])
        writer.writeheader()
        for row in results_rows:
            writer.writerow(row)
        writer.writerow({
            "scene": "overall",
            "no_fourier_error_pct": f"{overall_error * 100:.2f}",
        })
    logger.info("Results saved: %s", csv_path)


if __name__ == "__main__":
    main()
