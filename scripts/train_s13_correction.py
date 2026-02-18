"""Train a small residual-correction MLP for Scene 13.

Strategy:
    1. Run the S13 specialist ensemble on S13 training data
    2. Compute residuals: delta_T = T_gt - T_ensemble
    3. Train a small MLP to predict delta_T from input features
    4. Save as a correction checkpoint

The correction MLP is applied on top of ensemble predictions
to reduce the per-receiver error that linear calibration cannot fix.

Usage:
    python scripts/train_s13_correction.py --epochs 200 --d-hidden 256 --n-layers 4
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import RegularGridInterpolator
from scipy.special import hankel1

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.forward_model import TransferFunctionModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("s13_correction")

CKPT_DIR = Path("checkpoints/phase2")
DATA_DIR = Path("data/phase1")
SPEED_OF_SOUND_M_PER_S = 343.0


class CorrectionMLP(nn.Module):
    """Small MLP that predicts residual correction delta_T.

    Input: 9 features (same as forward model)
    Output: 2 (Re(delta_T), Im(delta_T))
    """

    def __init__(
        self,
        d_in: int = 9,
        d_hidden: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = [nn.Linear(d_in, d_hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(d_hidden, d_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(d_hidden, 2))
        self.net = nn.Sequential(*layers)

        # Initialize last layer to near-zero (start from no correction)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (B, 9) -> (B, 2)."""
        return self.net(x)


def generate_ensemble_predictions(
    specialist_names: list,
    h5_path: Path,
    device: torch.device,
) -> tuple:
    """Run ensemble on S13, return per-sample T predictions and GT targets.

    Returns:
        inputs_all: (N_total, 9) input features
        t_ens_all: (N_total, 2) ensemble T predictions [Re, Im]
        t_gt_all: (N_total, 2) ground truth T [Re, Im]
        weights_all: (N_total,) sample weights (|p_inc|^2 * scale^2)
    """
    with h5py.File(h5_path, "r") as f:
        freqs_hz = f["frequencies"][:]
        src_pos = f["sources/positions"][:]
        rcv_pos = f["receivers/positions"][:]
        sdf_gx = f["sdf/grid_x"][:]
        sdf_gy = f["sdf/grid_y"][:]
        sdf_vals = f["sdf/values"][:]
        n_src = src_pos.shape[0]
        n_freq = len(freqs_hz)
        n_rcv = rcv_pos.shape[0]

        p_total_bem_all = []
        for si in range(n_src):
            p_total_bem_all.append(f[f"pressure/src_{si:03d}/field"][:])

    k_arr = 2.0 * np.pi * freqs_hz / SPEED_OF_SOUND_M_PER_S
    sdf_interp = RegularGridInterpolator(
        (sdf_gx, sdf_gy), sdf_vals,
        method="linear", bounds_error=False, fill_value=1.0,
    )
    sdf_at_rcv = sdf_interp(rcv_pos)

    # Collect per-model T predictions, then average
    n_total = n_src * n_freq * n_rcv
    t_preds_sum = np.zeros((n_total, 2), dtype=np.float64)  # accumulate Re, Im
    n_models = len(specialist_names)

    for sname in specialist_names:
        ckpt = torch.load(CKPT_DIR / f"{sname}.pt", map_location=device, weights_only=False)
        config = ckpt["config"]
        tsl = config.get("trained_scene_list", sorted(ckpt["scene_scales"].keys()))
        scale = ckpt["scene_scales"][13]
        sid_0idx = tsl.index(13)

        mdl = TransferFunctionModel(
            d_in=9, d_hidden=config["d_hidden"], n_blocks=config["n_blocks"],
            d_out=config.get("d_out", 2), n_fourier=config["n_fourier"],
            fourier_sigma=config["fourier_sigma"], dropout=config["dropout"],
            n_scenes=config["n_scenes"],
            scene_emb_dim=config.get("scene_emb_dim", 32),
        ).to(device)
        mdl.load_state_dict(ckpt["model_state_dict"])
        mdl.eval()

        idx = 0
        for si in range(n_src):
            xs_m, ys_m = src_pos[si]
            dx_sr = rcv_pos[:, 0] - xs_m
            dy_sr = rcv_pos[:, 1] - ys_m
            dist_sr = np.sqrt(dx_sr**2 + dy_sr**2)

            for fi_s in range(0, n_freq, 50):
                fi_e = min(fi_s + 50, n_freq)
                n_f = fi_e - fi_s
                n = n_f * n_rcv
                inputs = np.column_stack([
                    np.full(n, xs_m), np.full(n, ys_m),
                    np.tile(rcv_pos[:, 0], n_f),
                    np.tile(rcv_pos[:, 1], n_f),
                    np.repeat(k_arr[fi_s:fi_e], n_rcv),
                    np.tile(sdf_at_rcv, n_f),
                    np.tile(dist_sr, n_f),
                    np.tile(dx_sr, n_f),
                    np.tile(dy_sr, n_f),
                ]).astype(np.float32)

                inp_t = torch.from_numpy(inputs).to(device)
                sid_t = torch.full((n,), sid_0idx, dtype=torch.long, device=device)
                with torch.no_grad():
                    pred_raw = mdl(inp_t, scene_ids=sid_t).cpu().numpy()

                # T_denorm = pred * scale
                t_preds_sum[idx:idx + n, 0] += pred_raw[:, 0] * scale
                t_preds_sum[idx:idx + n, 1] += pred_raw[:, 1] * scale
                idx += n

        logger.info("  %s: done", sname)

    t_ens = t_preds_sum / n_models  # (N_total, 2)

    # Build inputs, GT targets, weights
    inputs_all = np.zeros((n_total, 9), dtype=np.float32)
    t_gt_all = np.zeros((n_total, 2), dtype=np.float64)
    weights_all = np.zeros(n_total, dtype=np.float64)

    # Get scale from any checkpoint
    ckpt0 = torch.load(CKPT_DIR / f"{specialist_names[0]}.pt", map_location="cpu", weights_only=False)
    scale = ckpt0["scene_scales"][13]

    idx = 0
    for si in range(n_src):
        xs_m, ys_m = src_pos[si]
        dx_sr = rcv_pos[:, 0] - xs_m
        dy_sr = rcv_pos[:, 1] - ys_m
        dist_sr = np.sqrt(dx_sr**2 + dy_sr**2)
        dist_sr_safe = np.maximum(dist_sr, 1e-15)

        for fi in range(n_freq):
            n = n_rcv
            sl = slice(idx, idx + n)

            inputs_all[sl] = np.column_stack([
                np.full(n, xs_m), np.full(n, ys_m),
                rcv_pos[:, 0], rcv_pos[:, 1],
                np.full(n, k_arr[fi]),
                sdf_at_rcv, dist_sr, dx_sr, dy_sr,
            ]).astype(np.float32)

            # GT: T = (p_total / p_inc - 1) / scale
            kr = k_arr[fi] * dist_sr_safe
            p_inc = -0.25j * hankel1(0, kr)  # (R,)
            p_total_bem = p_total_bem_all[si][fi]  # (R,)
            T_gt = (p_total_bem / p_inc - 1.0) / scale
            t_gt_all[sl, 0] = np.real(T_gt)
            t_gt_all[sl, 1] = np.imag(T_gt)

            # Gate-aligned weight: |p_inc|^2 * scale^2
            p_inc_sq = np.abs(p_inc) ** 2
            weights_all[sl] = p_inc_sq * scale**2

            idx += n

    return inputs_all, t_ens.astype(np.float32), t_gt_all.astype(np.float32), weights_all.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train S13 correction MLP")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-hidden", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="s13_corr")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    specialist_names = [
        "best_v7_ft13", "best_v8", "best_v8_ft13",
        "best_v11_ft13", "best_v13_ft13", "best_v13_ft13b",
    ]

    h5_path = DATA_DIR / "scene_013.h5"
    logger.info("Generating ensemble predictions for S13...")
    inputs_all, t_ens_all, t_gt_all, weights_all = generate_ensemble_predictions(
        specialist_names, h5_path, device
    )

    # Residual target: delta_T = T_gt - T_ens
    delta_t = t_gt_all - t_ens_all  # (N, 2)
    logger.info("Residual stats: mean=%.4f, std=%.4f, max=%.4f",
                np.mean(np.abs(delta_t)), np.std(np.abs(delta_t)),
                np.max(np.abs(delta_t)))

    n_total = len(inputs_all)
    n_train = int(n_total * 0.8)

    # Shuffle
    perm = np.random.permutation(n_total)
    inputs_all = inputs_all[perm]
    delta_t = delta_t[perm]
    weights_all = weights_all[perm]

    # Split
    inputs_train = torch.tensor(inputs_all[:n_train], device=device)
    delta_train = torch.tensor(delta_t[:n_train], device=device)
    weights_train = torch.tensor(weights_all[:n_train], device=device)
    weights_train = weights_train / weights_train.mean()  # normalize

    inputs_val = torch.tensor(inputs_all[n_train:], device=device)
    delta_val = torch.tensor(delta_t[n_train:], device=device)
    weights_val = torch.tensor(weights_all[n_train:], device=device)
    weights_val = weights_val / weights_val.mean()

    logger.info("Train: %d, Val: %d", n_train, n_total - n_train)

    # Build correction model
    correction = CorrectionMLP(
        d_in=9, d_hidden=args.d_hidden,
        n_layers=args.n_layers, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in correction.parameters())
    logger.info("CorrectionMLP: %d params (%.2f KB)", n_params, n_params * 4 / 1024)

    optimizer = torch.optim.AdamW(
        correction.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_val = float("inf")
    best_epoch = 0
    patience_counter = 0

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        # Train
        correction.train()
        perm_t = torch.randperm(n_train, device=device)
        train_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, args.batch_size):
            idx = perm_t[i:i + args.batch_size]
            inp = inputs_train[idx]
            tgt = delta_train[idx]
            w = weights_train[idx]

            pred = correction(inp)
            diff = pred - tgt
            loss = (w[:, None] * diff**2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

        # Validate
        correction.eval()
        with torch.no_grad():
            n_val = len(inputs_val)
            val_loss = 0.0
            n_vb = 0
            for i in range(0, n_val, args.batch_size):
                inp = inputs_val[i:i + args.batch_size]
                tgt = delta_val[i:i + args.batch_size]
                w = weights_val[i:i + args.batch_size]
                pred = correction(inp)
                diff = pred - tgt
                loss = (w[:, None] * diff**2).mean()
                val_loss += loss.item()
                n_vb += 1
            val_loss /= max(n_vb, 1)

        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0
            ckpt_path = CKPT_DIR / f"best_{args.tag}.pt"
            torch.save({
                "model_state_dict": correction.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val,
                "config": {
                    "d_in": 9,
                    "d_hidden": args.d_hidden,
                    "n_layers": args.n_layers,
                    "dropout": args.dropout,
                    "n_params": n_params,
                    "specialist_names": specialist_names,
                },
            }, ckpt_path)
        else:
            patience_counter += 1

        if epoch <= 5 or epoch % 10 == 0 or epoch == args.epochs:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "Ep %4d: train=%.4e val=%.4e lr=%.2e %ds",
                epoch, train_loss, val_loss, lr, elapsed,
            )

        if patience_counter >= args.patience:
            logger.info("Early stop at epoch %d (patience=%d)", epoch, args.patience)
            break

    logger.info("Done. Best val=%.4e at epoch %d", best_val, best_epoch)
    logger.info("Saved: %s", CKPT_DIR / f"best_{args.tag}.pt")


if __name__ == "__main__":
    main()
