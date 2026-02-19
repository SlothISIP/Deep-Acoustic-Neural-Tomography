"""Phase 3 Training: Inverse Model (Sound -> Geometry).

Trains an auto-decoder SDF model to reconstruct scene geometry from
acoustic pressure observations, using a frozen Phase 2 forward model
for cycle-consistency.

Training Stages
---------------
    Stage 1 (epochs 0-200):    SDF supervision + Eikonal constraint
    Stage 2 (epochs 200-500):  + Cycle-consistency (ramped)
    Stage 3 (epochs 500-1000): + Helmholtz PDE residual

Loss
----
    L = w_sdf * L_sdf + w_eik * L_eikonal + w_cyc * L_cycle
        + w_helm * L_helmholtz + w_z * ||z||^2

Gate criterion
--------------
    SDF IoU > 0.8 AND Helmholtz residual < 1e-3
    (evaluated by scripts/eval_phase3.py)

Usage
-----
    python scripts/run_phase3.py                          # full training
    python scripts/run_phase3.py --epochs 200             # Stage 1 only
    python scripts/run_phase3.py --resume --epochs 1000   # resume
    python scripts/run_phase3.py --forward-ckpt best_v11  # specific forward model
"""

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.optim import AdamW

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.forward_model import TransferFunctionModel, build_transfer_model
from src.inverse_dataset import load_all_scenes
from src.inverse_model import (
    InverseModel,
    build_inverse_model,
    compute_sdf_iou,
    cycle_consistency_loss,
    eikonal_loss,
    helmholtz_residual,
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
logger = logging.getLogger("phase3_train")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "phase1"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase3"
FWD_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase2"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase3"

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
STAGE_BOUNDARIES = (200, 500)  # Stage 1->2 at epoch 200, Stage 2->3 at epoch 500

DEFAULTS = {
    "epochs": 1000,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "lr_warmup_epochs": 10,
    "patience": 200,
    "checkpoint_every": 50,
    "seed": 42,
    # Architecture
    "d_cond": 256,
    "d_hidden": 256,
    "n_blocks": 6,
    "n_fourier": 128,
    "fourier_sigma": 10.0,
    "dropout": 0.05,
    # Batch sizes
    "n_sdf_batch": 4096,
    "n_cycle_batch": 2048,
    "n_helmholtz_batch": 256,
    # Loss weights
    "w_sdf": 1.0,
    "w_eikonal": 0.1,
    "w_cycle": 0.01,
    "w_helmholtz": 1e-4,
    "w_z_reg": 1e-3,
    # Stage 2 ramp
    "cycle_ramp_epochs": 50,
}


# ---------------------------------------------------------------------------
# Load frozen forward model
# ---------------------------------------------------------------------------
def load_frozen_forward(
    ckpt_name: str,
    device: torch.device,
) -> tuple:
    """Load Phase 2 forward model in frozen eval mode.

    Returns
    -------
    model : TransferFunctionModel
    scene_scales : dict
    trained_scene_list : list
    """
    ckpt_path = FWD_CHECKPOINT_DIR / f"{ckpt_name}.pt"
    if not ckpt_path.exists():
        logger.error("Forward checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = build_transfer_model(
        d_hidden=cfg.get("d_hidden", cfg.get("hidden_dim", 768)),
        n_blocks=cfg.get("n_blocks", cfg.get("n_layers", 6)),
        n_fourier=cfg.get("n_fourier", 256),
        fourier_sigma=cfg.get("fourier_sigma", 30.0),
        dropout=cfg.get("dropout", 0.0),
        n_scenes=cfg.get("n_scenes", 0),
        scene_emb_dim=cfg.get("scene_emb_dim", 32),
        d_out=cfg.get("d_out", 2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    scene_scales = ckpt["scene_scales"]
    trained_scene_list = cfg.get(
        "trained_scene_list", sorted(scene_scales.keys())
    )

    logger.info(
        "Frozen forward model: %s (epoch %d, %.2f%% val_loss)",
        ckpt_path.name,
        ckpt["epoch"],
        ckpt["best_val_loss"],
    )

    return model, scene_scales, trained_scene_list


# ---------------------------------------------------------------------------
# Per-scene training step
# ---------------------------------------------------------------------------
def train_scene_step(
    inverse_model: InverseModel,
    forward_model: TransferFunctionModel,
    scene_data,
    scene_idx: int,
    device: torch.device,
    epoch: int,
    config: dict,
) -> Dict[str, float]:
    """One training step for a single scene.

    Returns dict of loss components.
    """
    n_sdf = config["n_sdf_batch"]
    n_cycle = config["n_cycle_batch"]
    n_helm = config["n_helmholtz_batch"]
    sd = scene_data

    losses = {}
    total_loss = torch.tensor(0.0, device=device)

    # ------------------------------------------------------------------
    # 1) SDF loss + Eikonal loss (all stages)
    # ------------------------------------------------------------------
    # Sample grid points (with optional boundary oversampling)
    n_grid = sd.n_grid
    bdy_factor = config.get("boundary_oversample", 1.0)

    if bdy_factor > 1.0:
        # Stratified sampling: oversample near boundary (|SDF| < 0.1m)
        bdy_mask = np.abs(sd.sdf_flat) < 0.1
        bdy_idx_np = np.where(bdy_mask)[0]
        far_idx_np = np.where(~bdy_mask)[0]

        n_bdy = min(
            int(n_sdf * bdy_factor / (1.0 + bdy_factor)),
            len(bdy_idx_np),
        )
        n_far = min(n_sdf - n_bdy, len(far_idx_np))

        if n_bdy > 0 and n_far > 0:
            chosen_bdy = np.random.choice(bdy_idx_np, n_bdy, replace=False)
            chosen_far = np.random.choice(far_idx_np, n_far, replace=False)
            sdf_idx_np = np.concatenate([chosen_bdy, chosen_far])
            sdf_idx = torch.from_numpy(sdf_idx_np).long().to(device)
        else:
            sdf_idx = torch.randint(n_grid, (min(n_sdf, n_grid),), device=device)
    else:
        sdf_idx = torch.randint(n_grid, (min(n_sdf, n_grid),), device=device)

    xy_m = torch.from_numpy(sd.grid_coords).float().to(device)  # (G, 2)
    sdf_gt_all = torch.from_numpy(sd.sdf_flat).float().to(device)  # (G,)

    xy_batch = xy_m[sdf_idx].clone().requires_grad_(True)  # (B_sdf, 2)
    sdf_gt_batch = sdf_gt_all[sdf_idx].unsqueeze(-1)  # (B_sdf, 1)

    sdf_pred = inverse_model.predict_sdf(scene_idx, xy_batch)  # (B_sdf, 1)

    # SDF weight schedule: 1.0 in Stage 1, 0.5 in Stage 2+
    w_sdf = config["w_sdf"] if epoch < STAGE_BOUNDARIES[0] else 0.5
    l_sdf = sdf_loss(sdf_pred, sdf_gt_batch)
    total_loss = total_loss + w_sdf * l_sdf
    losses["sdf"] = l_sdf.item()

    # Eikonal
    l_eik = eikonal_loss(sdf_pred, xy_batch)
    total_loss = total_loss + config["w_eikonal"] * l_eik
    losses["eikonal"] = l_eik.item()

    # ------------------------------------------------------------------
    # 2) Cycle-consistency loss (Stage 2+)
    # ------------------------------------------------------------------
    if epoch >= STAGE_BOUNDARIES[0]:
        # Ramp cycle weight over cycle_ramp_epochs
        ramp_start = STAGE_BOUNDARIES[0]
        ramp_end = ramp_start + config["cycle_ramp_epochs"]
        if epoch < ramp_end:
            ramp = (epoch - ramp_start) / config["cycle_ramp_epochs"]
        else:
            ramp = 1.0
        w_cycle = config["w_cycle"] * ramp

        # Sample random observations: (src_idx, freq_idx, rcv_idx)
        n_obs = sd.n_observations
        obs_idx = torch.randint(n_obs, (min(n_cycle, n_obs),))
        n_fr = sd.n_freqs * sd.n_receivers
        si_arr = (obs_idx // n_fr).numpy()
        fi_arr = ((obs_idx % n_fr) // sd.n_receivers).numpy()
        ri_arr = (obs_idx % sd.n_receivers).numpy()

        # Gather observation data
        x_src_np = sd.src_pos[si_arr]  # (B_cyc, 2)
        x_rcv_np = sd.rcv_pos[ri_arr]  # (B_cyc, 2)
        k_np = sd.k_arr[fi_arr]  # (B_cyc,)
        p_gt_np = sd.pressure[si_arr, fi_arr, ri_arr]  # (B_cyc,) complex

        x_src_t = torch.from_numpy(x_src_np).float().to(device)
        x_rcv_t = torch.from_numpy(x_rcv_np).float().to(device)
        k_t = torch.from_numpy(k_np).float().to(device).unsqueeze(-1)  # (B, 1)
        p_gt_re = torch.from_numpy(p_gt_np.real.copy()).float().to(device)
        p_gt_im = torch.from_numpy(p_gt_np.imag.copy()).float().to(device)

        fwd_ids = torch.full(
            (len(si_arr),), sd.fwd_scene_idx,
            dtype=torch.long, device=device,
        )

        l_cycle = cycle_consistency_loss(
            inverse_model, forward_model, scene_idx,
            x_src_t, x_rcv_t, k_t, p_gt_re, p_gt_im,
            sd.scene_scale, fwd_ids,
        )

        if torch.isfinite(l_cycle):
            total_loss = total_loss + w_cycle * l_cycle
            losses["cycle"] = l_cycle.item()
        else:
            losses["cycle"] = float("nan")

    # ------------------------------------------------------------------
    # 3) Helmholtz residual (Stage 3+, skipped when no_helmholtz)
    # ------------------------------------------------------------------
    if epoch >= STAGE_BOUNDARIES[1] and not config.get("no_helmholtz", False):
        w_helm = config["w_helmholtz"]

        # Sample exterior points where SDF > 0.05 m
        exterior_mask = sd.sdf_flat > 0.05
        exterior_idx = np.where(exterior_mask)[0]

        if len(exterior_idx) >= n_helm:
            helm_idx = np.random.choice(
                exterior_idx, size=n_helm, replace=False,
            )

            xy_helm = torch.from_numpy(
                sd.grid_coords[helm_idx]
            ).float().to(device)  # (B_helm, 2)

            # Random source and frequency per point
            si_helm = np.random.randint(0, sd.n_sources, size=n_helm)
            fi_helm = np.random.randint(0, sd.n_freqs, size=n_helm)

            x_src_helm = torch.from_numpy(
                sd.src_pos[si_helm]
            ).float().to(device)  # (B_helm, 2)
            k_helm = torch.from_numpy(
                sd.k_arr[fi_helm]
            ).float().to(device).unsqueeze(-1)  # (B_helm, 1)

            fwd_ids_helm = torch.full(
                (n_helm,), sd.fwd_scene_idx,
                dtype=torch.long, device=device,
            )

            z_code = inverse_model.get_code(scene_idx)

            l_helm = helmholtz_residual(
                forward_model, inverse_model.sdf_decoder,
                z_code, x_src_helm, xy_helm, k_helm,
                sd.scene_scale, fwd_ids_helm,
            )

            if torch.isfinite(l_helm):
                total_loss = total_loss + w_helm * l_helm
                losses["helmholtz"] = l_helm.item()
            else:
                losses["helmholtz"] = float("nan")

    # ------------------------------------------------------------------
    # 4) Latent code regularization (all stages)
    # ------------------------------------------------------------------
    z = inverse_model.get_code(scene_idx)
    # z is (d_cond,) for K=1, (K, d_cond) for K>1
    l_z = (z ** 2).mean()
    total_loss = total_loss + config["w_z_reg"] * l_z
    losses["z_reg"] = l_z.item()
    losses["total"] = total_loss.item()

    return total_loss, losses


# ---------------------------------------------------------------------------
# IoU validation
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate_iou(
    inverse_model: InverseModel,
    all_scenes: dict,
    device: torch.device,
) -> Dict[int, float]:
    """Compute per-scene SDF IoU.

    Returns dict scene_id -> IoU.
    """
    inverse_model.eval()
    ious = {}

    for sid, sd in sorted(all_scenes.items()):
        scene_idx = list(all_scenes.keys()).index(sid)
        xy_m = torch.from_numpy(sd.grid_coords).float().to(device)
        sdf_gt = torch.from_numpy(sd.sdf_flat).float().to(device)

        # Predict in chunks to avoid VRAM overflow
        chunk_size = 8192
        sdf_preds = []
        for i in range(0, len(xy_m), chunk_size):
            chunk = xy_m[i : i + chunk_size]
            pred = inverse_model.predict_sdf(scene_idx, chunk)
            sdf_preds.append(pred.squeeze(-1))

        sdf_pred_flat = torch.cat(sdf_preds, dim=0)
        iou = compute_sdf_iou(sdf_pred_flat, sdf_gt)
        ious[sid] = iou

    inverse_model.train()
    return ious


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: Path,
    inverse_model: InverseModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_mean_iou: float,
    config: dict,
) -> None:
    """Save Phase 3 training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": inverse_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_mean_iou": best_mean_iou,
            "config": config,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 Inverse Model Training"
    )
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument(
        "--forward-ckpt", type=str, default="best_v7",
        help="Phase 2 forward model checkpoint name (without .pt)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from latest checkpoint",
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="Tag for checkpoint naming: best_{tag}.pt",
    )
    parser.add_argument("--scenes", nargs="+", type=int, default=None)
    parser.add_argument(
        "--no-helmholtz", action="store_true",
        help="Disable Helmholtz loss entirely (stay in Stage 2 after epoch 500)",
    )
    parser.add_argument(
        "--boundary-oversample", type=float, default=1.0,
        help="Oversample factor for boundary points (|SDF|<0.1m). "
             "E.g. 3.0 = 3x more boundary points per batch.",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Resume from specific checkpoint name (without .pt), e.g. best_phase3",
    )
    parser.add_argument(
        "--multi-body", type=str, default=None,
        help="Multi-body scene specs: 'scene_id:K,...' (e.g., '12:2' for S12 with 2 codes)",
    )
    # Architecture
    parser.add_argument("--d-cond", type=int, default=DEFAULTS["d_cond"])
    parser.add_argument("--d-hidden", type=int, default=DEFAULTS["d_hidden"])
    parser.add_argument("--n-blocks", type=int, default=DEFAULTS["n_blocks"])
    # Batch sizes
    parser.add_argument("--n-sdf-batch", type=int, default=DEFAULTS["n_sdf_batch"])
    parser.add_argument("--n-cycle-batch", type=int, default=DEFAULTS["n_cycle_batch"])
    parser.add_argument("--n-helmholtz-batch", type=int, default=DEFAULTS["n_helmholtz_batch"])
    # Loss weights
    parser.add_argument("--w-cycle", type=float, default=DEFAULTS["w_cycle"])
    parser.add_argument("--w-helmholtz", type=float, default=DEFAULTS["w_helmholtz"])
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # ------------------------------------------------------------------
    # Load frozen forward model
    # ------------------------------------------------------------------
    forward_model, scene_scales, fwd_scene_list = load_frozen_forward(
        args.forward_ckpt, device,
    )
    fwd_scene_id_map = {sid: idx for idx, sid in enumerate(fwd_scene_list)}

    # ------------------------------------------------------------------
    # Load scene data
    # ------------------------------------------------------------------
    logger.info("Loading scene data...")
    t_load = time.time()
    all_scenes = load_all_scenes(
        DATA_DIR, scene_scales, fwd_scene_id_map,
        scene_ids=args.scenes,
    )
    logger.info("Loaded %d scenes in %.1fs", len(all_scenes), time.time() - t_load)

    if len(all_scenes) == 0:
        logger.error("No scenes loaded, exiting")
        sys.exit(1)

    scene_ids_sorted = sorted(all_scenes.keys())
    n_scenes = len(scene_ids_sorted)
    # Build inverse model scene indexing: sorted scene_ids -> 0-indexed
    inv_scene_id_map = {sid: idx for idx, sid in enumerate(scene_ids_sorted)}

    # ------------------------------------------------------------------
    # Parse multi-body scene specs
    # ------------------------------------------------------------------
    multi_body_scene_ids: Optional[Dict[int, int]] = None
    if args.multi_body:
        multi_body_scene_ids = {}
        for spec in args.multi_body.split(","):
            sid_str, k_str = spec.strip().split(":")
            multi_body_scene_ids[int(sid_str)] = int(k_str)
        logger.info("Multi-body scenes: %s", multi_body_scene_ids)

    # ------------------------------------------------------------------
    # Build inverse model
    # ------------------------------------------------------------------
    inverse_model = build_inverse_model(
        n_scenes=n_scenes,
        d_cond=args.d_cond,
        d_hidden=args.d_hidden,
        n_blocks=args.n_blocks,
        n_fourier=DEFAULTS["n_fourier"],
        fourier_sigma=DEFAULTS["fourier_sigma"],
        dropout=DEFAULTS["dropout"],
        multi_body_scene_ids=multi_body_scene_ids,
        inv_scene_id_map=inv_scene_id_map,
    ).to(device)

    n_params = inverse_model.count_parameters()
    logger.info("Inverse model: %d trainable parameters", n_params)

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    optimizer = AdamW(
        inverse_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    lr_warmup = DEFAULTS["lr_warmup_epochs"]
    no_helmholtz = args.no_helmholtz

    def lr_lambda(epoch: int) -> float:
        if epoch < lr_warmup:
            return (epoch + 1) / lr_warmup
        # Reduce LR at Stage 3 only when Helmholtz is active
        if epoch >= STAGE_BOUNDARIES[1] and not no_helmholtz:
            base_lr_scale = 0.2  # 5x reduction
        else:
            base_lr_scale = 1.0
        progress = (epoch - lr_warmup) / max(args.epochs - lr_warmup, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(1e-6 / args.lr, base_lr_scale * cosine)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    config = {
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "forward_checkpoint": args.forward_ckpt,
        "d_cond": args.d_cond,
        "d_hidden": args.d_hidden,
        "n_blocks": args.n_blocks,
        "n_fourier": DEFAULTS["n_fourier"],
        "fourier_sigma": DEFAULTS["fourier_sigma"],
        "dropout": DEFAULTS["dropout"],
        "n_scenes": n_scenes,
        "scene_ids": scene_ids_sorted,
        "inv_scene_id_map": inv_scene_id_map,
        "fwd_scene_id_map": fwd_scene_id_map,
        "fwd_scene_scales": dict(scene_scales),
        "n_sdf_batch": args.n_sdf_batch,
        "n_cycle_batch": args.n_cycle_batch,
        "n_helmholtz_batch": args.n_helmholtz_batch,
        "w_sdf": DEFAULTS["w_sdf"],
        "w_eikonal": DEFAULTS["w_eikonal"],
        "w_cycle": args.w_cycle,
        "w_helmholtz": args.w_helmholtz,
        "w_z_reg": DEFAULTS["w_z_reg"],
        "cycle_ramp_epochs": DEFAULTS["cycle_ramp_epochs"],
        "stage_boundaries": STAGE_BOUNDARIES,
        "no_helmholtz": args.no_helmholtz,
        "boundary_oversample": args.boundary_oversample,
        "multi_body_scene_ids": multi_body_scene_ids,
    }

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    tag_suffix = f"_{args.tag}" if args.tag else ""
    best_ckpt_name = f"best_phase3{tag_suffix}.pt"
    latest_ckpt_name = f"latest_phase3{tag_suffix}.pt"

    start_epoch = 0
    best_mean_iou = 0.0
    patience_counter = 0

    if args.resume_from or args.resume:
        if args.resume_from:
            ckpt_path = CHECKPOINT_DIR / f"{args.resume_from}.pt"
        else:
            ckpt_path = CHECKPOINT_DIR / latest_ckpt_name
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            # Use compat loader to handle old->new code table remapping
            inverse_model.load_state_dict_compat(ckpt["model_state_dict"])
            # Only load optimizer/scheduler if code table size matches
            old_codes = ckpt["model_state_dict"].get("auto_decoder_codes.weight")
            new_codes = inverse_model.auto_decoder_codes.weight
            if old_codes is not None and old_codes.shape[0] == new_codes.shape[0]:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            else:
                logger.info(
                    "Code table size changed (%d -> %d), resetting optimizer",
                    old_codes.shape[0] if old_codes is not None else 0,
                    new_codes.shape[0],
                )
            start_epoch = ckpt["epoch"] + 1
            best_mean_iou = ckpt["best_mean_iou"]
            logger.info(
                "Resumed from %s epoch %d (best IoU=%.4f)",
                ckpt_path.name, start_epoch, best_mean_iou,
            )
        else:
            logger.warning("No checkpoint at %s -- starting fresh", ckpt_path)

    # ------------------------------------------------------------------
    # Metrics CSV
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / "training_metrics.csv"
    if not metrics_path.exists() or (not args.resume):
        header = "epoch,stage,total_loss,sdf_loss,eikonal_loss"
        header += ",cycle_loss,helmholtz_loss,z_reg_loss"
        header += ",mean_iou,lr,time_s"
        with open(metrics_path, "w") as fh:
            fh.write(header + "\n")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    logger.info("Starting training from epoch %d ...", start_epoch)
    t0_total = time.time()

    # Logging schedule: every 10 epochs + milestones
    log_epochs = set(
        list(range(5))
        + list(range(9, args.epochs, 10))
        + [STAGE_BOUNDARIES[0], STAGE_BOUNDARIES[1]]
        + [args.epochs - 1]
    )

    for epoch in range(start_epoch, args.epochs):
        t0_ep = time.time()
        inverse_model.train()

        # Determine current stage
        if epoch < STAGE_BOUNDARIES[0]:
            stage = 1
        elif epoch < STAGE_BOUNDARIES[1] or config["no_helmholtz"]:
            stage = 2
        else:
            stage = 3

        # Shuffle scene order
        scene_order = list(scene_ids_sorted)
        np.random.shuffle(scene_order)

        # Accumulate losses across scenes
        epoch_losses: Dict[str, List[float]] = {
            "total": [], "sdf": [], "eikonal": [],
            "cycle": [], "helmholtz": [], "z_reg": [],
        }

        for sid in scene_order:
            sd = all_scenes[sid]
            scene_idx = inv_scene_id_map[sid]

            optimizer.zero_grad(set_to_none=True)

            total_loss, step_losses = train_scene_step(
                inverse_model, forward_model, sd, scene_idx,
                device, epoch, config,
            )

            if not torch.isfinite(total_loss):
                logger.warning(
                    "Non-finite loss at epoch %d scene %d, skipping",
                    epoch, sid,
                )
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                inverse_model.parameters(), max_norm=1.0,
            )
            optimizer.step()

            # Accumulate
            for key in epoch_losses:
                if key in step_losses:
                    val = step_losses[key]
                    if np.isfinite(val):
                        epoch_losses[key].append(val)

            # Free GPU memory between scenes
            if device.type == "cuda":
                torch.cuda.empty_cache()

        scheduler.step()

        # Average losses
        avg = {}
        for key, vals in epoch_losses.items():
            avg[key] = np.mean(vals) if vals else float("nan")

        # ------------------------------------------------------------------
        # Validation: IoU every 10 epochs
        # ------------------------------------------------------------------
        mean_iou = 0.0
        ious_str = ""
        if (epoch + 1) % 10 == 0 or epoch in log_epochs:
            ious = validate_iou(inverse_model, all_scenes, device)
            mean_iou = np.mean(list(ious.values()))
            ious_str = " ".join(
                f"S{sid}={iou:.3f}" for sid, iou in sorted(ious.items())
            )

        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0_ep

        # Log
        if epoch in log_epochs or (epoch + 1) % 10 == 0:
            logger.info(
                "Ep %4d [S%d]: loss=%.4e sdf=%.4e eik=%.4e "
                "cyc=%.4e helm=%.4e IoU=%.4f lr=%.2e %.1fs",
                epoch + 1, stage,
                avg["total"], avg["sdf"], avg["eikonal"],
                avg.get("cycle", float("nan")),
                avg.get("helmholtz", float("nan")),
                mean_iou, lr_now, elapsed,
            )
            if ious_str:
                logger.info("  IoU: %s", ious_str)

        # Save metrics
        with open(metrics_path, "a") as fh:
            fh.write(
                f"{epoch},{stage},{avg['total']:.6e},{avg['sdf']:.6e},"
                f"{avg['eikonal']:.6e},{avg.get('cycle', float('nan')):.6e},"
                f"{avg.get('helmholtz', float('nan')):.6e},"
                f"{avg['z_reg']:.6e},{mean_iou:.6f},{lr_now:.6e},{elapsed:.1f}\n"
            )

        # Periodic checkpoint
        if (epoch + 1) % DEFAULTS["checkpoint_every"] == 0:
            save_checkpoint(
                CHECKPOINT_DIR / latest_ckpt_name,
                inverse_model, optimizer, scheduler,
                epoch, best_mean_iou, config,
            )

        # Best model (by mean IoU, computed every 10 epochs)
        if mean_iou > best_mean_iou and mean_iou > 0:
            best_mean_iou = mean_iou
            patience_counter = 0
            save_checkpoint(
                CHECKPOINT_DIR / best_ckpt_name,
                inverse_model, optimizer, scheduler,
                epoch, best_mean_iou, config,
            )
            logger.info(
                "  ** New best: IoU=%.4f (saved %s)", best_mean_iou, best_ckpt_name,
            )
        elif mean_iou > 0:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            logger.info(
                "Early stopping at epoch %d (patience=%d)", epoch + 1, args.patience,
            )
            break

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    total_time = time.time() - t0_total
    logger.info("")
    logger.info("Done in %.1f min", total_time / 60)
    logger.info("Best mean IoU: %.4f", best_mean_iou)
    logger.info("Saved: %s", CHECKPOINT_DIR / best_ckpt_name)


if __name__ == "__main__":
    main()
