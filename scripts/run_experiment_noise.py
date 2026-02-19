"""Experiment C: Noise Robustness of Cycle-Consistency.

Tests how observation noise affects the cycle-consistency evaluation.
No model retraining -- noise is injected into BEM pressure observations
at test time.

Method
------
    For each SNR level:
    1. Inject complex Gaussian noise into BEM pressure: p_noisy = p + n
       where n ~ CN(0, sigma^2), sigma = ||p|| / (R * sqrt(10^(SNR/10)))
    2. Re-run cycle-consistency evaluation (eval_phase4 logic)
    3. Record Pearson r per scene

SNR Levels
----------
    {10, 20, 30, 40} dB + clean (inf)

Output
------
    results/experiments/noise_robustness.csv

Usage
-----
    python scripts/run_experiment_noise.py
    python scripts/run_experiment_noise.py --snr 10 20 30 40
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.special import hankel1

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.forward_model import build_transfer_model
from src.inverse_dataset import InverseSceneData, load_all_scenes
from src.inverse_model import InverseModel, build_inverse_model, compute_sdf_iou

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiment_noise")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "phase1"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase3"
FWD_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase2"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"

# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
SPEED_OF_SOUND_M_PER_S: float = 343.0
SEED: int = 42


# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------
def inject_noise(
    pressure: np.ndarray,
    snr_db: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Inject complex Gaussian noise at specified SNR.

    SNR = ||p||^2 / (N * sigma^2), where N is total number of elements.

    Parameters
    ----------
    pressure : np.ndarray, complex128
        Clean BEM pressure field (any shape).
    snr_db : float
        Signal-to-noise ratio [dB].
    rng : np.random.RandomState
        Random number generator for reproducibility.

    Returns
    -------
    noisy : np.ndarray, complex128
        Noisy pressure field (same shape).
    """
    signal_power = float(np.mean(np.abs(pressure) ** 2))
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    sigma = np.sqrt(noise_power / 2.0)  # /2 for complex (Re + Im)

    noise = sigma * (
        rng.randn(*pressure.shape) + 1j * rng.randn(*pressure.shape)
    )
    return pressure + noise


def verify_snr(
    clean: np.ndarray,
    noisy: np.ndarray,
    target_snr_db: float,
    tolerance_db: float = 1.0,
) -> float:
    """Verify actual SNR matches target within tolerance.

    Returns actual SNR in dB.
    """
    noise = noisy - clean
    signal_power = float(np.mean(np.abs(clean) ** 2))
    noise_power = float(np.mean(np.abs(noise) ** 2))

    if noise_power < 1e-30:
        return float("inf")

    actual_snr_db = 10.0 * np.log10(signal_power / noise_power)
    if abs(actual_snr_db - target_snr_db) > tolerance_db:
        logger.warning(
            "SNR mismatch: target=%.1f dB, actual=%.1f dB",
            target_snr_db, actual_snr_db,
        )
    return actual_snr_db


# ---------------------------------------------------------------------------
# Exact incident field (reused from eval_phase4)
# ---------------------------------------------------------------------------
def compute_p_inc_exact(
    x_src: np.ndarray,
    x_rcv: np.ndarray,
    k: np.ndarray,
) -> np.ndarray:
    """Exact 2D incident field: p_inc = -(i/4) H_0^{(1)}(kr)."""
    dx = x_rcv[:, 0] - x_src[:, 0]
    dy = x_rcv[:, 1] - x_src[:, 1]
    r = np.sqrt(dx ** 2 + dy ** 2)  # (B,)

    if not np.all(np.isfinite(r)):
        raise ValueError(f"Non-finite distances: {np.sum(~np.isfinite(r))}")
    if np.any(r < 1e-10):
        raise ValueError("Source-receiver distance near zero")

    kr = k * r
    return -0.25j * hankel1(0, kr)


# ---------------------------------------------------------------------------
# Cycle evaluation with noise
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_cycle_with_noise(
    inverse_model: InverseModel,
    forward_model: torch.nn.Module,
    scene_data: InverseSceneData,
    scene_idx: int,
    device: torch.device,
    snr_db: Optional[float] = None,
    rng: Optional[np.random.RandomState] = None,
    freq_chunk_size: int = 20,
) -> Dict:
    """Evaluate cycle-consistency, optionally with noisy observations.

    Parameters
    ----------
    snr_db : float or None
        If None, use clean observations.
    rng : RandomState
        For noise generation reproducibility.
    """
    sd = scene_data
    S = sd.n_sources
    F = sd.n_freqs
    R = sd.n_receivers

    # Optionally inject noise
    if snr_db is not None and rng is not None:
        pressure = inject_noise(sd.pressure, snr_db, rng)
        actual_snr = verify_snr(sd.pressure, pressure, snr_db)
    else:
        pressure = sd.pressure.copy()
        actual_snr = float("inf")

    # 1. Predict SDF at receivers
    rcv_t = torch.from_numpy(sd.rcv_pos).float().to(device)
    sdf_rcv = inverse_model.predict_sdf(scene_idx, rcv_t)  # (R, 1)

    # 2. Reconstruct pressure
    p_pred_all = np.zeros((S, F, R), dtype=np.complex128)
    p_gt_all = pressure.copy()  # noisy or clean

    fwd_scene_ids = torch.full(
        (R,), sd.fwd_scene_idx, dtype=torch.long, device=device,
    )

    for si in range(S):
        x_src_np = np.tile(sd.src_pos[si], (R, 1))
        x_src_t = torch.from_numpy(x_src_np).float().to(device)

        for fi_start in range(0, F, freq_chunk_size):
            fi_end = min(fi_start + freq_chunk_size, F)
            n_f = fi_end - fi_start

            x_src_batch = x_src_t.unsqueeze(0).expand(n_f, -1, -1).reshape(n_f * R, 2)
            x_rcv_batch = rcv_t.unsqueeze(0).expand(n_f, -1, -1).reshape(n_f * R, 2)
            sdf_batch = sdf_rcv.unsqueeze(0).expand(n_f, -1, -1).reshape(n_f * R, 1)

            k_chunk = sd.k_arr[fi_start:fi_end]
            k_batch = torch.from_numpy(
                np.repeat(k_chunk, R)
            ).float().to(device).unsqueeze(-1)

            fwd_ids_batch = fwd_scene_ids.unsqueeze(0).expand(n_f, -1).reshape(n_f * R)

            T_pred = forward_model.forward_from_coords(
                x_src_batch, x_rcv_batch, k_batch, sdf_batch,
                scene_ids=fwd_ids_batch,
            )  # (n_f*R, 2)

            T_re = T_pred[:, 0].cpu().numpy().reshape(n_f, R)
            T_im = T_pred[:, 1].cpu().numpy().reshape(n_f, R)
            T_complex = (T_re + 1j * T_im) * sd.scene_scale

            for fi_local in range(n_f):
                fi_global = fi_start + fi_local
                k_val = sd.k_arr[fi_global]
                k_arr_r = np.full(R, k_val)

                p_inc = compute_p_inc_exact(x_src_np, sd.rcv_pos, k_arr_r)
                p_pred_all[si, fi_global, :] = p_inc * (1.0 + T_complex[fi_local])

    # 3. Compute Pearson r (p_pred vs p_gt which may be noisy)
    p_pred_flat = p_pred_all.ravel()
    p_gt_flat = p_gt_all.ravel()

    pred_vec = np.concatenate([p_pred_flat.real, p_pred_flat.imag])
    gt_vec = np.concatenate([p_gt_flat.real, p_gt_flat.imag])
    r_pearson = float(np.corrcoef(pred_vec, gt_vec)[0, 1])

    # Relative L2
    rel_l2 = float(
        np.sqrt(np.sum(np.abs(p_pred_flat - p_gt_flat) ** 2))
        / np.sqrt(np.sum(np.abs(p_gt_flat) ** 2))
    )

    return {
        "r_pearson": r_pearson,
        "rel_l2": rel_l2,
        "actual_snr_db": actual_snr,
        "n_observations": S * F * R,
    }


# ---------------------------------------------------------------------------
# Model loading (reused from eval_phase4)
# ---------------------------------------------------------------------------
def load_models(
    checkpoint_name: str,
    forward_ckpt: Optional[str],
    device: torch.device,
):
    """Load inverse and forward models."""
    ckpt_path = CHECKPOINT_DIR / f"{checkpoint_name}.pt"
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    inv_scene_id_map = {int(k): v for k, v in cfg["inv_scene_id_map"].items()}
    inverse_model = build_inverse_model(
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
    inverse_model.load_state_dict(ckpt["model_state_dict"])
    inverse_model = inverse_model.to(device)
    inverse_model.eval()

    # Forward model
    fwd_ckpt_name = forward_ckpt or cfg["forward_checkpoint"]
    fwd_ckpt_path = FWD_CHECKPOINT_DIR / f"{fwd_ckpt_name}.pt"
    fwd_ckpt = torch.load(fwd_ckpt_path, map_location=device, weights_only=False)
    fwd_cfg = fwd_ckpt["config"]

    forward_model = build_transfer_model(
        d_hidden=fwd_cfg.get("d_hidden", 768),
        n_blocks=fwd_cfg.get("n_blocks", 6),
        n_fourier=fwd_cfg.get("n_fourier", 256),
        fourier_sigma=fwd_cfg.get("fourier_sigma", 30.0),
        dropout=fwd_cfg.get("dropout", 0.0),
        n_scenes=fwd_cfg.get("n_scenes", 0),
        scene_emb_dim=fwd_cfg.get("scene_emb_dim", 32),
        d_out=fwd_cfg.get("d_out", 2),
    )
    forward_model.load_state_dict(fwd_ckpt["model_state_dict"])
    forward_model = forward_model.to(device)
    forward_model.eval()
    for p in forward_model.parameters():
        p.requires_grad = False

    scene_scales = fwd_ckpt["scene_scales"]
    fwd_scene_list = fwd_cfg.get("trained_scene_list", sorted(scene_scales.keys()))
    fwd_scene_id_map = {sid: idx for idx, sid in enumerate(fwd_scene_list)}

    return (
        inverse_model, forward_model,
        inv_scene_id_map, scene_scales, fwd_scene_id_map,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment C: Noise Robustness"
    )
    parser.add_argument("--checkpoint", type=str, default="best_phase3_v3")
    parser.add_argument("--forward-ckpt", type=str, default=None)
    parser.add_argument(
        "--snr", nargs="+", type=float, default=[10, 20, 30, 40],
        help="SNR levels in dB",
    )
    parser.add_argument("--scenes", nargs="+", type=int, default=None)
    parser.add_argument("--freq-chunk", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load models
    (
        inverse_model, forward_model,
        inv_scene_id_map, scene_scales, fwd_scene_id_map,
    ) = load_models(args.checkpoint, args.forward_ckpt, device)

    # Load scene data
    all_scenes = load_all_scenes(
        DATA_DIR, scene_scales, fwd_scene_id_map,
        scene_ids=args.scenes,
    )
    logger.info("Loaded %d scenes", len(all_scenes))

    # SNR levels: clean + specified
    snr_levels: List[Optional[float]] = [None] + [float(s) for s in args.snr]
    snr_labels = ["clean"] + [f"{s:.0f}" for s in args.snr]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Results storage: {snr_label: {scene_id: r_pearson}}
    all_results: Dict[str, Dict[int, float]] = {}

    t0 = time.time()

    for snr_val, snr_label in zip(snr_levels, snr_labels):
        rng = np.random.RandomState(SEED)
        logger.info("=" * 50)
        logger.info("SNR: %s dB", snr_label)
        logger.info("=" * 50)

        scene_r_vals: Dict[int, float] = {}

        for sid in sorted(all_scenes.keys()):
            sd = all_scenes[sid]
            scene_idx = inv_scene_id_map[sid]

            results = evaluate_cycle_with_noise(
                inverse_model, forward_model, sd, scene_idx, device,
                snr_db=snr_val, rng=rng,
                freq_chunk_size=args.freq_chunk,
            )

            scene_r_vals[sid] = results["r_pearson"]
            logger.info(
                "  S%02d: r=%.4f (actual SNR=%.1f dB)",
                sid, results["r_pearson"], results["actual_snr_db"],
            )

        mean_r = float(np.mean(list(scene_r_vals.values())))
        logger.info("  Mean r = %.4f (SNR=%s dB)", mean_r, snr_label)
        all_results[snr_label] = scene_r_vals

    elapsed = time.time() - t0
    logger.info("Total time: %.1f min", elapsed / 60)

    # Write CSV
    sids = sorted(all_scenes.keys())
    csv_path = RESULTS_DIR / "noise_robustness.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        header = ["snr_db", "mean_r"] + [f"S{sid}" for sid in sids]
        writer.writerow(header)

        for snr_label in snr_labels:
            r_vals = all_results[snr_label]
            mean_r = float(np.mean(list(r_vals.values())))
            row = [snr_label, f"{mean_r:.4f}"]
            row += [f"{r_vals[sid]:.4f}" for sid in sids]
            writer.writerow(row)

    logger.info("Results CSV: %s", csv_path)

    # Print summary table
    print("\n" + "=" * 70)
    print("Experiment C: Noise Robustness")
    print("=" * 70)
    print(f"{'SNR (dB)':<10} {'Mean r':>8}", end="")
    for sid in sids[:5]:
        print(f"  S{sid:>2}", end="")
    print("  ...")
    print("-" * 50)
    for snr_label in snr_labels:
        r_vals = all_results[snr_label]
        mean_r = float(np.mean(list(r_vals.values())))
        print(f"{snr_label:<10} {mean_r:>8.4f}", end="")
        for sid in sids[:5]:
            print(f"  {r_vals[sid]:.2f}", end="")
        print()
    print("=" * 70)


if __name__ == "__main__":
    main()
