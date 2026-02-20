"""Inference Speed & T Dynamic Range Analysis.

Part 1 -- Inference speed: BEM vs Neural forward model.
    BEM:    65 min / 15 scenes (from Phase 1 data factory)
    Neural: batch inference wall-clock measurement

Part 2 -- T formulation dynamic range reduction.
    Compare |p_scat| vs |T| amplitude ranges in dB.

Usage
-----
    python scripts/measure_inference_speed.py
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict

import csv
import h5py
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.special import hankel1

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.forward_model import build_transfer_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("speed_analysis")

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
BEM_TOTAL_TIME_S: float = 65.0 * 60.0  # 65 minutes from Phase 1 data factory


# ---------------------------------------------------------------------------
# Part 1: Inference Speed
# ---------------------------------------------------------------------------
def measure_inference_speed(device: torch.device) -> Dict:
    """Measure neural forward model inference speed."""
    ckpt_path = CHECKPOINT_DIR / "best_v7.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = build_transfer_model(
        d_hidden=cfg.get("d_hidden", 768),
        n_blocks=cfg.get("n_blocks", 6),
        n_fourier=cfg.get("n_fourier", 256),
        fourier_sigma=cfg.get("fourier_sigma", 30.0),
        n_scenes=cfg.get("n_scenes", 15),
        scene_emb_dim=cfg.get("scene_emb_dim", 32),
        d_out=cfg.get("d_out", 2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Typical inference size: 200 freqs × 50 receivers = 10,000 samples per source
    batch_sizes = [10000, 50000, 100000]
    results = {}

    for bs in batch_sizes:
        dummy_input = torch.randn(bs, 9, device=device)
        dummy_ids = torch.zeros(bs, dtype=torch.long, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_input, scene_ids=dummy_ids)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Timed runs
        n_runs = 20
        timings = []
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(dummy_input, scene_ids=dummy_ids)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - t0)

        mean_ms = np.mean(timings) * 1000.0
        std_ms = np.std(timings) * 1000.0
        throughput = bs / np.mean(timings)

        results[bs] = {
            "mean_ms": float(mean_ms),
            "std_ms": float(std_ms),
            "throughput_per_s": float(throughput),
        }
        logger.info(
            "  Batch %d: %.2f +/- %.2f ms (%.0f samples/s)",
            bs, mean_ms, std_ms, throughput,
        )

    # Per-scene inference comparison
    # BEM: 65 min for 15 scenes × S sources × 200 freqs × R receivers
    # Neural: time for one scene's full evaluation
    n_scenes = 15
    bem_per_scene_s = BEM_TOTAL_TIME_S / n_scenes

    # Typical scene: 5 sources × 200 freqs × 50 receivers = 50,000 samples
    neural_scene_samples = 50000
    neural_scene_ms = results[50000]["mean_ms"]
    neural_scene_s = neural_scene_ms / 1000.0

    speedup = bem_per_scene_s / max(neural_scene_s, 1e-10)

    logger.info("=" * 60)
    logger.info("Inference Speed Comparison")
    logger.info("  BEM: %.1f s/scene (%.1f min total for %d scenes)",
                bem_per_scene_s, BEM_TOTAL_TIME_S / 60, n_scenes)
    logger.info("  Neural: %.3f s/scene (%d samples @ %.1f ms)",
                neural_scene_s, neural_scene_samples, neural_scene_ms)
    logger.info("  Speedup: %.0fx", speedup)
    logger.info("=" * 60)

    return {
        "batch_results": results,
        "bem_per_scene_s": float(bem_per_scene_s),
        "neural_per_scene_s": float(neural_scene_s),
        "speedup": float(speedup),
    }


# ---------------------------------------------------------------------------
# Part 2: T Dynamic Range Analysis
# ---------------------------------------------------------------------------
def analyze_dynamic_range() -> Dict:
    """Compare dynamic range of p_scat vs T = p_scat / p_inc."""
    scene_results = []

    for sid in range(1, 16):
        h5_path = DATA_DIR / f"scene_{sid:03d}.h5"
        if not h5_path.exists():
            continue

        with h5py.File(h5_path, "r") as f:
            freqs_hz = f["frequencies"][:]  # (F,)
            src_pos = f["sources/positions"][:]  # (S, 2)
            rcv_pos = f["receivers/positions"][:]  # (R, 2)
            n_freq = len(freqs_hz)
            n_src = src_pos.shape[0]
            n_rcv = rcv_pos.shape[0]
            k_arr = 2.0 * np.pi * freqs_hz / SPEED_OF_SOUND_M_PER_S

            all_p_scat = []
            all_T = []

            for si in range(n_src):
                p_total = f[f"pressure/src_{si:03d}/field"][:]  # (F, R) complex128

                xs, ys = src_pos[si]
                dx = rcv_pos[:, 0] - xs
                dy = rcv_pos[:, 1] - ys
                dist = np.sqrt(dx ** 2 + dy ** 2)
                dist_safe = np.maximum(dist, 1e-15)

                # Incident field
                kr = k_arr[:, None] * dist_safe[None, :]  # (F, R)
                p_inc = -0.25j * hankel1(0, kr)  # (F, R)

                # Scattered field
                p_scat = p_total - p_inc  # (F, R)

                # Transfer function
                p_inc_safe = np.where(
                    np.abs(p_inc) > 1e-30, p_inc, 1e-30 + 0j
                )
                T = p_scat / p_inc_safe  # (F, R)

                all_p_scat.append(np.abs(p_scat).ravel())
                all_T.append(np.abs(T).ravel())

        p_scat_all = np.concatenate(all_p_scat)
        T_all = np.concatenate(all_T)

        # Filter out zeros/nans
        p_scat_valid = p_scat_all[p_scat_all > 1e-30]
        T_valid = T_all[T_all > 1e-30]

        if len(p_scat_valid) == 0 or len(T_valid) == 0:
            continue

        # Dynamic range in dB
        p_scat_range_dB = 20.0 * np.log10(p_scat_valid.max() / p_scat_valid.min())
        T_range_dB = 20.0 * np.log10(T_valid.max() / T_valid.min())
        reduction_dB = p_scat_range_dB - T_range_dB

        # Standard deviation (complexity)
        p_scat_std = np.std(p_scat_valid)
        T_std = np.std(T_valid)

        scene_results.append({
            "scene_id": sid,
            "p_scat_range_dB": float(p_scat_range_dB),
            "T_range_dB": float(T_range_dB),
            "reduction_dB": float(reduction_dB),
            "p_scat_mean": float(np.mean(p_scat_valid)),
            "T_mean": float(np.mean(T_valid)),
            "p_scat_std": float(p_scat_std),
            "T_std": float(T_std),
        })
        logger.info(
            "  S%02d: p_scat range=%.1f dB, T range=%.1f dB, reduction=%.1f dB",
            sid, p_scat_range_dB, T_range_dB, reduction_dB,
        )

    # Aggregate
    mean_p_range = np.mean([r["p_scat_range_dB"] for r in scene_results])
    mean_T_range = np.mean([r["T_range_dB"] for r in scene_results])
    mean_reduction = np.mean([r["reduction_dB"] for r in scene_results])

    logger.info("=" * 60)
    logger.info("Dynamic Range Analysis")
    logger.info("  Mean |p_scat| range: %.1f dB", mean_p_range)
    logger.info("  Mean |T| range: %.1f dB", mean_T_range)
    logger.info("  Mean reduction: %.1f dB", mean_reduction)
    logger.info("=" * 60)

    return {
        "per_scene": scene_results,
        "mean_p_scat_range_dB": float(mean_p_range),
        "mean_T_range_dB": float(mean_T_range),
        "mean_reduction_dB": float(mean_reduction),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Part 1: Inference Speed
    logger.info("=" * 60)
    logger.info("Part 1: Inference Speed Measurement")
    logger.info("=" * 60)
    speed_results = measure_inference_speed(device)

    # Part 2: T Dynamic Range
    logger.info("=" * 60)
    logger.info("Part 2: T Formulation Dynamic Range")
    logger.info("=" * 60)
    range_results = analyze_dynamic_range()

    # Save results
    csv_path = RESULTS_DIR / "inference_speed.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["bem_per_scene_s", f"{speed_results['bem_per_scene_s']:.1f}"])
        writer.writerow(["neural_per_scene_s", f"{speed_results['neural_per_scene_s']:.4f}"])
        writer.writerow(["speedup", f"{speed_results['speedup']:.0f}"])
        for bs, br in speed_results["batch_results"].items():
            writer.writerow([f"batch_{bs}_ms", f"{br['mean_ms']:.2f}"])
            writer.writerow([f"batch_{bs}_throughput", f"{br['throughput_per_s']:.0f}"])
    logger.info("Saved: %s", csv_path)

    csv_path2 = RESULTS_DIR / "dynamic_range.csv"
    with open(csv_path2, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_id", "p_scat_range_dB", "T_range_dB", "reduction_dB",
                          "p_scat_mean", "T_mean"])
        for r in range_results["per_scene"]:
            writer.writerow([
                r["scene_id"],
                f"{r['p_scat_range_dB']:.1f}",
                f"{r['T_range_dB']:.1f}",
                f"{r['reduction_dB']:.1f}",
                f"{r['p_scat_mean']:.4e}",
                f"{r['T_mean']:.4e}",
            ])
        writer.writerow([
            "MEAN",
            f"{range_results['mean_p_scat_range_dB']:.1f}",
            f"{range_results['mean_T_range_dB']:.1f}",
            f"{range_results['mean_reduction_dB']:.1f}",
            "", "",
        ])
    logger.info("Saved: %s", csv_path2)

    # Final summary
    print("\n" + "=" * 60)
    print("INFERENCE SPEED & DYNAMIC RANGE SUMMARY")
    print("=" * 60)
    print(f"\nInference Speed:")
    print(f"  BEM:    {speed_results['bem_per_scene_s']:.1f} s/scene")
    print(f"  Neural: {speed_results['neural_per_scene_s']:.4f} s/scene")
    print(f"  Speedup: {speed_results['speedup']:.0f}x")
    print(f"\nDynamic Range Reduction:")
    print(f"  |p_scat| mean range: {range_results['mean_p_scat_range_dB']:.1f} dB")
    print(f"  |T| mean range:      {range_results['mean_T_range_dB']:.1f} dB")
    print(f"  Mean reduction:      {range_results['mean_reduction_dB']:.1f} dB")
    print("=" * 60)


if __name__ == "__main__":
    main()
