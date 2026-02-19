"""Seed Sweep: Run inverse model training with multiple seeds.

Measures variance across random initializations to validate
reproducibility of Phase 3 and Phase 4 gate results.

Method
------
    For each seed in {42, 123, 456}:
    1. Train Phase 3 inverse model from scratch (1000 epochs)
    2. Evaluate Phase 3 gate (SDF IoU)
    3. Evaluate Phase 4 gate (cycle-consistency Pearson r)

Output
------
    results/experiments/seed_sweep.csv

Usage
-----
    python scripts/run_seed_sweep.py
    python scripts/run_seed_sweep.py --epochs 1000 --seeds 42 123 456
"""

import argparse
import csv
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("seed_sweep")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase3"
PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEEDS: List[int] = [42, 123, 456]


def run_phase3_training(
    seed: int,
    epochs: int = 1000,
    forward_ckpt: str = "best_v11",
) -> str:
    """Train Phase 3 inverse model with given seed. Returns tag name."""
    tag = f"seed{seed}"
    cmd = [
        PYTHON, str(PROJECT_ROOT / "scripts" / "run_phase3.py"),
        "--seed", str(seed),
        "--tag", tag,
        "--epochs", str(epochs),
        "--no-helmholtz",
        "--boundary-oversample", "3.0",
        "--multi-body", "12:2",
        "--forward-ckpt", forward_ckpt,
        "--patience", "300",
    ]

    logger.info("=" * 60)
    logger.info("Phase 3 Training: seed=%d, tag=%s", seed, tag)
    logger.info("Command: %s", " ".join(cmd))
    logger.info("=" * 60)

    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error("Training FAILED (seed=%d):", seed)
        logger.error("STDERR: %s", result.stderr[-2000:])
        return ""

    logger.info("Training done: seed=%d, %.1fs", seed, elapsed)
    return tag


def run_phase3_eval(tag: str) -> Dict:
    """Evaluate Phase 3 gate (IoU). Returns per-scene IoU dict."""
    ckpt_name = f"best_phase3_{tag}" if tag else "best_phase3"
    cmd = [
        PYTHON, str(PROJECT_ROOT / "scripts" / "eval_phase3.py"),
        "--checkpoint", ckpt_name,
    ]

    logger.info("Phase 3 Eval: checkpoint=%s", ckpt_name)
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        logger.error("Phase 3 eval FAILED: %s", result.stderr[-2000:])
        return {}

    # Parse gate report
    report_path = RESULTS_DIR.parent / "phase3" / "phase3_gate_report.txt"
    ious: Dict[str, float] = {}
    mean_iou: float = 0.0

    if report_path.exists():
        with open(report_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Mean"):
                    parts = line.split()
                    mean_iou = float(parts[1])
                elif line and line[0].isdigit():
                    parts = line.split()
                    if len(parts) >= 2:
                        scene_id = parts[0]
                        iou = float(parts[1])
                        ious[f"S{scene_id}"] = iou

    ious["mean"] = mean_iou
    return ious


def run_phase4_eval(tag: str) -> Dict:
    """Evaluate Phase 4 gate (cycle-consistency r). Returns per-scene r dict."""
    ckpt_name = f"best_phase3_{tag}" if tag else "best_phase3"
    cmd = [
        PYTHON, str(PROJECT_ROOT / "scripts" / "eval_phase4.py"),
        "--checkpoint", ckpt_name,
    ]

    logger.info("Phase 4 Eval: checkpoint=%s", ckpt_name)
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        logger.error("Phase 4 eval FAILED: %s", result.stderr[-2000:])
        return {}

    # Parse gate report
    report_path = RESULTS_DIR.parent / "phase4" / "phase4_gate_report.txt"
    rs: Dict[str, float] = {}
    mean_r: float = 0.0

    if report_path.exists():
        with open(report_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Mean"):
                    parts = line.split()
                    mean_r = float(parts[1])
                elif line and line[0].isdigit():
                    parts = line.split()
                    if len(parts) >= 2:
                        scene_id = parts[0]
                        r_val = float(parts[1])
                        rs[f"S{scene_id}"] = r_val

    rs["mean"] = mean_r
    return rs


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Sweep for Reproducibility")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help="Seeds to test (default: 42 123 456)",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--forward-ckpt", type=str, default="best_v11")
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training, only evaluate existing checkpoints",
    )
    args = parser.parse_args()

    results: List[Dict] = []

    for seed in args.seeds:
        tag = f"seed{seed}"

        if not args.eval_only:
            tag = run_phase3_training(seed, args.epochs, args.forward_ckpt)
            if not tag:
                logger.error("Skipping seed %d (training failed)", seed)
                continue

        # Check if checkpoint exists
        ckpt_path = CHECKPOINT_DIR / f"best_phase3_{tag}.pt"
        if not ckpt_path.exists():
            logger.warning("Checkpoint not found: %s, skipping", ckpt_path)
            continue

        ious = run_phase3_eval(tag)
        rs = run_phase4_eval(tag)

        results.append({
            "seed": seed,
            "tag": tag,
            "mean_iou": ious.get("mean", 0.0),
            "mean_r": rs.get("mean", 0.0),
            "per_scene_iou": ious,
            "per_scene_r": rs,
        })

        logger.info(
            "Seed %d: IoU=%.4f, r=%.4f",
            seed, ious.get("mean", 0.0), rs.get("mean", 0.0),
        )

    if not results:
        logger.error("No results collected!")
        return

    # Summary statistics
    ious_list = [r["mean_iou"] for r in results]
    rs_list = [r["mean_r"] for r in results]

    print("\n" + "=" * 60)
    print("Seed Sweep Results")
    print("=" * 60)
    print(f"{'Seed':>6} {'Mean IoU':>10} {'Mean r':>10}")
    print("-" * 30)
    for r in results:
        print(f"{r['seed']:>6d} {r['mean_iou']:>10.4f} {r['mean_r']:>10.4f}")
    print("-" * 30)
    print(
        f"{'Mean':>6} {np.mean(ious_list):>10.4f} {np.mean(rs_list):>10.4f}"
    )
    print(
        f"{'Std':>6} {np.std(ious_list):>10.4f} {np.std(rs_list):>10.4f}"
    )
    print(
        f"{'Min':>6} {np.min(ious_list):>10.4f} {np.min(rs_list):>10.4f}"
    )
    print(
        f"{'Max':>6} {np.max(ious_list):>10.4f} {np.max(rs_list):>10.4f}"
    )
    print("=" * 60)

    # Write CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "seed_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "mean_iou", "mean_r"])
        for r in results:
            writer.writerow([
                r["seed"],
                f"{r['mean_iou']:.4f}",
                f"{r['mean_r']:.4f}",
            ])
        writer.writerow([
            "mean",
            f"{np.mean(ious_list):.4f}",
            f"{np.mean(rs_list):.4f}",
        ])
        writer.writerow([
            "std",
            f"{np.std(ious_list):.4f}",
            f"{np.std(rs_list):.4f}",
        ])

    logger.info("Results CSV: %s", csv_path)

    # Gate check
    gate_iou = 0.8
    gate_r = 0.8
    iou_pass = all(x > gate_iou for x in ious_list)
    r_pass = all(x > gate_r for x in rs_list)
    print(f"\nGate IoU > {gate_iou}: {'ALL PASS' if iou_pass else 'SOME FAIL'}")
    print(f"Gate r > {gate_r}: {'ALL PASS' if r_pass else 'SOME FAIL'}")


if __name__ == "__main__":
    main()
