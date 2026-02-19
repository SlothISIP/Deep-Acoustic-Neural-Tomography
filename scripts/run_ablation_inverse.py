"""Phase 3 Inverse Model Ablation Study.

Trains 2 ablation variants of the inverse model to measure
the contribution of each loss component:
    1. SDF-only: L_sdf + L_eikonal (Stage 1 only, 200 epochs)
    2. SDF+Eikonal, no cycle: L_sdf + L_eikonal (500 epochs, no cycle)

Reference configurations:
    v1: Helmholtz enabled (best_phase3.pt)
    v2: no Helmholtz, bdy 3x (best_phase3_v2.pt) -- baseline IoU=0.9388
    v3: multi-code S12 (best_phase3_v3.pt) -- if available

Usage
-----
    python scripts/run_ablation_inverse.py
"""

import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ablation_inverse")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_training(tag: str, epochs: int, extra_args: list) -> bool:
    """Run Phase 3 training with given config."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_phase3.py"),
        "--forward-ckpt", "best_v11",
        "--no-helmholtz",
        "--epochs", str(epochs),
        "--tag", tag,
    ] + extra_args

    logger.info("Training: %s (epochs=%d)", tag, epochs)
    logger.info("Command: %s", " ".join(cmd))

    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1800,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error("FAILED: %s\n%s", tag, result.stderr[-500:])
        return False

    logger.info("Done: %s in %.1f min", tag, elapsed / 60)
    return True


def run_eval(checkpoint: str) -> bool:
    """Run Phase 3 evaluation."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "eval_phase3.py"),
        "--checkpoint", checkpoint,
        "--forward-ckpt", "best_v11",
    ]
    logger.info("Evaluating: %s", checkpoint)
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        logger.error("Eval FAILED: %s\n%s", checkpoint, result.stderr[-500:])
        return False
    # Print output
    print(result.stdout[-2000:])
    return True


def main() -> None:
    configs = [
        {
            "tag": "ablation_sdf_only",
            "epochs": 200,
            "extra_args": ["--w-cycle", "0.0"],
            "description": "SDF + Eikonal only (200 epochs)",
        },
        {
            "tag": "ablation_no_cycle",
            "epochs": 500,
            "extra_args": ["--w-cycle", "0.0", "--boundary-oversample", "3.0"],
            "description": "SDF + Eikonal + bdy 3x (500 epochs, no cycle)",
        },
    ]

    for cfg in configs:
        logger.info("=" * 50)
        logger.info("Ablation: %s", cfg["description"])
        logger.info("=" * 50)

        success = run_training(cfg["tag"], cfg["epochs"], cfg["extra_args"])
        if success:
            run_eval(f"best_phase3_{cfg['tag']}")
        else:
            logger.warning("Skipping eval for %s due to training failure", cfg["tag"])

    logger.info("All ablation runs complete")


if __name__ == "__main__":
    main()
