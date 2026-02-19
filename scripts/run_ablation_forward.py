"""Phase 2 Forward Model Ablation Study.

Runs 5 evaluation configurations to measure the contribution of
ensemble size and per-source calibration to reconstruction accuracy.

Configurations
--------------
    A: single model (best_v11), no calibration
    B: single model (best_v11), per-source calibration
    C: 2-model ensemble (v11+v13), per-source calibration
    D: 4-model ensemble, no calibration
    E: 4-model ensemble, per-source calibration (baseline 4.47%)

Usage
-----
    python scripts/run_ablation_forward.py
"""

import csv
import logging
import subprocess
import sys
import re
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ablation_forward")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "ablations"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    {
        "name": "A_single_nocalib",
        "label": "Single (v11)",
        "ensemble": None,
        "checkpoint": "best_v11",
        "calibrate": False,
        "scene13_ckpt": None,
    },
    {
        "name": "B_single_calib",
        "label": "Single (v11) + calib",
        "ensemble": None,
        "checkpoint": "best_v11",
        "calibrate": True,
        "scene13_ckpt": None,
    },
    {
        "name": "C_duo_calib",
        "label": "Duo (v11,v13) + calib",
        "ensemble": "best_v11,best_v13",
        "checkpoint": None,
        "calibrate": True,
        "scene13_ckpt": None,
    },
    {
        "name": "D_quad_nocalib",
        "label": "Quad ensemble",
        "ensemble": "best_v7,best_v8,best_v11,best_v13",
        "checkpoint": None,
        "calibrate": False,
        "scene13_ckpt": "best_v18_s13",
    },
    {
        "name": "E_quad_calib",
        "label": "Quad ensemble + calib",
        "ensemble": "best_v7,best_v8,best_v11,best_v13",
        "checkpoint": None,
        "calibrate": True,
        "scene13_ckpt": "best_v18_s13",
    },
]


def run_eval(cfg: dict) -> dict:
    """Run eval_phase2.py with given config and parse results.

    Returns dict with per-scene errors and overall error.
    """
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "eval_phase2.py")]

    if cfg["ensemble"]:
        cmd.extend(["--ensemble", cfg["ensemble"]])
    elif cfg["checkpoint"]:
        cmd.extend(["--checkpoint", cfg["checkpoint"]])

    if cfg["calibrate"]:
        cmd.append("--calibrate")

    if cfg["scene13_ckpt"]:
        cmd.extend(["--scene13-checkpoint", cfg["scene13_ckpt"]])

    logger.info("Running: %s", cfg["name"])
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300,
    )

    if result.returncode != 0:
        logger.error("FAILED: %s\n%s", cfg["name"], result.stderr[-500:])
        return {"error": "FAILED"}

    output = result.stdout + result.stderr

    # Parse overall error from report
    overall_match = re.search(r"Result:\s+([\d.]+)%", output)
    overall_error = float(overall_match.group(1)) if overall_match else -1.0

    # Parse per-scene errors
    per_scene = {}
    for m in re.finditer(r"Scene\s+(\d+):\s+error=([\d.]+)%", output):
        sid = int(m.group(1))
        err = float(m.group(2))
        per_scene[sid] = err

    return {
        "overall": overall_error,
        "per_scene": per_scene,
        "status": "PASS" if overall_error < 5.0 else "FAIL",
    }


def main() -> None:
    all_results = {}

    for cfg in CONFIGS:
        result = run_eval(cfg)
        all_results[cfg["name"]] = {**result, "label": cfg["label"]}
        if result.get("overall", -1) > 0:
            logger.info(
                "  %s: overall=%.2f%% %s",
                cfg["name"], result["overall"], result["status"],
            )
        else:
            logger.warning("  %s: FAILED", cfg["name"])

    # Write CSV
    csv_path = RESULTS_DIR / "forward_ablation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Config", "Label", "Overall%", "Status"]
        header += [f"S{i}" for i in range(1, 16)]
        writer.writerow(header)

        for cfg in CONFIGS:
            name = cfg["name"]
            res = all_results[name]
            row = [name, res["label"], f"{res.get('overall', -1):.2f}", res.get("status", "N/A")]
            for sid in range(1, 16):
                row.append(f"{res.get('per_scene', {}).get(sid, -1):.2f}")
            writer.writerow(row)

    logger.info("Forward ablation CSV: %s", csv_path)

    # Print summary table
    print("\n" + "=" * 70)
    print("Forward Model Ablation Results")
    print("=" * 70)
    print(f"{'Config':<28} {'Overall%':>10} {'Status':>8}")
    print("-" * 70)
    for cfg in CONFIGS:
        name = cfg["name"]
        res = all_results[name]
        print(f"{res['label']:<28} {res.get('overall', -1):>9.2f}% {res.get('status', 'N/A'):>8}")
    print("=" * 70)


if __name__ == "__main__":
    main()
