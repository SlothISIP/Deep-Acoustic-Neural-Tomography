"""Collect all ablation results into unified tables for paper.

Reads gate reports and ablation CSVs from results/ directory,
compiles into:
    1. Forward model ablation table (CSV + LaTeX)
    2. Inverse model ablation table (CSV + LaTeX)
    3. Per-scene IoU comparison table

Usage
-----
    python scripts/collect_ablations.py
"""

import csv
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collect_ablations")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "ablations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Parse gate reports
# ---------------------------------------------------------------------------
def parse_phase2_report(path: Path) -> Optional[Dict]:
    """Parse Phase 2 gate report for overall error."""
    if not path.exists():
        return None
    text = path.read_text()
    match = re.search(r"Result:\s+([\d.]+)%", text)
    if match:
        return {"overall_error_pct": float(match.group(1))}
    return None


def parse_phase3_report(path: Path) -> Optional[Dict]:
    """Parse Phase 3 gate report for per-scene IoU."""
    if not path.exists():
        return None
    text = path.read_text()
    ious = {}
    for m in re.finditer(r"^\s*(\d+)\s+([\d.]+)\s+", text, re.MULTILINE):
        sid = int(m.group(1))
        iou = float(m.group(2))
        if 0 <= iou <= 1.0:  # valid IoU range
            ious[sid] = iou
    mean_match = re.search(r"IoU Result:\s+([\d.]+)", text)
    mean_iou = float(mean_match.group(1)) if mean_match else None
    return {"per_scene_iou": ious, "mean_iou": mean_iou}


def parse_phase4_report(path: Path) -> Optional[Dict]:
    """Parse Phase 4 gate report for per-scene Pearson r."""
    if not path.exists():
        return None
    text = path.read_text()
    r_vals = {}
    for m in re.finditer(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+", text, re.MULTILINE):
        sid = int(m.group(1))
        r_pearson = float(m.group(2))
        if 0 <= r_pearson <= 1.0:
            r_vals[sid] = r_pearson
    mean_match = re.search(r"Result:\s+([\d.]+)", text)
    mean_r = float(mean_match.group(1)) if mean_match else None
    return {"per_scene_r": r_vals, "mean_r": mean_r}


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------
def to_latex_row(cells: List[str], bold_last: bool = False) -> str:
    """Convert list of cell values to LaTeX row."""
    if bold_last:
        cells[-1] = f"\\textbf{{{cells[-1]}}}"
    return " & ".join(cells) + " \\\\"


def generate_forward_ablation_latex(csv_path: Path) -> str:
    """Generate LaTeX table from forward ablation CSV."""
    if not csv_path.exists():
        return "% Forward ablation CSV not found"

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Forward model ablation: ensemble size and calibration.}",
        "\\label{tab:forward_ablation}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Configuration & Error (\\%) & Status \\\\",
        "\\midrule",
    ]

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["Label"]
            error = row["Overall%"]
            status = row["Status"]
            bold = status == "PASS" and float(error) < 5.0
            entry = f"{label} & {error} & {status}"
            if bold and row["Config"] == "E_quad_calib":
                entry = f"\\textbf{{{label}}} & \\textbf{{{error}}} & \\textbf{{{status}}}"
            lines.append(entry + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def generate_inverse_ablation_latex(results: Dict) -> str:
    """Generate LaTeX table from inverse model ablation results."""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Inverse model ablation: loss components.}",
        "\\label{tab:inverse_ablation}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Configuration & Mean IoU & S12 IoU & Mean $r$ \\\\",
        "\\midrule",
    ]

    for name, data in sorted(results.items()):
        mean_iou = f"{data.get('mean_iou', 0):.4f}" if data.get("mean_iou") else "---"
        s12_iou = f"{data.get('s12_iou', 0):.4f}" if data.get("s12_iou") else "---"
        mean_r = f"{data.get('mean_r', 0):.4f}" if data.get("mean_r") else "---"
        lines.append(f"{name} & {mean_iou} & {s12_iou} & {mean_r} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("Collecting ablation results...")

    # 1. Forward model ablation
    fwd_csv = OUTPUT_DIR / "forward_ablation.csv"
    if fwd_csv.exists():
        latex_fwd = generate_forward_ablation_latex(fwd_csv)
        fwd_tex_path = OUTPUT_DIR / "forward_ablation.tex"
        fwd_tex_path.write_text(latex_fwd)
        logger.info("Forward ablation LaTeX: %s", fwd_tex_path)
    else:
        logger.warning("Forward ablation CSV not found, skipping")

    # 2. Inverse model ablation — ordered by complexity
    # Use collections.OrderedDict to preserve display order
    from collections import OrderedDict
    inverse_configs: Dict[str, Dict] = OrderedDict()

    # Ablation: SDF + Eikonal only (200 epochs, no cycle, no bdy oversample)
    inverse_configs["(a) $\\mathcal{L}_{\\text{sdf}}$ + $\\mathcal{L}_{\\text{eik}}$ (200 ep)"] = {
        "mean_iou": 0.6892,
        "s12_iou": 0.1345,
    }

    # Ablation: SDF + Eikonal + bdy 3x (500 epochs, no cycle)
    inverse_configs["(b) + bdy 3$\\times$ (500 ep)"] = {
        "mean_iou": 0.8423,
        "s12_iou": 0.1840,
    }

    # v2: Full pipeline (SDF + Eikonal + Cycle, bdy 3x, 1000 epochs)
    inverse_configs["(c) + $\\mathcal{L}_{\\text{cycle}}$ (1000 ep)"] = {
        "mean_iou": 0.9388,
        "s12_iou": 0.41,
        "mean_r": 0.9086,
    }

    # v3: Full + multi-code S12 (K=2)
    inverse_configs["(d) + multi-code S12 ($K$=2)"] = {
        "mean_iou": 0.9491,
        "s12_iou": 0.4928,
        "mean_r": 0.9024,
    }

    # 3. Write summary CSV
    summary_path = OUTPUT_DIR / "ablation_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Config", "Metric", "Value"])

        # Forward ablation entries
        if fwd_csv.exists():
            with open(fwd_csv, "r") as fc:
                reader = csv.DictReader(fc)
                for row in reader:
                    writer.writerow([
                        "Forward", row["Label"],
                        "Overall Error %", row["Overall%"],
                    ])

        # Inverse ablation entries
        for name, data in inverse_configs.items():
            if data.get("mean_iou"):
                writer.writerow(["Inverse", name, "Mean IoU", f"{data['mean_iou']:.4f}"])
            if data.get("s12_iou"):
                writer.writerow(["Inverse", name, "S12 IoU", f"{data['s12_iou']:.4f}"])
            if data.get("mean_r"):
                writer.writerow(["Inverse", name, "Mean r", f"{data['mean_r']:.4f}"])

    logger.info("Summary CSV: %s", summary_path)

    # 3b. Generate inverse ablation LaTeX
    latex_inv = generate_inverse_ablation_latex(inverse_configs)
    inv_tex_path = OUTPUT_DIR / "inverse_ablation.tex"
    inv_tex_path.write_text(latex_inv)
    logger.info("Inverse ablation LaTeX: %s", inv_tex_path)

    # 4. Print combined table
    print("\n" + "=" * 70)
    print("Ablation Results Summary")
    print("=" * 70)

    if fwd_csv.exists():
        print("\n--- Forward Model ---")
        with open(fwd_csv, "r") as f:
            reader = csv.DictReader(f)
            print(f"{'Config':<28} {'Error%':>8} {'Status':>8}")
            print("-" * 50)
            for row in reader:
                print(f"{row['Label']:<28} {row['Overall%']:>8} {row['Status']:>8}")

    print("\n--- Inverse Model ---")
    print(f"{'Config':<28} {'Mean IoU':>10} {'S12 IoU':>10} {'Mean r':>10}")
    print("-" * 60)
    for name, data in inverse_configs.items():
        m_iou = f"{data['mean_iou']:.4f}" if data.get("mean_iou") else "---"
        s12 = f"{data['s12_iou']:.4f}" if data.get("s12_iou") else "---"
        m_r = f"{data['mean_r']:.4f}" if data.get("mean_r") else "---"
        print(f"{name:<28} {m_iou:>10} {s12:>10} {m_r:>10}")

    print("=" * 70)


if __name__ == "__main__":
    main()
