"""Generate publication-quality figures for ICASSP paper.

All figures conform to IEEE 2-column format:
    - 1-column width: 3.35 inches
    - 2-column width: 6.875 inches
    - DPI: 300
    - Font: serif (DejaVu Serif), 8-9pt
    - Colorblind-safe palette

Figures
-------
    1. Architecture Diagram (2-col)
    2. BEM Validation (1-col)
    3. Forward Model Performance (1-col)
    4. SDF Gallery -- 4 representative scenes (2-col)
    5. Ablation Bar Charts (2-col)
    6. Cycle-Consistency Scatter (1-col)
    7. Generalization + Noise (2-col)

Output
------
    results/paper_figures/fig_{1..7}_*.{pdf,png}

Usage
-----
    python scripts/generate_paper_figures.py                 # all figures
    python scripts/generate_paper_figures.py --fig 2 5       # specific
    python scripts/generate_paper_figures.py --fig 4 --checkpoint best_phase3_v3
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("paper_figures")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "paper_figures"

# ---------------------------------------------------------------------------
# IEEE style constants
# ---------------------------------------------------------------------------
COL_WIDTH_IN: float = 3.35
TWO_COL_WIDTH_IN: float = 6.875
DPI: int = 300
FONT_SIZE_PT: int = 8
FONT_SIZE_TITLE_PT: int = 9

# Colorblind-safe palette (Tol bright)
COLORS = {
    "blue": "#4477AA",
    "cyan": "#66CCEE",
    "green": "#228833",
    "yellow": "#CCBB44",
    "red": "#EE6677",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
    "dark": "#332288",
}


def setup_rcparams() -> None:
    """Configure matplotlib rcParams for IEEE publication quality."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
        "font.size": FONT_SIZE_PT,
        "axes.titlesize": FONT_SIZE_TITLE_PT,
        "axes.labelsize": FONT_SIZE_PT,
        "xtick.labelsize": FONT_SIZE_PT - 1,
        "ytick.labelsize": FONT_SIZE_PT - 1,
        "legend.fontsize": FONT_SIZE_PT - 1,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "pdf.fonttype": 42,  # TrueType fonts in PDF
        "ps.fonttype": 42,
    })


def save_fig(fig: plt.Figure, name: str) -> None:
    """Save figure as both PDF and PNG."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        path = OUTPUT_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=DPI, format=ext)
    logger.info("Saved: %s.{pdf,png}", OUTPUT_DIR / name)
    plt.close(fig)


# ===================================================================
# Figure 1: Architecture Diagram (drawn programmatically)
# ===================================================================
def fig_1_architecture() -> None:
    """Architecture diagram showing forward + inverse cycle."""
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_IN, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.set_aspect("equal")
    ax.axis("off")

    box_style = dict(
        boxstyle="round,pad=0.15", linewidth=0.8,
    )

    # Forward path (top)
    blocks_fwd = [
        (0.8, 2.2, 1.4, 0.5, "Audio\n$p(f)$", COLORS["blue"]),
        (2.8, 2.2, 1.8, 0.5, "Inverse Model\n$f_\\theta(p) \\to \\hat{s}$", COLORS["red"]),
        (5.2, 2.2, 1.5, 0.5, "SDF\n$\\hat{s}(x)$", COLORS["green"]),
        (7.3, 2.2, 1.8, 0.5, "Forward Model\n$g_\\phi(\\hat{s}) \\to \\hat{p}$", COLORS["purple"]),
    ]

    for x, y, w, h, label, color in blocks_fwd:
        rect = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08",
            facecolor=color, edgecolor="black", linewidth=0.8, alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=7, color="white",
            fontweight="bold",
        )

    # Arrows (forward path)
    arrow_kw = dict(
        arrowstyle="->,head_width=0.12,head_length=0.08",
        color="black", linewidth=0.8,
    )
    ax.annotate("", xy=(2.7, 2.45), xytext=(2.2, 2.45), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.1, 2.45), xytext=(4.6, 2.45), arrowprops=arrow_kw)
    ax.annotate("", xy=(7.2, 2.45), xytext=(6.7, 2.45), arrowprops=arrow_kw)

    # Cycle arrow (bottom)
    ax.annotate(
        "", xy=(1.5, 1.9), xytext=(8.3, 1.9),
        arrowprops=dict(
            arrowstyle="->,head_width=0.12,head_length=0.08",
            color=COLORS["red"], linewidth=1.0,
            connectionstyle="arc3,rad=0.3",
        ),
    )
    ax.text(
        5.0, 1.35, "$\\mathcal{L}_{cycle} = \\|\\hat{p} - p\\|^2 / \\|p\\|^2$",
        ha="center", va="center", fontsize=7.5, color=COLORS["red"],
        style="italic",
    )

    # Loss annotations (bottom row)
    losses = [
        (1.0, 0.6, "$\\mathcal{L}_{data}$\nBEM supervision", COLORS["blue"], False),
        (3.2, 0.6, "$\\mathcal{L}_{eik}$\n$|\\nabla s| = 1$", COLORS["green"], False),
        (5.4, 0.6, "$\\mathcal{L}_{helm}$\n$\\nabla^2 p + k^2 p = 0$", "#999999", True),
        (7.6, 0.6, "$\\mathcal{L}_{sdf}$\nSDF supervision", COLORS["yellow"], False),
    ]

    for x, y, label, color, disabled in losses:
        ax.text(
            x, y, label, ha="center", va="center",
            fontsize=6.5, color=color,
            alpha=0.4 if disabled else 1.0,
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                      edgecolor=color, linewidth=0.5,
                      alpha=0.4 if disabled else 0.9,
                      linestyle="--" if disabled else "-"),
        )
        if disabled:
            ax.plot(
                [x - 0.7, x + 0.7], [y, y],
                color="#CC0000", linewidth=1.2, alpha=0.8,
            )
            ax.text(
                x, y - 0.3, "DISABLED", ha="center", va="center",
                fontsize=5, color="#CC0000", fontweight="bold",
            )

    ax.set_title("Deep Acoustic Diffraction Tomography: Architecture", fontsize=9)
    save_fig(fig, "fig_1_architecture")


# ===================================================================
# Figure 2: BEM Validation (Phase 0)
# ===================================================================
def fig_2_bem_validation() -> None:
    """BEM vs analytical validation with error metrics."""
    report_path = RESULTS_DIR / "phase0" / "phase0_report.txt"
    if not report_path.exists():
        logger.warning("Phase 0 report not found, skipping Fig 2")
        return

    # Read the existing comparison image
    img_path = RESULTS_DIR / "phase0" / "wedge_bem_vs_analytical.png"
    if not img_path.exists():
        logger.warning("BEM comparison image not found, skipping Fig 2")
        return

    img = plt.imread(str(img_path))

    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, 2.5))
    ax.imshow(img)
    ax.axis("off")

    # Add error annotation
    ax.text(
        0.02, 0.02,
        "BEM vs Macdonald: L2 error = 1.77%\n"
        "f = 2 kHz, N = 358 elements, 90° wedge",
        transform=ax.transAxes, fontsize=6,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
    )

    ax.set_title("Phase 0: BEM Validation")
    save_fig(fig, "fig_2_bem_validation")


# ===================================================================
# Figure 3: Forward Model Performance (Phase 2)
# ===================================================================
def fig_3_forward_performance() -> None:
    """Per-scene forward model error bar chart."""
    csv_path = RESULTS_DIR / "ablations" / "forward_ablation.csv"
    if not csv_path.exists():
        logger.warning("Forward ablation CSV not found, skipping Fig 3")
        return

    # Read best config (quad ensemble + calib)
    scene_ids = list(range(1, 16))
    errors = {}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Config"] == "E_quad_calib":
                for sid in scene_ids:
                    errors[sid] = float(row[f"S{sid}"])

    if not errors:
        logger.warning("Best forward config not found in CSV")
        return

    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, 2.2))

    x = np.arange(len(scene_ids))
    vals = [errors[sid] for sid in scene_ids]
    bar_colors = [
        COLORS["red"] if v > 5.0 else COLORS["blue"] for v in vals
    ]

    ax.bar(x, vals, color=bar_colors, edgecolor="black", linewidth=0.3, width=0.7)
    ax.axhline(y=5.0, color=COLORS["red"], linestyle="--", linewidth=0.8, label="Gate (5%)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s}" for s in scene_ids], rotation=45, ha="right")
    ax.set_ylabel("Relative L2 Error (%)")
    ax.set_title("Forward Model: Per-Scene Error")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, max(vals) * 1.1)

    # Overall annotation
    overall = np.mean(vals)
    ax.text(
        0.02, 0.95, f"Overall: {overall:.2f}%",
        transform=ax.transAxes, fontsize=7,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    save_fig(fig, "fig_3_forward_performance")


# ===================================================================
# Figure 4: SDF Gallery (4 representative scenes)
# ===================================================================
def fig_4_sdf_gallery(checkpoint_name: str = "best_phase3_v3") -> None:
    """2x4 grid: GT vs Predicted SDF for 4 scenes."""
    import torch
    from src.inverse_model import build_inverse_model, compute_sdf_iou
    from src.inverse_dataset import load_all_scenes
    from src.forward_model import build_transfer_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt_path = PROJECT_ROOT / "checkpoints" / "phase3" / f"{checkpoint_name}.pt"
    if not ckpt_path.exists():
        logger.warning("Checkpoint not found: %s, skipping Fig 4", ckpt_path)
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    inv_scene_id_map = {int(k): v for k, v in cfg["inv_scene_id_map"].items()}

    model = build_inverse_model(
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
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load forward model for scene_scales
    fwd_ckpt_name = cfg["forward_checkpoint"]
    fwd_ckpt_path = PROJECT_ROOT / "checkpoints" / "phase2" / f"{fwd_ckpt_name}.pt"
    fwd_ckpt = torch.load(fwd_ckpt_path, map_location=device, weights_only=False)
    scene_scales = fwd_ckpt["scene_scales"]
    fwd_cfg = fwd_ckpt["config"]
    fwd_scene_list = fwd_cfg.get("trained_scene_list", sorted(scene_scales.keys()))
    fwd_scene_id_map = {sid: idx for idx, sid in enumerate(fwd_scene_list)}

    # Load 4 representative scenes
    gallery_scenes = [1, 7, 10, 12]  # wedge, cylinder, triangle, dual bars
    scene_names = ["S1: Wedge", "S7: Cylinder", "S10: Triangle", "S12: Dual Bars"]

    all_scenes = load_all_scenes(
        PROJECT_ROOT / "data" / "phase1",
        scene_scales, fwd_scene_id_map,
        scene_ids=gallery_scenes,
    )

    fig, axes = plt.subplots(2, 4, figsize=(TWO_COL_WIDTH_IN, 3.0))

    for col, (sid, name) in enumerate(zip(gallery_scenes, scene_names)):
        if sid not in all_scenes:
            logger.warning("Scene %d not found, skipping", sid)
            continue

        sd = all_scenes[sid]
        scene_idx = inv_scene_id_map[sid]
        gx, gy = len(sd.grid_x), len(sd.grid_y)

        # GT SDF
        sdf_gt = sd.sdf_grid  # (Gx, Gy)

        # Predicted SDF
        with torch.no_grad():
            xy_m = torch.from_numpy(sd.grid_coords).float().to(device)
            preds = []
            for i in range(0, len(xy_m), 8192):
                chunk = xy_m[i : i + 8192]
                pred = model.predict_sdf(scene_idx, chunk)
                preds.append(pred.squeeze(-1))
            sdf_pred = torch.cat(preds, dim=0).cpu().numpy().reshape(gx, gy)

        # IoU
        iou = compute_sdf_iou(
            torch.from_numpy(sdf_pred.ravel()),
            torch.from_numpy(sdf_gt.ravel()),
        )

        # Plot GT (top row)
        ax_gt = axes[0, col]
        vmax = max(abs(sdf_gt.min()), abs(sdf_gt.max()), 0.5)
        im = ax_gt.contourf(
            sd.grid_x, sd.grid_y, sdf_gt.T,
            levels=np.linspace(-vmax, vmax, 21),
            cmap="RdBu_r",
        )
        ax_gt.contour(sd.grid_x, sd.grid_y, sdf_gt.T, levels=[0], colors="black", linewidths=0.8)
        ax_gt.set_aspect("equal")
        ax_gt.set_title(f"{name}" if col > 0 else f"GT: {name}", fontsize=7)
        if col == 0:
            ax_gt.set_ylabel("GT SDF", fontsize=7)
        ax_gt.tick_params(labelsize=5)

        # Plot Predicted (bottom row)
        ax_pred = axes[1, col]
        ax_pred.contourf(
            sd.grid_x, sd.grid_y, sdf_pred.T,
            levels=np.linspace(-vmax, vmax, 21),
            cmap="RdBu_r",
        )
        ax_pred.contour(
            sd.grid_x, sd.grid_y, sdf_pred.T,
            levels=[0], colors="black", linewidths=0.8,
        )
        ax_pred.set_aspect("equal")
        ax_pred.set_title(f"IoU = {iou:.3f}", fontsize=7)
        if col == 0:
            ax_pred.set_ylabel("Predicted", fontsize=7)
        ax_pred.tick_params(labelsize=5)

    fig.suptitle("SDF Reconstruction: Ground Truth vs Predicted", fontsize=9, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig_4_sdf_gallery")


# ===================================================================
# Figure 5: Ablation Bar Charts (forward + inverse)
# ===================================================================
def fig_5_ablation_bars() -> None:
    """Side-by-side ablation bar charts for forward and inverse models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TWO_COL_WIDTH_IN, 2.8))

    # --- Forward ablation (8 configs: 3 baseline + 5 ensemble) ---
    fwd_configs = [
        ("No scatter", 47.95, "baseline"),
        ("Vanilla MLP", 48.00, "baseline"),
        ("No FF (T)", 2.27, "baseline"),
        ("Single", 11.54, "ensemble"),
        ("+calib", 10.20, "ensemble"),
        ("Duo+cal", 9.89, "ensemble"),
        ("Quad", 4.57, "ensemble"),
        ("Quad+cal", 4.47, "ensemble"),
    ]
    x_fwd = np.arange(len(fwd_configs))
    labels_fwd = [c[0] for c in fwd_configs]
    vals_fwd = [c[1] for c in fwd_configs]
    groups_fwd = [c[2] for c in fwd_configs]

    bar_colors_fwd = []
    for val, group in zip(vals_fwd, groups_fwd):
        if group == "baseline":
            bar_colors_fwd.append(COLORS["grey"] if val > 5.0 else COLORS["cyan"])
        else:
            bar_colors_fwd.append(COLORS["green"] if val < 5.0 else COLORS["red"])

    bars_fwd = ax1.bar(
        x_fwd, vals_fwd, color=bar_colors_fwd,
        edgecolor="black", linewidth=0.3, width=0.6,
    )
    ax1.axhline(y=5.0, color=COLORS["red"], linestyle="--", linewidth=0.8, label="Gate (5%)")
    # Visual separator between baseline and ensemble groups
    ax1.axvline(x=2.5, color="black", linestyle=":", linewidth=0.6, alpha=0.5)
    ax1.set_xticks(x_fwd)
    ax1.set_xticklabels(labels_fwd, rotation=40, ha="right", fontsize=5.5)
    ax1.set_ylabel("Error (%)")
    ax1.set_title("(a) Forward Model Ablation")
    ax1.legend(loc="upper right", fontsize=6)

    # Value labels on bars
    for bar, val in zip(bars_fwd, vals_fwd):
        y_pos = bar.get_height() + 0.5
        label_text = f"{val:.1f}%"
        # For tall bars (>20%), place label inside
        if val > 20.0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.5,
                label_text, ha="center", va="center", fontsize=5, color="white",
                fontweight="bold",
            )
        else:
            ax1.text(
                bar.get_x() + bar.get_width() / 2, y_pos,
                label_text, ha="center", va="bottom", fontsize=5,
            )

    # Group labels
    ax1.text(1.0, -0.18, "Baselines", ha="center", va="top", fontsize=5.5,
             fontstyle="italic", transform=ax1.get_xaxis_transform())
    ax1.text(5.5, -0.18, "Ensemble (T formulation)", ha="center", va="top",
             fontsize=5.5, fontstyle="italic", transform=ax1.get_xaxis_transform())

    # --- Inverse ablation ---
    inv_configs = [
        ("SDF+Eik", 0.6892),
        ("+bdy 3x", 0.8423),
        ("+Cycle", 0.9388),
        ("+Multi-code", 0.9491),
    ]
    x_inv = np.arange(len(inv_configs))
    labels_inv = [c[0] for c in inv_configs]
    vals_inv = [c[1] for c in inv_configs]

    bars_inv = ax2.bar(
        x_inv, vals_inv, color=COLORS["blue"],
        edgecolor="black", linewidth=0.3, width=0.6,
    )
    ax2.axhline(y=0.8, color=COLORS["red"], linestyle="--", linewidth=0.8, label="Gate (0.8)")
    ax2.set_xticks(x_inv)
    ax2.set_xticklabels(labels_inv, rotation=30, ha="right", fontsize=6)
    ax2.set_ylabel("Mean IoU")
    ax2.set_title("(b) Inverse Model Ablation")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower right", fontsize=6)

    # Value labels
    for bar, val in zip(bars_inv, vals_inv):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", va="bottom", fontsize=5.5,
        )

    fig.tight_layout()
    save_fig(fig, "fig_5_ablation_bars")


# ===================================================================
# Figure 6: Cycle-Consistency (Phase 4)
# ===================================================================
def fig_6_cycle_consistency() -> None:
    """Per-scene Pearson r bar chart from Phase 4 CSV."""
    csv_path = RESULTS_DIR / "phase4" / "cycle_consistency_metrics.csv"
    if not csv_path.exists():
        logger.warning("Phase 4 CSV not found, skipping Fig 6")
        return

    scene_ids: List[int] = []
    r_vals: List[float] = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene_ids.append(int(row["scene_id"]))
            r_vals.append(float(row["r_pearson"]))

    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, 2.2))

    x = np.arange(len(scene_ids))
    bar_colors = [
        COLORS["red"] if r < 0.8 else COLORS["blue"] for r in r_vals
    ]

    ax.bar(x, r_vals, color=bar_colors, edgecolor="black", linewidth=0.3, width=0.7)
    ax.axhline(y=0.8, color=COLORS["red"], linestyle="--", linewidth=0.8, label="Gate (r=0.8)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s}" for s in scene_ids], rotation=45, ha="right")
    ax.set_ylabel("Pearson $r$")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cycle-Consistency Correlation")
    ax.legend(loc="lower right", framealpha=0.9)

    mean_r = np.mean(r_vals)
    ax.text(
        0.02, 0.95, f"Mean $r$ = {mean_r:.4f}",
        transform=ax.transAxes, fontsize=7,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    save_fig(fig, "fig_6_cycle_consistency")


# ===================================================================
# Figure 7: Generalization + Noise (Exp B + C)
# ===================================================================
def fig_7_generalization_noise() -> None:
    """Side-by-side: LOO generalization + noise robustness."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TWO_COL_WIDTH_IN, 2.5))

    # --- (a) LOO Generalization ---
    loo_csv = RESULTS_DIR / "experiments" / "loo_generalization.csv"
    if loo_csv.exists():
        scenes: List[str] = []
        pre_ious: List[float] = []
        final_ious: List[float] = []

        with open(loo_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scenes.append(f"S{row['fold_scene']}")
                pre_ious.append(float(row["pre_iou"]))
                final_ious.append(float(row["final_iou"]))

        x = np.arange(len(scenes))
        w = 0.35

        ax1.bar(x - w / 2, pre_ious, w, label="Trained", color=COLORS["blue"],
                edgecolor="black", linewidth=0.3)
        ax1.bar(x + w / 2, final_ious, w, label="LOO (code-only)", color=COLORS["cyan"],
                edgecolor="black", linewidth=0.3)

        ax1.set_xticks(x)
        ax1.set_xticklabels(scenes)
        ax1.set_ylabel("IoU")
        ax1.set_ylim(0, 1.1)
        ax1.legend(fontsize=6)
        ax1.set_title("(a) LOO Generalization")

        # Recovery annotation
        if pre_ious and final_ious:
            mean_recovery = np.mean(final_ious) / np.mean(pre_ious) * 100
            ax1.text(
                0.02, 0.02, f"Recovery: {mean_recovery:.0f}%",
                transform=ax1.transAxes, fontsize=6,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8),
            )
    else:
        ax1.text(
            0.5, 0.5, "LOO data not available\n(run experiment B first)",
            ha="center", va="center", transform=ax1.transAxes, fontsize=7,
        )
        ax1.set_title("(a) LOO Generalization")

    # --- (b) Noise Robustness ---
    noise_csv = RESULTS_DIR / "experiments" / "noise_robustness.csv"
    if noise_csv.exists():
        snr_labels: List[str] = []
        mean_r_vals: List[float] = []

        with open(noise_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                snr_labels.append(row["snr_db"])
                mean_r_vals.append(float(row["mean_r"]))

        x_noise = np.arange(len(snr_labels))
        bar_colors = [COLORS["blue"]] * len(mean_r_vals)
        # Highlight clean
        if snr_labels[0] == "clean":
            bar_colors[0] = COLORS["green"]

        ax2.bar(
            x_noise, mean_r_vals, color=bar_colors,
            edgecolor="black", linewidth=0.3, width=0.6,
        )
        ax2.axhline(y=0.8, color=COLORS["red"], linestyle="--", linewidth=0.8, label="Gate")

        ax2.set_xticks(x_noise)
        ax2.set_xticklabels(snr_labels)
        ax2.set_xlabel("SNR (dB)")
        ax2.set_ylabel("Mean Pearson $r$")
        ax2.set_ylim(0, 1.05)
        ax2.set_title("(b) Noise Robustness")
        ax2.legend(loc="lower left", fontsize=6)
    else:
        ax2.text(
            0.5, 0.5, "Noise data not available\n(run experiment C first)",
            ha="center", va="center", transform=ax2.transAxes, fontsize=7,
        )
        ax2.set_title("(b) Noise Robustness")

    fig.tight_layout()
    save_fig(fig, "fig_7_generalization_noise")


# ===================================================================
# Main
# ===================================================================
FIGURE_MAP = {
    1: ("Architecture Diagram", fig_1_architecture),
    2: ("BEM Validation", fig_2_bem_validation),
    3: ("Forward Performance", fig_3_forward_performance),
    4: ("SDF Gallery", fig_4_sdf_gallery),
    5: ("Ablation Bar Charts", fig_5_ablation_bars),
    6: ("Cycle-Consistency", fig_6_cycle_consistency),
    7: ("Generalization + Noise", fig_7_generalization_noise),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ICASSP paper figures"
    )
    parser.add_argument(
        "--fig", nargs="+", type=int, default=None,
        help="Figure numbers to generate (default: all)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="best_phase3_v3",
        help="Inverse model checkpoint for Fig 4",
    )
    args = parser.parse_args()

    setup_rcparams()

    figs_to_gen = args.fig if args.fig else sorted(FIGURE_MAP.keys())

    logger.info("Generating %d figures...", len(figs_to_gen))

    for fig_num in figs_to_gen:
        if fig_num not in FIGURE_MAP:
            logger.warning("Unknown figure number: %d", fig_num)
            continue

        name, func = FIGURE_MAP[fig_num]
        logger.info("Figure %d: %s", fig_num, name)

        if fig_num == 4:
            func(args.checkpoint)
        else:
            func()

    logger.info("Done. Output: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
