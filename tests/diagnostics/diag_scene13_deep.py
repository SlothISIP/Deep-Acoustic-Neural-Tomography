"""Deep diagnostic for Scene 13 (step discontinuity) Phase 2 error.

Scene 13 shows 19.53% reconstruction error (gate: 5%), with per-source
breakdown suggesting source 2 is the worst offender (~28.93%).

This diagnostic answers:
    1. What does the transfer function T = p_scat / p_inc look like?
    2. Where are the extreme |T| values and do they correlate with geometry?
    3. How does |T| distribute across regions (shadow, transition, lit)?
    4. Is source 2's high error caused by singularities, shadow geometry,
       or inter-body scattering?

Outputs
-------
    results/phase2/diag_scene13_T_maps.png        -- |T| spatial maps per source
    results/phase2/diag_scene13_T_stats.png        -- |T| statistics + distributions
    results/phase2/diag_scene13_T_freq_spectra.png -- |T| vs frequency for outlier receivers
    stdout: full diagnostic report table
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scipy.special import hankel1

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("diag_scene13")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPEED_OF_SOUND_M_PER_S: float = 343.0
REGION_NAMES: Dict[int, str] = {0: "shadow", 1: "transition", 2: "lit"}
DATA_PATH: Path = PROJECT_ROOT / "data" / "phase1" / "scene_013.h5"
RESULTS_DIR: Path = PROJECT_ROOT / "results" / "phase2"
REP_FREQ_HZ: float = 5000.0  # representative frequency for spatial maps

# Scene 13 geometry: step discontinuity (from scenes.py)
STEP_W: float = 0.4
STEP_H1: float = 0.15
STEP_H2: float = 0.30
STEP1_VERTS = np.array([
    [-STEP_W, -STEP_H1 / 2],
    [0.0, -STEP_H1 / 2],
    [0.0, STEP_H1 / 2],
    [-STEP_W, STEP_H1 / 2],
])
STEP2_VERTS = np.array([
    [0.0, -STEP_H2 / 2],
    [STEP_W, -STEP_H2 / 2],
    [STEP_W, STEP_H2 / 2],
    [0.0, STEP_H2 / 2],
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_incident_field_2d(
    rcv_pos_m: np.ndarray,
    src_pos_m: np.ndarray,
    k_rad_per_m: float,
) -> np.ndarray:
    """Compute 2D point source incident field.

    p_inc(x) = -(i/4) H_0^(1)(k |x - x_s|)

    Parameters
    ----------
    rcv_pos_m : np.ndarray, shape (R, 2)
    src_pos_m : np.ndarray, shape (2,)
    k_rad_per_m : float

    Returns
    -------
    p_inc : np.ndarray, complex128, shape (R,)
    """
    diff = rcv_pos_m - src_pos_m[None, :]  # (R, 2)
    dist = np.sqrt(np.sum(diff ** 2, axis=1))  # (R,)
    dist_safe = np.maximum(dist, 1e-15)  # (R,)
    return -0.25j * hankel1(0, k_rad_per_m * dist_safe)  # (R,), complex128


def draw_step_geometry(ax: plt.Axes) -> None:
    """Draw the step discontinuity bodies on an axis."""
    # Step 1: left block (shorter)
    rect1 = Rectangle(
        (-STEP_W, -STEP_H1 / 2), STEP_W, STEP_H1,
        linewidth=1.5, edgecolor="black", facecolor="lightgray", alpha=0.7,
    )
    # Step 2: right block (taller)
    rect2 = Rectangle(
        (0.0, -STEP_H2 / 2), STEP_W, STEP_H2,
        linewidth=1.5, edgecolor="black", facecolor="darkgray", alpha=0.7,
    )
    ax.add_patch(rect1)
    ax.add_patch(rect2)


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------
def run_diagnostic() -> None:
    """Run deep diagnostic on scene 13 transfer function."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        logger.error("Scene 13 data not found: %s", DATA_PATH)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    with h5py.File(DATA_PATH, "r") as f:
        freqs_hz = f["frequencies"][:]  # (F,), F=200
        src_pos = f["sources/positions"][:]  # (S, 2), S=3
        rcv_pos = f["receivers/positions"][:]  # (R, 2), R=198

        n_freq = len(freqs_hz)
        n_src = src_pos.shape[0]
        n_rcv = rcv_pos.shape[0]

        k_arr = 2.0 * np.pi * freqs_hz / SPEED_OF_SOUND_M_PER_S  # (F,)

        # Load all pressure + region data
        p_total_all = {}  # {src_idx: (F, R) complex128}
        regions_all = {}  # {src_idx: (R,) int}
        for si in range(n_src):
            p_total_all[si] = f[f"pressure/src_{si:03d}/field"][:]  # (F, R)
            regions_all[si] = f[f"regions/src_{si:03d}/labels"][:]  # (R,)

        scene_name = f["config"].attrs["name"]
        scene_category = f["config"].attrs["category"]

    logger.info("Loaded scene 13: name=%s, category=%s", scene_name, scene_category)
    logger.info("  F=%d, S=%d, R=%d, freq=[%.0f, %.0f] Hz",
                n_freq, n_src, n_rcv, freqs_hz[0], freqs_hz[-1])

    # ------------------------------------------------------------------
    # 2. Compute transfer function T for each source
    #    T = p_scat / p_inc = (p_total - p_inc) / p_inc = p_total/p_inc - 1
    # ------------------------------------------------------------------
    T_all = {}  # {src_idx: (F, R) complex128}
    T_abs_all = {}  # {src_idx: (F, R) float}
    p_inc_all = {}  # {src_idx: (F, R) complex128}

    for si in range(n_src):
        T_arr = np.zeros((n_freq, n_rcv), dtype=np.complex128)  # (F, R)
        p_inc_arr = np.zeros((n_freq, n_rcv), dtype=np.complex128)  # (F, R)

        for fi in range(n_freq):
            p_inc = compute_incident_field_2d(rcv_pos, src_pos[si], k_arr[fi])  # (R,)
            p_inc_arr[fi, :] = p_inc
            p_total = p_total_all[si][fi, :]  # (R,)

            # T = (p_total / p_inc) - 1 = p_scat / p_inc
            p_inc_safe = np.where(np.abs(p_inc) > 1e-20, p_inc, 1e-20)  # (R,)
            T_arr[fi, :] = (p_total / p_inc_safe) - 1.0  # (R,), complex128

        T_all[si] = T_arr
        T_abs_all[si] = np.abs(T_arr)  # (F, R)
        p_inc_all[si] = p_inc_arr

    # ------------------------------------------------------------------
    # 3. Compute |T| statistics per source
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCENE 13 DEEP DIAGNOSTIC: Transfer Function Analysis")
    print("=" * 80)
    print(f"Scene: {scene_name} ({scene_category})")
    print(f"Geometry: step discontinuity (step1: {STEP_W}x{STEP_H1} m, step2: {STEP_W}x{STEP_H2} m)")
    print(f"Frequencies: {n_freq} bins, [{freqs_hz[0]:.0f}, {freqs_hz[-1]:.0f}] Hz")
    print(f"Sources: {n_src}, Receivers: {n_rcv}")
    print()

    # Source positions
    print("Source Positions:")
    for si in range(n_src):
        print(f"  src_{si}: ({src_pos[si, 0]:.3f}, {src_pos[si, 1]:.3f}) m")
    print()

    # Per-source |T| statistics (averaged over all frequencies)
    print("-" * 80)
    print(f"{'Source':>8} {'mean|T|':>10} {'std|T|':>10} {'max|T|':>10} "
          f"{'p50|T|':>10} {'p95|T|':>10} {'p99|T|':>10} {'>5.0':>8} {'>10.0':>8}")
    print("-" * 80)

    for si in range(n_src):
        T_abs = T_abs_all[si].ravel()  # (F*R,)
        print(
            f"  src_{si}: "
            f"{np.mean(T_abs):10.4f} "
            f"{np.std(T_abs):10.4f} "
            f"{np.max(T_abs):10.4f} "
            f"{np.percentile(T_abs, 50):10.4f} "
            f"{np.percentile(T_abs, 95):10.4f} "
            f"{np.percentile(T_abs, 99):10.4f} "
            f"{int(np.sum(T_abs > 5.0)):8d} "
            f"{int(np.sum(T_abs > 10.0)):8d}"
        )
    print()

    # ------------------------------------------------------------------
    # 4. Per-region |T| statistics per source
    # ------------------------------------------------------------------
    print("-" * 80)
    print("Per-Region |T| Statistics (averaged over all frequencies)")
    print("-" * 80)
    print(f"{'Source':>8} {'Region':>12} {'N_rcv':>8} {'mean|T|':>10} {'std|T|':>10} "
          f"{'max|T|':>10} {'p95|T|':>10} {'p99|T|':>10}")
    print("-" * 80)

    region_stats = {}  # {(si, reg): dict}
    for si in range(n_src):
        labels = regions_all[si]  # (R,)
        for reg_id in [0, 1, 2]:
            mask = labels == reg_id
            n_reg = int(mask.sum())
            if n_reg == 0:
                print(f"  src_{si}  {REGION_NAMES[reg_id]:>12}  {n_reg:8d}  {'N/A':>10}")
                continue

            T_reg = T_abs_all[si][:, mask].ravel()  # (F * n_reg,)
            stats = {
                "n_rcv": n_reg,
                "mean": float(np.mean(T_reg)),
                "std": float(np.std(T_reg)),
                "max": float(np.max(T_reg)),
                "p95": float(np.percentile(T_reg, 95)),
                "p99": float(np.percentile(T_reg, 99)),
            }
            region_stats[(si, reg_id)] = stats

            print(
                f"  src_{si}  {REGION_NAMES[reg_id]:>12}  {n_reg:8d}  "
                f"{stats['mean']:10.4f} {stats['std']:10.4f} "
                f"{stats['max']:10.4f} {stats['p95']:10.4f} {stats['p99']:10.4f}"
            )
    print()

    # ------------------------------------------------------------------
    # 5. Per-source |p_inc| statistics to check for small denominators
    # ------------------------------------------------------------------
    print("-" * 80)
    print("|p_inc| Statistics (small p_inc -> T amplification)")
    print("-" * 80)
    print(f"{'Source':>8} {'mean|p_inc|':>12} {'min|p_inc|':>12} {'<1e-3':>8} {'<1e-2':>8}")
    print("-" * 80)

    for si in range(n_src):
        p_inc_abs = np.abs(p_inc_all[si]).ravel()  # (F*R,)
        print(
            f"  src_{si}: "
            f"{np.mean(p_inc_abs):12.6f} "
            f"{np.min(p_inc_abs):12.6f} "
            f"{int(np.sum(p_inc_abs < 1e-3)):8d} "
            f"{int(np.sum(p_inc_abs < 1e-2)):8d}"
        )
    print()

    # ------------------------------------------------------------------
    # 6. Identify the worst receivers per source
    # ------------------------------------------------------------------
    print("-" * 80)
    print("Top 10 Worst Receivers (highest mean |T| over frequency) per Source")
    print("-" * 80)

    worst_rcv = {}  # {si: sorted list of (rcv_idx, mean_T)}
    for si in range(n_src):
        mean_T_per_rcv = np.mean(T_abs_all[si], axis=0)  # (R,)
        sorted_idx = np.argsort(mean_T_per_rcv)[::-1]  # descending
        worst_rcv[si] = [(int(idx), float(mean_T_per_rcv[idx])) for idx in sorted_idx[:10]]

        labels = regions_all[si]
        print(f"\n  src_{si} (pos=({src_pos[si, 0]:.3f}, {src_pos[si, 1]:.3f})):")
        print(f"    {'Rank':>4} {'RcvIdx':>8} {'mean|T|':>10} {'max|T|':>10} "
              f"{'Region':>10} {'RcvPos':>20} {'Dist':>8}")
        for rank, (ridx, mt) in enumerate(worst_rcv[si]):
            max_t = float(np.max(T_abs_all[si][:, ridx]))
            reg_name = REGION_NAMES[int(labels[ridx])]
            rpos = rcv_pos[ridx]
            dist = float(np.linalg.norm(src_pos[si] - rpos))
            print(
                f"    {rank + 1:4d} {ridx:8d} {mt:10.4f} {max_t:10.4f} "
                f"{reg_name:>10} ({rpos[0]:+.3f}, {rpos[1]:+.3f}) {dist:8.4f}"
            )

    # ------------------------------------------------------------------
    # 7. Frequency-dependent analysis at representative freq
    # ------------------------------------------------------------------
    fi_rep = int(np.argmin(np.abs(freqs_hz - REP_FREQ_HZ)))
    freq_rep = freqs_hz[fi_rep]
    k_rep = k_arr[fi_rep]
    lambda_rep = SPEED_OF_SOUND_M_PER_S / freq_rep

    print(f"\n{'=' * 80}")
    print(f"Representative Frequency: {freq_rep:.0f} Hz (k={k_rep:.2f} rad/m, lambda={lambda_rep:.4f} m)")
    print(f"{'=' * 80}")

    for si in range(n_src):
        T_at_freq = T_abs_all[si][fi_rep, :]  # (R,)
        print(
            f"  src_{si}: mean|T|={np.mean(T_at_freq):.4f}, "
            f"max|T|={np.max(T_at_freq):.4f}, "
            f"std|T|={np.std(T_at_freq):.4f}, "
            f"|T|>2.0: {int(np.sum(T_at_freq > 2.0))}/{n_rcv}"
        )

    # ------------------------------------------------------------------
    # 8. Correlation analysis: source geometry vs difficulty
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("GEOMETRY ANALYSIS: Source position relative to step discontinuity")
    print(f"{'=' * 80}")
    print(f"Step 1 (left):  x in [{-STEP_W:.2f}, 0.00], y in [{-STEP_H1/2:.3f}, {STEP_H1/2:.3f}]")
    print(f"Step 2 (right): x in [0.00, {STEP_W:.2f}], y in [{-STEP_H2/2:.3f}, {STEP_H2/2:.3f}]")
    print()

    for si in range(n_src):
        sx, sy = src_pos[si]
        # Angle from origin to source
        angle_deg = float(np.degrees(np.arctan2(sy, sx)))

        # Shadow analysis: how many receivers are behind BOTH blocks from this source?
        labels = regions_all[si]
        n_shadow = int(np.sum(labels == 0))
        n_trans = int(np.sum(labels == 1))
        n_lit = int(np.sum(labels == 2))

        # Distance to step edge (x=0 line)
        dist_to_edge = abs(sx)

        # Is source behind the taller block?
        behind_tall = sx > 0
        behind_short = sx < 0

        print(f"  src_{si}: ({sx:.3f}, {sy:.3f})")
        print(f"    angle={angle_deg:.1f} deg, dist_to_step_edge={dist_to_edge:.3f} m")
        print(f"    behind_tall_block={behind_tall}, behind_short_block={behind_short}")
        print(f"    regions: shadow={n_shadow}, transition={n_trans}, lit={n_lit}")
        print(f"    shadow_fraction={n_shadow/n_rcv*100:.1f}%")
        print()

    # ------------------------------------------------------------------
    # 9. Inter-body scattering index: compare |p_total| / |p_inc| near the
    #    gap between the two blocks (x~0)
    # ------------------------------------------------------------------
    print(f"{'=' * 80}")
    print("INTER-BODY SCATTERING: Field amplification near step edge (x~0)")
    print(f"{'=' * 80}")

    # Receivers near the step edge (|x| < 0.1)
    near_edge_mask = np.abs(rcv_pos[:, 0]) < 0.1  # (R,)
    n_near_edge = int(near_edge_mask.sum())
    print(f"Receivers within |x| < 0.1 m of step edge: {n_near_edge}")

    if n_near_edge > 0:
        for si in range(n_src):
            T_near = T_abs_all[si][:, near_edge_mask]  # (F, n_near_edge)
            T_far = T_abs_all[si][:, ~near_edge_mask]  # (F, n_far)
            print(
                f"  src_{si}: near_edge mean|T|={np.mean(T_near):.4f}, "
                f"far mean|T|={np.mean(T_far):.4f}, "
                f"ratio={np.mean(T_near)/max(np.mean(T_far), 1e-10):.2f}x"
            )

    # Receivers between the two blocks (x in [-0.05, 0.05], |y| < STEP_H2/2)
    in_gap_mask = (
        (np.abs(rcv_pos[:, 0]) < 0.05) &
        (np.abs(rcv_pos[:, 1]) < STEP_H2 / 2)
    )
    n_in_gap = int(in_gap_mask.sum())
    print(f"\nReceivers in gap region (|x|<0.05, |y|<{STEP_H2/2:.2f}): {n_in_gap}")
    if n_in_gap > 0:
        for si in range(n_src):
            T_gap = T_abs_all[si][:, in_gap_mask]  # (F, n_gap)
            print(
                f"  src_{si}: gap mean|T|={np.mean(T_gap):.4f}, "
                f"max|T|={np.max(T_gap):.4f}"
            )

    # ------------------------------------------------------------------
    # 10. Dynamic range analysis
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("DYNAMIC RANGE: |T| distribution clipping analysis")
    print(f"{'=' * 80}")
    print("If max|T| >> 1, the transfer function has extreme dynamic range,")
    print("making it hard for an MLP to learn both small and large values.")
    print()

    for si in range(n_src):
        T_flat = T_abs_all[si].ravel()
        total_n = len(T_flat)
        bins = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, np.inf]
        print(f"  src_{si} |T| histogram:")
        for i in range(len(bins) - 1):
            count = int(np.sum((T_flat >= bins[i]) & (T_flat < bins[i + 1])))
            pct = count / total_n * 100
            bar = "#" * int(pct / 2)
            print(f"    [{bins[i]:7.1f}, {bins[i+1]:7.1f}): {count:8d} ({pct:5.1f}%) {bar}")
        print()

    # ------------------------------------------------------------------
    # FIGURE 1: |T| spatial maps at representative frequency
    # ------------------------------------------------------------------
    logger.info("Generating |T| spatial maps at f=%.0f Hz ...", freq_rep)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Scene 13 (step): |T| = |p_scat/p_inc| at f={freq_rep:.0f} Hz "
        f"(k={k_rep:.1f} rad/m)",
        fontsize=13,
    )

    vmin_global = 0.0
    vmax_global = min(
        np.percentile(
            np.concatenate([T_abs_all[si][fi_rep, :] for si in range(n_src)]),
            98
        ),
        10.0,
    )

    for si in range(n_src):
        ax = axes[si]
        T_at_freq = T_abs_all[si][fi_rep, :]  # (R,)
        labels = regions_all[si]

        sc = ax.scatter(
            rcv_pos[:, 0], rcv_pos[:, 1],
            c=T_at_freq, cmap="hot", s=15,
            vmin=vmin_global, vmax=vmax_global,
            edgecolors="none", alpha=0.8,
        )

        # Draw geometry
        draw_step_geometry(ax)

        # Mark source
        ax.plot(
            src_pos[si, 0], src_pos[si, 1],
            marker="*", markersize=15, color="cyan", markeredgecolor="black",
        )

        # Mark shadow boundary receivers
        shadow_mask = labels == 0
        ax.scatter(
            rcv_pos[shadow_mask, 0], rcv_pos[shadow_mask, 1],
            facecolors="none", edgecolors="blue", s=25, linewidths=0.5,
            label="shadow",
        )

        n_shadow = int(shadow_mask.sum())
        mean_T = float(np.mean(T_at_freq))
        max_T = float(np.max(T_at_freq))

        ax.set_title(
            f"src_{si} ({src_pos[si, 0]:.2f}, {src_pos[si, 1]:.2f})\n"
            f"mean|T|={mean_T:.3f}, max|T|={max_T:.3f}, shadow={n_shadow}/{n_rcv}",
            fontsize=9,
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.2)

        plt.colorbar(sc, ax=ax, shrink=0.8, label="|T|")

    fig.tight_layout()
    fig1_path = RESULTS_DIR / "diag_scene13_T_maps.png"
    fig.savefig(fig1_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", fig1_path)

    # ------------------------------------------------------------------
    # FIGURE 2: |T| statistics -- histograms + box plots
    # ------------------------------------------------------------------
    logger.info("Generating |T| statistics figure ...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Scene 13 (step): |T| Distribution Analysis", fontsize=13)

    # Row 1: histograms per source
    for si in range(n_src):
        ax = axes[0, si]
        T_flat = T_abs_all[si].ravel()
        T_clipped = np.clip(T_flat, 0, 10)
        ax.hist(T_clipped, bins=100, density=True, alpha=0.7, color="steelblue")
        ax.axvline(np.mean(T_flat), color="red", linestyle="--", label=f"mean={np.mean(T_flat):.3f}")
        ax.axvline(np.median(T_flat), color="orange", linestyle="--", label=f"median={np.median(T_flat):.3f}")
        ax.axvline(np.percentile(T_flat, 95), color="green", linestyle="--",
                   label=f"p95={np.percentile(T_flat, 95):.3f}")
        ax.set_xlabel("|T| (clipped at 10)")
        ax.set_ylabel("Density")
        ax.set_title(f"src_{si}: |T| histogram (all F x R)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    # Row 2: per-region box plots per source
    for si in range(n_src):
        ax = axes[1, si]
        labels = regions_all[si]
        box_data = []
        box_labels = []
        for reg_id in [0, 1, 2]:
            mask = labels == reg_id
            if mask.sum() == 0:
                continue
            T_reg = T_abs_all[si][:, mask].ravel()
            # Clip for visualization
            T_reg_clipped = np.clip(T_reg, 0, 15)
            box_data.append(T_reg_clipped)
            box_labels.append(f"{REGION_NAMES[reg_id]}\n(n={int(mask.sum())})")

        if box_data:
            bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                           showfliers=False, whis=[5, 95])
            colors = ["#ff6b6b", "#ffd93d", "#6bcf7f"]
            for patch, color in zip(bp["boxes"], colors[:len(box_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

        ax.set_ylabel("|T| (whisk 5-95%)")
        ax.set_title(f"src_{si}: |T| by region")
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig2_path = RESULTS_DIR / "diag_scene13_T_stats.png"
    fig.savefig(fig2_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", fig2_path)

    # ------------------------------------------------------------------
    # FIGURE 3: |T| vs frequency for top outlier receivers per source
    # ------------------------------------------------------------------
    logger.info("Generating |T| frequency spectra for outlier receivers ...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Scene 13 (step): |T| vs Frequency for Top-5 Outlier Receivers", fontsize=13)

    for si in range(n_src):
        ax = axes[si]
        mean_T_per_rcv = np.mean(T_abs_all[si], axis=0)  # (R,)
        top5_idx = np.argsort(mean_T_per_rcv)[-5:][::-1]

        for rank, ridx in enumerate(top5_idx):
            T_spec = T_abs_all[si][:, ridx]  # (F,)
            label = (
                f"rcv_{ridx} ({rcv_pos[ridx, 0]:+.2f},{rcv_pos[ridx, 1]:+.2f}) "
                f"mean={mean_T_per_rcv[ridx]:.2f}"
            )
            ax.plot(freqs_hz / 1000.0, T_spec, alpha=0.7, label=label, linewidth=1.0)

        # Also plot the median receiver for comparison
        median_idx = np.argsort(mean_T_per_rcv)[n_rcv // 2]
        ax.plot(
            freqs_hz / 1000.0, T_abs_all[si][:, median_idx],
            color="black", linestyle="--", alpha=0.5,
            label=f"median rcv_{median_idx} (mean={mean_T_per_rcv[median_idx]:.3f})",
            linewidth=1.5,
        )

        ax.set_xlabel("Frequency [kHz]")
        ax.set_ylabel("|T|")
        ax.set_title(f"src_{si}")
        ax.legend(fontsize=6, loc="upper left")
        ax.grid(True, alpha=0.2)
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig3_path = RESULTS_DIR / "diag_scene13_T_freq_spectra.png"
    fig.savefig(fig3_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", fig3_path)

    # ------------------------------------------------------------------
    # FIGURE 4: |T| spatial map comparison at multiple frequencies
    # ------------------------------------------------------------------
    logger.info("Generating multi-frequency |T| maps for the worst source ...")

    # Find worst source
    worst_si = int(np.argmax([np.mean(T_abs_all[si]) for si in range(n_src)]))
    freq_samples = [2000, 3000, 5000, 7000, 8000]
    freq_indices = [int(np.argmin(np.abs(freqs_hz - f))) for f in freq_samples]

    fig, axes = plt.subplots(1, len(freq_samples), figsize=(5 * len(freq_samples), 5))
    fig.suptitle(
        f"Scene 13 (step): src_{worst_si} |T| maps at multiple frequencies", fontsize=13
    )

    vmax_multi = min(
        np.percentile(
            np.concatenate([T_abs_all[worst_si][fi, :] for fi in freq_indices]),
            98,
        ),
        10.0,
    )

    for i, fi in enumerate(freq_indices):
        ax = axes[i]
        T_at_freq = T_abs_all[worst_si][fi, :]

        sc = ax.scatter(
            rcv_pos[:, 0], rcv_pos[:, 1],
            c=T_at_freq, cmap="hot", s=15,
            vmin=0, vmax=vmax_multi,
            edgecolors="none", alpha=0.8,
        )
        draw_step_geometry(ax)
        ax.plot(
            src_pos[worst_si, 0], src_pos[worst_si, 1],
            marker="*", markersize=12, color="cyan", markeredgecolor="black",
        )

        ax.set_title(f"f={freqs_hz[fi]:.0f} Hz\nmean|T|={np.mean(T_at_freq):.3f}", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.2)
        plt.colorbar(sc, ax=ax, shrink=0.7, label="|T|")

    fig.tight_layout()
    fig4_path = RESULTS_DIR / "diag_scene13_T_multifreq.png"
    fig.savefig(fig4_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", fig4_path)

    # ------------------------------------------------------------------
    # SUMMARY: Root cause analysis
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("ROOT CAUSE ANALYSIS SUMMARY")
    print(f"{'=' * 80}")

    # Compute per-source overall error proxy: mean |T|
    mean_T_per_src = [float(np.mean(T_abs_all[si])) for si in range(n_src)]
    max_T_per_src = [float(np.max(T_abs_all[si])) for si in range(n_src)]
    p99_T_per_src = [float(np.percentile(T_abs_all[si], 99)) for si in range(n_src)]

    worst_idx = int(np.argmax(mean_T_per_src))
    best_idx = int(np.argmin(mean_T_per_src))

    print(f"\n  Worst source: src_{worst_idx} (mean|T|={mean_T_per_src[worst_idx]:.4f})")
    print(f"  Best source:  src_{best_idx} (mean|T|={mean_T_per_src[best_idx]:.4f})")
    print(f"  Ratio worst/best: {mean_T_per_src[worst_idx]/max(mean_T_per_src[best_idx], 1e-10):.2f}x")

    print(f"\n  Key Findings:")

    # Check 1: extreme |T| values
    max_T_overall = max(max_T_per_src)
    if max_T_overall > 5.0:
        print(f"  [!] EXTREME |T| VALUES: max|T|={max_T_overall:.2f}")
        print(f"      MLP struggles with high dynamic range (small+large values simultaneously)")
    else:
        print(f"  [ok] |T| dynamic range moderate (max={max_T_overall:.2f})")

    # Check 2: shadow zone dominance
    for si in range(n_src):
        labels = regions_all[si]
        shadow_frac = np.sum(labels == 0) / n_rcv
        if shadow_frac > 0.4:
            shadow_mean_T = float(np.mean(T_abs_all[si][:, labels == 0]))
            lit_mean_T_vals = T_abs_all[si][:, labels == 2]
            lit_mean_T = float(np.mean(lit_mean_T_vals)) if lit_mean_T_vals.size > 0 else 0.0
            print(
                f"  [!] src_{si}: {shadow_frac*100:.0f}% in shadow "
                f"(shadow mean|T|={shadow_mean_T:.3f}, lit mean|T|={lit_mean_T:.3f})"
            )
            if shadow_mean_T > 2.0 * lit_mean_T and lit_mean_T > 0:
                print(f"      --> Shadow zone |T| is {shadow_mean_T/lit_mean_T:.1f}x higher than lit zone")

    # Check 3: inter-body scattering
    if n_near_edge > 0:
        for si in range(n_src):
            near_mean = float(np.mean(T_abs_all[si][:, near_edge_mask]))
            far_mean = float(np.mean(T_abs_all[si][:, ~near_edge_mask]))
            if near_mean > 1.5 * far_mean:
                print(
                    f"  [!] src_{si}: Inter-body scattering amplification "
                    f"near edge ({near_mean:.3f} vs {far_mean:.3f}, {near_mean/far_mean:.1f}x)"
                )

    # Check 4: frequency dependence
    for si in range(n_src):
        mean_T_per_freq = np.mean(T_abs_all[si], axis=1)  # (F,)
        freq_variation = float(np.std(mean_T_per_freq) / np.mean(mean_T_per_freq))
        if freq_variation > 0.3:
            peak_fi = int(np.argmax(mean_T_per_freq))
            print(
                f"  [!] src_{si}: Strong frequency dependence "
                f"(CV={freq_variation:.2f}, peak at f={freqs_hz[peak_fi]:.0f} Hz)"
            )

    print(f"\n{'=' * 80}")
    print("DIAGNOSTIC COMPLETE")
    print(f"Figures saved to: {RESULTS_DIR}")
    print(f"  - diag_scene13_T_maps.png")
    print(f"  - diag_scene13_T_stats.png")
    print(f"  - diag_scene13_T_freq_spectra.png")
    print(f"  - diag_scene13_T_multifreq.png")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    run_diagnostic()
