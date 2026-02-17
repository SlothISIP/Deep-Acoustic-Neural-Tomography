"""Phase 1 BEM Data Factory: generate acoustic datasets for 15 scenes.

Pipeline per scene
------------------
    1. Build mesh at freq_max_hz (Nyquist-safe for all frequencies)
    2. Frequency sweep with multi-source solve (shared matrix per freq)
    3. Per-frequency checkpointing to HDF5 (resume on interruption)
    4. RIR synthesis via IDFT (np.unwrap + np.fft.irfft)
    5. Causality gate validation (pre-arrival energy < 1e-4)
    6. SDF ground truth on dense grid
    7. Region labeling (shadow / transition / lit)

Gate criterion (Phase 1 → Phase 2)
-----------------------------------
    - Causality: h(t < t_arrival) energy ratio < 1e-4 for ALL source-receiver pairs
    - Dataset completeness: 15 scenes with SDF ground truth

HDF5 layout per scene
----------------------
    /config/scene_id, name, category, freq_min_hz, freq_max_hz, n_freqs
    /mesh/midpoints (N,2), normals (N,2), lengths (N,)
    /frequencies (F,)
    /sources/positions (S,2)
    /receivers/positions (R,2)
    /pressure/src_NNN/field (F,R) complex128
    /rir/src_NNN/waveform (T,), sample_rate_hz, causality_ratio, is_causal
    /sdf/grid_x (Gx,), grid_y (Gy,), values (Gx,Gy)
    /regions/src_NNN/labels (R,) int

Usage
-----
    python scripts/run_phase1.py                  # all 15 scenes
    python scripts/run_phase1.py --scenes 1 2 3   # specific scenes
    python scripts/run_phase1.py --resume          # resume interrupted run
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.bem2d import (
    SPEED_OF_SOUND_M_PER_S,
    Mesh2D,
    assemble_bem_matrix,
    compute_incident_field,
    evaluate_field,
    solve_bem,
)
from src.rir import (
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_RIR_LENGTH_S,
    synthesize_and_validate,
)
from src.scenes import SceneConfig, build_all_scenes, label_regions

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase1")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "phase1"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase1"
SDF_GRID_N: int = 200  # SDF ground-truth grid resolution per axis
SDF_GRID_EXTENT_M: float = 1.5  # +/- extent of SDF grid [m]
CAUSALITY_THRESHOLD: float = 1e-4


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------
def _scene_h5_path(scene_id: int) -> Path:
    """Return HDF5 file path for a scene."""
    return DATA_DIR / f"scene_{scene_id:03d}.h5"


def _init_h5(
    path: Path,
    scene: SceneConfig,
    mesh: Mesh2D,
    freqs_hz: np.ndarray,
) -> None:
    """Initialize HDF5 file with metadata, mesh, and frequency grid.

    Skips if file already exists with matching config.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "a") as f:
        # Config group
        if "config" not in f:
            cfg = f.create_group("config")
            cfg.attrs["scene_id"] = scene.scene_id
            cfg.attrs["name"] = scene.name
            cfg.attrs["category"] = scene.category
            cfg.attrs["freq_min_hz"] = scene.freq_min_hz
            cfg.attrs["freq_max_hz"] = scene.freq_max_hz
            cfg.attrs["n_freqs"] = scene.n_freqs

        # Mesh group
        if "mesh" not in f:
            mg = f.create_group("mesh")
            mg.create_dataset("midpoints", data=mesh.midpoints_m)
            mg.create_dataset("normals", data=mesh.normals)
            mg.create_dataset("lengths", data=mesh.lengths_m)

        # Frequency array
        if "frequencies" not in f:
            f.create_dataset("frequencies", data=freqs_hz)

        # Source / receiver positions
        if "sources" not in f:
            sg = f.create_group("sources")
            sg.create_dataset("positions", data=scene.source_positions_m)
        if "receivers" not in f:
            rg = f.create_group("receivers")
            rg.create_dataset("positions", data=scene.receiver_positions_m)


def _get_completed_freqs(path: Path, src_idx: int) -> set:
    """Return set of frequency indices already computed for a source."""
    completed = set()
    if not path.exists():
        return completed

    with h5py.File(path, "r") as f:
        key = f"pressure/src_{src_idx:03d}/field"
        if key not in f:
            return completed
        ds = f[key]
        # Check which rows have non-zero data
        for fi in range(ds.shape[0]):
            row = ds[fi, :]  # (R,)
            if np.any(row != 0.0):
                completed.add(fi)

    return completed


def _save_pressure_row(
    path: Path,
    src_idx: int,
    freq_idx: int,
    n_freqs: int,
    n_receivers: int,
    field_row: np.ndarray,
) -> None:
    """Save one frequency row of pressure data to HDF5.

    Creates the dataset on first call, then writes the row.
    """
    with h5py.File(path, "a") as f:
        key = f"pressure/src_{src_idx:03d}/field"
        if key not in f:
            grp_key = f"pressure/src_{src_idx:03d}"
            if grp_key not in f:
                f.create_group(grp_key)
            f.create_dataset(
                key,
                shape=(n_freqs, n_receivers),
                dtype=np.complex128,
                chunks=(1, n_receivers),
            )
        f[key][freq_idx, :] = field_row


def _save_rir(
    path: Path,
    src_idx: int,
    rcv_idx: int,
    waveform: np.ndarray,
    sample_rate_hz: float,
    causality_ratio: float,
    is_causal: bool,
) -> None:
    """Save RIR data for one source-receiver pair."""
    with h5py.File(path, "a") as f:
        key = f"rir/src_{src_idx:03d}/rcv_{rcv_idx:03d}"
        if key in f:
            del f[key]
        grp = f.create_group(key)
        grp.create_dataset("waveform", data=waveform)
        grp.attrs["sample_rate_hz"] = sample_rate_hz
        grp.attrs["causality_ratio"] = causality_ratio
        grp.attrs["is_causal"] = is_causal


def _save_sdf(
    path: Path,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    sdf_values: np.ndarray,
) -> None:
    """Save SDF ground truth grid."""
    with h5py.File(path, "a") as f:
        if "sdf" in f:
            del f["sdf"]
        sg = f.create_group("sdf")
        sg.create_dataset("grid_x", data=grid_x)
        sg.create_dataset("grid_y", data=grid_y)
        sg.create_dataset("values", data=sdf_values)


def _save_regions(
    path: Path,
    src_idx: int,
    labels: np.ndarray,
) -> None:
    """Save region labels for one source."""
    with h5py.File(path, "a") as f:
        key = f"regions/src_{src_idx:03d}/labels"
        if key in f:
            del f[key]
        parent = f"regions/src_{src_idx:03d}"
        if parent not in f:
            f.create_group(parent)
        f.create_dataset(key, data=labels)


# ---------------------------------------------------------------------------
# Core: frequency sweep with per-freq checkpointing
# ---------------------------------------------------------------------------
def run_frequency_sweep(
    scene: SceneConfig,
    mesh: Mesh2D,
    freqs_hz: np.ndarray,
    h5_path: Path,
) -> None:
    """Run BEM frequency sweep for all sources, saving per-frequency.

    At each frequency:
        1. Assemble BEM matrix (once per frequency)
        2. Solve for all sources (single factorization, multi-RHS)
        3. Evaluate field at all receivers for each source
        4. Save to HDF5 row-by-row

    Parameters
    ----------
    scene : SceneConfig
    mesh : Mesh2D
    freqs_hz : np.ndarray, shape (F,)
    h5_path : Path
    """
    S = len(scene.source_positions_m)
    R = len(scene.receiver_positions_m)
    F = len(freqs_hz)

    # Determine which frequencies are already done (for ALL sources)
    all_completed = None
    for s in range(S):
        completed_s = _get_completed_freqs(h5_path, s)
        if all_completed is None:
            all_completed = completed_s
        else:
            all_completed &= completed_s

    n_remaining = F - len(all_completed)
    if n_remaining == 0:
        logger.info("  All %d frequencies already computed. Skipping BEM sweep.", F)
        return

    logger.info(
        "  BEM sweep: F=%d total, %d remaining, S=%d sources, R=%d receivers, N=%d elements",
        F, n_remaining, S, R, mesh.n_elements,
    )

    t_sweep_start = time.time()

    for fi, freq_hz in enumerate(freqs_hz):
        if fi in all_completed:
            continue

        t_freq_start = time.time()
        k_rad_per_m = 2.0 * np.pi * freq_hz / SPEED_OF_SOUND_M_PER_S

        # 1. Assemble BEM matrix (shared across sources)
        A = assemble_bem_matrix(mesh, k_rad_per_m)  # (N, N)

        # 2. Build multi-source RHS
        rhs_cols = []
        for s in range(S):
            p_inc_bdy = compute_incident_field(
                mesh.midpoints_m, scene.source_positions_m[s], k_rad_per_m,
            )  # (N,)
            rhs_cols.append(p_inc_bdy)
        rhs = np.column_stack(rhs_cols)  # (N, S)

        # 3. Solve for all sources at once
        p_surface_all = solve_bem(A, rhs)  # (N, S)

        # 4. Evaluate and save for each source
        for s in range(S):
            p_eval = evaluate_field(
                scene.receiver_positions_m,
                mesh,
                p_surface_all[:, s],
                scene.source_positions_m[s],
                k_rad_per_m,
            )  # (R,)

            _save_pressure_row(h5_path, s, fi, F, R, p_eval)

        elapsed_s = time.time() - t_freq_start
        total_elapsed_s = time.time() - t_sweep_start

        # Progress logging (every 10 freqs or at end)
        if (fi + 1) % 10 == 0 or fi == F - 1:
            pct = (fi + 1) / F * 100
            avg_per_freq_s = total_elapsed_s / (fi + 1 - len(all_completed) + 1)
            eta_s = avg_per_freq_s * (F - fi - 1)
            logger.info(
                "  [%3d/%d] f=%.0f Hz, k=%.1f, %.1fs/freq, %.0f%% done, ETA %.0fs",
                fi + 1, F, freq_hz, k_rad_per_m, elapsed_s, pct, eta_s,
            )


# ---------------------------------------------------------------------------
# Core: RIR synthesis + causality validation
# ---------------------------------------------------------------------------
def run_rir_synthesis(
    scene: SceneConfig,
    freqs_hz: np.ndarray,
    h5_path: Path,
) -> List[dict]:
    """Synthesize RIRs for all source-receiver pairs and validate causality.

    Reads all pressure data first, closes HDF5, then processes and saves
    in a separate write pass (avoids concurrent read/write on Windows).

    Returns list of result dicts for the summary report.
    """
    S = len(scene.source_positions_m)
    R = len(scene.receiver_positions_m)

    # --- Read pass: load all pressure data into memory ---
    pressure_data = {}  # {src_idx: (F, R) complex128}
    with h5py.File(h5_path, "r") as f:
        for s in range(S):
            key = f"pressure/src_{s:03d}/field"
            pressure_data[s] = f[key][:]  # (F, R), complex128

    # --- Process pass: synthesize RIRs ---
    results = []
    rir_cache = []  # list of (s, r_idx, waveform, sr, ratio, is_causal)

    for s in range(S):
        pressure_all = pressure_data[s]  # (F, R)

        for r_idx in range(R):
            pressure_spectrum = pressure_all[:, r_idx]  # (F,)

            # Source-receiver distance
            dist_m = float(np.linalg.norm(
                scene.source_positions_m[s] - scene.receiver_positions_m[r_idx]
            ))

            rir_result = synthesize_and_validate(
                freqs_hz, pressure_spectrum, dist_m,
            )

            results.append({
                "src": s,
                "rcv": r_idx,
                "dist_m": dist_m,
                "causality_ratio": rir_result.causality_ratio,
                "is_causal": rir_result.is_causal,
                "parseval_error": rir_result.parseval_error,
            })

            rir_cache.append((
                s, r_idx, rir_result.waveform,
                rir_result.sample_rate_hz,
                rir_result.causality_ratio,
                rir_result.is_causal,
            ))

    # --- Write pass: save all RIRs to HDF5 ---
    with h5py.File(h5_path, "a") as f:
        for s, r_idx, waveform, sr, ratio, is_causal in rir_cache:
            key = f"rir/src_{s:03d}/rcv_{r_idx:03d}"
            if key in f:
                del f[key]
            grp = f.create_group(key)
            grp.create_dataset("waveform", data=waveform)
            grp.attrs["sample_rate_hz"] = sr
            grp.attrs["causality_ratio"] = ratio
            grp.attrs["is_causal"] = is_causal

    n_causal = sum(1 for r in results if r["is_causal"])
    n_total = len(results)
    max_ratio = max(r["causality_ratio"] for r in results)
    mean_ratio = np.mean([r["causality_ratio"] for r in results])

    logger.info(
        "  RIR synthesis: %d/%d causal (%.1f%%), max_ratio=%.2e, mean=%.2e",
        n_causal, n_total, 100 * n_causal / n_total, max_ratio, mean_ratio,
    )

    return results


# ---------------------------------------------------------------------------
# Core: SDF ground truth
# ---------------------------------------------------------------------------
def compute_and_save_sdf(scene: SceneConfig, h5_path: Path) -> None:
    """Compute SDF on dense grid and save to HDF5."""
    grid_x = np.linspace(-SDF_GRID_EXTENT_M, SDF_GRID_EXTENT_M, SDF_GRID_N)
    grid_y = np.linspace(-SDF_GRID_EXTENT_M, SDF_GRID_EXTENT_M, SDF_GRID_N)
    xx, yy = np.meshgrid(grid_x, grid_y, indexing="ij")  # (Gx, Gy)
    sdf_values = scene.sdf_func(xx, yy)  # (Gx, Gy)

    if not np.all(np.isfinite(sdf_values)):
        n_bad = int(np.sum(~np.isfinite(sdf_values)))
        logger.warning("  SDF has %d non-finite values", n_bad)

    _save_sdf(h5_path, grid_x, grid_y, sdf_values)
    logger.info("  SDF grid: %dx%d, extent=[%.1f, %.1f] m", SDF_GRID_N, SDF_GRID_N,
                -SDF_GRID_EXTENT_M, SDF_GRID_EXTENT_M)


# ---------------------------------------------------------------------------
# Core: region labeling
# ---------------------------------------------------------------------------
def compute_and_save_regions(
    scene: SceneConfig,
    mesh: Mesh2D,
    h5_path: Path,
) -> None:
    """Label receivers as shadow/transition/lit for each source."""
    S = len(scene.source_positions_m)
    for s in range(S):
        labels = label_regions(
            scene.source_positions_m[s],
            scene.receiver_positions_m,
            mesh,
        )  # (R,)
        _save_regions(h5_path, s, labels)

        n_shadow = int(np.sum(labels == 0))
        n_trans = int(np.sum(labels == 1))
        n_lit = int(np.sum(labels == 2))
        logger.info(
            "  Regions src_%03d: shadow=%d, transition=%d, lit=%d",
            s, n_shadow, n_trans, n_lit,
        )


# ---------------------------------------------------------------------------
# Per-scene pipeline
# ---------------------------------------------------------------------------
def process_scene(scene: SceneConfig) -> dict:
    """Full Phase 1 pipeline for one scene.

    Returns summary dict with gate results.
    """
    logger.info("=" * 70)
    logger.info("Scene %02d: %s (category=%s)", scene.scene_id, scene.name, scene.category)
    logger.info("=" * 70)

    h5_path = _scene_h5_path(scene.scene_id)
    t_scene_start = time.time()

    # 1. Build mesh at max frequency
    logger.info("Step 1: Building mesh at freq_max=%.0f Hz", scene.freq_max_hz)
    mesh = scene.mesh_builder(scene.freq_max_hz)
    mesh.validate()
    logger.info(
        "  Mesh: N=%d elements, h=[%.4f, %.4f] m",
        mesh.n_elements, np.min(mesh.lengths_m), np.max(mesh.lengths_m),
    )

    # 2. Frequency grid
    freqs_hz = np.linspace(scene.freq_min_hz, scene.freq_max_hz, scene.n_freqs)

    # 3. Initialize HDF5
    _init_h5(h5_path, scene, mesh, freqs_hz)

    # 4. BEM frequency sweep
    logger.info("Step 2: BEM frequency sweep")
    run_frequency_sweep(scene, mesh, freqs_hz, h5_path)

    # 5. RIR synthesis + causality check
    logger.info("Step 3: RIR synthesis + causality validation")
    rir_results = run_rir_synthesis(scene, freqs_hz, h5_path)

    # 6. SDF ground truth
    logger.info("Step 4: SDF ground truth")
    compute_and_save_sdf(scene, h5_path)

    # 7. Region labeling
    logger.info("Step 5: Region labeling")
    compute_and_save_regions(scene, mesh, h5_path)

    elapsed_s = time.time() - t_scene_start

    # Gate results
    n_total_pairs = len(rir_results)
    n_causal = sum(1 for r in rir_results if r["is_causal"])
    max_causality_ratio = max(r["causality_ratio"] for r in rir_results)
    mean_causality_ratio = float(np.mean([r["causality_ratio"] for r in rir_results]))
    gate_pass = n_causal == n_total_pairs

    summary = {
        "scene_id": scene.scene_id,
        "name": scene.name,
        "category": scene.category,
        "n_elements": mesh.n_elements,
        "n_freqs": len(freqs_hz),
        "n_sources": len(scene.source_positions_m),
        "n_receivers": len(scene.receiver_positions_m),
        "n_pairs": n_total_pairs,
        "n_causal": n_causal,
        "max_causality_ratio": max_causality_ratio,
        "mean_causality_ratio": mean_causality_ratio,
        "gate_pass": gate_pass,
        "elapsed_s": elapsed_s,
        "h5_path": str(h5_path),
    }

    status = "PASS" if gate_pass else "FAIL"
    logger.info(
        "Scene %02d %s: %d/%d causal, max_ratio=%.2e, time=%.0fs",
        scene.scene_id, status, n_causal, n_total_pairs,
        max_causality_ratio, elapsed_s,
    )

    return summary


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def print_summary(summaries: List[dict]) -> bool:
    """Print gate validation summary table.

    Returns True if ALL scenes pass.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 1 GATE VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(
        "%-4s %-20s %-6s %-5s %-5s %-8s %-10s %-10s %-6s",
        "ID", "Name", "N_el", "S", "R", "Pairs",
        "Max Ratio", "Mean Ratio", "Gate",
    )
    logger.info("-" * 80)

    all_pass = True
    for s in summaries:
        status = "PASS" if s["gate_pass"] else "FAIL"
        if not s["gate_pass"]:
            all_pass = False
        logger.info(
            "%4d %-20s %6d %5d %5d %8d %10.2e %10.2e %6s",
            s["scene_id"], s["name"], s["n_elements"],
            s["n_sources"], s["n_receivers"], s["n_pairs"],
            s["max_causality_ratio"], s["mean_causality_ratio"], status,
        )

    logger.info("-" * 80)

    total_pairs = sum(s["n_pairs"] for s in summaries)
    total_causal = sum(s["n_causal"] for s in summaries)
    total_time_s = sum(s["elapsed_s"] for s in summaries)
    overall_max = max(s["max_causality_ratio"] for s in summaries)

    logger.info(
        "Total: %d scenes, %d/%d pairs causal (%.2f%%), max_ratio=%.2e, time=%.0fs",
        len(summaries), total_causal, total_pairs,
        100 * total_causal / total_pairs if total_pairs > 0 else 0,
        overall_max, total_time_s,
    )

    overall = "PASS" if all_pass else "FAIL"
    logger.info("")
    logger.info("Phase 1 Gate: %s (threshold: causality ratio < %.0e)", overall, CAUSALITY_THRESHOLD)
    logger.info("=" * 80)

    # Save summary to text file
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "phase1_gate_report.txt"
    with open(report_path, "w") as fout:
        fout.write("Phase 1 Gate Validation Report\n")
        fout.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fout.write(f"Threshold: causality ratio < {CAUSALITY_THRESHOLD:.0e}\n\n")
        fout.write(f"{'ID':>4} {'Name':<20} {'N_el':>6} {'S':>3} {'R':>5} "
                    f"{'Pairs':>6} {'Max Ratio':>12} {'Mean Ratio':>12} {'Gate':>6}\n")
        fout.write("-" * 80 + "\n")
        for s in summaries:
            status = "PASS" if s["gate_pass"] else "FAIL"
            fout.write(
                f"{s['scene_id']:4d} {s['name']:<20} {s['n_elements']:6d} "
                f"{s['n_sources']:3d} {s['n_receivers']:5d} {s['n_pairs']:6d} "
                f"{s['max_causality_ratio']:12.2e} {s['mean_causality_ratio']:12.2e} "
                f"{status:>6}\n"
            )
        fout.write("-" * 80 + "\n")
        fout.write(f"\nOverall: {overall}\n")
        fout.write(f"Total pairs: {total_causal}/{total_pairs} causal\n")
        fout.write(f"Max causality ratio: {overall_max:.2e}\n")
        fout.write(f"Total time: {total_time_s:.0f}s\n")

    logger.info("Report saved: %s", report_path)

    return all_pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 1 BEM Data Factory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        type=int,
        default=None,
        help="Scene IDs to process (default: all 15)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted run (skip completed frequencies)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print scene info without running BEM",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for Phase 1 data factory."""
    args = parse_args()

    logger.info("Phase 1 BEM Data Factory")
    logger.info("Gate criterion: causality ratio < %.0e for ALL source-receiver pairs", CAUSALITY_THRESHOLD)
    logger.info("Output directory: %s", DATA_DIR)

    # Build all scene configs
    all_scenes = build_all_scenes()

    # Filter scenes if requested
    if args.scenes is not None:
        valid_ids = {s.scene_id for s in all_scenes}
        for sid in args.scenes:
            if sid not in valid_ids:
                logger.error("Invalid scene ID: %d (valid: %s)", sid, sorted(valid_ids))
                sys.exit(1)
        scenes = [s for s in all_scenes if s.scene_id in args.scenes]
    else:
        scenes = all_scenes

    logger.info("Processing %d scenes: %s", len(scenes), [s.scene_id for s in scenes])

    if args.dry_run:
        logger.info("Dry run -- printing scene info only")
        for sc in scenes:
            mesh = sc.mesh_builder(sc.freq_max_hz)
            logger.info(
                "  Scene %02d: %-20s N=%d, S=%d, R=%d, F=%d, freqs=[%.0f, %.0f] Hz",
                sc.scene_id, sc.name, mesh.n_elements,
                len(sc.source_positions_m), len(sc.receiver_positions_m),
                sc.n_freqs, sc.freq_min_hz, sc.freq_max_hz,
            )
        return

    # Process scenes
    summaries = []
    for scene in scenes:
        summary = process_scene(scene)
        summaries.append(summary)

    # Print and save summary
    all_pass = print_summary(summaries)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
