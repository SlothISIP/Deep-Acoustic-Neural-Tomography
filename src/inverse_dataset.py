"""Phase 3 Inverse Dataset: per-scene structured data for SDF reconstruction.

Unlike Phase 2's flattened dataset (all scenes concatenated, shuffled by
sample), the inverse model needs per-scene structured access:
  - SDF grid (Gx, Gy) for supervision and IoU
  - Pressure observations (S, F, R) for cycle-consistency
  - Scene metadata (scale, forward model scene index)

All 15 scenes are preloaded to CPU (~400 MB total) and transferred
per-scene to GPU during training to keep VRAM usage bounded.

Data schema per scene
---------------------
    SDF:     grid_coords (Gx*Gy, 2), sdf_flat (Gx*Gy,)
    Obs:     pressure (S, F, R) complex128
             src_pos (S, 2), rcv_pos (R, 2), freqs_hz (F,)
    Meta:    scene_scale (float), fwd_scene_idx (int)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPEED_OF_SOUND_M_PER_S: float = 343.0


# ---------------------------------------------------------------------------
# Per-scene data container
# ---------------------------------------------------------------------------
@dataclass
class InverseSceneData:
    """Structured data for one scene used in Phase 3 inverse training.

    Attributes
    ----------
    scene_id : int
        Scene number (1-15).
    sdf_grid : np.ndarray, shape (Gx, Gy)
        SDF values on regular grid [m].
    grid_x : np.ndarray, shape (Gx,)
        X coordinates of SDF grid [m].
    grid_y : np.ndarray, shape (Gy,)
        Y coordinates of SDF grid [m].
    grid_coords : np.ndarray, shape (Gx*Gy, 2)
        Flattened grid coordinates [m], row-major.
    sdf_flat : np.ndarray, shape (Gx*Gy,)
        Flattened SDF values [m], matching grid_coords.
    pressure : np.ndarray, shape (S, F, R), complex128
        Total pressure field p_total from BEM.
    src_pos : np.ndarray, shape (S, 2)
        Source positions [m].
    rcv_pos : np.ndarray, shape (R, 2)
        Receiver positions [m].
    freqs_hz : np.ndarray, shape (F,)
        Frequency array [Hz].
    k_arr : np.ndarray, shape (F,)
        Wavenumber array [rad/m].
    scene_scale : float
        Per-scene transfer function RMS scale from Phase 2 dataset.
    fwd_scene_idx : int
        0-indexed scene ID for the frozen forward model's scene embedding.
    """

    scene_id: int
    sdf_grid: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    grid_coords: np.ndarray
    sdf_flat: np.ndarray
    pressure: np.ndarray
    src_pos: np.ndarray
    rcv_pos: np.ndarray
    freqs_hz: np.ndarray
    k_arr: np.ndarray
    scene_scale: float
    fwd_scene_idx: int

    @property
    def n_sources(self) -> int:
        return self.src_pos.shape[0]

    @property
    def n_freqs(self) -> int:
        return len(self.freqs_hz)

    @property
    def n_receivers(self) -> int:
        return self.rcv_pos.shape[0]

    @property
    def n_grid(self) -> int:
        return len(self.sdf_flat)

    @property
    def n_observations(self) -> int:
        """Total number of (source, freq, receiver) tuples."""
        return self.n_sources * self.n_freqs * self.n_receivers

    def memory_bytes(self) -> int:
        """Approximate memory footprint in bytes."""
        return (
            self.sdf_grid.nbytes
            + self.grid_coords.nbytes
            + self.sdf_flat.nbytes
            + self.pressure.nbytes
            + self.src_pos.nbytes
            + self.rcv_pos.nbytes
            + self.freqs_hz.nbytes
            + self.k_arr.nbytes
        )


# ---------------------------------------------------------------------------
# Scene loader
# ---------------------------------------------------------------------------
def _load_one_scene(
    h5_path: Path,
    scene_id: int,
    scene_scale: float,
    fwd_scene_idx: int,
) -> InverseSceneData:
    """Load one scene from Phase 1 HDF5 into InverseSceneData.

    Parameters
    ----------
    h5_path : Path
        Path to scene_NNN.h5 file.
    scene_id : int
        Scene number (1-15).
    scene_scale : float
        Per-scene RMS scale from forward model checkpoint.
    fwd_scene_idx : int
        0-indexed forward model scene embedding index.

    Returns
    -------
    data : InverseSceneData
    """
    with h5py.File(h5_path, "r") as f:
        # SDF grid
        grid_x = f["sdf/grid_x"][:]  # (Gx,)
        grid_y = f["sdf/grid_y"][:]  # (Gy,)
        sdf_values = f["sdf/values"][:]  # (Gx, Gy)

        # Build flattened grid coordinates
        xx, yy = np.meshgrid(grid_x, grid_y, indexing="ij")  # (Gx, Gy)
        grid_coords = np.column_stack(
            [xx.ravel(), yy.ravel()]
        )  # (Gx*Gy, 2)
        sdf_flat = sdf_values.ravel()  # (Gx*Gy,)

        # Observation data
        freqs_hz = f["frequencies"][:]  # (F,)
        src_pos = f["sources/positions"][:]  # (S, 2)
        rcv_pos = f["receivers/positions"][:]  # (R, 2)

        n_src = src_pos.shape[0]
        n_freq = len(freqs_hz)
        n_rcv = rcv_pos.shape[0]

        # Load pressure for all sources
        pressure = np.zeros(
            (n_src, n_freq, n_rcv), dtype=np.complex128
        )  # (S, F, R)
        for si in range(n_src):
            pressure[si] = f[f"pressure/src_{si:03d}/field"][:]  # (F, R)

    k_arr = 2.0 * np.pi * freqs_hz / SPEED_OF_SOUND_M_PER_S  # (F,)

    # Validate
    if not np.all(np.isfinite(sdf_flat)):
        raise ValueError(
            f"Scene {scene_id}: non-finite SDF values: "
            f"{np.sum(~np.isfinite(sdf_flat))}"
        )
    if not np.all(np.isfinite(pressure)):
        raise ValueError(
            f"Scene {scene_id}: non-finite pressure: "
            f"{np.sum(~np.isfinite(pressure))}"
        )

    return InverseSceneData(
        scene_id=scene_id,
        sdf_grid=sdf_values,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_coords=grid_coords,
        sdf_flat=sdf_flat,
        pressure=pressure,
        src_pos=src_pos,
        rcv_pos=rcv_pos,
        freqs_hz=freqs_hz,
        k_arr=k_arr,
        scene_scale=scene_scale,
        fwd_scene_idx=fwd_scene_idx,
    )


def load_all_scenes(
    data_dir: Path,
    scene_scales: Dict[int, float],
    fwd_scene_id_map: Dict[int, int],
    scene_ids: Optional[List[int]] = None,
) -> Dict[int, InverseSceneData]:
    """Load all scenes for Phase 3 inverse model training.

    Parameters
    ----------
    data_dir : Path
        Directory containing scene_NNN.h5 files.
    scene_scales : dict
        Mapping scene_id -> RMS scale from forward model checkpoint.
    fwd_scene_id_map : dict
        Mapping scene_id -> 0-indexed forward model scene embedding index.
    scene_ids : list of int, optional
        Subset of scenes to load.  None loads all available.

    Returns
    -------
    scenes : dict
        Mapping scene_id -> InverseSceneData.
    """
    if scene_ids is None:
        scene_ids = sorted(scene_scales.keys())

    scenes: Dict[int, InverseSceneData] = {}
    total_bytes = 0

    for sid in scene_ids:
        h5_path = data_dir / f"scene_{sid:03d}.h5"
        if not h5_path.exists():
            logger.warning("Scene %d HDF5 not found: %s", sid, h5_path)
            continue

        scale = scene_scales.get(sid)
        fwd_idx = fwd_scene_id_map.get(sid)
        if scale is None or fwd_idx is None:
            logger.warning(
                "Scene %d missing from forward model metadata, skipping", sid
            )
            continue

        data = _load_one_scene(h5_path, sid, scale, fwd_idx)
        scenes[sid] = data
        total_bytes += data.memory_bytes()

        logger.info(
            "  Scene %02d: grid=%dx%d (%d pts), obs=%dx%dx%d (%d total), "
            "scale=%.4f",
            sid,
            len(data.grid_x),
            len(data.grid_y),
            data.n_grid,
            data.n_sources,
            data.n_freqs,
            data.n_receivers,
            data.n_observations,
            data.scene_scale,
        )

    logger.info(
        "Loaded %d scenes (%.1f MB CPU memory)",
        len(scenes),
        total_bytes / 1e6,
    )

    return scenes
