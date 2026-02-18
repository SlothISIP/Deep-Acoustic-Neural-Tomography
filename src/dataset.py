"""Phase 1 HDF5 dataset for Phase 2 forward model training.

Loads BEM pressure data from Phase 1 HDF5 files and computes the
**transfer function** target T = p_scat / p_inc, which normalizes out
the dominant 1/sqrt(r) decay and exp(ikr) phase of the incident field.

Data schema per sample
----------------------
    Input:  (x_src, y_src, x_rcv, y_rcv, k, sdf_rcv, dist, dx, dy)  -- 9 floats

    Target (cartesian mode): (Re(T), Im(T))  -- 2 floats
    Target (log_polar mode): (log|R|, cos(angle(R)), sin(angle(R)))  -- 3 floats
        where R = p_total / p_inc = 1 + T

    T = p_scat / p_inc
    p_scat = p_total(BEM) - p_inc(analytical)
    p_inc  = -(i/4) H_0^{(1)}(k |x - x_s|)

Why transfer function?
    Direct regression of complex p_scat fails because the field oscillates
    rapidly with wavenumber k (phase span ~ 12 cycles across 2-8 kHz).
    Dividing by p_inc removes the dominant phase variation, yielding a
    smoother target that neural networks can learn effectively.

Why log-polar?
    Near geometric edges, |T| can exceed 35 (3500% dynamic range vs typical
    receivers).  Predicting (Re, Im) with MSE loss over-penalizes these
    extreme values while under-fitting the majority.  Log-polar targets
    compress the dynamic range to ~5:1 and the MSE loss on (log|R|, cos, sin)
    directly approximates per-sample relative error -- aligning with the
    gate metric ||delta p||_2 / ||p||_2.

Normalization
-------------
    Cartesian: Per-scene RMS normalization of T.
    Log-polar: Global z-score normalization per channel.
"""

import logging
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.special import hankel1
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPEED_OF_SOUND_M_PER_S: float = 343.0
N_RAW_FEATURES: int = 9
DEFAULT_NUM_WORKERS: int = 0 if platform.system() == "Windows" else 4


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class Phase1Dataset(Dataset):
    """PyTorch dataset for Phase 1 BEM data with transfer function targets.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``scene_NNN.h5`` files.
    scene_ids : list of int, optional
        Scenes to load (1-15).  ``None`` loads all 15.
    """

    def __init__(
        self,
        data_dir: Path,
        scene_ids: Optional[List[int]] = None,
        log_compress: bool = False,
        target_mode: str = "cartesian",
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.log_compress = log_compress
        self.target_mode = target_mode  # "cartesian" or "log_polar"
        self.target_stats: Optional[Dict[str, np.ndarray]] = None

        if target_mode not in ("cartesian", "log_polar"):
            raise ValueError(f"Unknown target_mode: {target_mode}")

        if scene_ids is None:
            scene_ids = list(range(1, 16))

        all_inputs: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []
        all_scene_ids: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        self.scene_scales: Dict[int, float] = {}

        for sid in scene_ids:
            h5_path = self.data_dir / f"scene_{sid:03d}.h5"
            if not h5_path.exists():
                logger.warning("Scene %d HDF5 not found: %s", sid, h5_path)
                continue

            logger.info("Loading scene %d: %s", sid, h5_path.name)
            inputs, targets, labels = self._load_scene(h5_path)

            if target_mode == "cartesian":
                # Optional log compression: sign(x) * log(1 + |x|)
                if self.log_compress:
                    targets = np.sign(targets) * np.log1p(np.abs(targets))

                # Per-scene RMS normalization
                rms = float(np.sqrt(np.mean(np.sum(targets ** 2, axis=1))))
                rms = max(rms, 1e-10)
                self.scene_scales[sid] = rms
                targets_norm = targets / rms
            else:
                # log_polar: no per-scene normalization (done globally after loop)
                self.scene_scales[sid] = 1.0
                targets_norm = targets

            all_inputs.append(inputs)
            all_targets.append(targets_norm)
            all_scene_ids.append(np.full(len(inputs), sid, dtype=np.int32))
            all_labels.append(labels)

            logger.info(
                "  Scene %d: %d samples, scale=%.4e",
                sid,
                len(inputs),
                self.scene_scales[sid],
            )

        all_targets_np = np.concatenate(all_targets, axis=0)

        # Global z-score normalization for log_polar targets
        if target_mode == "log_polar":
            target_mean = all_targets_np.mean(axis=0)  # (3,)
            target_std = all_targets_np.std(axis=0)  # (3,)
            target_std = np.maximum(target_std, 1e-8)
            all_targets_np = (all_targets_np - target_mean) / target_std
            self.target_stats = {
                "mean": target_mean.tolist(),
                "std": target_std.tolist(),
            }
            logger.info(
                "  Log-polar target stats: mean=%s, std=%s",
                [f"{m:.4f}" for m in target_mean],
                [f"{s:.4f}" for s in target_std],
            )

        self.inputs = torch.from_numpy(
            np.concatenate(all_inputs, axis=0)
        ).float()  # (N, 9)
        self.targets = torch.from_numpy(
            all_targets_np
        ).float()  # (N, 2) for cartesian or (N, 3) for log_polar
        self.scene_ids = torch.from_numpy(
            np.concatenate(all_scene_ids, axis=0)
        ).int()  # (N,)
        self.region_labels = torch.from_numpy(
            np.concatenate(all_labels, axis=0)
        ).int()  # (N,)

        # Per-sample scale for fast retrieval
        scale_map = self.scene_scales
        self.scales = torch.tensor(
            [scale_map[int(sid)] for sid in self.scene_ids],
            dtype=torch.float32,
        )  # (N,)

        logger.info(
            "Dataset loaded: %d samples from %d scenes (target_mode=%s)",
            len(self),
            len(self.scene_scales),
            target_mode,
        )

    # ------------------------------------------------------------------
    def _load_scene(
        self,
        h5_path: Path,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load one scene and compute transfer function targets.

        T(x_r, k) = p_scat / p_inc

        Returns
        -------
        inputs : (S*F*R, 9) float64
        targets : (S*F*R, 2 or 3) float64
            cartesian: (Re(T), Im(T))
            log_polar: (log|R|, cos(angle(R)), sin(angle(R)))  where R = 1 + T
        labels : (S*F*R,) int32
        """
        with h5py.File(h5_path, "r") as f:
            freqs_hz = f["frequencies"][:]  # (F,)
            src_pos = f["sources/positions"][:]  # (S, 2)
            rcv_pos = f["receivers/positions"][:]  # (R, 2)

            sdf_grid_x = f["sdf/grid_x"][:]
            sdf_grid_y = f["sdf/grid_y"][:]
            sdf_values = f["sdf/values"][:]

            F = len(freqs_hz)
            S = src_pos.shape[0]
            R = rcv_pos.shape[0]

            k_arr = 2.0 * np.pi * freqs_hz / SPEED_OF_SOUND_M_PER_S  # (F,)

            sdf_interp = RegularGridInterpolator(
                (sdf_grid_x, sdf_grid_y),
                sdf_values,
                method="linear",
                bounds_error=False,
                fill_value=1.0,
            )
            sdf_at_rcv = sdf_interp(rcv_pos)  # (R,)

            src_inputs: List[np.ndarray] = []
            src_targets: List[np.ndarray] = []
            src_labels: List[np.ndarray] = []

            for si in range(S):
                p_total = f[f"pressure/src_{si:03d}/field"][:]  # (F, R) complex128
                region_lab = f[f"regions/src_{si:03d}/labels"][:]  # (R,)

                xs_m, ys_m = src_pos[si]
                dx_sr = rcv_pos[:, 0] - xs_m  # (R,)
                dy_sr = rcv_pos[:, 1] - ys_m  # (R,)
                dist_sr = np.sqrt(dx_sr ** 2 + dy_sr ** 2)  # (R,)
                dist_sr_safe = np.maximum(dist_sr, 1e-15)

                # Vectorized incident field: (F, R) complex128
                kr = k_arr[:, None] * dist_sr_safe[None, :]  # (F, R)
                p_inc = -0.25j * hankel1(0, kr)  # (F, R)

                # Scattered field
                p_scat = p_total - p_inc  # (F, R)

                # Transfer function: T = p_scat / p_inc
                p_inc_safe = np.where(
                    np.abs(p_inc) > 1e-15, p_inc, 1e-15 + 0j
                )
                T = p_scat / p_inc_safe  # (F, R) complex128

                # Build features -- row-major (freq-first)
                n = F * R
                inputs_src = np.column_stack(
                    [
                        np.full(n, xs_m),
                        np.full(n, ys_m),
                        np.tile(rcv_pos[:, 0], F),
                        np.tile(rcv_pos[:, 1], F),
                        np.repeat(k_arr, R),
                        np.tile(sdf_at_rcv, F),
                        np.tile(dist_sr, F),
                        np.tile(dx_sr, F),
                        np.tile(dy_sr, F),
                    ]
                )  # (F*R, 9)

                if self.target_mode == "log_polar":
                    # ratio = p_total / p_inc = 1 + T
                    ratio = 1.0 + T  # (F, n_rcv) complex128
                    abs_ratio = np.abs(ratio)  # (F, n_rcv)
                    abs_ratio_safe = np.maximum(abs_ratio, 1e-15)
                    log_abs_R = np.log(abs_ratio_safe)  # (F, n_rcv)
                    angle_R = np.angle(ratio)  # (F, n_rcv)
                    targets_src = np.column_stack([
                        log_abs_R.ravel(),
                        np.cos(angle_R).ravel(),
                        np.sin(angle_R).ravel(),
                    ])  # (F*R, 3)
                else:
                    targets_src = np.column_stack(
                        [T.ravel().real, T.ravel().imag]
                    )  # (F*R, 2)

                labels_src = np.tile(region_lab, F).astype(np.int32)

                src_inputs.append(inputs_src)
                src_targets.append(targets_src)
                src_labels.append(labels_src)

        inputs = np.concatenate(src_inputs, axis=0)
        targets = np.concatenate(src_targets, axis=0)
        labels = np.concatenate(src_labels, axis=0)

        if not np.all(np.isfinite(inputs)):
            raise ValueError(
                f"Non-finite inputs: {np.sum(~np.isfinite(inputs))}"
            )
        if not np.all(np.isfinite(targets)):
            raise ValueError(
                f"Non-finite targets: {np.sum(~np.isfinite(targets))}"
            )

        return inputs, targets, labels

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (inputs, targets, scale, region_label)."""
        return (
            self.inputs[idx],
            self.targets[idx],
            self.scales[idx],
            self.region_labels[idx],
        )

    def get_input_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-feature mean and std for z-score normalization."""
        mean = self.inputs.mean(dim=0)
        std = self.inputs.std(dim=0)
        std = torch.clamp(std, min=1e-8)
        return mean, std


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
def create_dataloaders(
    data_dir: Path,
    batch_size: int = 8192,
    val_fraction: float = 0.2,
    seed: int = 42,
    num_workers: Optional[int] = None,
    scene_ids: Optional[List[int]] = None,
) -> Tuple[DataLoader, DataLoader, Phase1Dataset]:
    """Create train / val DataLoaders from Phase 1 HDF5 data.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``scene_NNN.h5`` files.
    batch_size : int
        Training batch size.
    val_fraction : float
        Fraction of data held out for validation.
    seed : int
        Random seed for reproducible split.
    num_workers : int, optional
        DataLoader workers (default: 0 on Windows, 4 elsewhere).
    scene_ids : list of int, optional
        Subset of scenes to load.

    Returns
    -------
    train_loader, val_loader, dataset
    """
    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS

    dataset = Phase1Dataset(data_dir, scene_ids=scene_ids)

    n_total = len(dataset)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=generator
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        "DataLoaders: train=%d, val=%d, batch=%d",
        n_train,
        n_val,
        batch_size,
    )

    return train_loader, val_loader, dataset
