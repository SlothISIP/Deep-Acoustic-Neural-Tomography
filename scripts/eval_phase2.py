"""Phase 2 Gate Evaluation: BEM Reconstruction Error.

Loads the trained transfer function model and evaluates reconstruction
error against Phase 1 BEM ground truth.

Reconstruction
--------------
    T_pred = model(inputs)                     -- normalized (Re, Im) of transfer function
    T_complex = (Re + j*Im) * scene_scale      -- denormalized
    p_total = p_inc * (1 + T_complex)           -- total field

Gate criterion
--------------
    Overall relative L2 error < 5%:
        error = ||p_pred_total - p_BEM_total||_2 / ||p_BEM_total||_2

Evaluation breakdown
--------------------
    - Per-scene error
    - Per-region error (shadow / transition / lit)
    - Per-frequency error (averaged over receivers)

Usage
-----
    python scripts/eval_phase2.py                    # evaluate best model
    python scripts/eval_phase2.py --checkpoint latest # use latest checkpoint
    python scripts/eval_phase2.py --scenes 1 2 3      # specific scenes
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scipy.interpolate import RegularGridInterpolator
from scipy.special import hankel1
from src.forward_model import TransferFunctionModel, build_transfer_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase2_eval")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "phase1"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase2"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase2"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPEED_OF_SOUND_M_PER_S: float = 343.0
REGION_NAMES = {0: "shadow", 1: "transition", 2: "lit"}
GATE_THRESHOLD: float = 0.05  # 5% relative error


# ---------------------------------------------------------------------------
# Evaluation per scene
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_scene(
    model: TransferFunctionModel,
    h5_path: Path,
    scene_scale: float,
    scene_id_0idx: int,
    device: torch.device,
    log_compress: bool = False,
    target_mode: str = "cartesian",
    target_stats: Optional[Dict] = None,
    calibrate: bool = False,
    calibrate_region: bool = False,
) -> Dict:
    """Evaluate reconstruction error for one scene.

    Reconstruction:
        cartesian: p_total = p_inc * (1 + T_complex * scale)
        log_polar: p_total = p_inc * R,  where R = |R|*exp(j*angle_R)
                   |R| = exp(log_R_denorm), angle_R = atan2(sin_denorm, cos_denorm)

    Returns
    -------
    dict with per-source, per-region, and aggregate error metrics.
    """
    with h5py.File(h5_path, "r") as f:
        freqs_hz = f["frequencies"][:]  # (F,)
        src_pos = f["sources/positions"][:]  # (S, 2)
        rcv_pos = f["receivers/positions"][:]  # (R, 2)

        sdf_grid_x = f["sdf/grid_x"][:]
        sdf_grid_y = f["sdf/grid_y"][:]
        sdf_values = f["sdf/values"][:]

        n_freq = len(freqs_hz)
        n_src = src_pos.shape[0]
        n_rcv = rcv_pos.shape[0]

        k_arr = 2.0 * np.pi * freqs_hz / SPEED_OF_SOUND_M_PER_S  # (F,)

        sdf_interp = RegularGridInterpolator(
            (sdf_grid_x, sdf_grid_y),
            sdf_values,
            method="linear",
            bounds_error=False,
            fill_value=1.0,
        )
        sdf_at_rcv = sdf_interp(rcv_pos)  # (R,)

        scene_results = {
            "per_source": [],
            "per_region": {0: [], 1: [], 2: []},
            "per_freq": np.zeros(n_freq),
            "per_freq_norm": np.zeros(n_freq),
        }

        total_diff_sq = 0.0
        total_ref_sq = 0.0

        for si in range(n_src):
            p_total_bem = f[f"pressure/src_{si:03d}/field"][:]  # (F, R) complex128
            region_lab = f[f"regions/src_{si:03d}/labels"][:]  # (R,)

            xs_m, ys_m = src_pos[si]
            dx_sr = rcv_pos[:, 0] - xs_m  # (R,)
            dy_sr = rcv_pos[:, 1] - ys_m  # (R,)
            dist_sr = np.sqrt(dx_sr ** 2 + dy_sr ** 2)  # (R,)
            dist_sr_safe = np.maximum(dist_sr, 1e-15)

            p_total_pred_np = np.zeros((n_freq, n_rcv), dtype=np.complex128)

            # Process in frequency chunks for memory efficiency
            chunk_size = 50
            for fi_start in range(0, n_freq, chunk_size):
                fi_end = min(fi_start + chunk_size, n_freq)
                n_f = fi_end - fi_start

                # Build input features: (n_f*R, 9)
                n = n_f * n_rcv
                inputs = np.column_stack([
                    np.full(n, xs_m),
                    np.full(n, ys_m),
                    np.tile(rcv_pos[:, 0], n_f),
                    np.tile(rcv_pos[:, 1], n_f),
                    np.repeat(k_arr[fi_start:fi_end], n_rcv),
                    np.tile(sdf_at_rcv, n_f),
                    np.tile(dist_sr, n_f),
                    np.tile(dx_sr, n_f),
                    np.tile(dy_sr, n_f),
                ]).astype(np.float32)  # (n_f*R, 9)

                inputs_t = torch.from_numpy(inputs).to(device)
                sid_t = torch.full(
                    (n,), scene_id_0idx, dtype=torch.long, device=device
                )
                pred_raw = model(inputs_t, scene_ids=sid_t).cpu().numpy()

                # Incident field: p_inc = -(i/4) H_0^(1)(kr)
                kr = k_arr[fi_start:fi_end, None] * dist_sr_safe[None, :]  # (n_f, R)
                p_inc = -0.25j * hankel1(0, kr)  # (n_f, R)

                if target_mode == "log_polar":
                    # pred_raw: (n_f*R, 3) = (log|R|_norm, cos_norm, sin_norm)
                    ts = target_stats
                    ts_mean = np.array(ts["mean"])  # (3,)
                    ts_std = np.array(ts["std"])  # (3,)
                    pred_denorm = pred_raw * ts_std + ts_mean  # (n_f*R, 3)

                    abs_R = np.exp(pred_denorm[:, 0])  # (n_f*R,)
                    angle_R = np.arctan2(pred_denorm[:, 2], pred_denorm[:, 1])
                    R_complex = abs_R * np.exp(1j * angle_R)  # (n_f*R,)

                    p_pred_chunk = p_inc * R_complex.reshape(n_f, n_rcv)
                else:
                    # pred_raw: (n_f*R, 2) = (Re(T)_norm, Im(T)_norm)
                    t_re_denorm = pred_raw[:, 0] * scene_scale  # (n_f*R,)
                    t_im_denorm = pred_raw[:, 1] * scene_scale  # (n_f*R,)

                    # Decompress if log-compressed targets were used
                    if log_compress:
                        t_re_raw = np.sign(t_re_denorm) * np.expm1(
                            np.abs(t_re_denorm)
                        )
                        t_im_raw = np.sign(t_im_denorm) * np.expm1(
                            np.abs(t_im_denorm)
                        )
                    else:
                        t_re_raw = t_re_denorm
                        t_im_raw = t_im_denorm

                    t_complex = t_re_raw + 1j * t_im_raw  # (n_f*R,)
                    p_pred_chunk = p_inc * (1.0 + t_complex.reshape(n_f, n_rcv))

                p_total_pred_np[fi_start:fi_end] = p_pred_chunk

            # Calibration: find optimal T-scaling alpha per source
            if calibrate and target_mode != "log_polar":
                # p_scat_pred = p_total_pred - p_inc_all,  p_scat_gt = p_total_bem - p_inc_all
                p_inc_all = np.zeros((n_freq, n_rcv), dtype=np.complex128)
                for fi2 in range(0, n_freq, chunk_size):
                    fi2_end = min(fi2 + chunk_size, n_freq)
                    kr2 = k_arr[fi2:fi2_end, None] * dist_sr_safe[None, :]
                    p_inc_all[fi2:fi2_end] = -0.25j * hankel1(0, kr2)
                p_scat_pred = p_total_pred_np - p_inc_all  # (F, R)
                p_scat_gt = p_total_bem - p_inc_all  # (F, R)
                # Per-source alpha: minimize ||alpha * p_scat_pred - p_scat_gt||^2
                cal_numer_src = np.real(np.sum(np.conj(p_scat_pred) * p_scat_gt))
                cal_denom_src = np.sum(np.abs(p_scat_pred) ** 2)
                alpha_this = cal_numer_src / max(cal_denom_src, 1e-30)
                # Store per-source data for recomputation
                if "cal_per_source" not in scene_results:
                    scene_results["cal_per_source"] = []
                scene_results["cal_per_source"].append({
                    "alpha": alpha_this,
                    "p_scat_pred": p_scat_pred,
                    "p_inc": p_inc_all,
                    "p_total_bem": p_total_bem,
                    "region_lab": region_lab,
                })

            # Compute errors
            diff = p_total_pred_np - p_total_bem  # (F, R)
            diff_sq = np.abs(diff) ** 2  # (F, R)
            ref_sq = np.abs(p_total_bem) ** 2  # (F, R)

            # Overall L2 error for this source
            src_diff = np.sqrt(np.sum(diff_sq))
            src_ref = np.sqrt(np.sum(ref_sq))
            src_error = src_diff / max(src_ref, 1e-30)
            scene_results["per_source"].append(src_error)

            total_diff_sq += np.sum(diff_sq)
            total_ref_sq += np.sum(ref_sq)

            # Per-frequency error
            scene_results["per_freq"] += np.sum(diff_sq, axis=1)  # (F,)
            scene_results["per_freq_norm"] += np.sum(ref_sq, axis=1)  # (F,)

            # Per-region error
            for reg_id in [0, 1, 2]:
                mask = region_lab == reg_id
                if mask.sum() == 0:
                    continue
                reg_diff = np.sqrt(np.sum(diff_sq[:, mask]))
                reg_ref = np.sqrt(np.sum(ref_sq[:, mask]))
                reg_error = reg_diff / max(reg_ref, 1e-30)
                scene_results["per_region"][reg_id].append(reg_error)

    # Calibration: recompute errors with optimal alpha
    use_calibration = (calibrate or calibrate_region) and "cal_per_source" in scene_results
    if use_calibration:
        # Recompute all errors with calibrated predictions
        total_diff_sq = 0.0
        total_ref_sq = 0.0
        scene_results["per_source"] = []
        scene_results["per_region"] = {0: [], 1: [], 2: []}
        n_freq_cal = len(scene_results["per_freq"])
        scene_results["per_freq"] = np.zeros(n_freq_cal)
        scene_results["per_freq_norm"] = np.zeros(n_freq_cal)

        all_alphas_per_source = []  # type: list[float | dict]

        for si_c, src_data in enumerate(scene_results["cal_per_source"]):
            p_scat_pred_c = src_data["p_scat_pred"]  # (F, R)
            p_inc_c = src_data["p_inc"]  # (F, R)
            p_bem_c = src_data["p_total_bem"]  # (F, R)
            reg_c = src_data["region_lab"]  # (R,)

            if calibrate_region:
                # Per-source-per-region alpha
                p_cal = np.copy(p_inc_c)  # (F, R)
                region_alphas = {}
                for reg_id in [0, 1, 2]:
                    mask = reg_c == reg_id  # (R,)
                    if mask.sum() == 0:
                        continue
                    ps_pred_reg = p_scat_pred_c[:, mask]  # (F, R_reg)
                    ps_gt_reg = (p_bem_c[:, mask] - p_inc_c[:, mask])  # (F, R_reg)
                    numer = np.real(np.sum(np.conj(ps_pred_reg) * ps_gt_reg))
                    denom = np.sum(np.abs(ps_pred_reg) ** 2)
                    alpha_reg = numer / max(denom, 1e-30)
                    region_alphas[reg_id] = alpha_reg
                    p_cal[:, mask] += alpha_reg * p_scat_pred_c[:, mask]
                all_alphas_per_source.append(region_alphas)
            else:
                # Per-source alpha (original behavior)
                alpha_c = src_data["alpha"]
                p_cal = p_inc_c + alpha_c * p_scat_pred_c  # (F, R)
                all_alphas_per_source.append(alpha_c)

            diff_c = p_cal - p_bem_c
            diff_sq_c = np.abs(diff_c) ** 2
            ref_sq_c = np.abs(p_bem_c) ** 2

            src_err = np.sqrt(np.sum(diff_sq_c)) / max(np.sqrt(np.sum(ref_sq_c)), 1e-30)
            scene_results["per_source"].append(src_err)
            total_diff_sq += np.sum(diff_sq_c)
            total_ref_sq += np.sum(ref_sq_c)
            scene_results["per_freq"] += np.sum(diff_sq_c, axis=1)
            scene_results["per_freq_norm"] += np.sum(ref_sq_c, axis=1)

            for reg_id in [0, 1, 2]:
                mask = reg_c == reg_id
                if mask.sum() == 0:
                    continue
                reg_diff = np.sqrt(np.sum(diff_sq_c[:, mask]))
                reg_ref = np.sqrt(np.sum(ref_sq_c[:, mask]))
                scene_results["per_region"][reg_id].append(
                    reg_diff / max(reg_ref, 1e-30)
                )

        scene_results["calibration_alphas_per_source"] = all_alphas_per_source
        if calibrate_region:
            scene_results["calibration_mode"] = "region"
        else:
            scene_results["calibration_alpha"] = np.mean(
                [a for a in all_alphas_per_source if isinstance(a, (int, float))]
            )

        # Clean up stored data
        scene_results.pop("cal_per_source", None)

    # Aggregate
    scene_error = np.sqrt(total_diff_sq) / max(np.sqrt(total_ref_sq), 1e-30)
    scene_results["error"] = scene_error

    # Per-frequency relative error
    scene_results["per_freq_rel"] = np.sqrt(
        scene_results["per_freq"]
        / np.maximum(scene_results["per_freq_norm"], 1e-30)
    )

    return scene_results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(all_results: Dict[int, Dict]) -> str:
    """Generate Phase 2 gate report string."""
    lines = []
    lines.append("=" * 70)
    lines.append("Phase 2 Gate Report: BEM Reconstruction Error (Transfer Function)")
    lines.append("=" * 70)
    lines.append("")

    lines.append(
        f"{'Scene':>8} {'Error%':>10} {'Shadow%':>10} {'Trans%':>10} "
        f"{'Lit%':>10} {'Status':>8}"
    )
    lines.append("-" * 70)

    total_diff_sq = 0.0
    total_ref_sq = 0.0

    for sid in sorted(all_results.keys()):
        res = all_results[sid]
        err = res["error"]

        reg_strs = []
        for reg_id in [0, 1, 2]:
            if res["per_region"][reg_id]:
                reg_avg = np.mean(res["per_region"][reg_id])
                reg_strs.append(f"{reg_avg * 100:8.2f}%")
            else:
                reg_strs.append(f"{'N/A':>9}")

        status = "PASS" if err < GATE_THRESHOLD else "FAIL"
        lines.append(
            f"{sid:>8d} {err * 100:9.2f}% {reg_strs[0]} "
            f"{reg_strs[1]} {reg_strs[2]} {status:>8}"
        )

    lines.append("-" * 70)

    # Overall: weighted by energy (not simple mean)
    for sid in sorted(all_results.keys()):
        res = all_results[sid]
        total_diff_sq += np.sum(res["per_freq"])
        total_ref_sq += np.sum(res["per_freq_norm"])

    overall_error = np.sqrt(total_diff_sq / max(total_ref_sq, 1e-30))
    overall_status = "PASS" if overall_error < GATE_THRESHOLD else "FAIL"
    lines.append(
        f"{'Overall':>8} {overall_error * 100:9.2f}%"
        f"{'':>32} {overall_status:>8}"
    )
    lines.append("")

    lines.append("=" * 70)
    lines.append(f"Gate Criterion: overall error < {GATE_THRESHOLD * 100:.0f}%")
    lines.append(f"Result:         {overall_error * 100:.2f}%")
    lines.append(
        f"Decision:       "
        f"{'PASS -- Phase 3 UNLOCKED' if overall_status == 'PASS' else 'FAIL -- iterate Phase 2'}"
    )
    lines.append("=" * 70)

    return "\n".join(lines)


def plot_per_freq_error(
    all_results: Dict[int, Dict],
    freqs_hz: np.ndarray,
    output_path: Path,
) -> None:
    """Plot per-frequency reconstruction error across all scenes."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for sid in sorted(all_results.keys()):
        res = all_results[sid]
        ax.plot(
            freqs_hz / 1000.0,
            res["per_freq_rel"] * 100.0,
            alpha=0.5,
            label=f"Scene {sid}",
        )

    ax.axhline(
        y=GATE_THRESHOLD * 100, color="r", linestyle="--", label="5% gate"
    )
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel("Relative Error [%]")
    ax.set_title("Phase 2: Per-Frequency Reconstruction Error")
    ax.legend(ncol=3, fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Per-freq error plot: %s", output_path)


def plot_per_scene_error(
    all_results: Dict[int, Dict],
    output_path: Path,
) -> None:
    """Bar chart of per-scene reconstruction error."""
    sids = sorted(all_results.keys())
    errors = [all_results[s]["error"] * 100 for s in sids]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(
        range(len(sids)), errors, color="steelblue", edgecolor="navy"
    )

    for bar, err in zip(bars, errors):
        if err >= GATE_THRESHOLD * 100:
            bar.set_color("salmon")
            bar.set_edgecolor("darkred")

    ax.axhline(
        y=GATE_THRESHOLD * 100, color="r", linestyle="--", label="5% gate"
    )
    ax.set_xticks(range(len(sids)))
    ax.set_xticklabels([f"S{s}" for s in sids])
    ax.set_xlabel("Scene")
    ax.set_ylabel("Relative L2 Error [%]")
    ax.set_title("Phase 2: Per-Scene BEM Reconstruction Error")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Per-scene error plot: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 Gate Evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        help="Checkpoint name: 'best' or 'latest'",
    )
    parser.add_argument(
        "--scene13-checkpoint",
        type=str,
        default=None,
        help="Separate checkpoint for scene 13 (e.g., 'best_scene13')",
    )
    parser.add_argument(
        "--ensemble",
        type=str,
        default=None,
        help="Comma-separated checkpoint names to ensemble (e.g., 'best_v11,best_v13'). "
             "Predictions are averaged across models.",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Compute optimal per-source T-scaling factor alpha that minimizes "
             "L2(alpha*T_pred - T_gt). Corrects systematic magnitude bias.",
    )
    parser.add_argument(
        "--calibrate-region", action="store_true",
        help="Compute optimal per-source-per-region alpha. Separate scaling "
             "for shadow/transition/lit zones per source.",
    )
    parser.add_argument("--scenes", nargs="+", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---------------------------------------------------------------
    # Load checkpoint(s)
    # ---------------------------------------------------------------
    def load_model_from_ckpt(
        ckpt_name: str,
    ) -> tuple:
        """Load model and metadata from a checkpoint.

        Returns (model, scene_scales, trained_scene_list, config).
        """
        p = CHECKPOINT_DIR / f"{ckpt_name}.pt"
        if not p.exists():
            logger.error("Checkpoint not found: %s", p)
            sys.exit(1)

        ck = torch.load(p, map_location=device, weights_only=False)
        cfg = ck["config"]
        scales = ck["scene_scales"]

        logger.info(
            "Loaded checkpoint: %s (epoch %d, val_loss=%.4e)",
            p.name,
            ck["epoch"],
            ck["best_val_loss"],
        )

        mdl = build_transfer_model(
            d_hidden=cfg.get("d_hidden", cfg.get("hidden_dim", 768)),
            n_blocks=cfg.get("n_blocks", cfg.get("n_layers", 6)),
            n_fourier=cfg.get("n_fourier", 256),
            fourier_sigma=cfg.get("fourier_sigma", 30.0),
            dropout=cfg.get("dropout", 0.0),
            n_scenes=cfg.get("n_scenes", 0),
            scene_emb_dim=cfg.get("scene_emb_dim", 32),
            d_out=cfg.get("d_out", 2),
        )
        mdl.load_state_dict(ck["model_state_dict"])
        mdl = mdl.to(device)
        mdl.eval()

        # Scene ID mapping: trained_scene_list maps scene_id -> 0-indexed
        tsl = cfg.get("trained_scene_list", sorted(scales.keys()))
        return mdl, scales, tsl, cfg

    # Primary model (or ensemble)
    if args.ensemble is not None:
        # Ensemble mode: load multiple models and average predictions
        ckpt_names = [c.strip() for c in args.ensemble.split(",")]
        ensemble_models = []
        for cn in ckpt_names:
            mdl_e, scales_e, tsl_e, cfg_e = load_model_from_ckpt(cn)
            ensemble_models.append((mdl_e, scales_e, tsl_e, cfg_e))
        # Use first model's metadata for scene mapping
        _, scene_scales, trained_scene_list, config = ensemble_models[0]
        logger.info("Ensemble of %d models: %s", len(ensemble_models), ckpt_names)

        class _EnsembleWrapper:
            """Lightweight wrapper: averages predictions from multiple models."""

            def __init__(self, models_list):
                self.models_list = models_list  # list of (model, scales, tsl, cfg)

            @torch.no_grad()
            def __call__(self, inputs_t, scene_ids=None):
                preds = []
                for mdl_e, _, tsl_e, _ in self.models_list:
                    # Remap scene_ids if trained_scene_list differs
                    if scene_ids is not None and tsl_e != trained_scene_list:
                        # Build remapping: global_sid -> model's 0-index
                        # scene_ids are 0-indexed in PRIMARY model's tsl
                        # need to convert to this model's 0-index
                        sid_map_e = {s: i for i, s in enumerate(tsl_e)}
                        remap = []
                        for idx in range(len(trained_scene_list)):
                            gsid = trained_scene_list[idx]
                            remap.append(sid_map_e.get(gsid, 0))
                        remap_t = torch.tensor(remap, device=inputs_t.device)
                        remapped_ids = remap_t[scene_ids]
                        preds.append(mdl_e(inputs_t, scene_ids=remapped_ids))
                    else:
                        preds.append(mdl_e(inputs_t, scene_ids=scene_ids))
                return torch.stack(preds).mean(dim=0)

        model = _EnsembleWrapper(ensemble_models)
    else:
        model, scene_scales, trained_scene_list, config = load_model_from_ckpt(
            args.checkpoint
        )
    scene_id_map = {sid: idx for idx, sid in enumerate(trained_scene_list)}
    log_compress = config.get("log_compress", False)
    target_mode = config.get("target_mode", "cartesian")
    target_stats = config.get("target_stats", None)
    if log_compress:
        logger.info("Log compression enabled (targets were log-compressed)")
    if target_mode == "log_polar":
        logger.info("Log-polar target mode (d_out=3)")
        if target_stats is None:
            logger.error("Log-polar mode requires target_stats in config")
            sys.exit(1)

    # Optional scene 13 specialist model(s)
    model_s13 = None
    scene_id_map_s13: Dict[int, int] = {}
    scene_scales_s13: Dict[int, float] = {}
    log_compress_s13 = False
    target_mode_s13 = "cartesian"
    target_stats_s13 = None

    if args.scene13_checkpoint is not None:
        s13_ckpt_names = [c.strip() for c in args.scene13_checkpoint.split(",")]
        if len(s13_ckpt_names) == 1:
            # Single specialist
            (
                model_s13,
                scene_scales_s13,
                tsl_s13,
                config_s13,
            ) = load_model_from_ckpt(s13_ckpt_names[0])
            scene_id_map_s13 = {sid: idx for idx, sid in enumerate(tsl_s13)}
            log_compress_s13 = config_s13.get("log_compress", False)
            target_mode_s13 = config_s13.get("target_mode", "cartesian")
            target_stats_s13 = config_s13.get("target_stats", None)
            logger.info(
                "Scene 13 specialist loaded (trained on scenes: %s)", tsl_s13
            )
        else:
            # Multiple S13 specialists — ensemble them
            s13_models = []
            for cn in s13_ckpt_names:
                mdl_s, scales_s, tsl_s, cfg_s = load_model_from_ckpt(cn)
                s13_models.append((mdl_s, scales_s, tsl_s, cfg_s))
            _, scene_scales_s13, tsl_s13, config_s13 = s13_models[0]
            scene_id_map_s13 = {sid: idx for idx, sid in enumerate(tsl_s13)}
            log_compress_s13 = config_s13.get("log_compress", False)
            target_mode_s13 = config_s13.get("target_mode", "cartesian")
            target_stats_s13 = config_s13.get("target_stats", None)

            class _S13EnsembleWrapper:
                """Ensemble wrapper for S13 specialists."""

                def __init__(self, models_list, primary_tsl):
                    self.models_list = models_list
                    self.primary_tsl = primary_tsl

                @torch.no_grad()
                def __call__(self, inputs_t, scene_ids=None):
                    preds = []
                    for mdl_s, _, tsl_s, _ in self.models_list:
                        if scene_ids is not None and tsl_s != self.primary_tsl:
                            sid_map_s = {s: i for i, s in enumerate(tsl_s)}
                            remap = []
                            for idx2 in range(len(self.primary_tsl)):
                                gsid = self.primary_tsl[idx2]
                                remap.append(sid_map_s.get(gsid, 0))
                            remap_t = torch.tensor(
                                remap, device=inputs_t.device
                            )
                            remapped = remap_t[scene_ids]
                            preds.append(mdl_s(inputs_t, scene_ids=remapped))
                        else:
                            preds.append(mdl_s(inputs_t, scene_ids=scene_ids))
                    return torch.stack(preds).mean(dim=0)

            model_s13 = _S13EnsembleWrapper(s13_models, tsl_s13)
            logger.info(
                "Scene 13 ensemble: %d models (%s)", len(s13_models),
                ", ".join(s13_ckpt_names),
            )

    # ---------------------------------------------------------------
    # Evaluate each scene
    # ---------------------------------------------------------------
    scene_ids = args.scenes or list(range(1, 16))
    all_results: Dict[int, Dict] = {}
    freqs_hz_ref = None

    t0 = time.time()

    for sid in scene_ids:
        h5_path = DATA_DIR / f"scene_{sid:03d}.h5"
        if not h5_path.exists():
            logger.warning("Scene %d HDF5 not found: %s", sid, h5_path)
            continue

        # Choose model: specialist for scene 13 if available
        if sid == 13 and model_s13 is not None:
            cur_model = model_s13
            cur_scale = scene_scales_s13.get(sid)
            cur_sid_0idx = scene_id_map_s13.get(sid)
            cur_log_compress = log_compress_s13
            cur_target_mode = target_mode_s13
            cur_target_stats = target_stats_s13
        else:
            cur_model = model
            cur_scale = scene_scales.get(sid)
            cur_sid_0idx = scene_id_map.get(sid)
            cur_log_compress = log_compress
            cur_target_mode = target_mode
            cur_target_stats = target_stats

        if cur_scale is None or cur_sid_0idx is None:
            logger.warning(
                "Scene %d has no scale/ID in checkpoint -- skipping", sid
            )
            continue

        logger.info("Evaluating scene %d ...", sid)
        results = evaluate_scene(
            cur_model, h5_path, cur_scale, cur_sid_0idx, device,
            log_compress=cur_log_compress,
            target_mode=cur_target_mode,
            target_stats=cur_target_stats,
            calibrate=args.calibrate,
            calibrate_region=args.calibrate_region,
        )
        all_results[sid] = results

        cal_str = ""
        if "calibration_mode" in results and results["calibration_mode"] == "region":
            # Per-region alphas: show compact summary
            for si_al, al_dict in enumerate(results["calibration_alphas_per_source"]):
                parts = [f"s{si_al}:"]
                for rid, rname in [(0, "sh"), (1, "tr"), (2, "lt")]:
                    if rid in al_dict:
                        parts.append(f"{rname}={al_dict[rid]:.3f}")
                cal_str += " " + " ".join(parts)
        elif "calibration_alphas_per_source" in results:
            alphas_str = ",".join(
                f"{a:.4f}" for a in results["calibration_alphas_per_source"]
            )
            cal_str = f" alpha=[{alphas_str}]"
        elif "calibration_alpha" in results:
            cal_str = f" alpha={results['calibration_alpha']:.4f}"
        logger.info(
            "  Scene %d: error=%.2f%% (sources: %s)%s",
            sid,
            results["error"] * 100,
            ", ".join(f"{e * 100:.2f}%" for e in results["per_source"]),
            cal_str,
        )

        if freqs_hz_ref is None:
            with h5py.File(h5_path, "r") as f:
                freqs_hz_ref = f["frequencies"][:]

    elapsed = time.time() - t0
    logger.info("Evaluation complete in %.1f s", elapsed)

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report = generate_report(all_results)
    print(report)

    report_path = RESULTS_DIR / "phase2_gate_report.txt"
    with open(report_path, "w") as fh:
        fh.write(report)
    logger.info("Gate report: %s", report_path)

    # ---------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------
    if freqs_hz_ref is not None:
        plot_per_freq_error(
            all_results, freqs_hz_ref, RESULTS_DIR / "per_freq_error.png"
        )
    plot_per_scene_error(all_results, RESULTS_DIR / "per_scene_error.png")

    # ---------------------------------------------------------------
    # Gate decision
    # ---------------------------------------------------------------
    total_diff_sq = sum(np.sum(r["per_freq"]) for r in all_results.values())
    total_ref_sq = sum(
        np.sum(r["per_freq_norm"]) for r in all_results.values()
    )
    overall = np.sqrt(total_diff_sq / max(total_ref_sq, 1e-30))

    if overall < GATE_THRESHOLD:
        logger.info(
            "GATE PASSED: %.2f%% < %.0f%%",
            overall * 100,
            GATE_THRESHOLD * 100,
        )
        logger.info("Phase 3 UNLOCKED")
    else:
        logger.warning(
            "GATE FAILED: %.2f%% >= %.0f%%",
            overall * 100,
            GATE_THRESHOLD * 100,
        )
        logger.warning("Review training and adjust hyperparameters")


if __name__ == "__main__":
    main()
