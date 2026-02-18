"""Optimize S13 ensemble weights for gate pass."""
import torch
import numpy as np
import h5py
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.forward_model import TransferFunctionModel
from scipy.special import hankel1
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

device = torch.device("cuda")
CKPT_DIR = Path("checkpoints/phase2")
DATA_DIR = Path("data/phase1")
c_m_s = 343.0

specialist_names = [
    "best_v7_ft13", "best_v8", "best_v8_ft13",
    "best_v11_ft13", "best_v13_ft13", "best_v13_ft13b",
]

# Load S13 data
h5_path = DATA_DIR / "scene_013.h5"
with h5py.File(h5_path, "r") as f:
    freqs_hz = f["frequencies"][:]
    src_pos = f["sources/positions"][:]
    rcv_pos = f["receivers/positions"][:]
    sdf_gx = f["sdf/grid_x"][:]
    sdf_gy = f["sdf/grid_y"][:]
    sdf_vals = f["sdf/values"][:]
    n_src = src_pos.shape[0]
    n_freq = len(freqs_hz)
    n_rcv = rcv_pos.shape[0]

    p_total_bem_all = []
    region_all = []
    for si in range(n_src):
        p_total_bem_all.append(f[f"pressure/src_{si:03d}/field"][:])
        region_all.append(f[f"regions/src_{si:03d}/labels"][:])

k_arr = 2.0 * np.pi * freqs_hz / c_m_s
sdf_interp = RegularGridInterpolator(
    (sdf_gx, sdf_gy), sdf_vals,
    method="linear", bounds_error=False, fill_value=1.0,
)
sdf_at_rcv = sdf_interp(rcv_pos)  # (R,)

# Precompute p_inc for each source
p_inc_all = []
for si in range(n_src):
    xs_m, ys_m = src_pos[si]
    dx_sr = rcv_pos[:, 0] - xs_m
    dy_sr = rcv_pos[:, 1] - ys_m
    dist_sr = np.sqrt(dx_sr**2 + dy_sr**2)
    dist_sr_safe = np.maximum(dist_sr, 1e-15)
    kr = k_arr[:, None] * dist_sr_safe[None, :]
    p_inc_all.append(-0.25j * hankel1(0, kr))  # (F, R)

# Run each specialist
all_preds = {}

for sname in specialist_names:
    ckpt = torch.load(CKPT_DIR / f"{sname}.pt", map_location=device, weights_only=False)
    config = ckpt["config"]
    scene_scales = ckpt["scene_scales"]
    tsl = config.get("trained_scene_list", sorted(scene_scales.keys()))

    model = TransferFunctionModel(
        d_in=9, d_hidden=config["d_hidden"], n_blocks=config["n_blocks"],
        d_out=config.get("d_out", 2), n_fourier=config["n_fourier"],
        fourier_sigma=config["fourier_sigma"], dropout=config["dropout"],
        n_scenes=config["n_scenes"], scene_emb_dim=config.get("scene_emb_dim", 32),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sid_0idx = tsl.index(13)
    scale = scene_scales[13]

    preds = []
    for si in range(n_src):
        xs_m, ys_m = src_pos[si]
        dx_sr = rcv_pos[:, 0] - xs_m
        dy_sr = rcv_pos[:, 1] - ys_m
        dist_sr = np.sqrt(dx_sr**2 + dy_sr**2)

        p_total_pred = np.zeros((n_freq, n_rcv), dtype=np.complex128)
        chunk_size = 50
        for fi_start in range(0, n_freq, chunk_size):
            fi_end = min(fi_start + chunk_size, n_freq)
            n_f = fi_end - fi_start
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
            ]).astype(np.float32)

            inp_t = torch.from_numpy(inputs).to(device)
            sid_t = torch.full((n,), sid_0idx, dtype=torch.long, device=device)

            with torch.no_grad():
                pred_raw = model(inp_t, scene_ids=sid_t).cpu().numpy()

            t_re = pred_raw[:, 0] * scale
            t_im = pred_raw[:, 1] * scale
            t_complex = (t_re + 1j * t_im).reshape(n_f, n_rcv)
            p_total_pred[fi_start:fi_end] = p_inc_all[si][fi_start:fi_end] * (1.0 + t_complex)

        preds.append(p_total_pred)

    all_preds[sname] = preds
    print(f"{sname}: done")

# ---------------------------------------------------------------
print("\n=== Optimization ===")
n_sp = len(specialist_names)

def gate_error_w(weights):
    total_diff_sq = 0.0
    total_ref_sq = 0.0
    for si in range(n_src):
        p_avg = sum(w * all_preds[sn][si] for w, sn in zip(weights, specialist_names))
        diff_sq = np.abs(p_avg - p_total_bem_all[si]) ** 2
        ref_sq = np.abs(p_total_bem_all[si]) ** 2
        total_diff_sq += np.sum(diff_sq)
        total_ref_sq += np.sum(ref_sq)
    return np.sqrt(total_diff_sq / max(total_ref_sq, 1e-30))

eq_w = np.ones(n_sp) / n_sp
eq_err = gate_error_w(eq_w)
print(f"Equal weights S13 error: {eq_err*100:.2f}%")

# Individual model errors
for i, sn in enumerate(specialist_names):
    w_i = np.zeros(n_sp); w_i[i] = 1.0
    print(f"  {sn}: {gate_error_w(w_i)*100:.2f}%")

# Global optimal
constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
bounds = [(0, 1)] * n_sp
result = minimize(gate_error_w, eq_w, method="SLSQP", bounds=bounds, constraints=constraints)
opt_w = result.x
opt_err = gate_error_w(opt_w)
print(f"\nGlobal optimal S13 error: {opt_err*100:.2f}%")
for sn, w in zip(specialist_names, opt_w):
    if w > 0.001:
        print(f"  {sn}: {w:.4f}")

# Per-source optimal
print("\n--- Per-source optimal ---")
total_diff_sq = 0.0
total_ref_sq = 0.0
for si in range(n_src):
    def src_err(weights, si_=si):
        p_avg = sum(w * all_preds[sn][si_] for w, sn in zip(weights, specialist_names))
        diff_sq = np.abs(p_avg - p_total_bem_all[si_]) ** 2
        ref_sq = np.abs(p_total_bem_all[si_]) ** 2
        return np.sqrt(np.sum(diff_sq) / max(np.sum(ref_sq), 1e-30))

    res = minimize(src_err, eq_w, method="SLSQP", bounds=bounds, constraints=constraints)
    opt_w_src = res.x
    err = src_err(opt_w_src)
    wstr = ", ".join(
        f"{specialist_names[j].replace('best_', '')}={opt_w_src[j]:.3f}"
        for j in range(n_sp) if opt_w_src[j] > 0.01
    )
    print(f"  src{si}: {err*100:.2f}% [{wstr}]")

    p_avg = sum(w * all_preds[sn][si] for w, sn in zip(opt_w_src, specialist_names))
    diff_sq = np.abs(p_avg - p_total_bem_all[si]) ** 2
    ref_sq = np.abs(p_total_bem_all[si]) ** 2
    total_diff_sq += np.sum(diff_sq)
    total_ref_sq += np.sum(ref_sq)

per_src_err = np.sqrt(total_diff_sq / max(total_ref_sq, 1e-30))
print(f"\nPer-source optimal S13: {per_src_err*100:.2f}%")

# Per-source weights + alpha
print("\n--- Per-source optimal weights + alpha ---")
total_diff_sq = 0.0
total_ref_sq = 0.0
for si in range(n_src):
    def src_err_cal(params, si_=si):
        weights = params[:n_sp]
        wsum = np.sum(weights)
        if wsum < 1e-10:
            return 1.0
        weights = weights / wsum
        alpha = params[n_sp]
        p_avg = sum(w * all_preds[sn][si_] for w, sn in zip(weights, specialist_names))
        p_scat = p_avg - p_inc_all[si_]
        p_cal = p_inc_all[si_] + alpha * p_scat
        diff_sq = np.abs(p_cal - p_total_bem_all[si_]) ** 2
        ref_sq = np.abs(p_total_bem_all[si_]) ** 2
        return np.sqrt(np.sum(diff_sq) / max(np.sum(ref_sq), 1e-30))

    x0_cal = np.append(eq_w, 1.0)
    bounds_cal = [(0, 1)] * n_sp + [(0.5, 2.0)]
    res_cal = minimize(src_err_cal, x0_cal, method="SLSQP", bounds=bounds_cal)
    opt_params = res_cal.x
    err_cal = src_err_cal(opt_params)
    alpha_cal = opt_params[n_sp]
    print(f"  src{si}: {err_cal*100:.2f}% (alpha={alpha_cal:.3f})")

    # Compute
    weights = opt_params[:n_sp]
    weights = weights / max(np.sum(weights), 1e-10)
    p_avg = sum(w * all_preds[sn][si] for w, sn in zip(weights, specialist_names))
    p_scat = p_avg - p_inc_all[si]
    p_cal = p_inc_all[si] + alpha_cal * p_scat
    diff_sq = np.abs(p_cal - p_total_bem_all[si]) ** 2
    ref_sq = np.abs(p_total_bem_all[si]) ** 2
    total_diff_sq += np.sum(diff_sq)
    total_ref_sq += np.sum(ref_sq)

combined_err = np.sqrt(total_diff_sq / max(total_ref_sq, 1e-30))
print(f"\nPer-src weights+alpha S13: {combined_err*100:.2f}%")
print(f"vs equal: {eq_err*100:.2f}%")
