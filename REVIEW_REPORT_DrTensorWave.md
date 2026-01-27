# Dr. Tensor Wave's Critical Review Report
## Acoustic Neural Tomography Roadmap v3.1

**Review Date:** 2026-01-27
**Reviewer:** Dr. Tensor Wave (2050 AI Physicist)
**Verdict:** Conditionally Acceptable with Major Revisions

---

## Executive Summary

로드맵의 전반적 구조는 건전하나, **물리적 일관성에서 중대한 누락**이 발견되었다. 특히 복소수 압력장 처리, trivial solution 회피, phase unwrapping 문제는 구현 전 반드시 해결해야 한다.

---

## CRITICAL SEVERITY (P0 - 즉시 수정)

### C1. 복소수 압력장 처리 누락
**Location:** Phase 3 - Joint Output Network
**Problem:** `f_θ: (γ(x), t) → (Pressure p, SDF s)` - 시간 t를 입력으로 받으면서 실수 출력 가정

Helmholtz 방정식의 해 p(x)는 **복소수**다. 주파수 영역에서 작업하거나 explicit Re/Im heads가 필요.

**Fix:**
```python
# Option A: Frequency domain (Recommended)
class AcousticNeuralField(nn.Module):
    def forward(self, gamma_x, wavenumber_k):
        # gamma_x: [B, D_fourier] - Fourier features
        # wavenumber_k: [B, 1] - k = 2*pi*f/c
        features = self.backbone(gamma_x, wavenumber_k)  # [B, D_hidden]

        p_real = self.pressure_head_real(features)  # [B, 1]
        p_imag = self.pressure_head_imag(features)  # [B, 1]
        p_complex = torch.complex(p_real, p_imag)   # [B, 1] complex64

        sdf = self.sdf_head(features)  # [B, 1] real
        return p_complex, sdf

# Option B: Complex-valued neural network (CVNN)
# Use torch.nn layers with complex weights
```

### C2. Trivial Solution 회피 전략 부재
**Location:** Phase 3 전체
**Problem:** SDF=const, p=0이 모든 physics loss 만족

**Fix:**
```python
def compute_losses(model, coords, measurements, source_pos, k):
    p, sdf = model(coords, k)

    # 1. Data loss
    loss_data = F.mse_loss(p[mic_indices], measurements)

    # 2. Helmholtz with SOURCE TERM (not homogeneous!)
    laplacian_p = compute_laplacian(p, coords)
    source_term = gaussian_source(coords, source_pos, sigma=0.01)
    loss_helmholtz = torch.mean((laplacian_p + k**2 * p + source_term)**2)

    # 3. Eikonal
    grad_sdf = torch.autograd.grad(sdf.sum(), coords, create_graph=True)[0]
    loss_eikonal = torch.mean((grad_sdf.norm(dim=-1) - 1)**2)

    # 4. Surface existence constraint (CRITICAL!)
    # SDF must cross zero somewhere in the domain
    sdf_min, sdf_max = sdf.min(), sdf.max()
    loss_surface = F.relu(-sdf_min) + F.relu(sdf_max)  # Both signs must exist

    # 5. Boundary condition at surface
    near_surface = (sdf.abs() < 0.01)
    if near_surface.any():
        normal = grad_sdf[near_surface] / (grad_sdf[near_surface].norm(dim=-1, keepdim=True) + 1e-8)
        dp_dn = (torch.autograd.grad(p[near_surface].sum(), coords, create_graph=True)[0] * normal).sum(dim=-1)
        loss_bc = torch.mean(dp_dn**2)  # Neumann: dp/dn = 0
    else:
        loss_bc = torch.tensor(0.0)

    return loss_data, loss_helmholtz, loss_eikonal, loss_surface, loss_bc
```

### C3. Phase Unwrapping 미구현
**Location:** Phase 1.5 - IDFT Synthesis
**Problem:** Multi-frequency phase 누적 시 2π jump로 인한 acausal artifacts

**Fix:**
```python
import numpy as np
from scipy.signal import hilbert

def frequency_to_time_domain(P_freq, freqs_hz, t_samples):
    """
    Convert frequency-domain pressure to time-domain RIR.

    Args:
        P_freq: [N_freq,] complex - Pressure at each frequency
        freqs_hz: [N_freq,] - Frequency values
        t_samples: [N_time,] - Time samples for output

    Returns:
        h_t: [N_time,] real - Room impulse response
    """
    # Step 1: Phase unwrapping (CRITICAL!)
    phase_raw = np.angle(P_freq)  # [N_freq,]
    phase_unwrapped = np.unwrap(phase_raw)

    # Step 2: Reconstruct with unwrapped phase
    P_corrected = np.abs(P_freq) * np.exp(1j * phase_unwrapped)

    # Step 3: Ensure Hermitian symmetry for real output
    N = len(t_samples)
    P_full = np.zeros(N, dtype=complex)
    P_full[:len(P_freq)] = P_corrected
    P_full[N-len(P_freq)+1:] = np.conj(P_corrected[-1:0:-1])

    # Step 4: IDFT
    h_t = np.fft.irfft(P_full, n=N)

    # Step 5: Causality check
    t_negative = t_samples < 0
    acausal_energy = np.sum(h_t[t_negative]**2)
    total_energy = np.sum(h_t**2)
    if acausal_energy / total_energy > 1e-6:
        raise ValueError(f"Causality violated! Acausal energy ratio: {acausal_energy/total_energy:.2e}")

    return h_t
```

---

## HIGH SEVERITY (P1 - Phase 시작 전 수정)

### H1. Fourier Feature Scale 과소평가
**Current:** σ ≈ f_max/c ≈ 23 m⁻¹
**Problem:** Diffraction의 각도 의존성 미반영

**Analysis:**
- Wave vector magnitude: k = 2πf/c
- Diffraction pattern spatial frequency: k·sin(θ)
- At 8kHz, θ=60°: k·sin(60°) ≈ 127 rad/m
- Nyquist requires σ > k_max/(2π) ≈ 20, but diffraction needs higher

**Fix:**
```python
# Physics-informed Fourier Feature scale
def compute_fourier_scale(f_max_hz, c_m_per_s=343.0, max_angle_deg=60.0):
    """
    Compute optimal Fourier Feature scale for acoustic diffraction.

    The scale must capture spatial frequencies up to k*sin(theta_max)
    where k = 2*pi*f/c is the wavenumber.
    """
    k_max = 2 * np.pi * f_max_hz / c_m_per_s  # rad/m
    theta_max = np.radians(max_angle_deg)
    spatial_freq_max = k_max * np.sin(theta_max)  # rad/m

    # Fourier feature scale (with safety margin)
    sigma = spatial_freq_max / (2 * np.pi) * 1.5  # Safety factor 1.5
    return sigma  # Returns ~50-70 for 8kHz

# Usage
sigma = compute_fourier_scale(8000)  # ≈ 62 m⁻¹ (not 23!)
```

### H2. Mesh Resolution Near Edges
**Current:** λ/6 uniform
**Problem:** Diffraction requires finer resolution near edges

**Fix:**
```python
import pygmsh

def create_l_shape_mesh_adaptive(corner_pos, arm_length, f_max_hz, c=343.0):
    """
    Create L-shape mesh with adaptive refinement near edges.
    """
    wavelength_min = c / f_max_hz  # ~43mm at 8kHz

    # Element sizes
    edge_size = wavelength_min / 10   # 4.3mm near edges
    flat_size = wavelength_min / 6    # 7.2mm on flat surfaces

    with pygmsh.geo.Geometry() as geom:
        # Define L-shape vertices
        # ... geometry definition ...

        # Mesh size field: fine near edges, coarse elsewhere
        geom.set_mesh_size_callback(
            lambda dim, tag, x, y, z, lc:
                edge_size if is_near_edge(x, y, z, corner_pos, threshold=0.05)
                else flat_size
        )

        mesh = geom.generate_mesh()

    return mesh
```

### H3. Burton-Miller Parameter
**Problem:** Coupling parameter α not specified

**Fix:**
```python
import bempp.api

def create_helmholtz_operator(grid, wavenumber_k):
    """
    Create Burton-Miller combined field integral equation operator.

    The coupling parameter alpha = i/k is optimal for exterior problems,
    ensuring unique solution at all frequencies.
    """
    # Optimal coupling parameter
    alpha = 1j / wavenumber_k

    # Spaces
    space = bempp.api.function_space(grid, "P", 1)

    # Operators
    identity = bempp.api.operators.boundary.sparse.identity(space, space, space)
    dlp = bempp.api.operators.boundary.helmholtz.double_layer(space, space, space, wavenumber_k)
    slp = bempp.api.operators.boundary.helmholtz.single_layer(space, space, space, wavenumber_k)
    hyp = bempp.api.operators.boundary.helmholtz.hypersingular(space, space, space, wavenumber_k)
    adlp = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(space, space, space, wavenumber_k)

    # Burton-Miller: (1/2 I + D + alpha*H) * u = (S + alpha*(1/2 I + D')) * g
    lhs = 0.5 * identity + dlp + alpha * hyp
    rhs_op = slp + alpha * (0.5 * identity + adlp)

    return lhs, rhs_op
```

### H4. Energy Conservation Validation
**Problem:** Parseval's theorem check missing

**Fix:**
```python
def validate_energy_conservation(P_freq, h_t, delta_f, delta_t, rtol=0.01):
    """
    Validate energy conservation between frequency and time domains.

    Parseval's theorem: ∫|P(f)|²df = ∫|h(t)|²dt
    """
    E_freq = np.sum(np.abs(P_freq)**2) * delta_f
    E_time = np.sum(np.abs(h_t)**2) * delta_t

    relative_error = abs(E_freq - E_time) / max(E_freq, E_time)

    if relative_error > rtol:
        raise ValueError(
            f"Energy conservation violated!\n"
            f"  Frequency domain: {E_freq:.6e}\n"
            f"  Time domain: {E_time:.6e}\n"
            f"  Relative error: {relative_error:.2%}"
        )

    return True
```

---

## MEDIUM SEVERITY (P2 - 구현 중 고려)

### M1. Adaptive Frequency Sampling Strategy
```python
def adaptive_frequency_sampling(f_min_hz, f_max_hz, geometry, base_resolution_hz=20):
    """
    Non-uniform frequency sampling: dense near resonances, sparse elsewhere.
    """
    # Estimate room modes from geometry
    room_modes = estimate_room_modes(geometry)  # Returns list of frequencies

    frequencies = []
    f = f_min_hz
    while f <= f_max_hz:
        # Check proximity to resonances
        min_dist_to_mode = min(abs(f - mode) for mode in room_modes) if room_modes else np.inf

        if min_dist_to_mode < 50:  # Within 50Hz of a resonance
            step = 2  # Fine sampling: 2Hz
        elif min_dist_to_mode < 100:
            step = 5  # Medium: 5Hz
        else:
            step = base_resolution_hz  # Coarse: 20Hz

        frequencies.append(f)
        f += step

    return np.array(frequencies)
```

### M2. Sim2Real Domain Randomization
```python
class DomainRandomizer:
    """Apply domain randomization during training to bridge sim2real gap."""

    def __init__(self):
        self.absorption_range = (0.0, 0.3)  # MDF absorption coefficient
        self.snr_range = (10, 30)  # dB
        self.speed_of_sound_range = (340, 346)  # Temperature variation

    def randomize(self, rir, metadata):
        """Apply randomization to simulated RIR."""
        # 1. Random absorption (amplitude decay)
        absorption = np.random.uniform(*self.absorption_range)
        decay_factor = np.exp(-absorption * metadata['distance_m'])
        rir = rir * decay_factor

        # 2. Random noise (SNR)
        snr_db = np.random.uniform(*self.snr_range)
        noise_power = np.mean(rir**2) / (10**(snr_db/10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(rir))
        rir = rir + noise

        # 3. Random speed of sound (time stretch)
        c = np.random.uniform(*self.speed_of_sound_range)
        stretch_factor = 343.0 / c
        rir = time_stretch(rir, stretch_factor)

        return rir
```

### M3. ARCore Pose Refinement
```python
def refine_arcore_poses(poses_raw, audio_toa, source_pos, c=343.0):
    """
    Refine ARCore poses using Time-of-Arrival constraints in LOS region.

    Args:
        poses_raw: [N, 6] - Raw ARCore poses (x,y,z,qx,qy,qz,qw)
        audio_toa: [N,] - Time of arrival of direct sound
        source_pos: [3,] - Known source position
        c: Speed of sound

    Returns:
        poses_refined: [N, 6] - Refined poses
    """
    poses_refined = poses_raw.copy()

    for i in range(len(poses_raw)):
        pos_raw = poses_raw[i, :3]

        # Distance from ToA
        dist_toa = audio_toa[i] * c

        # Distance from pose
        dist_pose = np.linalg.norm(pos_raw - source_pos)

        # Refinement: adjust position along source-mic axis
        direction = (pos_raw - source_pos) / dist_pose
        correction = dist_toa - dist_pose
        poses_refined[i, :3] = pos_raw + correction * direction

    return poses_refined
```

---

## JSX Visualization Corrections

### Timeline Bar Fix
```jsx
// Current (incorrect proportions)
<div className="flex mt-2 h-3 rounded-full overflow-hidden">
  <div className="bg-blue-500 flex-1" title="Phase 1"></div>  {/* 3 months */}
  <div className="bg-green-500 flex-1" title="Phase 2"></div> {/* 3 months */}
  <div className="bg-yellow-500 flex-1" title="Phase 3"></div>{/* 4 months */}
  <div className="bg-red-500 w-1/6" title="Phase 4"></div>    {/* 3 months - WRONG */}
</div>

// Fixed (correct proportions based on 13 total months)
<div className="flex mt-2 h-3 rounded-full overflow-hidden">
  <div className="bg-blue-500" style={{flexGrow: 3}} title="Phase 1: 3 months"></div>
  <div className="bg-green-500" style={{flexGrow: 3}} title="Phase 2: 3 months"></div>
  <div className="bg-yellow-500" style={{flexGrow: 4}} title="Phase 3: 4 months"></div>
  <div className="bg-red-500" style={{flexGrow: 3}} title="Phase 4: 3 months"></div>
</div>
```

### Add Risk Section
```jsx
const RiskItem = ({ severity, title, mitigation }) => (
  <div className={`p-3 rounded-lg border-l-4 ${
    severity === 'critical' ? 'border-red-500 bg-red-50' :
    severity === 'high' ? 'border-orange-500 bg-orange-50' :
    'border-yellow-500 bg-yellow-50'
  }`}>
    <div className="flex items-center gap-2">
      <span className={`text-xs font-bold uppercase ${
        severity === 'critical' ? 'text-red-600' :
        severity === 'high' ? 'text-orange-600' : 'text-yellow-600'
      }`}>{severity}</span>
      <span className="font-semibold">{title}</span>
    </div>
    <p className="text-sm text-gray-600 mt-1">Mitigation: {mitigation}</p>
  </div>
);

// Add to roadmap
<div className="bg-slate-700 rounded-lg p-6 mt-6">
  <h3 className="text-xl font-bold text-white mb-4">Risks & Mitigations</h3>
  <div className="space-y-3">
    <RiskItem
      severity="critical"
      title="BEM Numerical Instability at Resonance"
      mitigation="Burton-Miller formulation with α=i/k"
    />
    <RiskItem
      severity="high"
      title="PINN Trivial Solution"
      mitigation="Source term + surface existence constraint"
    />
    <RiskItem
      severity="high"
      title="Sim2Real Gap"
      mitigation="Domain randomization + calibration"
    />
  </div>
</div>
```

---

## Computational Requirements Summary

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | 1x RTX 3090 (24GB) | 4x A100 (40GB) |
| CPU | 16 cores | 64 cores |
| RAM | 64 GB | 256 GB |
| Storage | 500 GB SSD | 2 TB NVMe |
| Time (Phase 1) | 3 months | 1.5 months |
| Total GPU-hours | ~5,000 | ~1,500 (parallel) |

---

## Final Verdict

| Aspect | Score | Notes |
|--------|-------|-------|
| Physics Rigor | 6/10 | Complex field handling, trivial solution issues |
| Implementation Plan | 7/10 | Good structure, missing details |
| Computational Feasibility | 5/10 | Underestimated, needs cluster |
| Sim2Real Readiness | 4/10 | Many unaddressed gaps |
| Overall | 5.5/10 | **Needs revision before implementation** |

**Recommendation:** Address all P0 (Critical) issues before Phase 1. Secure computational resources early. Plan for 18 months instead of 13 if single-GPU.

---

*Review completed by Dr. Tensor Wave*
*"Physics is eternal. Your code should be too."*
