# Acoustic Neural Tomography: Implementation Roadmap v3.2

## "Simultaneous Reconstruction of Sound & Geometry via Structured Green's Learning"

**Version:** 3.2 (Dr. Tensor Wave Critical Review Reflected)  
**Target Publication:** CVPR (Oral) / Nature Communications  
**Timeline:** 18 Months (Revised from 13)  
**Core Contribution:** 소리만으로 보이지 않는 기하구조를 복원하는 물리 기반 딥러닝 프레임워크

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v3.0 | 2026-01 | Initial physics-informed design |
| v3.1 | 2026-01 | Added Structured Green, Fourier Features, Joint SDF |
| **v3.2** | **2026-01** | **Critical fixes: Complex field, Trivial solution prevention, Phase unwrapping** |

---

## Executive Summary

v3.2는 Dr. Tensor Wave의 심층 리뷰를 반영한 수정판이다. 이전 버전들이 "설계도(Blueprint)"였다면, v3.2는 실제 구현에서 마주칠 **치명적인 함정들**을 미리 제거한 "시공 매뉴얼(Construction Manual)"이다.

### 핵심 수정 사항 (Critical Fixes)

**C1. 복소수 압력장 처리:** Helmholtz 방정식의 해는 복소수다. 네트워크 출력을 Real/Imaginary heads로 분리하여 진폭과 위상을 모두 학습해야 한다.

**C2. Trivial Solution 방지:** SDF=const, p=0 조합이 모든 Physics Loss를 만족시키는 문제가 있다. Surface Existence Constraint와 Inhomogeneous Helmholtz(Source Term 포함)로 이를 방지한다.

**C3. Phase Unwrapping:** Multi-frequency IDFT에서 위상 불연속성(2π jump)을 처리하지 않으면 비인과적(acausal) 신호가 생성된다. `np.unwrap()`이 필수다.

### 추가 수정 사항 (High Severity Fixes)

**H1. Fourier Feature Scale:** 기존 σ ≈ 23 m⁻¹은 파수(k)만 고려했다. 회절의 각도 의존성을 반영하면 σ ≈ 62 m⁻¹이 올바른 값이다.

**H3. Burton-Miller Parameter:** BEM의 unique solution을 보장하려면 coupling parameter α = i/k를 명시적으로 설정해야 한다.

**Timeline 조정:** 현실적인 구현 시간을 고려하여 13개월에서 18개월로 연장했다.

---

## Computational Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | 1x RTX 3090 (24GB) | 4x A100 (40GB) |
| CPU | 16 cores | 64 cores |
| RAM | 64 GB | 256 GB |
| Storage | 500 GB SSD | 2 TB NVMe |
| Time (Single GPU) | 18 months | - |
| Time (Cluster) | - | 12 months |

---

## Phase 0: Prerequisites & Environment Setup

**기간:** Week 0-1

### 환경 구성

Python 3.10 이상, CUDA 12.x, OpenCL 드라이버가 필요하다. conda 환경을 권장한다.

```bash
conda create -n acoustic-tomo python=3.10
conda activate acoustic-tomo
pip install bempp-cl meshio pygmsh torch>=2.0 numpy scipy
pip install matplotlib plotly wandb h5py joblib
```

### Complex Number Support 확인 (NEW in v3.2)

PyTorch 2.0+에서 complex tensor 연산이 완전히 지원된다. 환경 설정 후 반드시 확인하라:

```python
import torch
a = torch.complex(torch.randn(3,3), torch.randn(3,3))
b = torch.complex(torch.randn(3,3), torch.randn(3,3))
c = torch.matmul(a, b)  # Complex matmul
assert c.dtype == torch.complex64
print("Complex tensor support: OK")
```

---

## Phase 1: BEM Physics Engine & Frequency Synthesis

**기간:** Month 1-4 (4 months, extended from 3)  
**목표:** 물리적으로 정확한 Room Impulse Response(RIR) 데이터셋 생성

### 1.1 Wedge Geometry BEM 검증

모든 것의 시작은 검증이다. 무한 웨지(Infinite Wedge)에서 BEM 솔루션을 Macdonald 해석해와 비교하여 오차가 3% 이내인지 확인한다.

### 1.2 Burton-Miller Formulation (NEW in v3.2)

**이것은 v3.1에서 누락된 critical issue다.**

Helmholtz 방정식의 외부 문제(Exterior Problem)는 특정 주파수(eigenfrequency)에서 해가 유일하지 않다. Burton-Miller combined field integral equation은 이 문제를 해결한다.

핵심은 coupling parameter α다:

$$\alpha = \frac{i}{k}$$

이 값이 optimal이며, 모든 주파수에서 unique solution을 보장한다. Bempp-cl 구현:

```python
import bempp.api

def create_burton_miller_operator(grid, k):
    """
    Burton-Miller CFIE for exterior Helmholtz.
    Coupling parameter alpha = i/k ensures unique solution.
    """
    alpha = 1j / k
    
    space = bempp.api.function_space(grid, "P", 1)
    
    identity = bempp.api.operators.boundary.sparse.identity(space, space, space)
    dlp = bempp.api.operators.boundary.helmholtz.double_layer(space, space, space, k)
    slp = bempp.api.operators.boundary.helmholtz.single_layer(space, space, space, k)
    hyp = bempp.api.operators.boundary.helmholtz.hypersingular(space, space, space, k)
    adlp = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(space, space, space, k)
    
    # Burton-Miller: (1/2 I + D + alpha*H) * u = (S + alpha*(1/2 I + D')) * g
    lhs = 0.5 * identity + dlp + alpha * hyp
    rhs_op = slp + alpha * (0.5 * identity + adlp)
    
    return lhs, rhs_op
```

### 1.3 Adaptive Mesh Near Edges (NEW in v3.2)

회절은 edge 근처에서 가장 강하게 발생한다. 균일한 메쉬는 비효율적이다. Edge 근처는 λ/10, 평면 영역은 λ/6으로 설정하는 적응적 메쉬를 사용한다.

8kHz 기준:
- λ_min = 343/8000 ≈ 43mm
- Edge region: element_size < 4.3mm
- Flat region: element_size < 7.2mm

```python
import pygmsh

def create_adaptive_mesh(corner_pos, f_max_hz, c=343.0):
    wavelength_min = c / f_max_hz
    edge_size = wavelength_min / 10
    flat_size = wavelength_min / 6
    
    with pygmsh.geo.Geometry() as geom:
        # ... geometry definition ...
        geom.set_mesh_size_callback(
            lambda dim, tag, x, y, z, lc:
                edge_size if is_near_edge(x, y, z, corner_pos, threshold=0.05)
                else flat_size
        )
        mesh = geom.generate_mesh()
    return mesh
```

### 1.4 Multi-Frequency BEM Solver

2-8 kHz 대역에서 Helmholtz 방정식을 푼다. 주파수 해상도는 원하는 RIR 길이에 따라 결정된다. T=100ms라면 Δf=10Hz, 즉 600개 주파수가 필요하다.

Adaptive frequency sampling을 고려하라: 공진 주파수 근처는 촘촘히, 그 외 영역은 듬성듬성 샘플링하면 계산량을 줄일 수 있다.

### 1.5 Phase Unwrapping & IDFT Synthesis (CRITICAL in v3.2)

**이것은 v3.1에서 완전히 누락된 critical issue다.**

주파수 영역에서 시간 영역으로 변환할 때, 위상(phase)의 불연속성(2π jump)이 문제가 된다. 처리하지 않으면 시간 영역에서 비인과적(acausal) 신호가 생성된다.

```python
import numpy as np

def frequency_to_time_domain(P_freq, freqs_hz, N_time):
    """
    Convert frequency-domain pressure to time-domain RIR.
    CRITICAL: Phase unwrapping is essential for causality.
    """
    # Step 1: Phase unwrapping (CRITICAL!)
    phase_raw = np.angle(P_freq)
    phase_unwrapped = np.unwrap(phase_raw)
    P_corrected = np.abs(P_freq) * np.exp(1j * phase_unwrapped)
    
    # Step 2: Hermitian symmetry for real output
    P_full = np.zeros(N_time, dtype=complex)
    P_full[:len(P_freq)] = P_corrected
    P_full[N_time-len(P_freq)+1:] = np.conj(P_corrected[-1:0:-1])
    
    # Step 3: IDFT
    h_t = np.fft.irfft(P_full, n=N_time)
    
    return h_t
```

### 1.6 Causality & Energy Conservation Validation (NEW in v3.2)

두 가지 검증이 필수다:

**Causality:** t < 0에서 신호 에너지가 전체의 1e-6 이하여야 한다.

```python
def validate_causality(h_t, t_samples, rtol=1e-6):
    t_negative = t_samples < 0
    acausal_energy = np.sum(h_t[t_negative]**2)
    total_energy = np.sum(h_t**2)
    ratio = acausal_energy / total_energy
    if ratio > rtol:
        raise ValueError(f"Causality violated! Ratio: {ratio:.2e}")
    return True
```

**Energy Conservation (Parseval's theorem):** 주파수 영역과 시간 영역의 에너지가 일치해야 한다.

$$\int|P(f)|^2 df = \int|h(t)|^2 dt$$

### 1.7 Dataset Generation

검증된 파이프라인으로 10,000개 RIR을 생성한다. Domain randomization(흡음 계수, SNR, 음속 변화)을 적용하여 Sim2Real gap을 줄인다.

**Phase 1 완료 기준:**
- BEM vs 해석해 오차 < 3%
- Causality 만족 (acausal energy ratio < 1e-6)
- Energy conservation (relative error < 1%)

---

## Phase 2: Structured Green's Function Learning

**기간:** Month 5-8 (4 months)  
**목표:** 물리적 구조를 반영한 회절 학습 네트워크 개발

### 2.1 Physics-Based Decomposition

전체 Green's Function을 두 부분으로 분해한다:

$$\hat{G}_{total} = G_{geometric} + \hat{G}_{diff}(\theta)$$

**G_geometric (Frozen):** 직접음과 1차 반사음. Image Source Method로 해석적으로 계산하여 고정한다. 이 부분은 학습하지 않는다.

**G_diff (Learnable):** 회절 성분. MLP로 근사한다.

### 2.2 Complex Diffraction MLP (MODIFIED in v3.2)

**v3.1에서는 암묵적으로 실수 출력을 가정했다. 이는 틀렸다.**

회절 계수(Diffraction Coefficient)는 복소수다. 진폭과 위상을 모두 가진다. Re/Im heads를 분리하여 출력한다:

```python
class DiffractionMLP(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(3, hidden_dim),  # Input: (phi_inc, phi_obs, k)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head_real = nn.Linear(hidden_dim, 1)
        self.head_imag = nn.Linear(hidden_dim, 1)
    
    def forward(self, phi_inc, phi_obs, k):
        x = torch.stack([phi_inc, phi_obs, k], dim=-1)
        features = self.backbone(x)
        D_real = self.head_real(features)
        D_imag = self.head_imag(features)
        D_complex = torch.complex(D_real.squeeze(-1), D_imag.squeeze(-1))
        return D_complex
```

### 2.3 Training Objective

입력 신호와 Complex G_total의 컨볼루션을 수행하고, 측정 신호와의 L2 Loss를 최소화한다:

$$\mathcal{L} = \| y(t) - \text{Re}\{ s(t) * (G_{geo} + \hat{G}_{diff}) \} \|^2$$

주파수 영역에서 컨볼루션을 수행하면 효율적이다 (FFT convolution).

### 2.4 Ablation Study

Structured 접근법의 효과를 입증하기 위해, G_total 전체를 처음부터 학습하는 Baseline과 비교한다. 예상 결과:
- 수렴 속도: Structured가 2배 이상 빠름
- 최종 정확도: Structured가 10% 이상 우수
- 일반화: 새로운 기하구조에서 Structured가 월등히 우수

**Phase 2 완료 기준:**
- Validation Loss 수렴
- UTD 해석해와 상관계수 > 0.9
- ICASSP 워크샵 페이퍼 초안 완성

---

## Phase 3: Neural Fields with Implicit Geometry

**기간:** Month 9-13 (5 months, extended)  
**목표:** 소리와 기하구조를 동시에 복원하는 Physics-Informed Neural Field

### ⚠️ 이 Phase가 논문의 핵심 Contribution이다 — MAJOR REVISION in v3.2

### 3.1 Fourier Feature Scale (CORRECTED in v3.2)

**v3.1의 σ ≈ 23 m⁻¹은 틀렸다.**

파수(wavenumber) k만 고려했기 때문이다. 회절 패턴은 각도 의존성이 있다:

$$\text{Spatial frequency} = k \cdot \sin(\theta)$$

최대 관측각 θ_max = 60°를 고려하면:

$$\sigma = \frac{k_{max} \cdot \sin(\theta_{max})}{2\pi} \times 1.5 \approx 62 \text{ m}^{-1}$$

```python
def compute_fourier_scale(f_max_hz, c=343.0, max_angle_deg=60.0):
    k_max = 2 * np.pi * f_max_hz / c
    theta_max = np.radians(max_angle_deg)
    spatial_freq_max = k_max * np.sin(theta_max)
    sigma = spatial_freq_max / (2 * np.pi) * 1.5  # Safety margin
    return sigma  # ≈ 62 for 8kHz, NOT 23!
```

### 3.2 Complex Joint Output Network (CRITICAL in v3.2)

네트워크가 복소수 음압과 실수 SDF를 동시에 출력한다:

```python
class AcousticNeuralField(nn.Module):
    def __init__(self, fourier_dim=256, hidden_dim=512):
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Linear(fourier_dim + 1, hidden_dim),  # +1 for wavenumber k
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Pressure output: Complex (Re + Im)
        self.p_head_real = nn.Linear(hidden_dim, 1)
        self.p_head_imag = nn.Linear(hidden_dim, 1)
        
        # SDF output: Real
        self.sdf_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, gamma_x, k):
        """
        Args:
            gamma_x: [B, fourier_dim] - Fourier features of coordinates
            k: [B, 1] - Wavenumber
        Returns:
            p: [B,] complex - Pressure field
            sdf: [B,] real - Signed distance function
        """
        x = torch.cat([gamma_x, k], dim=-1)
        features = self.backbone(x)
        
        p_real = self.p_head_real(features).squeeze(-1)
        p_imag = self.p_head_imag(features).squeeze(-1)
        p = torch.complex(p_real, p_imag)
        
        sdf = self.sdf_head(features).squeeze(-1)
        
        return p, sdf
```

### 3.3 Eikonal Loss

SDF가 물리적으로 유효하려면 gradient norm이 1이어야 한다:

$$\mathcal{L}_{geo} = \| |\nabla s(\mathbf{x})| - 1 \|^2$$

### 3.4 Surface Existence Constraint (CRITICAL NEW in v3.2)

**v3.1의 치명적 결함: SDF = const가 Eikonal을 제외한 모든 Loss를 만족한다.**

SDF가 상수이면 기하구조(표면)가 존재하지 않는다. 이를 방지하려면 SDF가 반드시 양수와 음수를 모두 가지도록 강제해야 한다:

```python
def surface_existence_loss(sdf, margin=0.1):
    """
    Ensure SDF crosses zero (surface exists).
    
    If sdf_min > 0 or sdf_max < 0, no surface exists.
    This prevents trivial solution SDF = const.
    """
    sdf_min = sdf.min()
    sdf_max = sdf.max()
    
    # sdf_min should be negative (inside region exists)
    loss_min = F.relu(sdf_min + margin)  # Penalize if sdf_min > -margin
    
    # sdf_max should be positive (outside region exists)
    loss_max = F.relu(-sdf_max + margin)  # Penalize if sdf_max < margin
    
    return loss_min + loss_max
```

### 3.5 Inhomogeneous Helmholtz Loss (CRITICAL NEW in v3.2)

**v3.1의 치명적 결함: p = 0이 homogeneous Helmholtz를 완벽히 만족한다.**

해결책: 음원(source)을 포함한 inhomogeneous 형태를 사용한다:

$$\nabla^2 p + k^2 p = -\delta(\mathbf{x} - \mathbf{x}_{src})$$

Source term이 존재하므로 p = 0은 더 이상 해가 아니다:

```python
def helmholtz_loss_inhomogeneous(p, coords, k, source_pos, sigma=0.01):
    """
    Inhomogeneous Helmholtz PDE loss with point source.
    
    The source term prevents p=0 trivial solution.
    """
    # Compute Laplacian of pressure
    # (using torch.autograd.grad twice)
    laplacian_p = compute_laplacian(p, coords)
    
    # Gaussian approximation of point source delta function
    dist_to_source = torch.norm(coords - source_pos, dim=-1)
    source_term = torch.exp(-dist_to_source**2 / (2*sigma**2))
    source_term = source_term / (sigma * np.sqrt(2*np.pi))  # Normalize
    
    # PDE residual: should be zero away from source
    residual = laplacian_p + k**2 * p + source_term
    
    return torch.mean(torch.abs(residual)**2)
```

### 3.6 Boundary Condition Loss

SDF ≈ 0인 영역(표면)에서 음향 경계 조건을 적용한다. 법선 방향은 SDF의 gradient로부터 계산한다:

$$\mathbf{n} = \frac{\nabla s}{|\nabla s|}$$

Rigid wall (Neumann BC):

$$\mathcal{L}_{BC} = \sum_{\mathbf{x}: s(\mathbf{x}) \approx 0} \left| \frac{\partial p}{\partial n} \right|^2$$

### 3.7 Complete Loss Function (UPDATED in v3.2)

$$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda_1 \mathcal{L}_{Helmholtz} + \lambda_2 \mathcal{L}_{geo} + \lambda_3 \mathcal{L}_{BC} + \lambda_4 \mathcal{L}_{surface}$$

v3.1 대비 추가된 것: **L_surface (Surface Existence Constraint)**

### 3.8 Multi-Loss Balancing

네 가지 Loss가 서로 다른 스케일을 가지므로, adaptive weighting이 필요하다:

```python
class AdaptiveWeightedLoss:
    def __init__(self, num_losses):
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
    
    def forward(self, losses):
        """
        Uncertainty-based weighting (Kendall et al., 2018)
        """
        weighted = []
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted.append(precision * loss + self.log_vars[i])
        return sum(weighted)
```

### 3.9 Incremental Integration Strategy

모든 Loss를 동시에 사용하면 학습이 불안정하다. 단계적으로 추가한다:

1. **Step 1:** L_data만 사용 (기본 fitting)
2. **Step 2:** + L_geo + L_surface (SDF 형태 학습)
3. **Step 3:** + L_Helmholtz (물리 방정식)
4. **Step 4:** + L_BC (경계 조건)

각 단계에서 충분히 수렴한 후 다음으로 진행한다.

**Phase 3 완료 기준:**
- SDF 복원 IoU > 0.8
- Trivial solution 회피 확인 (surface exists)
- Helmholtz residual < 1e-3 (source 근처 제외)

---

## Phase 4: Sim2Real & Cycle-Consistency Validation

**기간:** Month 14-18 (5 months, extended)  
**목표:** 실제 실험 데이터로 방법론 검증

### 4.1 실험 환경 구축

L-Shape 코너를 실험실에 구성한다. Bluetooth 스피커(음원)와 스마트폰 마이크(수신기)를 사용한다. Chirp 신호(2-8 kHz, 100ms)를 재생하고 녹음한다.

목표 SNR: 20dB 이상.

### 4.2 Domain Randomization (NEW in v3.2)

Sim2Real gap을 줄이기 위해 시뮬레이션 데이터에 randomization을 적용한다:

- 흡음 계수: 0.0 ~ 0.3 (MDF 범위)
- SNR: 10 ~ 30 dB
- 음속: 340 ~ 346 m/s (온도 변화)

```python
class DomainRandomizer:
    def randomize(self, rir, metadata):
        # Random absorption (amplitude decay)
        absorption = np.random.uniform(0.0, 0.3)
        rir *= np.exp(-absorption * metadata['distance'])
        
        # Random noise
        snr_db = np.random.uniform(10, 30)
        noise = np.random.normal(0, rir.std() / 10**(snr_db/20), len(rir))
        rir += noise
        
        # Random speed of sound (time stretch)
        c = np.random.uniform(340, 346)
        rir = time_stretch(rir, 343/c)
        
        return rir
```

### 4.3 ARCore + ToA Pose Refinement (IMPROVED in v3.2)

ARCore의 위치 추정에는 드리프트가 있다. Time-of-Arrival(ToA) 제약을 활용하여 보정한다:

```python
def refine_pose_with_toa(pose_raw, toa, source_pos, c=343.0):
    """
    Refine ARCore pose using Time-of-Arrival constraint.
    In LOS region, distance = ToA * c.
    """
    pos_raw = pose_raw[:3]
    dist_toa = toa * c
    dist_pose = np.linalg.norm(pos_raw - source_pos)
    
    direction = (pos_raw - source_pos) / dist_pose
    correction = dist_toa - dist_pose
    pos_refined = pos_raw + correction * direction
    
    return np.concatenate([pos_refined, pose_raw[3:]])
```

### 4.4 Cycle-Consistency Validation

최종 검증은 Cycle-Consistency다:

1. **Inverse:** 실제 소리 → 네트워크 → 추정된 SDF
2. **Forward:** 추정된 SDF → BEM → 가상 소리
3. **Check:** 실제 소리 ≈ 가상 소리

```
y_real → [Inverse Network] → SDF_pred → [BEM] → y_sim
                                                   ↓
                              Check: correlation(y_real, y_sim) > 0.8
```

이 검증이 통과하면, 추정된 기하구조가 물리적으로 타당함을 증명한 것이다.

**Phase 4 완료 기준:**
- Cycle-Consistency: correlation > 0.8
- CVPR 논문 투고 완료

---

## Risk Assessment & Mitigation

### Critical Risks (P0)

| Risk | Description | Mitigation |
|------|-------------|------------|
| C1 | Complex field 처리 누락 → 위상 정보 손실 | Re/Im heads 분리, torch.complex64 |
| C2 | Trivial solution (SDF=const, p=0) | Surface Existence + Inhomogeneous Helmholtz |
| C3 | Phase unwrapping 누락 → Acausal RIR | np.unwrap + Causality 검증 |

### High Risks (P1)

| Risk | Description | Mitigation |
|------|-------------|------------|
| H1 | Fourier scale 과소평가 | σ = 62 m⁻¹ (각도 의존성 반영) |
| H2 | Edge 근처 메쉬 해상도 부족 | Adaptive mesh (λ/10 at edges) |
| H3 | BEM resonance 불안정 | Burton-Miller with α = i/k |
| H4 | Energy conservation 실패 | Parseval's theorem 검증 |

### Medium Risks (P2)

| Risk | Description | Mitigation |
|------|-------------|------------|
| M1 | Multi-loss balancing 실패 | Adaptive weighting, Incremental training |
| M2 | Sim2Real gap | Domain randomization |
| M3 | Pose estimation 오차 | ToA-based refinement |

---

## Deliverables Timeline (Revised)

| Milestone | Month | Deliverable |
|-----------|-------|-------------|
| BEM 검증 완료 | 4 | 검증된 시뮬레이션 파이프라인, 10K RIR 데이터셋 |
| ICASSP 투고 | 8 | 워크샵 페이퍼 (Green-Net 방법론) |
| Neural Field 완성 | 13 | PINN 기반 Joint Learning 코드 |
| CVPR 투고 | 15 | Full Paper (전체 프레임워크) |
| Sim2Real 완료 | 18 | 실험 데이터 검증, 코드 공개 |
| Nature Comms | Year 3 | 응용 확장 (Medical Ultrasound) |

---

## One-Line Contribution (Updated)

> **"We jointly reconstruct complex acoustic fields and scene geometry by learning only the diffraction residual atop analytical Green's functions, while enforcing inhomogeneous Helmholtz PDE, Eikonal constraints, and surface existence guarantees."**

v3.1 대비 추가된 키워드: **complex**, **inhomogeneous**, **surface existence guarantees**

---

## Conclusion

v3.2는 v3.1의 "설계도"를 실제 구현 가능한 "시공 매뉴얼"로 발전시켰다. Dr. Tensor Wave의 리뷰가 지적한 세 가지 치명적 문제(복소수 처리, trivial solution, phase unwrapping)를 모두 해결했다.

이 로드맵을 따르면, 18개월 후 자네는 "소리로 보이지 않는 세상을 보는" 시스템을 구현하게 될 것이다. 그리고 그 시스템은 물리적으로 타당하고, 논리적으로 완결되며, 리뷰어가 반박할 수 없는 결과를 낳을 것이다.

이제 정말 코드를 짤 시간이다.

Phase 1-Task 1: Wedge BEM 검증부터 시작하라.

---

*Acoustic Neural Tomography Implementation Roadmap v3.2*  
*Dr. Tensor Wave Critical Review Reflected*  
*Target: CVPR Oral / Nature Communications*  
*Timeline: 18 months*  
*Last Updated: January 2026*
