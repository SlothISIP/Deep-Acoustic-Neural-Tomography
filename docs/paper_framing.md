# Paper Framing Document (ICASSP 2026)

## Final Title

**"Neural Acoustic Diffraction Tomography: Cycle-Consistent Geometry Reconstruction from 2D BEM Data"**

### Title Rationale
- "Neural" — honest about method being neural-network-based, no "Physics-Informed" claim
- "Acoustic Diffraction Tomography" — domain-specific, distinguishes from optical/seismic
- "Cycle-Consistent" — key technical mechanism that validates the inverse
- "Geometry Reconstruction" — the actual task (SDF from acoustic data)
- "2D BEM Data" — honest scope qualifier (synthetic, not real-world)

---

## One-Line Summary

> We propose a cycle-consistent neural framework for 2D acoustic diffraction
> tomography that learns transfer functions from BEM simulations and reconstructs
> scene geometry as signed distance functions, achieving 0.95 mean IoU across
> 15 synthetic scenes without Helmholtz PDE enforcement.

### Why This Is Honest
- "cycle-consistent neural" — no physics-informed/rigorous claim
- "2D" — explicit dimensionality qualifier
- "BEM simulations" — synthetic data, not measured
- "without Helmholtz PDE enforcement" — addresses the key negative result upfront
- "0.95 mean IoU" — quantitative, verifiable

---

## Contributions (3)

### C1: Transfer Function Formulation for Neural Acoustic Modeling
**Claim**: Learning the scattered-to-incident ratio T = p_scat/p_inc removes
dominant phase oscillations, compressing effective data variance from 13% to
89.6% of the total and enabling a compact MLP to approximate BEM-quality
acoustic fields at 4.47% error.

**Evidence**:
- Phase 2 gate: 4.47% overall error (quad ensemble + calibration)
- 14/15 scenes < 3.6% error; Scene 13 (step discontinuity) = 18.6%
- Forward ablation: single model 11.54% → ensemble 4.57% → +calib 4.47%

**Scope Qualifier**: Evaluated on 2D BEM data only (15 scenes, 200 frequencies,
2-8 kHz). Generalization to 3D or measured data not demonstrated.

### C2: Auto-Decoder Inverse Model with Eikonal SDF
**Claim**: A per-scene auto-decoder optimizing latent codes into an SDF decoder
with Eikonal regularization reconstructs 2D geometry at IoU 0.95 (14/15 scenes
> 0.92). Helmholtz PDE loss is demonstrated as incompatible with neural
surrogate forward models (negative result).

**Evidence**:
- Phase 3 gate: Mean IoU = 0.9491 (14/15 > 0.92, S12 = 0.49)
- Helmholtz loss disabled: neural ∇²p = network curvature (~10^5 residual),
  NOT physical Laplacian. Enabling it degrades IoU from 0.82 to 0.19 in 30 epochs.
- Eikonal constraint works (|∇s|=1 is network-native)
- Inverse ablation: base 0.69 → +bdy3x 0.84 → +cycle 0.94 → +multi-code 0.95

**Scope Qualifier**: Auto-decoder = per-scene optimization (not amortized
inference). 15 scenes is too few for encoder-based generalization.

### C3: Cycle-Consistency Validation with Robustness Analysis
**Claim**: Forward-inverse cycle achieves Pearson r = 0.90 across all 15 scenes,
robust to 10 dB SNR additive noise (r = 0.86). Leave-one-out analysis shows
partial decoder generalization (52% mean IoU recovery).

**Evidence**:
- Phase 4 gate: mean r = 0.9024, all 15 scenes individually pass (r > 0.83)
- Noise robustness: r = 0.90 (clean) → 0.86 (10 dB), graceful degradation
- LOO generalization: wedge-like shapes recover well (S1: 92%, S14: 97%),
  novel geometries struggle (S5: 9%, S10: 22%)
- S12 IoU = 0.49 but r = 0.92 demonstrates cycle-consistency ≠ geometry accuracy

**Scope Qualifier**: Cycle-consistency is necessary but not sufficient for
geometry accuracy. Forward model compensates via 8 non-SDF features.

---

## Known Limitations (Paper Discussion Section)

| # | Limitation | Honest Statement |
|---|-----------|------------------|
| 1 | **Helmholtz Failure** | Neural surrogate ∇²p is network curvature, not physical Laplacian. PDE enforcement requires differentiable physics-based solvers. |
| 2 | **S12 Multi-Body** | Disjoint geometry (IoU 0.49) remains unsolved despite frozen-decoder improvement (→0.62). Smooth-min composition has fundamental limits. |
| 3 | **2D Only** | All results on 2D BEM synthetic data. 3D extension requires ~100× BEM cost and mesh complexity. |
| 4 | **No Encoder** | Auto-decoder optimizes per-scene, no amortized inference. Encoder requires >100 scenes (dataset bottleneck). |
| 5 | **Scene 13** | Step discontinuity gives 18.6% forward error (others < 3.6%). Sharp features challenge the smooth MLP. |
| 6 | **Single Seed** | Results from seed=42 only. Seed sweep in progress for reproducibility (seeds: 42, 123, 456). |
| 7 | **Cycle ≠ Geometry** | S12 shows r=0.92 with IoU=0.49 — forward model compensates for geometry errors via spectral features. |

---

## Figure Plan (7 Figures)

| # | Figure | Content | Key Message |
|---|--------|---------|-------------|
| 1 | Architecture | Forward + Inverse + Cycle diagram | System overview |
| 2 | BEM Validation | Analytical vs BEM pressure field | Phase 0: 1.77% error |
| 3 | Forward Performance | Per-scene error bars + ensemble ablation | C1: 4.47% |
| 4 | SDF Gallery | GT vs predicted SDF contours (6 scenes) | C2: IoU 0.95 |
| 5 | Ablation Bars | Forward (5 config) + Inverse (4 stage) | Component contributions |
| 6 | Cycle Consistency | Scatter plot (p_pred vs p_gt) + per-scene r | C3: r = 0.90 |
| 7 | Generalization | LOO recovery + noise robustness | C3: robustness |

---

## Table Plan (3 Tables)

### Table 1: Forward Model Ablation
| Config | Error (%) |
|--------|-----------|
| Single model | 11.54 |
| + calibration | 10.20 |
| Duo ensemble + calib | 9.89 |
| Quad ensemble | 4.57 |
| Quad ensemble + calib | **4.47** |

### Table 2: Inverse Model Ablation
| Config | Mean IoU | S12 IoU | Mean r |
|--------|----------|---------|--------|
| L_sdf + L_eik (200 ep) | 0.689 | 0.135 | -- |
| + bdy 3x (500 ep) | 0.842 | 0.184 | -- |
| + L_cycle (1000 ep) | 0.939 | 0.410 | 0.909 |
| + multi-code K=2 | **0.949** | 0.493 | 0.902 |

### Table 3: Noise Robustness
| SNR (dB) | Mean r | Delta |
|----------|--------|-------|
| Clean | 0.9024 | -- |
| 40 | 0.9024 | -0.000 |
| 30 | 0.9020 | -0.000 |
| 20 | 0.8980 | -0.005 |
| 10 | 0.8604 | -0.042 |

---

## Abstract Draft (~150 words)

We present a neural framework for 2D acoustic diffraction tomography that
reconstructs scene geometry from boundary element method (BEM) simulations.
Our approach consists of three components: (1) a transfer function formulation
that learns the scattered-to-incident pressure ratio, eliminating dominant
phase oscillations and achieving 4.47% BEM reconstruction error across 15
synthetic scenes; (2) an auto-decoder inverse model that maps acoustic
observations to signed distance functions (SDF) with Eikonal regularization,
yielding 0.95 mean intersection-over-union; and (3) a cycle-consistency
mechanism that validates geometry through forward-inverse agreement (Pearson
r = 0.90). We demonstrate robustness to additive noise down to 10 dB SNR
(r = 0.86) and analyze partial generalization via leave-one-out evaluation.
Notably, we report that Helmholtz PDE enforcement through neural surrogates
fails due to the gap between network curvature and physical Laplacians--a
finding with implications for physics-informed acoustic learning.

---

## What We DON'T Claim

1. ~~"Physics-Informed"~~ — Helmholtz disabled, only Eikonal works
2. ~~"First-ever"~~ — First *in this specific 2D synthetic setup*
3. ~~"Real-time inference"~~ — Auto-decoder requires per-scene optimization
4. ~~"Generalizes to unseen geometries"~~ — LOO shows 52% recovery only
5. ~~"Audio-to-geometry"~~ — Input is complex pressure, not raw audio
6. ~~"3D capable"~~ — Exclusively 2D
7. ~~"Helmholtz-consistent"~~ — Residuals ~10^5, all FAIL

---

## Related Work Positioning

| Category | Key Papers | Our Difference |
|----------|-----------|----------------|
| Neural acoustics | NAF (Luo+ 2022), INRAS (Su+ 2023) | Transfer function target vs direct field |
| Acoustic inverse | AcousticNeRF (Liang+ 2024) | SDF output vs NeRF implicit field |
| Physics-informed | PINN (Raissi+ 2019) | Negative result: Helmholtz fails for surrogates |
| DeepSDF | Park+ 2019 | Applied to acoustic inverse, not vision |
| Diffraction | Kouyoumjian-Pathak UTD | BEM ground truth, neural approximation |

---

*Document Version: 1.0 (2026-02-19)*
*Status: Ready for review before manuscript writing*
