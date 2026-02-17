# acoustic-validate

BEM validation and acoustic physics diagnostics for Deep Acoustic Diffraction Tomography.

## Trigger Conditions

Activate this skill when the user:
- Says "검증", "validate", "gate check", "BEM 비교", "Phase N 검증"
- Asks to check phase gate criteria
- Reports NaN, Inf, or BEM solve failures
- Wants to verify physics constraints (Helmholtz, causality, energy)
- Says "해석해 비교", "Macdonald", "analytical"

## Description

Phase-aware validation for the acoustic diffraction tomography pipeline.
Each phase has specific gate criteria that must be met before proceeding.

## Instructions

When this skill is triggered, follow these steps:

### 1. Determine Current Phase and Validation Type

| Phase | Gate Criterion | Validation Script |
|-------|----------------|-------------------|
| **0** | BEM vs Macdonald analytical < 3% error | `validate_wedge_bem.py` |
| 1 | Causality h(t<0) ~ 0, 15 scenes generated | `validate_phase1.py` (future) |
| 2 | Forward model BEM reconstruction error < 5% | `validate_phase2.py` (future) |
| 3 | SDF IoU > 0.8, Helmholtz residual < 1e-3 | `validate_phase3.py` (future) |
| 4 | Cycle-consistency r > 0.8 | `validate_phase4.py` (future) |

### 2. Phase 0 Validation (Current)

Run the validation script:

```bash
python validate_wedge_bem.py
```

If the script does not exist yet, perform manual validation:

#### 2a. Check Environment
```bash
python -c "import bempp.api; print('bempp-cl:', bempp.api.__version__)"
python -c "import gmsh; print('gmsh OK')"
python -c "import meshio; print('meshio OK')"
```

#### 2b. Verify Mesh Quality
Check the generated wedge mesh:
- Total elements N < 10,000
- Element size at flat surface: <= lambda_min / 6
- Element size at wedge edge: <= lambda_min / 10
- lambda_min at f=2kHz: c/f = 343/2000 = 0.1715 m
- Max element size (flat): 0.1715/6 = 0.0286 m
- Max element size (edge): 0.1715/10 = 0.01715 m

#### 2c. Verify BEM Solution
- No NaN or Inf in pressure field
- Pressure magnitude physically reasonable (not diverging)
- Reciprocity: G(r, r') approximately equals G(r', r)

#### 2d. Compare Against Analytical
- Compute Macdonald analytical solution at same observation points
- Compute relative L2 error: ||p_BEM - p_analytical||_2 / ||p_analytical||_2
- **Gate: error < 3%**

### 3. Physics Diagnostics (Any Phase)

| Check | Method | Threshold |
|-------|--------|-----------|
| NaN/Inf | `np.isfinite()` on all arrays | Zero non-finite values |
| Causality | Pre-arrival energy ratio | < 1e-4 |
| Energy (Parseval) | Time vs frequency domain energy | < 1% difference |
| Helmholtz residual | nabla^2 p + k^2 p | < 1e-3 (Phase 3) |
| Eikonal | \|\|nabla s\|\| - 1 | < 1e-2 (Phase 3) |
| Mesh Nyquist | h_max vs lambda_min | h < lambda/6 |

### 4. Report Format

Always present results in this format:

```
## Phase N Validation Report

| Check | Value | Threshold | Status |
|-------|-------|-----------|--------|
| BEM vs analytical error | X.XX% | < 3% | PASS/FAIL |
| Mesh N elements | NNNN | < 10,000 | PASS/FAIL |
| Element size (flat) | X.XXX m | < 0.0286 m | PASS/FAIL |
| Element size (edge) | X.XXX m | < 0.0172 m | PASS/FAIL |
| NaN/Inf count | 0 | 0 | PASS/FAIL |

### Gate Decision
**Phase 0 → Phase 1**: PASS / FAIL

### Issues Found
- [Issue description with file:line reference]

### Recommendations
1. [Actionable fix if FAIL]
```

## Gate Thresholds Reference

| Phase | Metric | Threshold | Physical Basis |
|-------|--------|-----------|----------------|
| 0 → 1 | BEM vs analytical L2 error | < 3% | BEM numerical accuracy |
| 0 → 1 | Mesh element count | < 10,000 | RAM constraint (32GB) |
| 1 → 2 | Causality ratio h(t<0) | < 1e-4 | Physical impossibility |
| 1 → 2 | Scenes generated | >= 15 | Statistical significance |
| 2 → 3 | Forward reconstruction error | < 5% | Model accuracy |
| 3 → 4 | SDF IoU | > 0.8 | Geometry reconstruction quality |
| 3 → 4 | Helmholtz residual | < 1e-3 | PDE satisfaction |
| 4 → 5 | Cycle-consistency r | > 0.8 | Self-consistency |

## Physical Constants

```python
SPEED_OF_SOUND_M_S = 343.0       # at 20 deg C, 1 atm
DENSITY_AIR_KG_M3 = 1.225        # at 20 deg C, 1 atm
FREQ_HZ = 2000.0                 # Phase 0 test frequency
WAVELENGTH_M = SPEED_OF_SOUND_M_S / FREQ_HZ  # 0.1715 m
WAVENUMBER_RAD_M = 2 * np.pi * FREQ_HZ / SPEED_OF_SOUND_M_S  # 36.6 rad/m
```

## Key Equations

### Macdonald Analytical Solution (Infinite Wedge)
The diffracted field around an infinite wedge of exterior angle nu*pi:
```
p_diff(r, theta) = sum over n of J_n/nu(kr) * [A_n cos(n*theta/nu) + B_n sin(n*theta/nu)]
```
where J_n/nu are fractional-order Bessel functions.

### Free-Space Green's Function (2D)
```
G_0(r, r') = (i/4) * H_0^(1)(k|r - r'|)
```
where H_0^(1) is the zeroth-order Hankel function of the first kind.

### BEM Error Metric
```
error = ||p_BEM - p_analytical||_2 / ||p_analytical||_2
```
