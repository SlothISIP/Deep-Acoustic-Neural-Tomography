# Gate Criteria Reference

## Phase 0 → Phase 1: Foundation Validation

### Primary Gate
- **BEM vs Macdonald Analytical**: Relative L2 error < 3%
  - Computed at observation points around infinite wedge
  - Frequency: f = 2 kHz (k = 36.6 rad/m)
  - Wedge angle: 90 degrees (quarter-plane, exterior angle = 270 degrees)

### Secondary Checks
- Mesh element count N < 10,000
- Element size at flat surface <= lambda/6 = 28.6 mm
- Element size at wedge edge <= lambda/10 = 17.15 mm
- No NaN/Inf in BEM solution
- BEM matrix condition number logged (warning if > 1e10)

### Physical Constants (Phase 0)
- Speed of sound: c = 343 m/s (20 deg C, 1 atm)
- Frequency: f = 2000 Hz
- Wavelength: lambda = 0.1715 m
- Wavenumber: k = 36.6 rad/m

### Common Failure Modes
1. **Error > 3%**: Mesh too coarse near edge. Refine to lambda/10.
2. **NaN in BEM**: Singular matrix at resonance. Apply Burton-Miller (alpha = i/k).
3. **Large condition number**: Near-degenerate mesh elements. Check aspect ratios.
4. **Wrong analytical solution**: Verify wedge angle parameter nu. For 90-deg wedge, nu = 3/2 (exterior = 270 deg).

---

## Phase 1 → Phase 2: BEM Data Factory

### Primary Gate
- **Causality**: h(t < 0) energy ratio < 1e-4
- **Dataset completeness**: 15 scenes generated with SDF ground truth

### Secondary Checks
- IDFT uses np.fft.irfft() with np.unwrap() on phase
- Frequency sampling covers 2-8 kHz band
- Shadow/Lit/Transition regions labeled

---

## Phase 2 → Phase 3: Forward Model

### Primary Gate
- **BEM reconstruction error**: < 5% across 15 scenes
- **Green-Net vs UTD correlation**: r > 0.9

### Secondary Checks
- Structured Green: G_0 + G_ref frozen, only MLP_diff trained
- SIREN omega_0 scales with k
- Gradient checkpointing active (VRAM < 8GB)

---

## Phase 3 → Phase 4: Inverse Model

### Primary Gate
- **SDF recovery IoU**: > 0.8
- **Helmholtz residual**: < 1e-3

### Secondary Checks
- Eikonal constraint |nabla s| = 1 holds within 1e-2
- SDF does NOT receive frequency input
- BC Loss applied at SDF ~ 0 regions
- Robin BC: dp/dn + ik*beta*p = 0

---

## Phase 4 → Phase 5: Validation

### Primary Gate
- **Cycle-consistency correlation**: r > 0.8
- **Ablation studies**: 4 ablations complete

### Secondary Checks
- Forward Surrogate error vs BEM < 3%
- RAF real data comparison included
- Baseline comparisons: NAF, MESH2IR, pyroomacoustics
