"""Verify the BEM representation formula for a HALF-PLANE (single face).

For a rigid half-plane at y=0 (x>0), with sound in y>0:
- Analytical: p_total = G_0(x, x_s) + G_0(x, x_s_image)
  where x_s_image is the mirror of x_s about y=0.
- Surface pressure on face: p(s,0) = 2 * G_0((s,0), x_s)
- Representation: p_total(x) = p_inc(x) + int_0^L dG/dn * p ds

This tests whether the representation formula works for a semi-infinite face.
"""
import numpy as np
from scipy.special import hankel1

k = 36.64  # rad/m
SOURCE = np.array([0.0, 0.5])  # Source at (0, 0.5)
SOURCE_IMAGE = np.array([0.0, -0.5])  # Image at (0, -0.5)


def G0(x, y, k_val):
    """Free-space 2D Green's function."""
    r = np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
    r = max(r, 1e-15)
    return -0.25j * hankel1(0, k_val * r)


def dG0_dn(x, y, n, k_val):
    """Normal derivative of free-space Green's function at y."""
    dx = np.array([x[0] - y[0], x[1] - y[1]])
    r = np.sqrt(dx[0]**2 + dx[1]**2)
    if r < 1e-12:
        return 0.0
    kr = k_val * r
    H1 = hankel1(1, kr)
    dot = dx[0] * n[0] + dx[1] * n[1]
    return -0.25j * k_val * H1 * dot / r


# ==============================================
# Test 1: Single point, numerical integration
# ==============================================
print("=" * 60)
print("Test: Half-plane representation formula")
print("=" * 60)

field_pts = [
    np.array([0.3, 0.3]),
    np.array([0.1, 0.5]),
    np.array([0.5, 0.1]),
    np.array([-0.3, 0.3]),
    np.array([0.0, 0.8]),
]

normal = np.array([0.0, 1.0])  # Face normal (upward)

# Use very fine integration over truncated face [0, L]
L = 10.0
N_int = 5000
s_arr = np.linspace(0, L, N_int + 1)
s_mid = 0.5 * (s_arr[:-1] + s_arr[1:])
ds = s_arr[1] - s_arr[0]

print(f"\nL = {L} m, N_int = {N_int}, ds = {ds:.4f} m")
print(f"Source: {SOURCE}")
print(f"\n{'Point':<20} {'|p_exact|':>10} {'|p_repr|':>10} {'rel_err':>10}")
print("-" * 55)

for pt in field_pts:
    # Exact solution: direct + image
    p_exact = G0(pt, SOURCE, k) + G0(pt, SOURCE_IMAGE, k)

    # Representation formula: p_inc + integral of dG/dn * p_surface
    p_inc = G0(pt, SOURCE, k)

    integral = 0.0 + 0.0j
    for i in range(N_int):
        y = np.array([s_mid[i], 0.0])
        # Surface pressure = 2 * G_0(y, source)
        p_surf = 2.0 * G0(y, SOURCE, k)
        # Kernel
        kernel = dG0_dn(pt, y, normal, k)
        integral += kernel * p_surf * ds

    p_repr = p_inc + integral

    err = abs(p_repr - p_exact) / max(abs(p_exact), 1e-30)
    print(f"({pt[0]:5.1f}, {pt[1]:4.1f})       "
          f" {abs(p_exact):10.6f} {abs(p_repr):10.6f} {err:10.6f}")

# ==============================================
# Test 2: Check how integral changes with L
# ==============================================
print(f"\n{'='*60}")
print("Effect of truncation length on representation formula")
print(f"{'='*60}")

pt = np.array([0.3, 0.3])
p_exact = G0(pt, SOURCE, k) + G0(pt, SOURCE_IMAGE, k)
p_inc = G0(pt, SOURCE, k)

print(f"Field point: {pt}, |p_exact| = {abs(p_exact):.6f}")
print(f"\n{'L':>6} {'N':>6} {'|p_repr|':>10} {'err':>10}")
for L_test in [1, 2, 3, 5, 10, 20]:
    N_test = max(int(L_test / 0.002), 500)
    s_test = np.linspace(0, L_test, N_test + 1)
    s_mid_test = 0.5 * (s_test[:-1] + s_test[1:])
    ds_test = s_test[1] - s_test[0]

    integral_test = 0.0 + 0.0j
    for i in range(N_test):
        y = np.array([s_mid_test[i], 0.0])
        p_surf = 2.0 * G0(y, SOURCE, k)
        kernel = dG0_dn(pt, y, normal, k)
        integral_test += kernel * p_surf * ds_test

    p_repr_test = p_inc + integral_test
    err_test = abs(p_repr_test - p_exact) / abs(p_exact)
    print(f"{L_test:6.0f} {N_test:6d} {abs(p_repr_test):10.6f} {err_test:10.6f}")

# ==============================================
# Test 3: Now try the WEDGE (90-deg, two faces)
# ==============================================
print(f"\n{'='*60}")
print("Wedge representation formula (90-deg wedge)")
print(f"{'='*60}")

from diagnose_wedge3 import macdonald_green

EXTERIOR_ANGLE = 3.0 * np.pi / 2.0
source_r = 0.5
source_theta = np.pi / 2.0
source_x = source_r * np.cos(source_theta)
source_y = source_r * np.sin(source_theta)

# Analytical surface pressure from Macdonald on Face 1 (theta=0) and Face 2 (theta=3pi/2)
L_wedge = 10.0
N_wedge = 5000

# Face 1: along +x, normal (0,1)
s1 = np.linspace(0.001, L_wedge, N_wedge)
ds1 = s1[1] - s1[0]
p_f1 = macdonald_green(s1, np.zeros_like(s1), source_r, source_theta, k, EXTERIOR_ANGLE)

# Face 2: along -y, normal (-1, 0)
s2 = np.linspace(0.001, L_wedge, N_wedge)
ds2 = s2[1] - s2[0]
p_f2 = macdonald_green(s2, np.full_like(s2, EXTERIOR_ANGLE), source_r, source_theta, k, EXTERIOR_ANGLE)

test_wedge_pts = [
    (0.15, np.pi / 4),
    (0.3, np.pi / 4),
    (0.3, np.pi),
    (0.5, np.pi),
    (0.3, 5 * np.pi / 4),
]

print(f"L = {L_wedge} m, N per face = {N_wedge}")
print(f"\n{'r':>5} {'th':>6} {'|G_ana|':>10} {'|G_repr|':>10} {'err':>10}")
print("-" * 45)

for r_val, th_val in test_wedge_pts:
    x = r_val * np.cos(th_val)
    y = r_val * np.sin(th_val)
    pt = np.array([x, y])

    G_ana = macdonald_green(
        np.array([r_val]), np.array([th_val]),
        source_r, source_theta, k, EXTERIOR_ANGLE
    )[0]

    p_inc_val = G0(pt, np.array([source_x, source_y]), k)

    # Face 1 integral
    integral_f1 = 0.0 + 0.0j
    for i in range(N_wedge):
        y_pt = np.array([s1[i], 0.0])
        kernel = dG0_dn(pt, y_pt, np.array([0.0, 1.0]), k)
        integral_f1 += kernel * p_f1[i] * ds1

    # Face 2 integral
    integral_f2 = 0.0 + 0.0j
    for i in range(N_wedge):
        y_pt = np.array([0.0, -s2[i]])
        kernel = dG0_dn(pt, y_pt, np.array([-1.0, 0.0]), k)
        integral_f2 += kernel * p_f2[i] * ds2

    p_repr = p_inc_val + integral_f1 + integral_f2

    err = abs(p_repr - G_ana) / max(abs(G_ana), 1e-30)
    print(f"{r_val:5.2f} {np.degrees(th_val):5.0f}d {abs(G_ana):10.6f}"
          f" {abs(p_repr):10.6f} {err:10.6f}")
