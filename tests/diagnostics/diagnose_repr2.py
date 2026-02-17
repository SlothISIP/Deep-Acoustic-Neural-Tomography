"""Verify representation formula with CORRECT boundaries.

Half-space: the FULL x-axis (both x>0 and x<0), not just x>0.
Wedge: Face1 + Face2 (both semi-infinite faces from origin).
"""
import numpy as np
from scipy.special import hankel1, jv
import logging
logging.basicConfig(level=logging.WARNING)

k = 36.64


def G0_scalar(x, y, k_val):
    """Free-space 2D Green's function at two points."""
    r = np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
    r = max(r, 1e-15)
    return -0.25j * hankel1(0, k_val * r)


def dG0_dn_scalar(x, y, n, k_val):
    """dG_0/dn_y at y with normal n."""
    dx = np.array([x[0] - y[0], x[1] - y[1]], dtype=float)
    r = np.sqrt(dx[0]**2 + dx[1]**2)
    if r < 1e-12:
        return 0.0 + 0.0j
    return -0.25j * k_val * hankel1(1, k_val * r) * (dx @ n) / r


def macdonald_green(r, theta, r0, theta0, k_val, Phi, n_terms=300):
    nu = np.pi / Phi
    r_less = np.minimum(r, r0)
    r_greater = np.maximum(r, r0)
    G = np.zeros_like(r, dtype=np.complex128)
    for m in range(n_terms):
        order = m * nu
        eps_m = 1.0 if m == 0 else 2.0
        term = eps_m * jv(order, k_val * r_less) * hankel1(order, k_val * r_greater)
        term *= np.cos(order * theta) * np.cos(order * theta0)
        G += term
        if m > 20 and np.max(np.abs(term)) < 1e-15 * np.max(np.abs(G)):
            break
    G *= -1j * nu / 2.0
    return G


# ============================================================
# TEST 1: Half-space with FULL x-axis
# ============================================================
print("=" * 60)
print("HALF-SPACE: Representation formula with FULL x-axis")
print("=" * 60)

SOURCE = np.array([0.0, 0.5])
SOURCE_IMG = np.array([0.0, -0.5])
n_face = np.array([0.0, 1.0])  # body outward = up

field_pt = np.array([0.3, 0.3])
p_exact = G0_scalar(field_pt, SOURCE, k) + G0_scalar(field_pt, SOURCE_IMG, k)
p_inc = G0_scalar(field_pt, SOURCE, k)

print(f"Field point: ({field_pt[0]}, {field_pt[1]})")
print(f"|p_exact| = {abs(p_exact):.6f}")
print()

for L_val in [3.0, 5.0, 10.0, 20.0]:
    N = int(L_val / 0.001)
    # Integration from -L to +L (FULL x-axis)
    s_arr = np.linspace(-L_val, L_val, N + 1)
    s_mid = 0.5 * (s_arr[:-1] + s_arr[1:])
    ds = s_arr[1] - s_arr[0]

    integral = 0.0 + 0.0j
    for i in range(N):
        y = np.array([s_mid[i], 0.0])
        p_surf = 2.0 * G0_scalar(y, SOURCE, k)
        kernel_val = dG0_dn_scalar(field_pt, y, n_face, k)
        integral += kernel_val * p_surf * ds

    p_repr = p_inc + integral
    err = abs(p_repr - p_exact) / abs(p_exact)
    print(f"  L={L_val:5.1f}, N={N:6d}: |p_repr|={abs(p_repr):.6f}, err={100*err:.4f}%")

# Compare with one-sided (x>0 only)
print("\n  (x>0 only, for comparison):")
for L_val in [10.0]:
    N = int(L_val / 0.001)
    s_arr = np.linspace(0.001, L_val, N)
    ds = s_arr[1] - s_arr[0]

    integral = 0.0 + 0.0j
    for i in range(N):
        y = np.array([s_arr[i], 0.0])
        p_surf = 2.0 * G0_scalar(y, SOURCE, k)
        kernel_val = dG0_dn_scalar(field_pt, y, n_face, k)
        integral += kernel_val * p_surf * ds

    p_repr = p_inc + integral
    err = abs(p_repr - p_exact) / abs(p_exact)
    print(f"  L={L_val:5.1f}, N={N:6d}: |p_repr|={abs(p_repr):.6f}, err={100*err:.4f}% (x>0 only)")


# ============================================================
# TEST 2: Half-space using Macdonald with Phi=pi, TWO faces
# ============================================================
print()
print("=" * 60)
print("HALF-SPACE via Macdonald (Phi=pi): two faces")
print("=" * 60)

# For Phi=pi, nu=1. Face1: theta=0 (+x axis), Face2: theta=pi (-x axis)
# Both have normal (0,1) pointing up
Phi_hs = np.pi

L_val = 10.0
N = 5000
s_arr = np.linspace(0.001, L_val, N)
ds = s_arr[1] - s_arr[0]

# Surface pressure from Macdonald on Face1 (theta=0) and Face2 (theta=pi)
p_f1 = macdonald_green(s_arr, np.zeros(N), 0.5, np.pi/2, k, Phi_hs)
p_f2 = macdonald_green(s_arr, np.full(N, np.pi), 0.5, np.pi/2, k, Phi_hs)

# Representation formula with both faces
integral_f1 = 0.0 + 0.0j
integral_f2 = 0.0 + 0.0j

for i in range(N):
    # Face 1: boundary point at (s, 0), normal (0, 1)
    y1 = np.array([s_arr[i], 0.0])
    k1 = dG0_dn_scalar(field_pt, y1, np.array([0.0, 1.0]), k)
    integral_f1 += k1 * p_f1[i] * ds

    # Face 2: boundary point at (-s, 0), normal (0, 1)
    # Wait -- for Phi=pi, the body is pi < theta < 2pi = lower half-plane
    # Face 2 is theta=pi, which is the negative x-axis
    # Normal pointing from body (below) to exterior (above) = (0, 1)
    y2 = np.array([-s_arr[i], 0.0])
    k2 = dG0_dn_scalar(field_pt, y2, np.array([0.0, 1.0]), k)
    integral_f2 += k2 * p_f2[i] * ds

p_repr_2face = p_inc + integral_f1 + integral_f2
err_2face = abs(p_repr_2face - p_exact) / abs(p_exact)
print(f"Two faces (F1+F2): |p_repr|={abs(p_repr_2face):.6f}, err={100*err_2face:.4f}%")
print(f"  |integral_f1|={abs(integral_f1):.6f}, |integral_f2|={abs(integral_f2):.6f}")
print(f"  |p_inc|={abs(p_inc):.6f}, |p_exact|={abs(p_exact):.6f}")


# ============================================================
# TEST 3: 90-degree wedge with CORRECT Macdonald surface pressure
# ============================================================
print()
print("=" * 60)
print("90-DEG WEDGE: Representation formula with Macdonald surface pressure")
print("=" * 60)

EXTERIOR_ANGLE = 3.0 * np.pi / 2.0
source_r = 0.5
source_theta = np.pi / 2.0
source_x = source_r * np.cos(source_theta)
source_y = source_r * np.sin(source_theta)

L_val = 10.0
N = 5000
s_arr = np.linspace(0.001, L_val, N)
ds = s_arr[1] - s_arr[0]

# Surface pressure from Macdonald
p_f1_w = macdonald_green(s_arr, np.zeros(N), source_r, source_theta, k, EXTERIOR_ANGLE)
p_f2_w = macdonald_green(s_arr, np.full(N, EXTERIOR_ANGLE), source_r, source_theta, k, EXTERIOR_ANGLE)

test_pts = [
    (0.15, np.pi/4), (0.3, np.pi/4), (0.3, np.pi),
    (0.5, np.pi), (0.3, 5*np.pi/4),
]

print(f"L = {L_val}, N = {N}")
print(f"\n{'r':>5} {'th':>6} {'|G_ana|':>10} {'|p_repr|':>10} {'err%':>10}"
      f" {'|I_f1|':>10} {'|I_f2|':>10}")
print("-" * 75)

for r_val, th_val in test_pts:
    x = r_val * np.cos(th_val)
    y = r_val * np.sin(th_val)
    pt = np.array([x, y])

    G_ana = macdonald_green(
        np.array([r_val]), np.array([th_val]),
        source_r, source_theta, k, EXTERIOR_ANGLE
    )[0]
    p_inc_val = G0_scalar(pt, np.array([source_x, source_y]), k)

    # Face 1: boundary at (s, 0), normal (0, 1)
    I_f1 = 0.0 + 0.0j
    for i in range(N):
        y1 = np.array([s_arr[i], 0.0])
        kv = dG0_dn_scalar(pt, y1, np.array([0.0, 1.0]), k)
        I_f1 += kv * p_f1_w[i] * ds

    # Face 2: boundary at (0, -s), normal (-1, 0)
    I_f2 = 0.0 + 0.0j
    for i in range(N):
        y2 = np.array([0.0, -s_arr[i]])
        kv = dG0_dn_scalar(pt, y2, np.array([-1.0, 0.0]), k)
        I_f2 += kv * p_f2_w[i] * ds

    p_repr = p_inc_val + I_f1 + I_f2
    err = abs(p_repr - G_ana) / max(abs(G_ana), 1e-30)
    print(f"{r_val:5.2f} {np.degrees(th_val):5.0f}d {abs(G_ana):10.6f}"
          f" {abs(p_repr):10.6f} {100*err:9.2f}%"
          f" {abs(I_f1):10.6f} {abs(I_f2):10.6f}")

# Check if repr formula = p_inc + I_f1 + I_f2 is missing something
print(f"\nFor reference:")
r_val, th_val = 0.3, np.pi / 4
pt = np.array([r_val * np.cos(th_val), r_val * np.sin(th_val)])
G_ana = macdonald_green(np.array([r_val]), np.array([th_val]),
                         source_r, source_theta, k, EXTERIOR_ANGLE)[0]
p_inc_val = G0_scalar(pt, np.array([source_x, source_y]), k)
print(f"  p_inc  = {p_inc_val}")
print(f"  G_ana  = {G_ana}")
print(f"  p_scat_expected = G_ana - p_inc = {G_ana - p_inc_val}")
print(f"  |p_scat_expected| = {abs(G_ana - p_inc_val):.6f}")
