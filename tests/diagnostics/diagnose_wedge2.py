"""Diagnose wedge BEM: isolate whether the BIE solve or representation formula is wrong.

Strategy:
1. Compute analytical surface pressure on both faces (from Macdonald series)
2. Compare with BEM surface pressure -> isolates BIE solve correctness
3. Evaluate BEM field using ANALYTICAL surface pressure -> isolates repr formula
4. Test with just two faces (no hypotenuse) to see if closure is the problem
"""
import numpy as np
from scipy.special import hankel1, jv
import logging

logging.basicConfig(level=logging.WARNING)

k = 2.0 * np.pi * 2000.0 / 343.0
EXTERIOR_ANGLE = 3.0 * np.pi / 2.0
NU = np.pi / EXTERIOR_ANGLE
SOURCE_R = 0.5
SOURCE_THETA = np.pi / 2.0
SOURCE_X = SOURCE_R * np.cos(SOURCE_THETA)
SOURCE_Y = SOURCE_R * np.sin(SOURCE_THETA)


def macdonald_green(r, theta, r0, theta0, k_val, Phi, n_terms=300):
    """Macdonald Neumann Green's function for a rigid wedge."""
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
# TEST A: Analytical surface pressure on both faces
# ============================================================
print("=" * 70)
print("TEST A: Analytical surface pressure (Macdonald series)")
print("=" * 70)
print(f"Source: r0={SOURCE_R}, theta0={np.degrees(SOURCE_THETA):.0f} deg")

# Face 1: theta = 0 (along +x)
r_face1 = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
theta_face1 = np.zeros_like(r_face1)
p_ana_f1 = macdonald_green(r_face1, theta_face1, SOURCE_R, SOURCE_THETA, k, EXTERIOR_ANGLE)
p_inc_f1 = -0.25j * hankel1(0, k * np.sqrt(r_face1**2 + SOURCE_Y**2))

print("\nFace 1 (theta=0, along +x):")
print(f"  {'r':>6} {'|p_ana|':>10} {'|p_inc|':>10} {'ratio':>8} {'phase_diff':>12}")
for i in range(len(r_face1)):
    ratio = np.abs(p_ana_f1[i]) / max(np.abs(p_inc_f1[i]), 1e-30)
    pdiff = np.degrees(np.angle(p_ana_f1[i]) - np.angle(p_inc_f1[i]))
    print(f"  {r_face1[i]:6.2f} {np.abs(p_ana_f1[i]):10.6f} {np.abs(p_inc_f1[i]):10.6f}"
          f" {ratio:8.4f} {pdiff:10.2f} deg")

# Face 2: theta = 3*pi/2 (along -y)
r_face2 = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
theta_face2 = np.full_like(r_face2, EXTERIOR_ANGLE)
p_ana_f2 = macdonald_green(r_face2, theta_face2, SOURCE_R, SOURCE_THETA, k, EXTERIOR_ANGLE)
# Incident field at (0, -r): distance = r + SOURCE_Y
p_inc_f2 = -0.25j * hankel1(0, k * (r_face2 + SOURCE_Y))

print("\nFace 2 (theta=270deg, along -y):")
print(f"  {'r':>6} {'|p_ana|':>10} {'|p_inc|':>10} {'ratio':>8} {'phase_diff':>12}")
for i in range(len(r_face2)):
    ratio = np.abs(p_ana_f2[i]) / max(np.abs(p_inc_f2[i]), 1e-30)
    pdiff = np.degrees(np.angle(p_ana_f2[i]) - np.angle(p_inc_f2[i]))
    print(f"  {r_face2[i]:6.2f} {np.abs(p_ana_f2[i]):10.6f} {np.abs(p_inc_f2[i]):10.6f}"
          f" {ratio:8.4f} {pdiff:10.2f} deg")

# ============================================================
# TEST B: BEM surface pressure vs analytical surface pressure
# ============================================================
print()
print("=" * 70)
print("TEST B: BEM surface pressure vs analytical surface pressure")
print("=" * 70)

from validate_wedge_bem import (
    generate_wedge_mesh_2d, assemble_bem_matrix_2d,
    compute_incident_field_2d, evaluate_bem_field_2d, WAVELENGTH_M,
)

midpoints, normals, lengths, n_elem = generate_wedge_mesh_2d(
    face_length_m=3.0,
    element_size_flat_m=WAVELENGTH_M / 6.0,
    element_size_edge_m=WAVELENGTH_M / 10.0,
    grading_transition_m=0.3,
    exterior_angle_rad=EXTERIOR_ANGLE,
)
tangents = np.column_stack([-normals[:, 1], normals[:, 0]])

A = assemble_bem_matrix_2d(midpoints, normals, lengths, k)
p_inc_bdy = compute_incident_field_2d(midpoints, SOURCE_X, SOURCE_Y, k)
p_surface_bem = np.linalg.solve(A, p_inc_bdy)

# Compute analytical surface pressure at all collocation points
r_coll = np.sqrt(midpoints[:, 0]**2 + midpoints[:, 1]**2)
theta_coll = np.arctan2(midpoints[:, 1], midpoints[:, 0])
theta_coll = np.where(theta_coll < 0, theta_coll + 2 * np.pi, theta_coll)

p_surface_ana = macdonald_green(r_coll, theta_coll, SOURCE_R, SOURCE_THETA, k, EXTERIOR_ANGLE)

# Compare on each face
face1_mask = np.abs(midpoints[:, 1]) < 1e-6
face2_mask = np.abs(midpoints[:, 0]) < 1e-6
hyp_mask = ~face1_mask & ~face2_mask

for name, mask in [("Face1", face1_mask), ("Face2", face2_mask), ("Hyp", hyp_mask)]:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        continue
    err = np.abs(p_surface_bem[idx] - p_surface_ana[idx])
    rel_err = err / np.maximum(np.abs(p_surface_ana[idx]), 1e-30)
    print(f"\n{name}: {len(idx)} elements")
    print(f"  Mean |p_bem|={np.mean(np.abs(p_surface_bem[idx])):.6f}")
    print(f"  Mean |p_ana|={np.mean(np.abs(p_surface_ana[idx])):.6f}")
    print(f"  L2 rel error: {np.sqrt(np.sum(err**2))/np.sqrt(np.sum(np.abs(p_surface_ana[idx])**2)):.4f}")
    print(f"  Max pointwise rel error: {np.max(rel_err):.4f}")
    print(f"  Mean pointwise rel error: {np.mean(rel_err):.4f}")

    # Show a few representative elements
    print(f"  {'i':>5} {'r':>6} {'th_deg':>8} {'|p_bem|':>10} {'|p_ana|':>10} {'rel_err':>10}")
    sample_idx = np.linspace(0, len(idx) - 1, min(8, len(idx)), dtype=int)
    for si in sample_idx:
        i = idx[si]
        r_i = r_coll[i]
        th_i = np.degrees(theta_coll[i])
        print(f"  {i:5d} {r_i:6.3f} {th_i:8.1f} {np.abs(p_surface_bem[i]):10.6f}"
              f" {np.abs(p_surface_ana[i]):10.6f} {rel_err[si]:10.4f}")

# ============================================================
# TEST C: Evaluate exterior field using ANALYTICAL surface pressure
# ============================================================
print()
print("=" * 70)
print("TEST C: Exterior field using analytical surface pressure vs BEM surface")
print("=" * 70)

test_pts_polar = [
    (0.3, np.pi / 4),
    (0.8, np.pi / 2),
    (0.5, np.pi),
    (0.5, 5 * np.pi / 4),
]

print(f"  {'r':>5} {'th':>6} {'|G_ana|':>10} {'|G_bem_p|':>10} {'|G_ana_p|':>10}"
      f" {'err_bem':>10} {'err_ana_p':>10}")
print("-" * 80)

for r_val, th_val in test_pts_polar:
    x_val = r_val * np.cos(th_val)
    y_val = r_val * np.sin(th_val)

    # True analytical
    G_ana = macdonald_green(
        np.array([r_val]), np.array([th_val]),
        SOURCE_R, SOURCE_THETA, k, EXTERIOR_ANGLE
    )[0]

    eval_pt = np.array([[x_val, y_val]])

    # BEM field using BEM surface pressure
    G_bem_p = evaluate_bem_field_2d(
        eval_pt, midpoints, normals, lengths, tangents,
        p_surface_bem, SOURCE_X, SOURCE_Y, k
    )[0]

    # BEM field using ANALYTICAL surface pressure
    G_ana_p = evaluate_bem_field_2d(
        eval_pt, midpoints, normals, lengths, tangents,
        p_surface_ana, SOURCE_X, SOURCE_Y, k
    )[0]

    err_bem = np.abs(G_bem_p - G_ana) / max(np.abs(G_ana), 1e-30)
    err_ana_p = np.abs(G_ana_p - G_ana) / max(np.abs(G_ana), 1e-30)

    print(f"  {r_val:5.2f} {np.degrees(th_val):5.0f}d {np.abs(G_ana):10.6f}"
          f" {np.abs(G_bem_p):10.6f} {np.abs(G_ana_p):10.6f}"
          f" {err_bem:10.4f} {err_ana_p:10.4f}")

print()
print("Legend:")
print("  G_ana    = Macdonald analytical (true answer)")
print("  G_bem_p  = BEM repr formula + BEM surface pressure")
print("  G_ana_p  = BEM repr formula + analytical surface pressure")
print("  err_bem  = relative error of G_bem_p vs G_ana")
print("  err_ana_p = relative error of G_ana_p vs G_ana (isolates repr formula)")
print()
print("If err_ana_p is small: repr formula is correct, BEM solve is wrong")
print("If err_ana_p is large: repr formula is wrong (or mesh truncation issue)")

# ============================================================
# TEST D: Quick check -- open surface (no hypotenuse)
# ============================================================
print()
print("=" * 70)
print("TEST D: Open surface (no hypotenuse) -- just two faces")
print("=" * 70)

# Generate mesh with just two faces (drop hypotenuse elements)
n_f1 = np.sum(face1_mask)
n_f2 = np.sum(face2_mask)
f1_idx = np.where(face1_mask)[0]
f2_idx = np.where(face2_mask)[0]
open_idx = np.concatenate([f1_idx, f2_idx])

mid_open = midpoints[open_idx]
norm_open = normals[open_idx]
len_open = lengths[open_idx]
tan_open = tangents[open_idx]
n_open = len(open_idx)

print(f"Open mesh: {n_open} elements (Face1={n_f1}, Face2={n_f2})")

# Assemble BEM matrix for open surface (still using c=1/2)
A_open = assemble_bem_matrix_2d(mid_open, norm_open, len_open, k)
p_inc_open = compute_incident_field_2d(mid_open, SOURCE_X, SOURCE_Y, k)
p_surf_open = np.linalg.solve(A_open, p_inc_open)

# Compare surface pressures
p_ana_open = macdonald_green(
    np.sqrt(mid_open[:, 0]**2 + mid_open[:, 1]**2),
    np.where(
        np.arctan2(mid_open[:, 1], mid_open[:, 0]) < 0,
        np.arctan2(mid_open[:, 1], mid_open[:, 0]) + 2 * np.pi,
        np.arctan2(mid_open[:, 1], mid_open[:, 0])
    ),
    SOURCE_R, SOURCE_THETA, k, EXTERIOR_ANGLE
)

err_open_f1 = np.abs(p_surf_open[:n_f1] - p_ana_open[:n_f1])
rel_open_f1 = np.sqrt(np.sum(err_open_f1**2)) / np.sqrt(np.sum(np.abs(p_ana_open[:n_f1])**2))

err_open_f2 = np.abs(p_surf_open[n_f1:] - p_ana_open[n_f1:])
rel_open_f2 = np.sqrt(np.sum(err_open_f2**2)) / np.sqrt(np.sum(np.abs(p_ana_open[n_f1:])**2))

print(f"Surface pressure L2 error -- Face1: {rel_open_f1:.4f}, Face2: {rel_open_f2:.4f}")

# Evaluate at exterior points
print(f"\n  {'r':>5} {'th':>6} {'|G_ana|':>10} {'|G_open|':>10} {'err':>10}")
print("-" * 50)
for r_val, th_val in test_pts_polar:
    x_val = r_val * np.cos(th_val)
    y_val = r_val * np.sin(th_val)

    G_ana = macdonald_green(
        np.array([r_val]), np.array([th_val]),
        SOURCE_R, SOURCE_THETA, k, EXTERIOR_ANGLE
    )[0]

    eval_pt = np.array([[x_val, y_val]])
    G_open = evaluate_bem_field_2d(
        eval_pt, mid_open, norm_open, len_open, tan_open,
        p_surf_open, SOURCE_X, SOURCE_Y, k
    )[0]

    err = np.abs(G_open - G_ana) / max(np.abs(G_ana), 1e-30)
    print(f"  {r_val:5.2f} {np.degrees(th_val):5.0f}d {np.abs(G_ana):10.6f}"
          f" {np.abs(G_open):10.6f} {err:10.4f}")
