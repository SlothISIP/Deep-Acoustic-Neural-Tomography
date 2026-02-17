"""Diagnose the wedge BEM vs analytical discrepancy.

Tests:
1. Check surface pressure on flat face far from tip (should be ~2*p_inc)
2. Compare analytical solution at a few points via direct computation
3. Check if the hypotenuse is causing problems by comparing with/without it
4. Verify normal directions visually
"""
import numpy as np
from scipy.special import hankel1, jv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Physical constants
k = 2.0 * np.pi * 2000.0 / 343.0  # 36.64 rad/m
EXTERIOR_ANGLE = 3.0 * np.pi / 2.0  # 270 deg
NU = np.pi / EXTERIOR_ANGLE  # 2/3
SOURCE_R = 0.5
SOURCE_THETA = np.pi / 2.0  # 90 deg, along +y axis
SOURCE_X = SOURCE_R * np.cos(SOURCE_THETA)  # ~0
SOURCE_Y = SOURCE_R * np.sin(SOURCE_THETA)  # 0.5


def macdonald_green(r, theta, r0, theta0, k_val, Phi, n_terms=300):
    """Macdonald series for rigid wedge Neumann Green's function."""
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


def green_2d(x1, y1, x2, y2, k_val):
    """Free-space 2D Green's function: -(i/4) H_0^(1)(k*r)."""
    r = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    r = np.maximum(r, 1e-15)
    return -0.25j * hankel1(0, k_val * r)


# ============================================================
# TEST 1: Analytical solution sanity check at specific points
# ============================================================
print("=" * 60)
print("TEST 1: Analytical solution spot checks")
print("=" * 60)
print(f"Source: r0={SOURCE_R}, theta0={np.degrees(SOURCE_THETA):.0f} deg")
print(f"Wedge: exterior={np.degrees(EXTERIOR_ANGLE):.0f} deg, nu={NU:.4f}")
print()

# Check at a point in the lit region (should be close to free-space + image)
test_points = [
    (0.3, np.pi / 4, "Lit region (45 deg)"),
    (0.8, np.pi / 2, "Same angle as source (90 deg)"),
    (0.5, np.pi, "Opposite side (180 deg)"),
    (0.5, 5 * np.pi / 4, "Shadow region (225 deg)"),
    (0.3, 0.05, "Near Face 1 (theta~0)"),
    (0.3, EXTERIOR_ANGLE - 0.05, "Near Face 2 (theta~270 deg)"),
]

print(f"  {'Point':<30} {'r':>5} {'theta':>8} {'|G_N|':>10} {'|G_0|':>10} {'ratio':>8}")
print("-" * 80)
for r_val, th_val, label in test_points:
    r_arr = np.array([r_val])
    th_arr = np.array([th_val])
    G_N = macdonald_green(r_arr, th_arr, SOURCE_R, SOURCE_THETA, k, EXTERIOR_ANGLE)[0]
    G_0 = green_2d(
        r_val * np.cos(th_val), r_val * np.sin(th_val),
        SOURCE_X, SOURCE_Y, k
    )
    if isinstance(G_0, np.ndarray):
        G_0 = G_0[0]
    ratio = np.abs(G_N) / max(np.abs(G_0), 1e-30)
    print(f"  {label:<30} {r_val:5.2f} {np.degrees(th_val):7.1f}d"
          f"  {np.abs(G_N):10.6f}  {np.abs(G_0):10.6f}  {ratio:8.4f}")

# ============================================================
# TEST 2: Import the actual BEM solver and check surface pressure
# ============================================================
print()
print("=" * 60)
print("TEST 2: Surface pressure check (should be ~2*p_inc on flat face)")
print("=" * 60)

from validate_wedge_bem import (
    generate_wedge_mesh_2d, assemble_bem_matrix_2d,
    compute_incident_field_2d, WAVELENGTH_M,
)

midpoints, normals, lengths, n_elem = generate_wedge_mesh_2d(
    face_length_m=3.0,
    element_size_flat_m=WAVELENGTH_M / 6.0,
    element_size_edge_m=WAVELENGTH_M / 10.0,
    grading_transition_m=0.3,
    exterior_angle_rad=EXTERIOR_ANGLE,
)

# Identify which elements are on each face
face1_mask = np.abs(midpoints[:, 1]) < 1e-6  # y ~ 0 (Face 1 along +x)
face2_mask = np.abs(midpoints[:, 0]) < 1e-6  # x ~ 0 (Face 2 along -y)
hyp_mask = ~face1_mask & ~face2_mask

n_f1 = np.sum(face1_mask)
n_f2 = np.sum(face2_mask)
n_hyp = np.sum(hyp_mask)
print(f"Mesh: {n_elem} elements (Face1={n_f1}, Face2={n_f2}, Hyp={n_hyp})")
print()

# Check normal directions
print("Normal direction samples:")
for name, mask in [("Face1", face1_mask), ("Face2", face2_mask), ("Hyp", hyp_mask)]:
    idx = np.where(mask)[0]
    if len(idx) > 0:
        i_mid = idx[len(idx) // 2]
        print(f"  {name}: midpt=({midpoints[i_mid,0]:.3f}, {midpoints[i_mid,1]:.3f})"
              f" normal=({normals[i_mid,0]:.3f}, {normals[i_mid,1]:.3f})")

# Assemble and solve
A = assemble_bem_matrix_2d(midpoints, normals, lengths, k)
p_inc = compute_incident_field_2d(midpoints, SOURCE_X, SOURCE_Y, k)
p_surface = np.linalg.solve(A, p_inc)

print()
print("Surface pressure ratio |p_surface / p_inc| (should be ~2.0 on flat faces):")
print(f"  {'Face':<8} {'i':>4} {'x':>7} {'y':>7} {'|p_s/p_i|':>10} {'phase_diff':>12}")
print("-" * 60)

# Check a few elements on each face, far from tip
for name, mask in [("Face1", face1_mask), ("Face2", face2_mask)]:
    idx = np.where(mask)[0]
    # Pick elements far from tip (last few)
    far_idx = idx[-5:]
    for i in far_idx:
        ratio = np.abs(p_surface[i]) / max(np.abs(p_inc[i]), 1e-30)
        phase_diff = np.angle(p_surface[i]) - np.angle(p_inc[i])
        print(f"  {name:<8} {i:4d} {midpoints[i,0]:7.3f} {midpoints[i,1]:7.3f}"
              f"  {ratio:10.4f}  {np.degrees(phase_diff):10.2f} deg")

# Also check near the tip
print("\nNear tip (should still be ~2 for half-space, but wedge enhances):")
for name, mask in [("Face1", face1_mask), ("Face2", face2_mask)]:
    idx = np.where(mask)[0]
    near_idx = idx[:3]
    for i in near_idx:
        ratio = np.abs(p_surface[i]) / max(np.abs(p_inc[i]), 1e-30)
        print(f"  {name:<8} {i:4d} ({midpoints[i,0]:7.4f}, {midpoints[i,1]:7.4f})"
              f"  ratio={ratio:.4f}")

# ============================================================
# TEST 3: Compare analytical vs BEM at a single exterior point
# ============================================================
print()
print("=" * 60)
print("TEST 3: Single-point comparison (analytical vs BEM)")
print("=" * 60)

from validate_wedge_bem import evaluate_bem_field_2d

tangents = np.column_stack([-normals[:, 1], normals[:, 0]])

# A few test points in the exterior
test_ext = [
    (0.3, np.pi / 4),
    (0.8, np.pi / 2),
    (0.5, np.pi),
    (0.5, 5 * np.pi / 4),
]

print(f"  {'r':>5} {'theta':>8} {'|G_ana|':>10} {'|G_bem|':>10} {'rel_err':>10}")
print("-" * 60)
for r_val, th_val in test_ext:
    x_val = r_val * np.cos(th_val)
    y_val = r_val * np.sin(th_val)

    # Analytical
    G_ana = macdonald_green(
        np.array([r_val]), np.array([th_val]),
        SOURCE_R, SOURCE_THETA, k, EXTERIOR_ANGLE
    )[0]

    # BEM
    eval_pt = np.array([[x_val, y_val]])
    G_bem = evaluate_bem_field_2d(
        eval_pt, midpoints, normals, lengths, tangents,
        p_surface, SOURCE_X, SOURCE_Y, k
    )[0]

    err = np.abs(G_bem - G_ana) / max(np.abs(G_ana), 1e-30)
    print(f"  {r_val:5.2f} {np.degrees(th_val):7.1f}d"
          f"  {np.abs(G_ana):10.6f}  {np.abs(G_bem):10.6f}  {err:10.4f}")

# ============================================================
# TEST 4: Visualize mesh and normals
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
colors = {'Face1': 'blue', 'Face2': 'red', 'Hyp': 'green'}
for name, mask, col in [("Face1", face1_mask, 'blue'), ("Face2", face2_mask, 'red'),
                         ("Hyp", hyp_mask, 'green')]:
    idx = np.where(mask)[0]
    ax.scatter(midpoints[idx, 0], midpoints[idx, 1], c=col, s=10, label=name)
    # Draw normals (scaled)
    scale = 0.05
    for i in idx[::5]:  # every 5th element
        ax.arrow(midpoints[i, 0], midpoints[i, 1],
                 normals[i, 0] * scale, normals[i, 1] * scale,
                 head_width=0.01, color=col, alpha=0.5)

ax.plot(SOURCE_X, SOURCE_Y, 'r*', markersize=15, label='Source')
ax.set_xlim([-0.5, 3.5])
ax.set_ylim([-3.5, 0.5])
ax.set_aspect('equal')
ax.legend()
ax.set_title("Wedge mesh with normals")
ax.grid(True, alpha=0.3)
plt.savefig("results/phase0/wedge_mesh_normals.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nMesh visualization saved to results/phase0/wedge_mesh_normals.png")
