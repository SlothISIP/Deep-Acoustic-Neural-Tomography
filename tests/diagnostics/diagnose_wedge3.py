"""Test if longer faces + open surface gives better exterior field near tip.

Also compute combined surface pressure L2 error for different face lengths.
"""
import numpy as np
from scipy.special import hankel1, jv
import logging
import time

logging.basicConfig(level=logging.WARNING)

k = 2.0 * np.pi * 2000.0 / 343.0
WAVELENGTH = 343.0 / 2000.0
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


def generate_open_wedge_mesh(face_length_m, h_edge, h_flat, transition_m):
    """Generate open-surface wedge mesh (Face1 + Face2 only, no hypotenuse)."""
    def graded_nodes(length, h_min, h_max, trans):
        nodes = [0.0]
        pos = 0.0
        while pos < length:
            t = min(pos / trans, 1.0)
            h = h_min + t * (h_max - h_min)
            pos += h
            if pos > length:
                pos = length
            nodes.append(pos)
        return np.array(nodes)

    L = face_length_m
    nodes1 = graded_nodes(L, h_edge, h_flat, transition_m)
    n1 = len(nodes1) - 1
    mid1_x = 0.5 * (nodes1[:-1] + nodes1[1:])
    mid1_y = np.zeros(n1)
    len1 = nodes1[1:] - nodes1[:-1]
    norm1 = np.column_stack([np.zeros(n1), np.ones(n1)])

    nodes2 = graded_nodes(L, h_edge, h_flat, transition_m)
    n2 = len(nodes2) - 1
    mid2_dist = 0.5 * (nodes2[:-1] + nodes2[1:])
    mid2_x = np.zeros(n2)
    mid2_y = -mid2_dist
    len2 = nodes2[1:] - nodes2[:-1]
    norm2 = np.column_stack([-np.ones(n2), np.zeros(n2)])

    midpoints = np.vstack([
        np.column_stack([mid1_x, mid1_y]),
        np.column_stack([mid2_x, mid2_y]),
    ])
    normals = np.vstack([norm1, norm2])
    lengths = np.concatenate([len1, len2])
    return midpoints, normals, lengths, n1, n2


def assemble_and_solve(midpoints, normals, lengths, k_val, sx, sy):
    """Assemble BEM matrix and solve."""
    n = len(lengths)
    gauss_pts, gauss_wts = np.polynomial.legendre.leggauss(4)
    tangents = np.column_stack([-normals[:, 1], normals[:, 0]])

    D = np.zeros((n, n), dtype=np.complex128)
    for j in range(n):
        half_len = lengths[j] / 2.0
        qpts = midpoints[j][None, :] + gauss_pts[:, None] * half_len * tangents[j][None, :]
        nj = normals[j]
        for i in range(n):
            if i == j:
                continue
            dx = midpoints[i][None, :] - qpts
            dist = np.sqrt(np.sum(dx**2, axis=1))
            if np.any(dist < 1e-12):
                continue
            kr = k_val * dist
            H1 = hankel1(1, kr)
            dot_dn = np.sum(dx * nj[None, :], axis=1)
            kernel = -0.25j * k_val * H1 * dot_dn / dist
            D[i, j] = half_len * np.sum(gauss_wts * kernel)

    A = 0.5 * np.eye(n, dtype=np.complex128) + D

    dx_s = midpoints[:, 0] - sx
    dy_s = midpoints[:, 1] - sy
    dist_s = np.maximum(np.sqrt(dx_s**2 + dy_s**2), 1e-15)
    p_inc = -0.25j * hankel1(0, k_val * dist_s)

    p_surface = np.linalg.solve(A, p_inc)
    return p_surface, tangents


def evaluate_field(eval_pts, midpoints, normals, lengths, tangents, p_surface, sx, sy, k_val):
    """Evaluate BEM field at exterior points."""
    gauss_pts, gauss_wts = np.polynomial.legendre.leggauss(4)
    n_eval = eval_pts.shape[0]
    n_elem = midpoints.shape[0]

    dx_s = eval_pts[:, 0] - sx
    dy_s = eval_pts[:, 1] - sy
    dist_s = np.maximum(np.sqrt(dx_s**2 + dy_s**2), 1e-15)
    p_inc = -0.25j * hankel1(0, k_val * dist_s)

    p_scat = np.zeros(n_eval, dtype=np.complex128)
    for j in range(n_elem):
        half_len = lengths[j] / 2.0
        qpts = midpoints[j][None, :] + gauss_pts[:, None] * half_len * tangents[j][None, :]
        nj = normals[j]
        for q in range(len(gauss_pts)):
            yq = qpts[q]
            dx = eval_pts - yq[None, :]
            dist = np.maximum(np.sqrt(np.sum(dx**2, axis=1)), 1e-15)
            kr = k_val * dist
            H1 = hankel1(1, kr)
            dot_dn = np.sum(dx * nj[None, :], axis=1)
            kernel = -0.25j * k_val * H1 * dot_dn / dist
            p_scat += half_len * gauss_wts[q] * kernel * p_surface[j]

    return p_inc + p_scat


# ============================================================
# Test different face lengths
# ============================================================
h_edge = WAVELENGTH / 10.0
h_flat = WAVELENGTH / 6.0
transition = 0.3

face_lengths = [3.0, 5.0, 8.0]

for L in face_lengths:
    print(f"\n{'='*70}")
    print(f"Face length L = {L:.1f} m (L/lambda = {L/WAVELENGTH:.1f})")
    print(f"{'='*70}")

    t0 = time.time()
    midpts, norms, lens, n1, n2 = generate_open_wedge_mesh(L, h_edge, h_flat, transition)
    n_total = len(lens)
    print(f"Mesh: {n_total} elements (F1={n1}, F2={n2})")

    p_surf, tans = assemble_and_solve(midpts, norms, lens, k, SOURCE_X, SOURCE_Y)
    t_solve = time.time() - t0
    print(f"Solve time: {t_solve:.1f} s")

    # Analytical surface pressure
    r_coll = np.sqrt(midpts[:, 0]**2 + midpts[:, 1]**2)
    theta_coll = np.arctan2(midpts[:, 1], midpts[:, 0])
    theta_coll = np.where(theta_coll < 0, theta_coll + 2 * np.pi, theta_coll)
    p_ana = macdonald_green(r_coll, theta_coll, SOURCE_R, SOURCE_THETA, k, EXTERIOR_ANGLE)

    # Surface pressure L2 error
    err_f1 = np.sqrt(np.sum(np.abs(p_surf[:n1] - p_ana[:n1])**2))
    norm_f1 = np.sqrt(np.sum(np.abs(p_ana[:n1])**2))
    err_f2 = np.sqrt(np.sum(np.abs(p_surf[n1:] - p_ana[n1:])**2))
    norm_f2 = np.sqrt(np.sum(np.abs(p_ana[n1:])**2))
    err_total = np.sqrt(err_f1**2 + err_f2**2) / np.sqrt(norm_f1**2 + norm_f2**2)

    print(f"\nSurface pressure L2 error:")
    print(f"  Face1: {100*err_f1/norm_f1:.2f}%")
    print(f"  Face2: {100*err_f2/norm_f2:.2f}%")
    print(f"  Combined: {100*err_total:.2f}%")

    # Exterior field comparison (near tip only)
    test_pts = [
        (0.15, np.pi / 4), (0.15, np.pi / 2), (0.15, np.pi), (0.15, 5*np.pi/4),
        (0.3, np.pi / 4), (0.3, np.pi / 2), (0.3, np.pi), (0.3, 5*np.pi/4),
        (0.5, np.pi / 4), (0.5, np.pi / 2), (0.5, np.pi), (0.5, 5*np.pi/4),
    ]

    print(f"\nExterior field comparison:")
    print(f"  {'r':>5} {'th':>6} {'|G_ana|':>10} {'|G_bem|':>10} {'err%':>8}")
    print(f"  {'-'*45}")

    all_errs = []
    all_ana = []
    for r_val, th_val in test_pts:
        x = r_val * np.cos(th_val)
        y = r_val * np.sin(th_val)
        G_ana_val = macdonald_green(np.array([r_val]), np.array([th_val]),
                                     SOURCE_R, SOURCE_THETA, k, EXTERIOR_ANGLE)[0]
        G_bem_val = evaluate_field(np.array([[x, y]]), midpts, norms, lens, tans,
                                    p_surf, SOURCE_X, SOURCE_Y, k)[0]
        err = np.abs(G_bem_val - G_ana_val) / max(np.abs(G_ana_val), 1e-30)
        all_errs.append(np.abs(G_bem_val - G_ana_val)**2)
        all_ana.append(np.abs(G_ana_val)**2)
        print(f"  {r_val:5.2f} {np.degrees(th_val):5.0f}d {np.abs(G_ana_val):10.6f}"
              f" {np.abs(G_bem_val):10.6f} {100*err:7.2f}%")

    # L2 error over all test points
    l2_err = np.sqrt(sum(all_errs)) / np.sqrt(sum(all_ana))
    print(f"\n  Overall exterior L2 error: {100*l2_err:.2f}%")
