"""Diagnostic: Macdonald series sign + BIE sign convention for 2D rigid wedge BEM.

Test 1: Macdonald series vs image-source for half-space (Phi=pi, nu=1).
Test 2: BIE sign convention via rigid circular cylinder (a=0.2m).
"""
import numpy as np
from scipy.special import hankel1, jv

SPEED_OF_SOUND_M_PER_S: float = 343.0
FREQ_HZ: float = 2000.0
K: float = 2.0 * np.pi * FREQ_HZ / SPEED_OF_SOUND_M_PER_S  # ~36.64 rad/m


def macdonald_series(r_m, theta_rad, r0_m, theta0_rad, k, Phi, n_terms=400, sign=-1.0):
    """Macdonald Neumann Green's function: (sign*i*nu/2) * sum eps_m J H cos cos."""
    nu = np.pi / Phi  # (dimensionless)
    kr_lt = k * np.minimum(r_m, r0_m)  # (N,)
    kr_gt = k * np.maximum(r_m, r0_m)  # (N,)
    G = np.zeros(r_m.shape, dtype=np.complex128)  # (N,)
    for m in range(n_terms):
        order = m * nu
        eps = 1.0 if m == 0 else 2.0
        term = eps * jv(order, kr_lt) * hankel1(order, kr_gt) \
            * np.cos(order * theta_rad) * np.cos(order * theta0_rad)  # (N,)
        G += term
        if m > 30 and np.max(np.abs(term)) < 1e-15 * np.max(np.abs(G)):
            break
    return G * (sign * 1j * nu / 2.0)  # (N,), complex128


def green_2d(xf, yf, xs, ys, k):
    """G_0 = -(i/4) H_0^(1)(k|x-xs|)."""
    d = np.maximum(np.sqrt((xf - xs)**2 + (yf - ys)**2), 1e-15)  # (N,)
    return -0.25j * hankel1(0, k * d)  # (N,), complex128


def assemble_D(midpts, normals, tangents, lengths, k, n_gauss=6):
    """Assemble double-layer operator D_{ij} = int_j dG/dn_y(x_i, y) dGamma."""
    n = len(lengths)
    gp, gw = np.polynomial.legendre.leggauss(n_gauss)  # (Q,), (Q,)
    D = np.zeros((n, n), dtype=np.complex128)  # (N, N)
    for j in range(n):
        hl = lengths[j] / 2.0
        qpts = midpts[j][None, :] + gp[:, None] * hl * tangents[j][None, :]  # (Q, 2)
        nj = normals[j]  # (2,)
        for i in range(n):
            if i == j:
                continue
            dx = midpts[i][None, :] - qpts  # (Q, 2)
            dist = np.sqrt(np.sum(dx**2, axis=1))  # (Q,)
            if np.any(dist < 1e-12):
                continue
            dot_n = np.sum(dx * nj[None, :], axis=1)  # (Q,)
            kern = -0.25j * k * hankel1(1, k * dist) * dot_n / dist  # (Q,)
            D[i, j] = hl * np.sum(gw * kern)
    return D  # (N, N), complex128


# ===== TEST 1: Macdonald vs image-source (half-space) =====
def test1():
    print("=" * 70)
    print("TEST 1: Macdonald series vs image-source (Phi=pi, nu=1)")
    print("=" * 70)
    Phi = np.pi
    r0, th0 = 0.5, np.pi / 4.0
    xs, ys = r0 * np.cos(th0), r0 * np.sin(th0)
    r_pts = np.array([0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5])  # (8,)
    th_pts = np.array([0.1, np.pi/6, np.pi/4, np.pi/3,
                       np.pi/2, 2*np.pi/3, 5*np.pi/6, np.pi-0.1])  # (8,)
    xf = r_pts * np.cos(th_pts)  # (8,)
    yf = r_pts * np.sin(th_pts)  # (8,)

    G_img = green_2d(xf, yf, xs, ys, K) + green_2d(xf, yf, xs, -ys, K)  # (8,)
    G_neg = macdonald_series(r_pts, th_pts, r0, th0, K, Phi, sign=-1.0)  # (8,)
    G_pos = macdonald_series(r_pts, th_pts, r0, th0, K, Phi, sign=+1.0)  # (8,)

    norm = np.sqrt(np.sum(np.abs(G_img)**2))
    l2_neg = np.sqrt(np.sum(np.abs(G_neg - G_img)**2)) / norm
    l2_pos = np.sqrt(np.sum(np.abs(G_pos - G_img)**2)) / norm
    pw_neg = np.abs(G_neg - G_img) / (np.abs(G_img) + 1e-30)  # (8,)
    pw_pos = np.abs(G_pos - G_img) / (np.abs(G_img) + 1e-30)  # (8,)

    print(f"\nk={K:.2f}, source=(r0={r0}, th0={np.degrees(th0):.0f}deg)")
    print(f"{'pt':>3} {'r':>5} {'th_deg':>7} {'|G_img|':>9} {'err(-i)':>10} {'err(+i)':>10}")
    print("-" * 50)
    for i in range(len(r_pts)):
        print(f"{i:3d} {r_pts[i]:5.2f} {np.degrees(th_pts[i]):7.1f} "
              f"{np.abs(G_img[i]):9.6f} {pw_neg[i]:10.2e} {pw_pos[i]:10.2e}")
    print(f"\nL2 error -(i*nu/2): {l2_neg:.4e}")
    print(f"L2 error +(i*nu/2): {l2_pos:.4e}")

    if l2_neg < 0.01 and l2_pos > 0.5:
        print("--> ORIGINAL sign -(i*nu/2) is CORRECT.")
    elif l2_pos < 0.01 and l2_neg > 0.5:
        print("--> FLIPPED sign +(i*nu/2) is CORRECT. Existing code has sign error!")
    else:
        print(f"--> Ambiguous: neg={l2_neg:.4e}, pos={l2_pos:.4e}")


# ===== TEST 2: BIE sign via rigid circular cylinder =====
def test2():
    print("\n" + "=" * 70)
    print("TEST 2: BIE sign -- rigid circular cylinder (a=0.2m)")
    print("=" * 70)
    a = 0.2  # radius [m]
    ka = K * a  # ~7.33
    r0, kr0 = 0.8, K * 0.8
    sx, sy = r0, 0.0  # source on +x axis

    # -- Mesh: 120 constant elements around circle --
    ne = 120
    dphi = 2 * np.pi / ne
    phi = np.linspace(0.5 * dphi, 2 * np.pi - 0.5 * dphi, ne)  # (N,)
    mid = np.column_stack([a * np.cos(phi), a * np.sin(phi)])  # (N, 2)
    nrm = np.column_stack([np.cos(phi), np.sin(phi)])  # (N, 2)
    tng = np.column_stack([-np.sin(phi), np.cos(phi)])  # (N, 2)
    lens = np.full(ne, a * dphi)  # (N,)

    # -- Analytical: p_total(a,phi) via addition theorem + Neumann BC --
    # p = -(i/4) sum_n eps_n cos(n*phi) H_n(kr0) [J_n(ka) - J_n'(ka)/H_n'(ka) * H_n(ka)]
    def Jp(n, z):
        return -jv(1, z) if n == 0 else jv(n - 1, z) - (n / z) * jv(n, z)

    def Hp(n, z):
        return -hankel1(1, z) if n == 0 else hankel1(n - 1, z) - (n / z) * hankel1(n, z)

    p_exact = np.zeros(ne, dtype=np.complex128)  # (N,)
    for n in range(60):
        eps = 1.0 if n == 0 else 2.0
        bracket = jv(n, ka) - Jp(n, ka) * hankel1(n, ka) / Hp(n, ka)
        p_exact += eps * np.cos(n * phi) * hankel1(n, kr0) * bracket  # (N,)
    p_exact *= -0.25j  # (N,)

    # -- Assemble D and solve both conventions --
    D = assemble_D(mid, nrm, tng, lens, K)  # (N, N)
    dist_s = np.sqrt((mid[:, 0] - sx)**2 + (mid[:, 1] - sy)**2)  # (N,)
    p_inc = -0.25j * hankel1(0, K * dist_s)  # (N,)

    p_plus = np.linalg.solve(0.5 * np.eye(ne) + D, p_inc)   # (0.5I+D)p = p_inc
    p_minus = np.linalg.solve(0.5 * np.eye(ne) - D, p_inc)  # (0.5I-D)p = p_inc

    norm_ex = np.sqrt(np.sum(np.abs(p_exact)**2))
    l2_plus = np.sqrt(np.sum(np.abs(p_plus - p_exact)**2)) / norm_ex
    l2_minus = np.sqrt(np.sum(np.abs(p_minus - p_exact)**2)) / norm_ex
    pw_plus = np.abs(p_plus - p_exact) / (np.abs(p_exact) + 1e-30)  # (N,)
    pw_minus = np.abs(p_minus - p_exact) / (np.abs(p_exact) + 1e-30)  # (N,)

    print(f"\na={a}m, ka={ka:.2f}, source r0={r0}m, {ne} elements")
    print(f"\n(0.5I+D): L2={l2_plus:.4e}, max_pw={np.max(pw_plus):.4e}")
    print(f"(0.5I-D): L2={l2_minus:.4e}, max_pw={np.max(pw_minus):.4e}")

    print(f"\n{'elem':>5} {'phi_deg':>8} {'|exact|':>9} {'|+D|':>9} "
          f"{'|-D|':>9} {'err+D':>9} {'err-D':>9}")
    print("-" * 65)
    for idx in np.linspace(0, ne - 1, 10, dtype=int):
        print(f"{idx:5d} {np.degrees(phi[idx]):8.1f} {np.abs(p_exact[idx]):9.6f} "
              f"{np.abs(p_plus[idx]):9.6f} {np.abs(p_minus[idx]):9.6f} "
              f"{pw_plus[idx]:9.2e} {pw_minus[idx]:9.2e}")

    print("\n--- VERDICT ---")
    if l2_plus < 0.05 and l2_minus > 0.10:
        print("(0.5I+D)p=p_inc is CORRECT. Existing BEM sign is right.")
    elif l2_minus < 0.05 and l2_plus > 0.10:
        print("(0.5I-D)p=p_inc is CORRECT. Existing BEM sign is WRONG -- flip D!")
    else:
        print(f"Unexpected: L2(+D)={l2_plus:.3e}, L2(-D)={l2_minus:.3e}. Check kernel.")
    print(f"cond(0.5I+D)={np.linalg.cond(0.5*np.eye(ne)+D):.2e}, "
          f"cond(0.5I-D)={np.linalg.cond(0.5*np.eye(ne)-D):.2e}")


if __name__ == "__main__":
    print(f"diagnose_bem.py | f={FREQ_HZ:.0f}Hz, c={SPEED_OF_SOUND_M_PER_S:.0f}m/s, "
          f"k={K:.2f}rad/m\n")
    test1()
    test2()
    print("\n" + "=" * 70)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 70)
