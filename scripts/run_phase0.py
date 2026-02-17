"""
Phase 0 Gate Validation: Wedge BEM vs Macdonald Analytical Solution

Gate Criterion: Relative L2 error < 3%

Geometry: 90-degree rigid wedge (interior angle = pi/2)
    - Exterior angle Phi = 3*pi/2 (270 degrees, where sound propagates)
    - Wedge parameter nu = pi / Phi = 2/3
    - Face 1: theta = 0 (along +x axis)
    - Face 2: theta = 3*pi/2 (along -y axis)
    - Rigid (Neumann) BC: dp/dn = 0 on both faces

Physics:
    - 2D Helmholtz equation: nabla^2 p + k^2 p = -delta(x - x_s)
    - Free-space Green's function: G_0 = -(i/4) H_0^(1)(k|x-y|)
    - Macdonald series: eigenfunction expansion in wedge domain
    - BEM: direct boundary integral equation with constant elements

Reference:
    - Macdonald, H.M. (1902). "Electric Waves." Cambridge.
    - Bowman, Senior, Uslenghi (1969). "EM and Acoustic Scattering
      by Simple Shapes." Ch. 6.
"""

import logging
import time
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import hankel1, jv

# ---------------------------------------------------------------------------
# Physical Constants
# ---------------------------------------------------------------------------
SPEED_OF_SOUND_M_PER_S: float = 343.0
FREQ_HZ: float = 2000.0
WAVELENGTH_M: float = SPEED_OF_SOUND_M_PER_S / FREQ_HZ  # 0.1715 m
WAVENUMBER_RAD_PER_M: float = 2.0 * np.pi * FREQ_HZ / SPEED_OF_SOUND_M_PER_S  # 36.6 rad/m

# Wedge geometry
INTERIOR_ANGLE_RAD: float = np.pi / 2.0  # 90-degree wedge
EXTERIOR_ANGLE_RAD: float = 2.0 * np.pi - INTERIOR_ANGLE_RAD  # 3*pi/2 = 270 deg
NU: float = np.pi / EXTERIOR_ANGLE_RAD  # 2/3

# Source position (in wedge exterior domain)
SOURCE_R_M: float = 0.5
SOURCE_THETA_RAD: float = np.pi / 2.0  # 90 degrees (along +y axis)

# Mesh parameters
FACE_LENGTH_M: float = 5.0  # truncation length per face (>> eval radius for open mesh)
ELEMENT_SIZE_FLAT_M: float = WAVELENGTH_M / 6.0  # 28.6 mm
ELEMENT_SIZE_EDGE_M: float = WAVELENGTH_M / 10.0  # 17.15 mm
GRADING_TRANSITION_M: float = 0.3  # graded zone near tip

# Evaluation grid
EVAL_R_MIN_M: float = 0.05
EVAL_R_MAX_M: float = 1.5
EVAL_N_R: int = 60
EVAL_N_THETA: int = 120

# Series truncation
N_SERIES_TERMS: int = 250

# Output
OUTPUT_DIR: Path = Path("results/phase0")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Macdonald Analytical Solution
# ---------------------------------------------------------------------------
def macdonald_wedge_green_neumann(
    r_m: np.ndarray,
    theta_rad: np.ndarray,
    r0_m: float,
    theta0_rad: float,
    k_rad_per_m: float,
    exterior_angle_rad: float,
    n_terms: int = 250,
) -> np.ndarray:
    """Macdonald eigenfunction series for the 2D rigid wedge Green's function.

    G_N(r, theta; r0, theta0) =
        -(i * nu / 2) * sum_{m=0}^{n_terms-1} epsilon_m
        * J_{m*nu}(k*r_<) * H^(1)_{m*nu}(k*r_>)
        * cos(m*nu*theta) * cos(m*nu*theta0)

    where nu = pi / Phi, epsilon_0 = 1, epsilon_m = 2 for m >= 1,
    r_< = min(r, r0), r_> = max(r, r0).

    Parameters
    ----------
    r_m : np.ndarray
        Field point radial distances [m]. Shape: (N,)
    theta_rad : np.ndarray
        Field point angles [rad], in [0, Phi]. Shape: (N,)
    r0_m : float
        Source radial distance [m].
    theta0_rad : float
        Source angle [rad], in [0, Phi].
    k_rad_per_m : float
        Wavenumber k = 2*pi*f/c [rad/m].
    exterior_angle_rad : float
        Exterior wedge angle Phi [rad].
    n_terms : int
        Number of series terms.

    Returns
    -------
    G_N : np.ndarray, complex128, shape (N,)
        Green's function values at field points.
    """
    nu = np.pi / exterior_angle_rad  # (dimensionless)

    r_less_m = np.minimum(r_m, r0_m)  # (N,)
    r_greater_m = np.maximum(r_m, r0_m)  # (N,)

    kr_less = k_rad_per_m * r_less_m  # (N,)
    kr_greater = k_rad_per_m * r_greater_m  # (N,)

    G_N = np.zeros(r_m.shape, dtype=np.complex128)  # (N,)

    for m in range(n_terms):
        order = m * nu  # fractional Bessel order (dimensionless)
        epsilon_m = 1.0 if m == 0 else 2.0

        J_term = jv(order, kr_less)  # (N,)
        H_term = hankel1(order, kr_greater)  # (N,)
        cos_field = np.cos(order * theta_rad)  # (N,)
        cos_source = np.cos(order * theta0_rad)  # scalar

        term = epsilon_m * J_term * H_term * cos_field * cos_source  # (N,)
        G_N += term

        # Convergence check: stop if terms are negligible
        if m > 20:
            term_magnitude = np.max(np.abs(term))
            if term_magnitude < 1e-15 * np.max(np.abs(G_N)):
                logger.debug(
                    "Macdonald series converged at m=%d (term_mag=%.2e)", m, term_magnitude
                )
                break

    G_N *= -1j * nu / 2.0  # (N,), complex128
    return G_N


def free_space_green_2d(
    x_field_m: np.ndarray,
    y_field_m: np.ndarray,
    x_source_m: float,
    y_source_m: float,
    k_rad_per_m: float,
) -> np.ndarray:
    """2D free-space Helmholtz Green's function.

    G_0 = -(i/4) * H_0^(1)(k * |x - x_s|)

    Parameters
    ----------
    x_field_m, y_field_m : np.ndarray
        Field point Cartesian coordinates [m]. Shape: (N,)
    x_source_m, y_source_m : float
        Source Cartesian coordinates [m].
    k_rad_per_m : float
        Wavenumber [rad/m].

    Returns
    -------
    G_0 : np.ndarray, complex128, shape (N,)
    """
    dx_m = x_field_m - x_source_m  # (N,)
    dy_m = y_field_m - y_source_m  # (N,)
    dist_m = np.sqrt(dx_m**2 + dy_m**2)  # (N,)

    dist_m = np.maximum(dist_m, 1e-15)  # avoid singularity

    G_0 = -0.25j * hankel1(0, k_rad_per_m * dist_m)  # (N,), complex128
    return G_0


# ---------------------------------------------------------------------------
# 2. 2D BEM Solver (Constant Elements, Direct BIE)
# ---------------------------------------------------------------------------
def generate_wedge_mesh_2d(
    face_length_m: float,
    element_size_flat_m: float,
    element_size_edge_m: float,
    grading_transition_m: float,
    exterior_angle_rad: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Generate an OPEN mesh for a 2D rigid wedge (Face 1 + Face 2 only).

    The wedge consists of two semi-infinite rigid faces meeting at the origin.
    We truncate each face at face_length_m. No hypotenuse closure is used,
    because the analytical (Macdonald) solution assumes an infinite wedge.
    A closed mesh with hypotenuse introduces spurious diffraction from the
    artificial boundary, contaminating both the BIE solve and the
    representation formula evaluation.

    Geometry:
        Vertex A: (0, 0) -- wedge tip
        Vertex B: (L, 0) -- end of face 1
        Vertex C: (0, -L) -- end of face 2

    Faces:
        Face 1: A -> B along +x axis       normal = (0, +1)  [outward from body]
        Face 2: C -> A along +y axis        normal = (-1, 0)  [outward from body]

    Graded meshing: fine elements (lambda/10) near the tip, coarse (lambda/6)
    away from the tip, with a smooth transition.

    Parameters
    ----------
    face_length_m : float
        Truncation length of each face [m].
    element_size_flat_m : float
        Max element size on flat region [m].
    element_size_edge_m : float
        Element size near wedge tip [m].
    grading_transition_m : float
        Distance from tip where grading transitions [m].
    exterior_angle_rad : float
        Exterior wedge angle [rad]. Used only for validation logging.

    Returns
    -------
    midpoints_m : np.ndarray, shape (N_elem, 2)
        Collocation points (element midpoints) in Cartesian [m].
    normals : np.ndarray, shape (N_elem, 2)
        Outward normals (pointing from body into exterior domain).
    lengths_m : np.ndarray, shape (N_elem,)
        Element lengths [m].
    n_elements : int
        Total number of boundary elements.
    """
    L = face_length_m

    def graded_nodes_1d(length_m: float, h_min_m: float, h_max_m: float,
                        transition_m: float) -> np.ndarray:
        """Generate graded 1D node positions from 0 to length_m."""
        nodes = [0.0]
        pos = 0.0
        while pos < length_m:
            t = min(pos / transition_m, 1.0)  # (dimensionless)
            h = h_min_m + t * (h_max_m - h_min_m)  # (m)
            pos += h
            if pos > length_m:
                pos = length_m
            nodes.append(pos)
        return np.array(nodes)  # (N_nodes,)

    # --- Face 1: A(0,0) -> B(L,0) along +x axis ---
    # Outward normal: (0, +1) -- body is below, exterior above
    nodes_f1 = graded_nodes_1d(L, element_size_edge_m, element_size_flat_m,
                                grading_transition_m)  # (N1_nodes,)
    n1 = len(nodes_f1) - 1
    mid_f1_x = 0.5 * (nodes_f1[:-1] + nodes_f1[1:])  # (n1,)
    mid_f1_y = np.zeros(n1)  # (n1,)
    len_f1 = nodes_f1[1:] - nodes_f1[:-1]  # (n1,)
    norm_f1 = np.column_stack([np.zeros(n1), np.ones(n1)])  # (n1, 2), (0, +1)

    # --- Face 2: C(0,-L) -> A(0,0) along +y axis ---
    # Outward normal: body is to the RIGHT (x>0), so outward = LEFT = (-1, 0)
    nodes_f2_dist = graded_nodes_1d(L, element_size_edge_m, element_size_flat_m,
                                     grading_transition_m)  # distance from C toward A
    n2 = len(nodes_f2_dist) - 1

    # C is at (0, -L), A is at (0, 0). Parametrize from C to A: y = -L + dist
    mid_dist_f2 = 0.5 * (nodes_f2_dist[:-1] + nodes_f2_dist[1:])  # (n2,)
    mid_f2_x = np.zeros(n2)  # (n2,)
    mid_f2_y = -L + mid_dist_f2  # (n2,), goes from near -L to near 0
    len_f2 = nodes_f2_dist[1:] - nodes_f2_dist[:-1]  # (n2,)
    norm_f2 = np.column_stack([-np.ones(n2), np.zeros(n2)])  # (n2, 2), (-1, 0)

    # Combine: Face1 + Face2 (no hypotenuse -- open mesh for infinite wedge)
    midpoints_m = np.vstack([
        np.column_stack([mid_f1_x, mid_f1_y]),  # (n1, 2)
        np.column_stack([mid_f2_x, mid_f2_y]),  # (n2, 2)
    ])  # (N_elem, 2)

    normals = np.vstack([norm_f1, norm_f2])  # (N_elem, 2)
    lengths_m = np.concatenate([len_f1, len_f2])  # (N_elem,)
    n_elements = n1 + n2

    logger.info(
        "Open mesh: Face1=%d elements, Face2=%d elements, total=%d "
        "(no hypotenuse -- infinite wedge approximation)",
        n1, n2, n_elements,
    )

    return midpoints_m, normals, lengths_m, n_elements


def assemble_bem_matrix_2d(
    midpoints_m: np.ndarray,
    normals: np.ndarray,
    lengths_m: np.ndarray,
    k_rad_per_m: float,
) -> np.ndarray:
    """Assemble the 2D BEM matrix for the exterior Neumann problem.

    Solves: (1/2 I + D) p = p_inc

    where D_ij = integral over element j of dG_0/dn_y(x_i, y) dGamma(y)
    using 4-point Gauss-Legendre quadrature for off-diagonal terms.
    Diagonal terms D_ii = 0 for flat constant elements.

    Parameters
    ----------
    midpoints_m : np.ndarray, shape (N, 2)
        Collocation points [m].
    normals : np.ndarray, shape (N, 2)
        Outward normals at collocation points.
    lengths_m : np.ndarray, shape (N,)
        Element lengths [m].
    k_rad_per_m : float
        Wavenumber [rad/m].

    Returns
    -------
    A : np.ndarray, complex128, shape (N, N)
        System matrix (1/2 I + D).
    """
    n_elem = len(lengths_m)
    k = k_rad_per_m

    # Gauss-Legendre quadrature points and weights on [-1, 1]
    gauss_pts, gauss_wts = np.polynomial.legendre.leggauss(4)  # (4,), (4,)

    # For each element j, compute quadrature points in physical space
    # Element j: centered at midpoints_m[j], length lengths_m[j],
    # oriented along the face direction (tangent = perpendicular to normal)
    tangents = np.column_stack([-normals[:, 1], normals[:, 0]])  # (N, 2)

    D = np.zeros((n_elem, n_elem), dtype=np.complex128)  # (N, N)

    for j in range(n_elem):
        half_len = lengths_m[j] / 2.0  # (m)
        # Quadrature points on element j: midpoint + t * tangent * half_len
        # t in gauss_pts (mapped from [-1,1])
        # quad_pts shape: (4, 2)
        quad_pts_m = (
            midpoints_m[j, :][None, :]
            + gauss_pts[:, None] * half_len * tangents[j, :][None, :]
        )  # (4, 2)

        n_j = normals[j, :]  # (2,), normal at element j

        for i in range(n_elem):
            if i == j:
                # Diagonal: D_ii = 0 for flat constant elements
                continue

            x_i = midpoints_m[i, :]  # (2,), collocation point

            # Vector from quadrature points to collocation point
            dx_m = x_i[None, :] - quad_pts_m  # (4, 2)
            dist_m = np.sqrt(np.sum(dx_m**2, axis=1))  # (4,)

            if np.any(dist_m < 1e-12):
                logger.warning(
                    "Near-zero distance in BEM matrix: elements %d, %d", i, j
                )
                continue

            # dG_0/dn_y = -(ik/4) * H_1^(1)(kr) * (dx . n_y) / r
            # Note: dx = x_i - y, and dn_y is the normal at y (element j)
            kr = k * dist_m  # (4,)
            H1_vals = hankel1(1, kr)  # (4,), complex128
            dot_dn = np.sum(dx_m * n_j[None, :], axis=1)  # (4,)

            kernel_vals = -0.25j * k * H1_vals * dot_dn / dist_m  # (4,)

            # Integrate: sum over quadrature points with weights and Jacobian
            D[i, j] = half_len * np.sum(gauss_wts * kernel_vals)  # complex128

    # System matrix: A = (1/2) I + D
    A = 0.5 * np.eye(n_elem, dtype=np.complex128) + D  # (N, N)

    return A


def compute_incident_field_2d(
    points_m: np.ndarray,
    source_x_m: float,
    source_y_m: float,
    k_rad_per_m: float,
) -> np.ndarray:
    """Incident field from a 2D point source (line source in 3D).

    p_inc = -(i/4) * H_0^(1)(k * |x - x_s|)

    Parameters
    ----------
    points_m : np.ndarray, shape (N, 2)
        Evaluation points [m].
    source_x_m, source_y_m : float
        Source position [m].
    k_rad_per_m : float
        Wavenumber [rad/m].

    Returns
    -------
    p_inc : np.ndarray, complex128, shape (N,)
    """
    dx = points_m[:, 0] - source_x_m  # (N,)
    dy = points_m[:, 1] - source_y_m  # (N,)
    dist_m = np.sqrt(dx**2 + dy**2)  # (N,)
    dist_m = np.maximum(dist_m, 1e-15)

    p_inc = -0.25j * hankel1(0, k_rad_per_m * dist_m)  # (N,), complex128
    return p_inc


def evaluate_bem_field_2d(
    eval_points_m: np.ndarray,
    midpoints_m: np.ndarray,
    normals: np.ndarray,
    lengths_m: np.ndarray,
    tangents: np.ndarray,
    surface_pressure: np.ndarray,
    source_x_m: float,
    source_y_m: float,
    k_rad_per_m: float,
) -> np.ndarray:
    """Evaluate the total BEM field at exterior points using representation formula.

    p_total(x) = p_inc(x) - integral_Gamma dG_0/dn_body(x,y) * p(y) dGamma(y)

    The MINUS sign arises from Green's theorem with body-outward normals:
    the outward normal from the exterior domain is -n_body at the surface.

    Parameters
    ----------
    eval_points_m : np.ndarray, shape (M, 2)
        Exterior evaluation points [m].
    midpoints_m : np.ndarray, shape (N, 2)
        Boundary element midpoints [m].
    normals : np.ndarray, shape (N, 2)
        Outward normals.
    lengths_m : np.ndarray, shape (N,)
        Element lengths [m].
    tangents : np.ndarray, shape (N, 2)
        Tangent vectors.
    surface_pressure : np.ndarray, complex128, shape (N,)
        Solved surface pressure.
    source_x_m, source_y_m : float
        Source position [m].
    k_rad_per_m : float
        Wavenumber [rad/m].

    Returns
    -------
    p_total : np.ndarray, complex128, shape (M,)
    """
    k = k_rad_per_m
    n_eval = eval_points_m.shape[0]
    n_elem = midpoints_m.shape[0]

    # Incident field at evaluation points
    p_inc = compute_incident_field_2d(
        eval_points_m, source_x_m, source_y_m, k
    )  # (M,)

    # Scattered field via double-layer potential
    gauss_pts, gauss_wts = np.polynomial.legendre.leggauss(4)  # (4,), (4,)

    p_scat = np.zeros(n_eval, dtype=np.complex128)  # (M,)

    for j in range(n_elem):
        half_len = lengths_m[j] / 2.0
        # Quadrature points on element j
        quad_pts_m = (
            midpoints_m[j, :][None, :]
            + gauss_pts[:, None] * half_len * tangents[j, :][None, :]
        )  # (4, 2)
        n_j = normals[j, :]  # (2,)

        for q in range(len(gauss_pts)):
            y_q = quad_pts_m[q, :]  # (2,)
            dx_m = eval_points_m - y_q[None, :]  # (M, 2)
            dist_m = np.sqrt(np.sum(dx_m**2, axis=1))  # (M,)
            dist_m = np.maximum(dist_m, 1e-15)

            kr = k * dist_m  # (M,)
            H1_vals = hankel1(1, kr)  # (M,), complex128
            dot_dn = np.sum(dx_m * n_j[None, :], axis=1)  # (M,)

            kernel_vals = -0.25j * k * H1_vals * dot_dn / dist_m  # (M,)

            p_scat += half_len * gauss_wts[q] * kernel_vals * surface_pressure[j]

    # Representation formula with body-outward normals uses MINUS sign:
    # p_total(x) = p_inc(x) - int_Gamma dG_0/dn_body(x,y) p(y) dGamma(y)
    p_total = p_inc - p_scat  # (M,), complex128
    return p_total


def solve_wedge_bem_2d(
    k_rad_per_m: float,
    source_r_m: float,
    source_theta_rad: float,
    exterior_angle_rad: float,
    face_length_m: float,
    element_size_flat_m: float,
    element_size_edge_m: float,
    grading_transition_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full 2D BEM solve for a rigid wedge.

    Returns mesh info and solved surface pressure.
    """
    logger.info("Generating wedge mesh...")
    midpoints_m, normals, lengths_m, n_elements = generate_wedge_mesh_2d(
        face_length_m, element_size_flat_m, element_size_edge_m,
        grading_transition_m, exterior_angle_rad,
    )
    tangents = np.column_stack([-normals[:, 1], normals[:, 0]])  # (N, 2)

    logger.info(
        "Mesh: N=%d elements, min_h=%.4f m, max_h=%.4f m",
        n_elements, np.min(lengths_m), np.max(lengths_m),
    )

    # Source position in Cartesian
    source_x_m = source_r_m * np.cos(source_theta_rad)
    source_y_m = source_r_m * np.sin(source_theta_rad)

    logger.info("Assembling BEM matrix (%d x %d)...", n_elements, n_elements)
    t0 = time.time()
    A = assemble_bem_matrix_2d(midpoints_m, normals, lengths_m, k_rad_per_m)
    t_assemble_s = time.time() - t0
    logger.info("Assembly time: %.1f s", t_assemble_s)

    # Check matrix condition
    cond = np.linalg.cond(A)
    logger.info("Matrix condition number: %.2e", cond)
    if cond > 1e10:
        logger.warning("High condition number (%.2e) -- solution may be inaccurate", cond)

    # Right-hand side: incident field at boundary collocation points
    p_inc_boundary = compute_incident_field_2d(
        midpoints_m, source_x_m, source_y_m, k_rad_per_m
    )  # (N,), complex128

    # Solve linear system
    logger.info("Solving linear system...")
    t0 = time.time()
    surface_pressure = np.linalg.solve(A, p_inc_boundary)  # (N,), complex128
    t_solve_s = time.time() - t0
    logger.info("Solve time: %.2f s", t_solve_s)

    # NaN check
    if not np.all(np.isfinite(surface_pressure)):
        n_bad = np.sum(~np.isfinite(surface_pressure))
        raise ValueError(
            f"BEM solution contains {n_bad} non-finite values out of {n_elements}"
        )

    return midpoints_m, normals, lengths_m, tangents, surface_pressure


# ---------------------------------------------------------------------------
# 3. Evaluation Grid and Comparison
# ---------------------------------------------------------------------------
def create_polar_eval_grid(
    r_min_m: float,
    r_max_m: float,
    n_r: int,
    n_theta: int,
    exterior_angle_rad: float,
    source_r_m: float,
    source_theta_rad: float,
    exclusion_radius_m: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a polar evaluation grid in the wedge exterior, excluding source vicinity.

    Returns
    -------
    r_grid_m : np.ndarray, shape (n_r, n_theta)
    theta_grid_rad : np.ndarray, shape (n_r, n_theta)
    x_grid_m : np.ndarray, shape (n_r, n_theta)
    y_grid_m : np.ndarray, shape (n_r, n_theta)
    mask_valid : np.ndarray, bool, shape (n_r, n_theta)
        True for valid evaluation points (outside source exclusion zone).
    """
    r_arr_m = np.linspace(r_min_m, r_max_m, n_r)  # (n_r,)
    # Slightly inside the exterior angle to avoid boundary points
    theta_arr_rad = np.linspace(
        0.02, exterior_angle_rad - 0.02, n_theta
    )  # (n_theta,)

    r_grid_m, theta_grid_rad = np.meshgrid(r_arr_m, theta_arr_rad, indexing="ij")
    x_grid_m = r_grid_m * np.cos(theta_grid_rad)  # (n_r, n_theta)
    y_grid_m = r_grid_m * np.sin(theta_grid_rad)  # (n_r, n_theta)

    # Exclude points near the source (singularity)
    source_x_m = source_r_m * np.cos(source_theta_rad)
    source_y_m = source_r_m * np.sin(source_theta_rad)
    dist_to_source_m = np.sqrt(
        (x_grid_m - source_x_m)**2 + (y_grid_m - source_y_m)**2
    )  # (n_r, n_theta)
    mask_valid = dist_to_source_m > exclusion_radius_m  # (n_r, n_theta)

    return r_grid_m, theta_grid_rad, x_grid_m, y_grid_m, mask_valid


def compute_relative_l2_error(
    p_bem: np.ndarray,
    p_analytical: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Compute relative L2 error between BEM and analytical solutions.

    error = ||p_BEM - p_analytical||_2 / ||p_analytical||_2

    Only considers points where mask is True.

    Parameters
    ----------
    p_bem : np.ndarray, complex128
    p_analytical : np.ndarray, complex128
    mask : np.ndarray, bool

    Returns
    -------
    error : float
        Relative L2 error (dimensionless).
    """
    diff = p_bem[mask] - p_analytical[mask]
    numerator = np.sqrt(np.sum(np.abs(diff)**2))
    denominator = np.sqrt(np.sum(np.abs(p_analytical[mask])**2))

    if denominator < 1e-30:
        raise ValueError("Analytical solution norm is near zero -- cannot compute relative error")

    error = float(numerator / denominator)
    return error


# ---------------------------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------------------------
def plot_comparison(
    r_grid_m: np.ndarray,
    theta_grid_rad: np.ndarray,
    p_analytical: np.ndarray,
    p_bem: np.ndarray,
    mask_valid: np.ndarray,
    error_pct: float,
    output_path: Path,
) -> None:
    """Plot analytical vs BEM pressure fields and error distribution.

    Saves figure to output_path.
    """
    x_grid_m = r_grid_m * np.cos(theta_grid_rad)
    y_grid_m = r_grid_m * np.sin(theta_grid_rad)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Analytical
    p_ana_db = 20.0 * np.log10(np.abs(p_analytical) + 1e-30)
    im0 = axes[0].pcolormesh(
        x_grid_m, y_grid_m, p_ana_db,
        shading="auto", cmap="RdBu_r", vmin=-60, vmax=-10,
    )
    axes[0].set_title("Analytical (Macdonald)")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_aspect("equal")
    plt.colorbar(im0, ax=axes[0], label="|p| [dB]")

    # BEM
    p_bem_db = 20.0 * np.log10(np.abs(p_bem) + 1e-30)
    im1 = axes[1].pcolormesh(
        x_grid_m, y_grid_m, p_bem_db,
        shading="auto", cmap="RdBu_r", vmin=-60, vmax=-10,
    )
    axes[1].set_title("BEM (2D)")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")
    axes[1].set_aspect("equal")
    plt.colorbar(im1, ax=axes[1], label="|p| [dB]")

    # Pointwise error (only valid points)
    pointwise_error = np.full_like(r_grid_m, np.nan)
    denom = np.abs(p_analytical)
    valid_denom = mask_valid & (denom > 1e-30)
    pointwise_error[valid_denom] = (
        np.abs(p_bem[valid_denom] - p_analytical[valid_denom]) / denom[valid_denom]
    )

    im2 = axes[2].pcolormesh(
        x_grid_m, y_grid_m, pointwise_error * 100.0,
        shading="auto", cmap="hot_r", vmin=0, vmax=10,
    )
    axes[2].set_title(f"Pointwise Error (%) -- L2={error_pct:.2f}%")
    axes[2].set_xlabel("x [m]")
    axes[2].set_ylabel("y [m]")
    axes[2].set_aspect("equal")
    plt.colorbar(im2, ax=axes[2], label="Error [%]")

    # Draw wedge faces
    for ax in axes:
        ax.plot([0, FACE_LENGTH_M], [0, 0], "k-", linewidth=2, label="Face 1")
        ax.plot(
            [0, 0], [0, -FACE_LENGTH_M], "k-", linewidth=2, label="Face 2"
        )
        ax.plot(
            SOURCE_R_M * np.cos(SOURCE_THETA_RAD),
            SOURCE_R_M * np.sin(SOURCE_THETA_RAD),
            "r*", markersize=12, label="Source",
        )
        ax.set_xlim([-1.6, 1.6])
        ax.set_ylim([-1.6, 1.6])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s", output_path)


def plot_radial_comparison(
    r_grid_m: np.ndarray,
    theta_grid_rad: np.ndarray,
    p_analytical: np.ndarray,
    p_bem: np.ndarray,
    output_path: Path,
) -> None:
    """Plot radial cross-sections at selected angles."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    test_angles_rad = [np.pi / 4, np.pi / 2, np.pi, 5 * np.pi / 4]
    test_angle_labels = ["45 deg", "90 deg (source)", "180 deg", "225 deg (shadow)"]

    for idx, (angle_rad, label) in enumerate(zip(test_angles_rad, test_angle_labels)):
        ax = axes[idx // 2, idx % 2]

        # Find nearest theta column
        theta_arr = theta_grid_rad[0, :]  # (n_theta,)
        j_closest = np.argmin(np.abs(theta_arr - angle_rad))
        actual_angle_deg = np.degrees(theta_arr[j_closest])

        r_arr = r_grid_m[:, j_closest]  # (n_r,)
        p_ana_slice = np.abs(p_analytical[:, j_closest])  # (n_r,)
        p_bem_slice = np.abs(p_bem[:, j_closest])  # (n_r,)

        ax.semilogy(r_arr, p_ana_slice, "b-", linewidth=2, label="Analytical")
        ax.semilogy(r_arr, p_bem_slice, "r--", linewidth=1.5, label="BEM")
        ax.set_title(f"theta = {label} (actual: {actual_angle_deg:.1f} deg)")
        ax.set_xlabel("r [m]")
        ax.set_ylabel("|p|")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s", output_path)


# ---------------------------------------------------------------------------
# 5. Main Validation
# ---------------------------------------------------------------------------
def run_phase0_validation() -> dict:
    """Execute Phase 0 gate validation.

    Returns
    -------
    report : dict
        Validation results including error, pass/fail, timing.
    """
    k = WAVENUMBER_RAD_PER_M
    logger.info("=" * 60)
    logger.info("Phase 0: Wedge BEM vs Macdonald Analytical Validation")
    logger.info("=" * 60)
    logger.info("f = %.0f Hz, k = %.2f rad/m, lambda = %.4f m", FREQ_HZ, k, WAVELENGTH_M)
    logger.info("Wedge: interior = 90 deg, exterior = 270 deg, nu = %.4f", NU)
    logger.info(
        "Source: r0 = %.2f m, theta0 = %.1f deg",
        SOURCE_R_M, np.degrees(SOURCE_THETA_RAD),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- BEM Solve ---
    t_total_start = time.time()

    midpoints_m, normals, lengths_m, tangents, surface_pressure = solve_wedge_bem_2d(
        k_rad_per_m=k,
        source_r_m=SOURCE_R_M,
        source_theta_rad=SOURCE_THETA_RAD,
        exterior_angle_rad=EXTERIOR_ANGLE_RAD,
        face_length_m=FACE_LENGTH_M,
        element_size_flat_m=ELEMENT_SIZE_FLAT_M,
        element_size_edge_m=ELEMENT_SIZE_EDGE_M,
        grading_transition_m=GRADING_TRANSITION_M,
    )
    n_elements = len(lengths_m)

    # --- Evaluation Grid ---
    logger.info("Creating evaluation grid...")
    r_grid_m, theta_grid_rad, x_grid_m, y_grid_m, mask_valid = create_polar_eval_grid(
        EVAL_R_MIN_M, EVAL_R_MAX_M, EVAL_N_R, EVAL_N_THETA,
        EXTERIOR_ANGLE_RAD, SOURCE_R_M, SOURCE_THETA_RAD,
    )
    n_eval = r_grid_m.size
    logger.info("Evaluation grid: %d x %d = %d points", EVAL_N_R, EVAL_N_THETA, n_eval)

    # --- Analytical Solution ---
    logger.info("Computing Macdonald analytical solution (n_terms=%d)...", N_SERIES_TERMS)
    t0 = time.time()
    p_analytical = macdonald_wedge_green_neumann(
        r_grid_m.ravel(), theta_grid_rad.ravel(),
        SOURCE_R_M, SOURCE_THETA_RAD,
        k, EXTERIOR_ANGLE_RAD,
        n_terms=N_SERIES_TERMS,
    ).reshape(r_grid_m.shape)  # (n_r, n_theta), complex128
    t_analytical_s = time.time() - t0
    logger.info("Analytical computation: %.1f s", t_analytical_s)

    # --- BEM Field Evaluation ---
    logger.info("Evaluating BEM field at %d exterior points...", n_eval)
    source_x_m = SOURCE_R_M * np.cos(SOURCE_THETA_RAD)
    source_y_m = SOURCE_R_M * np.sin(SOURCE_THETA_RAD)

    t0 = time.time()
    eval_pts_flat = np.column_stack([x_grid_m.ravel(), y_grid_m.ravel()])  # (M, 2)
    p_bem_flat = evaluate_bem_field_2d(
        eval_pts_flat, midpoints_m, normals, lengths_m, tangents,
        surface_pressure, source_x_m, source_y_m, k,
    )  # (M,), complex128
    p_bem = p_bem_flat.reshape(r_grid_m.shape)  # (n_r, n_theta)
    t_eval_s = time.time() - t0
    logger.info("BEM evaluation: %.1f s", t_eval_s)

    t_total_s = time.time() - t_total_start

    # --- NaN Check ---
    n_nan_analytical = np.sum(~np.isfinite(p_analytical))
    n_nan_bem = np.sum(~np.isfinite(p_bem))
    logger.info("NaN/Inf check: analytical=%d, BEM=%d", n_nan_analytical, n_nan_bem)

    # --- Error Computation ---
    error_l2 = compute_relative_l2_error(p_bem, p_analytical, mask_valid)
    error_pct = error_l2 * 100.0
    gate_pass = error_pct < 3.0

    logger.info("-" * 60)
    logger.info("RELATIVE L2 ERROR: %.4f%%", error_pct)
    logger.info("GATE CRITERION: < 3%%")
    logger.info("RESULT: %s", "PASS" if gate_pass else "FAIL")
    logger.info("-" * 60)

    # --- Visualization ---
    plot_comparison(
        r_grid_m, theta_grid_rad, p_analytical, p_bem, mask_valid,
        error_pct, OUTPUT_DIR / "wedge_bem_vs_analytical.png",
    )
    plot_radial_comparison(
        r_grid_m, theta_grid_rad, p_analytical, p_bem,
        OUTPUT_DIR / "wedge_radial_comparison.png",
    )

    # --- Report ---
    report = {
        "frequency_hz": FREQ_HZ,
        "wavenumber_rad_per_m": k,
        "wavelength_m": WAVELENGTH_M,
        "wedge_interior_angle_deg": 90.0,
        "wedge_exterior_angle_deg": 270.0,
        "wedge_nu": NU,
        "source_r_m": SOURCE_R_M,
        "source_theta_deg": np.degrees(SOURCE_THETA_RAD),
        "n_elements": n_elements,
        "min_element_size_m": float(np.min(lengths_m)),
        "max_element_size_m": float(np.max(lengths_m)),
        "n_eval_points": n_eval,
        "n_valid_eval_points": int(np.sum(mask_valid)),
        "n_series_terms": N_SERIES_TERMS,
        "relative_l2_error_pct": error_pct,
        "gate_threshold_pct": 3.0,
        "gate_pass": gate_pass,
        "n_nan_analytical": int(n_nan_analytical),
        "n_nan_bem": int(n_nan_bem),
        "total_time_s": t_total_s,
    }

    # Print summary table
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 0 VALIDATION REPORT")
    logger.info("=" * 60)
    logger.info("%-30s %s", "Parameter", "Value")
    logger.info("-" * 60)
    for key, val in report.items():
        if isinstance(val, float):
            logger.info("%-30s %.4f", key, val)
        else:
            logger.info("%-30s %s", key, val)
    logger.info("=" * 60)

    # Save report as text
    report_path = OUTPUT_DIR / "phase0_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Phase 0 Gate Validation Report\n")
        f.write("=" * 50 + "\n\n")
        for key, val in report.items():
            f.write(f"{key}: {val}\n")
        f.write(f"\nGATE: {'PASS' if gate_pass else 'FAIL'}\n")
    logger.info("Report saved: %s", report_path)

    return report


if __name__ == "__main__":
    report = run_phase0_validation()

    if report["gate_pass"]:
        logger.info("")
        logger.info("Phase 0 PASSED. Phase 1 unlocked.")
    else:
        logger.info("")
        logger.info(
            "Phase 0 FAILED (error=%.2f%%, threshold=3.0%%). "
            "Investigate mesh quality, truncation length, or series convergence.",
            report["relative_l2_error_pct"],
        )
