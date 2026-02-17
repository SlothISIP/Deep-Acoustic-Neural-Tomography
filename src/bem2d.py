"""Vectorized 2D Boundary Element Method (BEM) for Helmholtz equation.

Solves exterior Neumann (rigid body) scattering problems in 2D using
constant boundary elements with Gauss-Legendre quadrature.

Mathematical formulation
------------------------
    BIE:  (1/2 I + D) p = p_inc   (body-outward normal convention)
    Repr: p(x) = p_inc(x) - D[p](x)   for x in exterior domain

where
    D[p](x) = integral_Gamma  dG/dn_y(x, y) p(y) dGamma(y)
    G(x,y)  = -(i/4) H_0^(1)(k|x-y|)          (2D free-space Green)
    dG/dn_y = -(ik/4) H_1^(1)(k|x-y|) (x-y).n_y / |x-y|

Performance
-----------
    Assembly  : O(N^2 * Q) fully vectorized (NumPy broadcasting)
    Solve     : O(N^3) direct (numpy.linalg.solve, reuses LU for multi-RHS)
    Evaluation: O(M * N * Q) vectorized with chunking for large M

Reference
---------
    Kirkup S. (2007) "The Boundary Element Method in Acoustics"
    Colton D., Kress R. (2013) "Inverse Acoustic and EM Scattering Theory"
"""

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.special import hankel1

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants and mesh defaults
# ---------------------------------------------------------------------------
SPEED_OF_SOUND_M_PER_S: float = 343.0
ELEMENTS_PER_WAVELENGTH_FLAT: float = 6.0
ELEMENTS_PER_WAVELENGTH_EDGE: float = 10.0
DEFAULT_GRADING_TRANSITION_M: float = 0.3
DEFAULT_N_GAUSS: int = 4
EVAL_CHUNK_SIZE: int = 1000


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------
@dataclass
class Mesh2D:
    """2D boundary element mesh with constant elements.

    Attributes
    ----------
    midpoints_m : np.ndarray, shape (N, 2)
        Collocation points (element midpoints) in Cartesian [m].
    normals : np.ndarray, shape (N, 2)
        Unit outward normals pointing from body into exterior domain.
    lengths_m : np.ndarray, shape (N,)
        Element lengths [m].
    tangents : np.ndarray, shape (N, 2)
        Tangent vectors along each element.
    """

    midpoints_m: np.ndarray
    normals: np.ndarray
    lengths_m: np.ndarray
    tangents: np.ndarray

    @property
    def n_elements(self) -> int:
        """Total number of boundary elements."""
        return len(self.lengths_m)

    def validate(self) -> None:
        """Check mesh integrity. Raises ValueError on failure."""
        N = self.n_elements
        if self.midpoints_m.shape != (N, 2):
            raise ValueError(f"midpoints shape {self.midpoints_m.shape} != ({N}, 2)")
        if self.normals.shape != (N, 2):
            raise ValueError(f"normals shape {self.normals.shape} != ({N}, 2)")
        if self.lengths_m.shape != (N,):
            raise ValueError(f"lengths shape {self.lengths_m.shape} != ({N},)")
        if self.tangents.shape != (N, 2):
            raise ValueError(f"tangents shape {self.tangents.shape} != ({N}, 2)")

        norms = np.linalg.norm(self.normals, axis=1)  # (N,)
        if not np.allclose(norms, 1.0, atol=1e-10):
            raise ValueError(f"Normal vectors not unit: max deviation {np.max(np.abs(norms - 1.0)):.2e}")
        if np.any(self.lengths_m <= 0):
            raise ValueError("Non-positive element lengths detected")
        if not np.all(np.isfinite(self.midpoints_m)):
            raise ValueError("Non-finite midpoint coordinates")
        if not np.all(np.isfinite(self.normals)):
            raise ValueError("Non-finite normal components")


# ---------------------------------------------------------------------------
# Mesh generation helpers (private)
# ---------------------------------------------------------------------------
def _compute_element_sizes(
    freq_max_hz: float,
    speed_of_sound_m_per_s: float = SPEED_OF_SOUND_M_PER_S,
) -> Tuple[float, float, float]:
    """Compute element sizes from max frequency.

    Returns (h_flat_m, h_edge_m, lambda_min_m).
    """
    lambda_min_m = speed_of_sound_m_per_s / freq_max_hz
    h_flat_m = lambda_min_m / ELEMENTS_PER_WAVELENGTH_FLAT
    h_edge_m = lambda_min_m / ELEMENTS_PER_WAVELENGTH_EDGE
    return h_flat_m, h_edge_m, lambda_min_m


def _graded_nodes(
    length_m: float,
    h_min_m: float,
    h_max_m: float,
    transition_m: float,
) -> np.ndarray:
    """Generate 1D node positions with grading from one end.

    Fine elements (h_min) at position 0, transitioning to coarse (h_max)
    beyond transition_m.

    Returns array of node positions from 0 to length_m.
    """
    nodes = [0.0]
    pos = 0.0
    while pos < length_m:
        t = min(pos / transition_m, 1.0)  # (dimensionless)
        h = h_min_m + t * (h_max_m - h_min_m)  # (m)
        pos += h
        if pos > length_m:
            pos = length_m
        nodes.append(pos)
    return np.array(nodes)


def _graded_nodes_both_ends(
    length_m: float,
    h_min_m: float,
    h_max_m: float,
    transition_m: float,
) -> np.ndarray:
    """Generate 1D node positions with grading at both ends.

    Fine elements (h_min) near positions 0 and length_m (e.g. polygon
    vertices), coarse (h_max) in the middle.

    Returns array of node positions from 0 to length_m.
    """
    nodes = [0.0]
    pos = 0.0
    while pos < length_m:
        d_near_m = min(pos, length_m - pos)  # distance to nearest end
        t = min(d_near_m / transition_m, 1.0)  # (dimensionless)
        h = h_min_m + t * (h_max_m - h_min_m)  # (m)
        pos += h
        if pos > length_m:
            pos = length_m
        nodes.append(pos)
    return np.array(nodes)


def _build_mesh_from_normals(
    midpoints_m: np.ndarray,
    normals: np.ndarray,
    lengths_m: np.ndarray,
) -> Mesh2D:
    """Construct Mesh2D from midpoints, normals, lengths (computes tangents)."""
    tangents = np.column_stack([-normals[:, 1], normals[:, 0]])  # (N, 2)
    return Mesh2D(
        midpoints_m=midpoints_m,
        normals=normals,
        lengths_m=lengths_m,
        tangents=tangents,
    )


# ---------------------------------------------------------------------------
# Mesh generators
# ---------------------------------------------------------------------------
def generate_mesh_wedge(
    interior_angle_rad: float,
    face_length_m: float,
    freq_max_hz: float,
    speed_of_sound_m_per_s: float = SPEED_OF_SOUND_M_PER_S,
    grading_transition_m: float = DEFAULT_GRADING_TRANSITION_M,
) -> Mesh2D:
    """Generate open mesh for a 2D rigid wedge.

    Two semi-infinite rigid faces meeting at the origin. Truncated at
    face_length_m. No hypotenuse (open mesh for infinite wedge approx).

    Face 1: along +x axis, normal = (0, +1)
    Face 2: along angle Phi from +x, normal = (sin Phi, -cos Phi)
    where Phi = 2*pi - interior_angle_rad (exterior angle).

    Parameters
    ----------
    interior_angle_rad : float
        Interior (body) angle at the wedge tip [rad].
    face_length_m : float
        Truncation length of each face [m].
    freq_max_hz : float
        Maximum frequency [Hz] -- determines element sizes.
    speed_of_sound_m_per_s : float
        Speed of sound [m/s].
    grading_transition_m : float
        Distance from tip where element grading transitions [m].

    Returns
    -------
    Mesh2D
    """
    exterior_angle_rad = 2.0 * np.pi - interior_angle_rad
    h_flat, h_edge, lambda_min = _compute_element_sizes(freq_max_hz, speed_of_sound_m_per_s)

    # --- Face 1: origin → (L, 0), normal = (0, +1) ---
    nodes_f1 = _graded_nodes(face_length_m, h_edge, h_flat, grading_transition_m)
    n1 = len(nodes_f1) - 1
    mid_dist_f1 = 0.5 * (nodes_f1[:-1] + nodes_f1[1:])  # (n1,)
    mids_f1 = np.column_stack([mid_dist_f1, np.zeros(n1)])  # (n1, 2)
    norms_f1 = np.tile([0.0, 1.0], (n1, 1))  # (n1, 2)
    lens_f1 = nodes_f1[1:] - nodes_f1[:-1]  # (n1,)

    # --- Face 2: origin → L*(cos Phi, sin Phi), normal = (sin Phi, -cos Phi) ---
    Phi = exterior_angle_rad
    face2_dir = np.array([np.cos(Phi), np.sin(Phi)])  # (2,)
    face2_normal = np.array([np.sin(Phi), -np.cos(Phi)])  # (2,)

    nodes_f2 = _graded_nodes(face_length_m, h_edge, h_flat, grading_transition_m)
    n2 = len(nodes_f2) - 1
    mid_dist_f2 = 0.5 * (nodes_f2[:-1] + nodes_f2[1:])  # (n2,)
    mids_f2 = mid_dist_f2[:, None] * face2_dir[None, :]  # (n2, 2)
    norms_f2 = np.tile(face2_normal, (n2, 1))  # (n2, 2)
    lens_f2 = nodes_f2[1:] - nodes_f2[:-1]  # (n2,)

    # Combine
    midpoints = np.vstack([mids_f1, mids_f2])  # (N, 2)
    normals = np.vstack([norms_f1, norms_f2])  # (N, 2)
    lengths = np.concatenate([lens_f1, lens_f2])  # (N,)

    mesh = _build_mesh_from_normals(midpoints, normals, lengths)
    logger.info(
        "Wedge mesh: interior=%.0f deg, Phi=%.0f deg, N=%d (F1=%d, F2=%d), "
        "h=[%.4f, %.4f] m, lambda_min=%.4f m",
        np.degrees(interior_angle_rad), np.degrees(Phi),
        mesh.n_elements, n1, n2,
        np.min(lengths), np.max(lengths), lambda_min,
    )
    return mesh


def generate_mesh_cylinder(
    radius_m: float,
    center_m: np.ndarray,
    freq_max_hz: float,
    speed_of_sound_m_per_s: float = SPEED_OF_SOUND_M_PER_S,
) -> Mesh2D:
    """Generate mesh for a rigid circular cylinder.

    Uniform angular spacing, outward normals point radially outward.

    Parameters
    ----------
    radius_m : float
        Cylinder radius [m].
    center_m : np.ndarray, shape (2,)
        Cylinder center position [m].
    freq_max_hz : float
        Maximum frequency [Hz].
    speed_of_sound_m_per_s : float
        Speed of sound [m/s].

    Returns
    -------
    Mesh2D
    """
    h_flat, _, lambda_min = _compute_element_sizes(freq_max_hz, speed_of_sound_m_per_s)
    circumference_m = 2.0 * np.pi * radius_m
    n_elem = max(int(np.ceil(circumference_m / h_flat)), 12)

    # Uniform angular spacing
    theta_edges = np.linspace(0.0, 2.0 * np.pi, n_elem + 1)  # (n_elem+1,)
    theta_mid = 0.5 * (theta_edges[:-1] + theta_edges[1:])  # (n_elem,)

    cx, cy = float(center_m[0]), float(center_m[1])
    midpoints = np.column_stack([
        cx + radius_m * np.cos(theta_mid),
        cy + radius_m * np.sin(theta_mid),
    ])  # (n_elem, 2)

    # Outward normals: radially outward
    normals = np.column_stack([
        np.cos(theta_mid), np.sin(theta_mid),
    ])  # (n_elem, 2)

    # Arc length per element
    d_theta = 2.0 * np.pi / n_elem
    lengths = np.full(n_elem, radius_m * d_theta)  # (n_elem,)

    mesh = _build_mesh_from_normals(midpoints, normals, lengths)
    logger.info(
        "Cylinder mesh: R=%.3f m, center=(%.2f, %.2f), N=%d, "
        "h=%.4f m, lambda_min=%.4f m",
        radius_m, cx, cy, n_elem, radius_m * d_theta, lambda_min,
    )
    return mesh


def generate_mesh_polygon(
    vertices_m: np.ndarray,
    freq_max_hz: float,
    speed_of_sound_m_per_s: float = SPEED_OF_SOUND_M_PER_S,
    grading_transition_m: float = DEFAULT_GRADING_TRANSITION_M,
) -> Mesh2D:
    """Generate mesh for a rigid polygon body.

    Vertices must be in CCW order. Outward normals are computed per-edge
    using the formula n = (dy, -dx) / |edge| which gives body-outward
    normals for CCW-ordered vertices.

    Graded meshing: fine elements near vertices (corners), coarse on
    flat regions.

    Parameters
    ----------
    vertices_m : np.ndarray, shape (V, 2)
        Polygon vertices in CCW order [m].
    freq_max_hz : float
        Maximum frequency [Hz].
    speed_of_sound_m_per_s : float
        Speed of sound [m/s].
    grading_transition_m : float
        Grading transition distance near vertices [m].

    Returns
    -------
    Mesh2D
    """
    h_flat, h_edge, _ = _compute_element_sizes(freq_max_hz, speed_of_sound_m_per_s)
    V = len(vertices_m)

    all_mids: List[np.ndarray] = []
    all_norms: List[np.ndarray] = []
    all_lens: List[np.ndarray] = []

    for i in range(V):
        v1 = vertices_m[i]  # (2,)
        v2 = vertices_m[(i + 1) % V]  # (2,)
        edge = v2 - v1  # (2,)
        edge_len_m = float(np.linalg.norm(edge))

        if edge_len_m < 1e-10:
            raise ValueError(
                f"Zero-length edge between vertices {i} and {(i + 1) % V}: "
                f"v1={v1}, v2={v2}"
            )

        tangent = edge / edge_len_m  # (2,)
        # CCW outward normal: n = (tangent_y, -tangent_x)
        normal = np.array([tangent[1], -tangent[0]])  # (2,)

        # Graded nodes along edge (fine near both vertices)
        nodes = _graded_nodes_both_ends(edge_len_m, h_edge, h_flat, grading_transition_m)
        n_elem = len(nodes) - 1

        mid_dists = 0.5 * (nodes[:-1] + nodes[1:])  # (n_elem,)
        elem_mids = v1[None, :] + mid_dists[:, None] * tangent[None, :]  # (n_elem, 2)
        elem_lens = nodes[1:] - nodes[:-1]  # (n_elem,)
        elem_norms = np.tile(normal, (n_elem, 1))  # (n_elem, 2)

        all_mids.append(elem_mids)
        all_norms.append(elem_norms)
        all_lens.append(elem_lens)

    midpoints = np.vstack(all_mids)  # (N, 2)
    normals = np.vstack(all_norms)  # (N, 2)
    lengths = np.concatenate(all_lens)  # (N,)

    mesh = _build_mesh_from_normals(midpoints, normals, lengths)
    logger.info(
        "Polygon mesh: %d vertices, N=%d, h=[%.4f, %.4f] m",
        V, mesh.n_elements, np.min(lengths), np.max(lengths),
    )
    return mesh


def combine_meshes(meshes: List[Mesh2D]) -> Mesh2D:
    """Combine multiple Mesh2D objects into a single mesh.

    Used for multi-body scenes (e.g. wedge + cylinder).
    """
    if not meshes:
        raise ValueError("Empty mesh list")

    return Mesh2D(
        midpoints_m=np.vstack([m.midpoints_m for m in meshes]),
        normals=np.vstack([m.normals for m in meshes]),
        lengths_m=np.concatenate([m.lengths_m for m in meshes]),
        tangents=np.vstack([m.tangents for m in meshes]),
    )


# ---------------------------------------------------------------------------
# BEM core: assembly
# ---------------------------------------------------------------------------
def assemble_bem_matrix(
    mesh: Mesh2D,
    k_rad_per_m: float,
    n_gauss: int = DEFAULT_N_GAUSS,
) -> np.ndarray:
    """Vectorized BEM matrix assembly for 2D exterior Neumann problem.

    Assembles A = (1/2 I + D) where

        D_ij = integral_{elem_j} dG/dn_y(x_i, y) dGamma(y)
        dG/dn_y = -(ik/4) H_1^(1)(k|x-y|) (x-y).n_y / |x-y|

    Diagonal D_ii = 0 (flat constant elements).

    Fully vectorized: O(N^2 * Q) with no Python loops.

    Parameters
    ----------
    mesh : Mesh2D
        Boundary element mesh.
    k_rad_per_m : float
        Wavenumber k = 2*pi*f/c [rad/m].
    n_gauss : int
        Number of Gauss-Legendre quadrature points per element.

    Returns
    -------
    A : np.ndarray, complex128, shape (N, N)
        BEM system matrix.
    """
    N = mesh.n_elements
    k = k_rad_per_m
    Q = n_gauss

    # Gauss-Legendre quadrature on [-1, 1]
    gauss_pts, gauss_wts = np.polynomial.legendre.leggauss(Q)  # (Q,), (Q,)
    half_lens = mesh.lengths_m / 2.0  # (N,)

    # Quadrature points for all elements
    # quad_pts[j, q, :] = midpoints[j] + gauss_pts[q] * half_len[j] * tangent[j]
    quad_pts = (
        mesh.midpoints_m[:, None, :]           # (N, 1, 2)
        + gauss_pts[None, :, None]             # (1, Q, 1)
        * half_lens[:, None, None]             # (N, 1, 1)
        * mesh.tangents[:, None, :]            # (N, 1, 2)
    )  # (N, Q, 2)

    # Pairwise difference: diff[i, j, q, :] = collocation[i] - quad_pt[j, q]
    diff = (
        mesh.midpoints_m[:, None, None, :]     # (N, 1, 1, 2)
        - quad_pts[None, :, :, :]              # (1, N, Q, 2)
    )  # (N, N, Q, 2)

    # Distances
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))  # (N, N, Q)
    dist = np.maximum(dist, 1e-15)

    # Kernel: dG/dn_y = -(ik/4) H_1^(1)(kr) (x-y).n_y / |x-y|
    kr = k * dist  # (N, N, Q)
    H1 = hankel1(1, kr)  # (N, N, Q), complex128

    # (collocation[i] - quad_pt[j,q]) . normal[j]
    dot_dn = np.sum(
        diff * mesh.normals[None, :, None, :],  # (1, N, 1, 2)
        axis=-1,
    )  # (N, N, Q)

    kernel = -0.25j * k * H1 * dot_dn / dist  # (N, N, Q), complex128

    # Integrate: sum over quadrature with weights * Jacobian
    D = (
        np.sum(kernel * gauss_wts[None, None, :], axis=2)  # (N, N)
        * half_lens[None, :]                                # (1, N)
    )  # (N, N)

    # Diagonal: D_ii = 0 for flat constant elements
    np.fill_diagonal(D, 0.0)

    # System matrix: A = (1/2) I + D
    A = 0.5 * np.eye(N, dtype=np.complex128) + D  # (N, N)

    # Condition number check
    cond = np.linalg.cond(A)
    if cond > 1e10:
        logger.warning(
            "High condition number: %.2e (N=%d, k=%.2f rad/m)", cond, N, k,
        )
    else:
        logger.debug("Matrix condition: %.2e (N=%d)", cond, N)

    if not np.all(np.isfinite(A)):
        raise ValueError("BEM matrix contains non-finite values")

    return A


# ---------------------------------------------------------------------------
# BEM core: incident field, solve, evaluate
# ---------------------------------------------------------------------------
def compute_incident_field(
    points_m: np.ndarray,
    source_pos_m: np.ndarray,
    k_rad_per_m: float,
) -> np.ndarray:
    """Incident field from a 2D point source (line source in 3D).

    p_inc(x) = -(i/4) H_0^(1)(k |x - x_s|)

    Parameters
    ----------
    points_m : np.ndarray, shape (M, 2)
        Evaluation points [m].
    source_pos_m : np.ndarray, shape (2,)
        Source position [m].
    k_rad_per_m : float
        Wavenumber [rad/m].

    Returns
    -------
    p_inc : np.ndarray, complex128, shape (M,)
    """
    diff = points_m - source_pos_m[None, :]  # (M, 2)
    dist = np.sqrt(np.sum(diff ** 2, axis=1))  # (M,)
    dist = np.maximum(dist, 1e-15)
    return -0.25j * hankel1(0, k_rad_per_m * dist)  # (M,), complex128


def solve_bem(
    A: np.ndarray,
    p_inc_boundary: np.ndarray,
) -> np.ndarray:
    """Solve BEM linear system: A p = p_inc.

    Supports single RHS (N,) or multiple RHS (N, S) for multi-source.

    Parameters
    ----------
    A : np.ndarray, complex128, shape (N, N)
        BEM system matrix.
    p_inc_boundary : np.ndarray, complex128, shape (N,) or (N, S)
        Right-hand side (incident field at boundary collocation points).

    Returns
    -------
    p_surface : np.ndarray, complex128, shape (N,) or (N, S)
        Solved surface pressure.
    """
    if not np.all(np.isfinite(A)):
        raise ValueError("BEM matrix contains non-finite values")
    if not np.all(np.isfinite(p_inc_boundary)):
        raise ValueError("RHS contains non-finite values")

    p_surface = np.linalg.solve(A, p_inc_boundary)

    if not np.all(np.isfinite(p_surface)):
        n_bad = int(np.sum(~np.isfinite(p_surface)))
        raise ValueError(f"BEM solution contains {n_bad} non-finite values")

    return p_surface


def evaluate_field(
    eval_points_m: np.ndarray,
    mesh: Mesh2D,
    surface_pressure: np.ndarray,
    source_pos_m: np.ndarray,
    k_rad_per_m: float,
    n_gauss: int = DEFAULT_N_GAUSS,
    chunk_size: int = EVAL_CHUNK_SIZE,
) -> np.ndarray:
    """Evaluate total BEM field at exterior points.

    p_total(x) = p_inc(x) - integral_Gamma dG/dn_y(x, y) p(y) dGamma(y)

    Vectorized with chunking for memory efficiency when M is large.

    Parameters
    ----------
    eval_points_m : np.ndarray, shape (M, 2)
        Exterior evaluation points [m].
    mesh : Mesh2D
        Boundary element mesh.
    surface_pressure : np.ndarray, complex128, shape (N,)
        Solved surface pressure from solve_bem().
    source_pos_m : np.ndarray, shape (2,)
        Source position [m].
    k_rad_per_m : float
        Wavenumber [rad/m].
    n_gauss : int
        Number of quadrature points.
    chunk_size : int
        Max evaluation points per chunk (controls memory usage).

    Returns
    -------
    p_total : np.ndarray, complex128, shape (M,)
    """
    M = eval_points_m.shape[0]
    k = k_rad_per_m
    N = mesh.n_elements
    Q = n_gauss

    # Incident field at all eval points
    p_inc = compute_incident_field(eval_points_m, source_pos_m, k)  # (M,)

    # Quadrature setup
    gauss_pts, gauss_wts = np.polynomial.legendre.leggauss(Q)  # (Q,), (Q,)
    half_lens = mesh.lengths_m / 2.0  # (N,)

    # Quadrature points for all boundary elements
    quad_pts = (
        mesh.midpoints_m[:, None, :]
        + gauss_pts[None, :, None] * half_lens[:, None, None] * mesh.tangents[:, None, :]
    )  # (N, Q, 2)

    p_scat = np.zeros(M, dtype=np.complex128)  # (M,)

    # Process in chunks for memory efficiency
    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        chunk_pts = eval_points_m[start:end]  # (C, 2)

        # diff[c, j, q, :] = eval_pt[c] - quad_pt[j, q]
        diff = chunk_pts[:, None, None, :] - quad_pts[None, :, :, :]  # (C, N, Q, 2)
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))  # (C, N, Q)
        dist = np.maximum(dist, 1e-15)

        kr = k * dist  # (C, N, Q)
        H1 = hankel1(1, kr)  # (C, N, Q)
        dot_dn = np.sum(
            diff * mesh.normals[None, :, None, :], axis=-1
        )  # (C, N, Q)

        kernel = -0.25j * k * H1 * dot_dn / dist  # (C, N, Q)

        # Integrate per element: sum over quadrature with weights * Jacobian
        integral = (
            np.sum(kernel * gauss_wts[None, None, :], axis=2)  # (C, N)
            * half_lens[None, :]                                # (1, N)
        )  # (C, N)

        # Multiply by surface pressure, sum over elements
        p_scat[start:end] = integral @ surface_pressure  # (C,)

    # Representation formula: p_total = p_inc - D[p]
    p_total = p_inc - p_scat  # (M,)
    return p_total


# ---------------------------------------------------------------------------
# Convenience: full pipeline
# ---------------------------------------------------------------------------
def full_solve(
    mesh: Mesh2D,
    source_pos_m: np.ndarray,
    eval_points_m: np.ndarray,
    k_rad_per_m: float,
    n_gauss: int = DEFAULT_N_GAUSS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Complete BEM pipeline: assemble -> solve -> evaluate.

    Parameters
    ----------
    mesh : Mesh2D
    source_pos_m : np.ndarray, shape (2,)
    eval_points_m : np.ndarray, shape (M, 2)
    k_rad_per_m : float

    Returns
    -------
    surface_pressure : np.ndarray, complex128, shape (N,)
    p_eval : np.ndarray, complex128, shape (M,)
    """
    A = assemble_bem_matrix(mesh, k_rad_per_m, n_gauss)
    p_inc_bdy = compute_incident_field(mesh.midpoints_m, source_pos_m, k_rad_per_m)
    p_surface = solve_bem(A, p_inc_bdy)
    p_eval = evaluate_field(
        eval_points_m, mesh, p_surface, source_pos_m, k_rad_per_m, n_gauss,
    )
    return p_surface, p_eval


def solve_multi_source(
    mesh: Mesh2D,
    sources_m: np.ndarray,
    eval_points_m: np.ndarray,
    k_rad_per_m: float,
    n_gauss: int = DEFAULT_N_GAUSS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve for multiple sources at one frequency (single matrix factorization).

    Parameters
    ----------
    mesh : Mesh2D
    sources_m : np.ndarray, shape (S, 2)
        Source positions.
    eval_points_m : np.ndarray, shape (M, 2)
        Evaluation points.
    k_rad_per_m : float

    Returns
    -------
    surface_pressures : np.ndarray, complex128, shape (N, S)
    field : np.ndarray, complex128, shape (S, M)
    """
    S = sources_m.shape[0]
    M = eval_points_m.shape[0]
    N = mesh.n_elements

    # Assemble once
    A = assemble_bem_matrix(mesh, k_rad_per_m, n_gauss)

    # RHS for all sources: (N, S)
    rhs = np.column_stack([
        compute_incident_field(mesh.midpoints_m, sources_m[s], k_rad_per_m)
        for s in range(S)
    ])  # (N, S)

    # Solve all sources at once
    p_surface_all = solve_bem(A, rhs)  # (N, S)

    # Evaluate field for each source
    field = np.zeros((S, M), dtype=np.complex128)  # (S, M)
    for s in range(S):
        field[s] = evaluate_field(
            eval_points_m, mesh, p_surface_all[:, s],
            sources_m[s], k_rad_per_m, n_gauss,
        )

    return p_surface_all, field


def solve_frequency_sweep(
    mesh: Mesh2D,
    source_pos_m: np.ndarray,
    eval_points_m: np.ndarray,
    freqs_hz: np.ndarray,
    speed_of_sound_m_per_s: float = SPEED_OF_SOUND_M_PER_S,
    n_gauss: int = DEFAULT_N_GAUSS,
    callback: Optional[Callable[[int, int, float], None]] = None,
) -> np.ndarray:
    """BEM solve across multiple frequencies for a single source.

    The mesh is fixed (designed for freq_max). At lower frequencies the
    mesh over-resolves, which is safe (more accuracy, no instability).

    Parameters
    ----------
    mesh : Mesh2D
    source_pos_m : np.ndarray, shape (2,)
    eval_points_m : np.ndarray, shape (M, 2)
    freqs_hz : np.ndarray, shape (F,)
    speed_of_sound_m_per_s : float
    n_gauss : int
    callback : optional
        Called as callback(freq_idx, n_freqs, freq_hz) for progress.

    Returns
    -------
    pressure : np.ndarray, complex128, shape (F, M)
    """
    F = len(freqs_hz)
    M = eval_points_m.shape[0]
    pressure = np.zeros((F, M), dtype=np.complex128)  # (F, M)

    for i, freq_hz in enumerate(freqs_hz):
        k = 2.0 * np.pi * freq_hz / speed_of_sound_m_per_s  # (rad/m)
        _, p_eval = full_solve(mesh, source_pos_m, eval_points_m, k, n_gauss)
        pressure[i, :] = p_eval

        if callback is not None:
            callback(i, F, freq_hz)

    return pressure
