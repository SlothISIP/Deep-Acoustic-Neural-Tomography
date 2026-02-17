"""Scene definitions for Phase 1 BEM Data Factory.

Defines 15 diverse acoustic scenes covering wedges, cylinders, polygons,
and multi-body configurations. Each scene includes geometry, source/receiver
positions, SDF ground truth, and shadow/lit/transition region labeling.

Scene categories
----------------
    1-4:   Wedge variants (60, 90, 120, 150 deg interior)
    5:     Thin barrier (rectangular screen)
    6-7:   Circular cylinders (small, large)
    8-9:   Square and rectangular blocks
    10:    Equilateral triangle
    11:    L-shaped barrier
    12:    Two parallel plates
    13:    Step discontinuity
    14:    Wedge + cylinder (multi-body)
    15:    Three-cylinder cluster
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.bem2d import (
    SPEED_OF_SOUND_M_PER_S,
    Mesh2D,
    combine_meshes,
    generate_mesh_cylinder,
    generate_mesh_polygon,
    generate_mesh_wedge,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Region labels
# ---------------------------------------------------------------------------
REGION_SHADOW: int = 0
REGION_TRANSITION: int = 1
REGION_LIT: int = 2


# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------
@dataclass
class SceneConfig:
    """Configuration for a single acoustic scene.

    Attributes
    ----------
    scene_id : int
        Scene number (1-15).
    name : str
        Human-readable scene name.
    category : str
        Scene category (wedge, cylinder, polygon, multi_body).
    mesh_builder : callable
        Function(freq_max_hz) -> Mesh2D.
    source_positions_m : np.ndarray, shape (S, 2)
        Source positions [m].
    receiver_positions_m : np.ndarray, shape (R, 2)
        Receiver positions [m].
    sdf_func : callable
        Function(x, y) -> signed distance to body surface [m].
        Negative inside body, positive outside.
    freq_min_hz : float
        Minimum BEM frequency [Hz].
    freq_max_hz : float
        Maximum BEM frequency [Hz].
    n_freqs : int
        Number of BEM frequencies.
    face_length_m : float
        Face truncation length for wedge scenes [m].
    """

    scene_id: int
    name: str
    category: str
    mesh_builder: Callable[[float], Mesh2D]
    source_positions_m: np.ndarray
    receiver_positions_m: np.ndarray
    sdf_func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    freq_min_hz: float = 2000.0
    freq_max_hz: float = 8000.0
    n_freqs: int = 200
    face_length_m: float = 3.0


# ---------------------------------------------------------------------------
# Source / receiver layout helpers
# ---------------------------------------------------------------------------
def _polar_receivers(
    n_r: int,
    n_theta: int,
    r_min_m: float,
    r_max_m: float,
    theta_min_rad: float,
    theta_max_rad: float,
    exclusion_zones: Optional[List[Tuple[np.ndarray, float]]] = None,
) -> np.ndarray:
    """Generate receiver positions on a polar grid.

    Parameters
    ----------
    exclusion_zones : list of (center, radius) pairs
        Exclude receivers within radius of each center.

    Returns
    -------
    receivers : np.ndarray, shape (M, 2)
    """
    r_arr = np.linspace(r_min_m, r_max_m, n_r)
    theta_arr = np.linspace(theta_min_rad, theta_max_rad, n_theta)
    r_grid, theta_grid = np.meshgrid(r_arr, theta_arr, indexing="ij")

    x = r_grid.ravel() * np.cos(theta_grid.ravel())
    y = r_grid.ravel() * np.sin(theta_grid.ravel())
    pts = np.column_stack([x, y])  # (M, 2)

    if exclusion_zones:
        mask = np.ones(len(pts), dtype=bool)
        for center, radius in exclusion_zones:
            dist = np.sqrt(np.sum((pts - center[None, :]) ** 2, axis=1))
            mask &= dist > radius
        pts = pts[mask]

    return pts


def _wedge_sources(
    exterior_angle_rad: float,
    r_m: float = 0.5,
) -> np.ndarray:
    """Generate 3 source positions spread across the wedge exterior."""
    angles = np.array([
        exterior_angle_rad * 0.25,
        exterior_angle_rad * 0.50,
        exterior_angle_rad * 0.75,
    ])
    return np.column_stack([r_m * np.cos(angles), r_m * np.sin(angles)])  # (3, 2)


def _body_exterior_sources(
    body_center_m: np.ndarray,
    r_m: float = 0.6,
    n_sources: int = 3,
) -> np.ndarray:
    """Generate sources evenly spaced around a body center."""
    angles = np.linspace(0, 2 * np.pi, n_sources, endpoint=False)
    cx, cy = body_center_m
    return np.column_stack([
        cx + r_m * np.cos(angles),
        cy + r_m * np.sin(angles),
    ])  # (S, 2)


# ---------------------------------------------------------------------------
# SDF functions
# ---------------------------------------------------------------------------
def _sdf_wedge(
    x: np.ndarray,
    y: np.ndarray,
    interior_angle_rad: float,
) -> np.ndarray:
    """SDF for a wedge body occupying the angular sector [Phi, 2*pi].

    Phi = 2*pi - interior_angle. The body is the infinite wedge with
    tip at origin.

    Returns signed distance: negative inside body, positive outside.
    """
    Phi = 2.0 * np.pi - interior_angle_rad
    theta = np.arctan2(y, x)
    theta = np.where(theta < 0, theta + 2.0 * np.pi, theta)

    # Distance to each face
    # Face 1 at theta=0: distance = |y| for x>0
    # Face 2 at theta=Phi: distance = perpendicular distance to the line
    r = np.sqrt(x ** 2 + y ** 2)

    # Distance to Face 1 (y=0, x>0)
    d_face1 = np.abs(y)

    # Distance to Face 2 (direction = (cos(Phi), sin(Phi)))
    # Perpendicular distance = |x*sin(Phi) - y*cos(Phi)|
    d_face2 = np.abs(x * np.sin(Phi) - y * np.cos(Phi))

    # Distance to tip (origin)
    d_tip = r

    # Min distance to body boundary
    d_boundary = np.minimum(d_face1, np.minimum(d_face2, d_tip))

    # Sign: positive in exterior (0 < theta < Phi), negative in body
    in_exterior = (theta > 0) & (theta < Phi)
    sdf = np.where(in_exterior, d_boundary, -d_boundary)

    return sdf


def _sdf_cylinder(
    x: np.ndarray,
    y: np.ndarray,
    radius_m: float,
    center_m: np.ndarray,
) -> np.ndarray:
    """SDF for a circular cylinder. Positive outside, negative inside."""
    dx = x - center_m[0]
    dy = y - center_m[1]
    return np.sqrt(dx ** 2 + dy ** 2) - radius_m


def _sdf_polygon(
    x: np.ndarray,
    y: np.ndarray,
    vertices_m: np.ndarray,
) -> np.ndarray:
    """SDF for a convex/concave polygon. Positive outside, negative inside.

    Uses winding number for inside/outside test and minimum edge distance.
    """
    V = len(vertices_m)
    pts = np.column_stack([x.ravel(), y.ravel()])  # (M, 2)
    M = len(pts)

    # Minimum distance to any edge
    min_dist = np.full(M, np.inf)

    # Winding number (ray casting for inside/outside)
    inside = np.zeros(M, dtype=bool)

    for i in range(V):
        v1 = vertices_m[i]
        v2 = vertices_m[(i + 1) % V]
        edge = v2 - v1
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-10:
            continue

        # Project point onto edge to find closest point
        t = np.sum((pts - v1[None, :]) * edge[None, :], axis=1) / (edge_len ** 2)
        t_clamped = np.clip(t, 0.0, 1.0)
        closest = v1[None, :] + t_clamped[:, None] * edge[None, :]  # (M, 2)
        dist = np.sqrt(np.sum((pts - closest) ** 2, axis=1))  # (M,)
        min_dist = np.minimum(min_dist, dist)

        # Ray casting (horizontal ray to the right)
        yi, yj = v1[1], v2[1]
        xi, xj = v1[0], v2[0]
        cond = ((yi > pts[:, 1]) != (yj > pts[:, 1])) & (
            pts[:, 0] < (xj - xi) * (pts[:, 1] - yi) / (yj - yi + 1e-30) + xi
        )
        inside ^= cond

    sdf = np.where(inside, -min_dist, min_dist)
    return sdf.reshape(x.shape)


def _sdf_multi_body(
    x: np.ndarray,
    y: np.ndarray,
    sdf_funcs: List[Callable],
) -> np.ndarray:
    """SDF for multiple bodies: min of individual SDFs (union)."""
    result = sdf_funcs[0](x, y)
    for sdf_func in sdf_funcs[1:]:
        result = np.minimum(result, sdf_func(x, y))
    return result


# ---------------------------------------------------------------------------
# Region labeling
# ---------------------------------------------------------------------------
def label_regions(
    source_pos_m: np.ndarray,
    receiver_positions_m: np.ndarray,
    mesh: Mesh2D,
    transition_angle_rad: float = 0.15,
) -> np.ndarray:
    """Label receivers as shadow (0), transition (1), or lit (2).

    Uses line-of-sight test against body boundary elements. If the
    line from source to receiver intersects any element, the receiver
    is in shadow. Near-grazing angles are labeled as transition.

    Parameters
    ----------
    source_pos_m : np.ndarray, shape (2,)
    receiver_positions_m : np.ndarray, shape (R, 2)
    mesh : Mesh2D
    transition_angle_rad : float
        Angular half-width of transition zone [rad].

    Returns
    -------
    labels : np.ndarray, shape (R,), int
        0=shadow, 1=transition, 2=lit.
    """
    R = receiver_positions_m.shape[0]
    labels = np.full(R, REGION_LIT, dtype=np.int32)

    src = source_pos_m  # (2,)
    N = mesh.n_elements

    for r_idx in range(R):
        rcv = receiver_positions_m[r_idx]  # (2,)
        direction = rcv - src  # (2,)
        seg_len = np.linalg.norm(direction)
        if seg_len < 1e-10:
            continue
        d_hat = direction / seg_len  # (2,)

        # Check intersection with each boundary element
        # Element j: segment from (midpoint - half_len * tangent) to
        #            (midpoint + half_len * tangent)
        half_lens = mesh.lengths_m / 2.0  # (N,)
        p1 = mesh.midpoints_m - half_lens[:, None] * mesh.tangents  # (N, 2)
        p2 = mesh.midpoints_m + half_lens[:, None] * mesh.tangents  # (N, 2)

        # Segment-segment intersection test (source→receiver vs p1→p2)
        # Using cross-product method
        r_vec = direction  # source → receiver
        for j in range(N):
            s_vec = p2[j] - p1[j]  # element direction
            q_minus_p = p1[j] - src

            denom = r_vec[0] * s_vec[1] - r_vec[1] * s_vec[0]
            if abs(denom) < 1e-15:
                continue  # parallel

            t_val = (q_minus_p[0] * s_vec[1] - q_minus_p[1] * s_vec[0]) / denom
            u_val = (q_minus_p[0] * r_vec[1] - q_minus_p[1] * r_vec[0]) / denom

            if 0.0 < t_val < 1.0 and 0.0 < u_val < 1.0:
                labels[r_idx] = REGION_SHADOW
                break

        # Check for transition zone (near shadow boundary)
        if labels[r_idx] == REGION_LIT:
            # Angle between source-receiver line and nearest face normal
            dot_normals = np.sum(d_hat[None, :] * mesh.normals, axis=1)  # (N,)
            min_angle = np.min(np.abs(np.arcsin(np.clip(dot_normals, -1, 1))))
            if min_angle < transition_angle_rad:
                labels[r_idx] = REGION_TRANSITION

    return labels


# ---------------------------------------------------------------------------
# Scene factory
# ---------------------------------------------------------------------------
def _make_wedge_scene(
    scene_id: int,
    interior_angle_deg: float,
    face_length_m: float = 3.0,
) -> SceneConfig:
    """Create a wedge scene configuration."""
    interior_angle_rad = np.radians(interior_angle_deg)
    exterior_angle_rad = 2.0 * np.pi - interior_angle_rad

    def mesh_builder(freq_max_hz: float) -> Mesh2D:
        return generate_mesh_wedge(
            interior_angle_rad=interior_angle_rad,
            face_length_m=face_length_m,
            freq_max_hz=freq_max_hz,
        )

    sources = _wedge_sources(exterior_angle_rad, r_m=0.5)

    # Receivers on polar grid in exterior domain
    exclusion_zones = [(s, 0.05) for s in sources]
    receivers = _polar_receivers(
        n_r=10, n_theta=20,
        r_min_m=0.15, r_max_m=1.0,
        theta_min_rad=0.05, theta_max_rad=exterior_angle_rad - 0.05,
        exclusion_zones=exclusion_zones,
    )

    def sdf_func(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return _sdf_wedge(x, y, interior_angle_rad)

    return SceneConfig(
        scene_id=scene_id,
        name=f"wedge_{int(interior_angle_deg)}deg",
        category="wedge",
        mesh_builder=mesh_builder,
        source_positions_m=sources,
        receiver_positions_m=receivers,
        sdf_func=sdf_func,
        face_length_m=face_length_m,
    )


def _make_cylinder_scene(
    scene_id: int,
    radius_m: float,
    name_suffix: str,
) -> SceneConfig:
    """Create a cylinder scene configuration."""
    center = np.array([0.0, 0.0])

    def mesh_builder(freq_max_hz: float) -> Mesh2D:
        return generate_mesh_cylinder(
            radius_m=radius_m,
            center_m=center,
            freq_max_hz=freq_max_hz,
        )

    sources = _body_exterior_sources(center, r_m=radius_m + 0.4, n_sources=3)

    exclusion_zones = [(s, 0.05) for s in sources]
    exclusion_zones.append((center, radius_m + 0.02))
    receivers = _polar_receivers(
        n_r=10, n_theta=20,
        r_min_m=radius_m + 0.05, r_max_m=radius_m + 0.8,
        theta_min_rad=0.0, theta_max_rad=2.0 * np.pi - 0.1,
        exclusion_zones=exclusion_zones,
    )

    def sdf_func(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return _sdf_cylinder(x, y, radius_m, center)

    return SceneConfig(
        scene_id=scene_id,
        name=f"cylinder_{name_suffix}",
        category="cylinder",
        mesh_builder=mesh_builder,
        source_positions_m=sources,
        receiver_positions_m=receivers,
        sdf_func=sdf_func,
    )


def _make_polygon_scene(
    scene_id: int,
    name: str,
    vertices_m: np.ndarray,
) -> SceneConfig:
    """Create a polygon scene configuration."""
    centroid = np.mean(vertices_m, axis=0)
    max_extent = np.max(np.linalg.norm(vertices_m - centroid, axis=1))

    def mesh_builder(freq_max_hz: float) -> Mesh2D:
        return generate_mesh_polygon(
            vertices_m=vertices_m,
            freq_max_hz=freq_max_hz,
        )

    sources = _body_exterior_sources(centroid, r_m=max_extent + 0.3, n_sources=3)

    exclusion_zones = [(s, 0.05) for s in sources]
    receivers = _polar_receivers(
        n_r=10, n_theta=20,
        r_min_m=max_extent + 0.05, r_max_m=max_extent + 0.7,
        theta_min_rad=0.0, theta_max_rad=2.0 * np.pi - 0.1,
        exclusion_zones=exclusion_zones,
    )

    def sdf_func(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return _sdf_polygon(x, y, vertices_m)

    return SceneConfig(
        scene_id=scene_id,
        name=name,
        category="polygon",
        mesh_builder=mesh_builder,
        source_positions_m=sources,
        receiver_positions_m=receivers,
        sdf_func=sdf_func,
    )


def _make_multi_body_scene(
    scene_id: int,
    name: str,
    mesh_builders: List[Callable[[float], Mesh2D]],
    sdf_funcs: List[Callable],
    sources_m: np.ndarray,
    receivers_m: np.ndarray,
) -> SceneConfig:
    """Create a multi-body scene configuration."""

    def combined_mesh_builder(freq_max_hz: float) -> Mesh2D:
        meshes = [builder(freq_max_hz) for builder in mesh_builders]
        return combine_meshes(meshes)

    def combined_sdf(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return _sdf_multi_body(x, y, sdf_funcs)

    return SceneConfig(
        scene_id=scene_id,
        name=name,
        category="multi_body",
        mesh_builder=combined_mesh_builder,
        source_positions_m=sources_m,
        receiver_positions_m=receivers_m,
        sdf_func=combined_sdf,
    )


# ---------------------------------------------------------------------------
# The 15 scenes
# ---------------------------------------------------------------------------
def build_all_scenes() -> List[SceneConfig]:
    """Build all 15 scene configurations for Phase 1.

    Returns list of SceneConfig, sorted by scene_id.
    """
    scenes: List[SceneConfig] = []

    # --- Category 1: Wedge variants (scenes 1-4) ---
    for sid, angle_deg in [(1, 60), (2, 90), (3, 120), (4, 150)]:
        scenes.append(_make_wedge_scene(sid, angle_deg))

    # --- Scene 5: Thin barrier (rectangular screen) ---
    barrier_w = 0.5
    barrier_h = 0.02
    barrier_verts = np.array([
        [-barrier_w / 2, -barrier_h / 2],
        [barrier_w / 2, -barrier_h / 2],
        [barrier_w / 2, barrier_h / 2],
        [-barrier_w / 2, barrier_h / 2],
    ])
    scenes.append(_make_polygon_scene(5, "thin_barrier", barrier_verts))

    # --- Scenes 6-7: Circular cylinders ---
    scenes.append(_make_cylinder_scene(6, radius_m=0.15, name_suffix="small"))
    scenes.append(_make_cylinder_scene(7, radius_m=0.40, name_suffix="large"))

    # --- Scene 8: Square block ---
    sq_side = 0.3
    sq_verts = np.array([
        [-sq_side / 2, -sq_side / 2],
        [sq_side / 2, -sq_side / 2],
        [sq_side / 2, sq_side / 2],
        [-sq_side / 2, sq_side / 2],
    ])
    scenes.append(_make_polygon_scene(8, "square_block", sq_verts))

    # --- Scene 9: Rectangle ---
    rect_w, rect_h = 0.5, 0.2
    rect_verts = np.array([
        [-rect_w / 2, -rect_h / 2],
        [rect_w / 2, -rect_h / 2],
        [rect_w / 2, rect_h / 2],
        [-rect_w / 2, rect_h / 2],
    ])
    scenes.append(_make_polygon_scene(9, "rectangle", rect_verts))

    # --- Scene 10: Equilateral triangle ---
    tri_side = 0.3
    tri_h = tri_side * np.sqrt(3) / 2
    tri_verts = np.array([
        [0.0, tri_h * 2 / 3],
        [-tri_side / 2, -tri_h / 3],
        [tri_side / 2, -tri_h / 3],
    ])
    scenes.append(_make_polygon_scene(10, "triangle", tri_verts))

    # --- Scene 11: L-shaped barrier ---
    l_verts = np.array([
        [0.0, 0.0],
        [0.5, 0.0],
        [0.5, 0.25],
        [0.25, 0.25],
        [0.25, 0.5],
        [0.0, 0.5],
    ])
    # Shift to center at centroid
    l_centroid = np.mean(l_verts, axis=0)
    l_verts_centered = l_verts - l_centroid
    scenes.append(_make_polygon_scene(11, "l_shape", l_verts_centered))

    # --- Scene 12: Two parallel plates ---
    plate_len = 0.4
    plate_h = 0.015
    gap = 0.1

    plate1_verts = np.array([
        [-plate_len / 2, gap / 2],
        [plate_len / 2, gap / 2],
        [plate_len / 2, gap / 2 + plate_h],
        [-plate_len / 2, gap / 2 + plate_h],
    ])
    plate2_verts = np.array([
        [-plate_len / 2, -gap / 2 - plate_h],
        [plate_len / 2, -gap / 2 - plate_h],
        [plate_len / 2, -gap / 2],
        [-plate_len / 2, -gap / 2],
    ])

    center_12 = np.array([0.0, 0.0])
    src_12 = _body_exterior_sources(center_12, r_m=0.5, n_sources=3)
    rcv_12 = _polar_receivers(
        n_r=10, n_theta=20,
        r_min_m=0.15, r_max_m=0.8,
        theta_min_rad=0.0, theta_max_rad=2.0 * np.pi - 0.1,
        exclusion_zones=[(s, 0.05) for s in src_12],
    )
    scenes.append(_make_multi_body_scene(
        12, "two_plates",
        mesh_builders=[
            lambda f, v=plate1_verts: generate_mesh_polygon(v, f),
            lambda f, v=plate2_verts: generate_mesh_polygon(v, f),
        ],
        sdf_funcs=[
            lambda x, y, v=plate1_verts: _sdf_polygon(x, y, v),
            lambda x, y, v=plate2_verts: _sdf_polygon(x, y, v),
        ],
        sources_m=src_12,
        receivers_m=rcv_12,
    ))

    # --- Scene 13: Step discontinuity ---
    step_w = 0.4
    step_h1 = 0.15
    step_h2 = 0.30
    step1_verts = np.array([
        [-step_w, -step_h1 / 2],
        [0.0, -step_h1 / 2],
        [0.0, step_h1 / 2],
        [-step_w, step_h1 / 2],
    ])
    step2_verts = np.array([
        [0.0, -step_h2 / 2],
        [step_w, -step_h2 / 2],
        [step_w, step_h2 / 2],
        [0.0, step_h2 / 2],
    ])

    center_13 = np.array([0.0, 0.0])
    src_13 = _body_exterior_sources(center_13, r_m=0.6, n_sources=3)
    rcv_13 = _polar_receivers(
        n_r=10, n_theta=20,
        r_min_m=0.2, r_max_m=0.9,
        theta_min_rad=0.0, theta_max_rad=2.0 * np.pi - 0.1,
        exclusion_zones=[(s, 0.05) for s in src_13],
    )
    scenes.append(_make_multi_body_scene(
        13, "step",
        mesh_builders=[
            lambda f, v=step1_verts: generate_mesh_polygon(v, f),
            lambda f, v=step2_verts: generate_mesh_polygon(v, f),
        ],
        sdf_funcs=[
            lambda x, y, v=step1_verts: _sdf_polygon(x, y, v),
            lambda x, y, v=step2_verts: _sdf_polygon(x, y, v),
        ],
        sources_m=src_13,
        receivers_m=rcv_13,
    ))

    # --- Scene 14: Wedge (90 deg) + cylinder ---
    wedge_14_angle = np.pi / 2.0
    cyl_14_center = np.array([0.4, 0.4])
    cyl_14_radius = 0.10

    src_14 = np.array([
        [0.0, 0.5],
        [-0.4, 0.3],
        [0.3, 0.8],
    ])
    rcv_14 = _polar_receivers(
        n_r=10, n_theta=20,
        r_min_m=0.15, r_max_m=1.0,
        theta_min_rad=0.05, theta_max_rad=3 * np.pi / 2 - 0.05,
        exclusion_zones=[(s, 0.05) for s in src_14] + [(cyl_14_center, cyl_14_radius + 0.02)],
    )
    scenes.append(_make_multi_body_scene(
        14, "wedge_cylinder",
        mesh_builders=[
            lambda f: generate_mesh_wedge(wedge_14_angle, 3.0, f),
            lambda f: generate_mesh_cylinder(cyl_14_radius, cyl_14_center, f),
        ],
        sdf_funcs=[
            lambda x, y: _sdf_wedge(x, y, wedge_14_angle),
            lambda x, y: _sdf_cylinder(x, y, cyl_14_radius, cyl_14_center),
        ],
        sources_m=src_14,
        receivers_m=rcv_14,
    ))

    # --- Scene 15: Three-cylinder cluster ---
    cyl15_r = 0.08
    cyl15_sep = 0.25
    cyl15_centers = np.array([
        [0.0, cyl15_sep / np.sqrt(3)],
        [-cyl15_sep / 2, -cyl15_sep / (2 * np.sqrt(3))],
        [cyl15_sep / 2, -cyl15_sep / (2 * np.sqrt(3))],
    ])

    cluster_center = np.mean(cyl15_centers, axis=0)
    src_15 = _body_exterior_sources(cluster_center, r_m=0.5, n_sources=3)
    excl_15 = [(s, 0.05) for s in src_15]
    for c in cyl15_centers:
        excl_15.append((c, cyl15_r + 0.02))
    rcv_15 = _polar_receivers(
        n_r=10, n_theta=20,
        r_min_m=0.2, r_max_m=0.8,
        theta_min_rad=0.0, theta_max_rad=2.0 * np.pi - 0.1,
        exclusion_zones=excl_15,
    )
    scenes.append(_make_multi_body_scene(
        15, "three_cylinders",
        mesh_builders=[
            lambda f, c=c: generate_mesh_cylinder(cyl15_r, c, f)
            for c in cyl15_centers
        ],
        sdf_funcs=[
            lambda x, y, c=c: _sdf_cylinder(x, y, cyl15_r, c)
            for c in cyl15_centers
        ],
        sources_m=src_15,
        receivers_m=rcv_15,
    ))

    # Sort by scene_id
    scenes.sort(key=lambda s: s.scene_id)

    logger.info("Built %d scene configurations", len(scenes))
    for sc in scenes:
        logger.info(
            "  Scene %02d: %-20s  S=%d sources, R=%d receivers",
            sc.scene_id, sc.name,
            len(sc.source_positions_m), len(sc.receiver_positions_m),
        )

    return scenes
