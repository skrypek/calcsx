# physics/geometry.py
"""
Differential geometry for 3-D coil paths and parametric coil generators.

Provides:
  - Frenet-Serret frame (T, N, B) with parallel-transport fallback for
    straight segments — gives the true tape-face normal for any 3-D coil
  - Parametric generators: solenoid, racetrack, D-shape, saddle, CCT
  - STEP/IGES import via cadquery (optional dependency)
"""

from __future__ import annotations
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Frenet-Serret frame with parallel-transport fallback
# ─────────────────────────────────────────────────────────────────────────────

def compute_frenet_frame(coords: np.ndarray) -> dict:
    """
    Compute T, N, B (tangent, normal, binormal) at each segment midpoint.

    Uses centered finite differences on the arc-length parametrised path.
    Where curvature drops below a threshold (straight or near-straight
    segments), falls back to a **parallel-transported** frame that smoothly
    propagates the normal from the last curved region.

    Parameters
    ----------
    coords : (n+1, 3) ordered vertices of the coil path

    Returns
    -------
    dict with:
        'tangent'  : (n, 3) unit tangent at each segment midpoint
        'normal'   : (n, 3) unit principal normal (tape-face normal)
        'binormal' : (n, 3) unit binormal
        'kappa'    : (n,) curvature magnitude (1/m)
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = len(coords) - 1
    if n < 2:
        # Degenerate: single segment
        T = coords[1:] - coords[:-1]
        T /= np.linalg.norm(T, axis=1, keepdims=True).clip(1e-12)
        N = _arbitrary_perp(T)
        B = np.cross(T, N)
        return {'tangent': T, 'normal': N, 'binormal': B, 'kappa': np.zeros(n)}

    # Segment tangents and arc lengths
    dl = coords[1:] - coords[:-1]                        # (n, 3)
    seg_len = np.linalg.norm(dl, axis=1).clip(1e-12)     # (n,)
    T_raw = dl / seg_len[:, None]                         # (n, 3) unit tangents

    # Curvature via central differences of the tangent vector
    # dT/ds ≈ (T[i+1] - T[i-1]) / (ds[i-1] + ds[i])
    kappa = np.zeros(n, dtype=np.float64)
    dT = np.zeros((n, 3), dtype=np.float64)

    for i in range(n):
        if i == 0:
            dT[i] = (T_raw[1] - T_raw[0]) / seg_len[0]
        elif i == n - 1:
            dT[i] = (T_raw[-1] - T_raw[-2]) / seg_len[-1]
        else:
            ds = 0.5 * (seg_len[i - 1] + seg_len[i])
            dT[i] = (T_raw[i] - T_raw[i - 1]) / max(ds, 1e-12)

    kappa = np.linalg.norm(dT, axis=1)

    # Principal normal from dT/ds (where curvature is nonzero)
    N_frenet = np.zeros((n, 3), dtype=np.float64)
    kappa_thresh = kappa.max() * 1e-4 if kappa.max() > 0 else 1e-10

    for i in range(n):
        if kappa[i] > kappa_thresh:
            N_frenet[i] = dT[i] / kappa[i]
        # else: leave as zero — will be filled by parallel transport

    # Parallel transport for straight/low-curvature segments
    N_out = _parallel_transport(T_raw, N_frenet, kappa, kappa_thresh)

    B_out = np.cross(T_raw, N_out)
    # Re-normalise for safety
    B_norms = np.linalg.norm(B_out, axis=1, keepdims=True).clip(1e-12)
    B_out /= B_norms

    return {
        'tangent': T_raw,
        'normal': N_out,
        'binormal': B_out,
        'kappa': kappa,
    }


def _parallel_transport(
    T: np.ndarray,
    N_frenet: np.ndarray,
    kappa: np.ndarray,
    kappa_thresh: float,
) -> np.ndarray:
    """
    Fill in normals for straight segments by parallel-transporting the
    last known Frenet normal along the curve.

    At each step, the transported normal is rotated minimally to stay
    perpendicular to the new tangent (Rodrigues rotation).
    """
    n = len(T)
    N = N_frenet.copy()

    # Find the first segment with nonzero curvature as seed
    seed = -1
    for i in range(n):
        if kappa[i] > kappa_thresh:
            seed = i
            break
    if seed < 0:
        # Entirely straight coil — use an arbitrary perpendicular
        return _arbitrary_perp(T)

    # Forward pass from seed
    for i in range(seed + 1, n):
        if kappa[i] > kappa_thresh:
            continue  # Frenet normal is valid here
        # Transport N[i-1] to be perpendicular to T[i]
        N[i] = _rotate_normal_to_tangent(N[i - 1], T[i - 1], T[i])

    # Backward pass from seed
    for i in range(seed - 1, -1, -1):
        if kappa[i] > kappa_thresh:
            continue
        N[i] = _rotate_normal_to_tangent(N[i + 1], T[i + 1], T[i])

    return N


def _rotate_normal_to_tangent(
    N_prev: np.ndarray,
    T_prev: np.ndarray,
    T_curr: np.ndarray,
) -> np.ndarray:
    """
    Minimally rotate N_prev so it's perpendicular to T_curr.

    Uses Rodrigues' rotation formula with the rotation axis = T_prev × T_curr.
    """
    cross = np.cross(T_prev, T_curr)
    sin_a = np.linalg.norm(cross)
    cos_a = np.dot(T_prev, T_curr)

    if sin_a < 1e-12:
        # Tangents are parallel — just project out T_curr component
        N_new = N_prev - np.dot(N_prev, T_curr) * T_curr
        norm = np.linalg.norm(N_new)
        return N_new / max(norm, 1e-12) if norm > 1e-12 else N_prev
    else:
        # Rodrigues rotation of N_prev around axis (cross/sin_a)
        axis = cross / sin_a
        N_new = (N_prev * cos_a
                 + np.cross(axis, N_prev) * sin_a
                 + axis * np.dot(axis, N_prev) * (1.0 - cos_a))
        # Project out any T_curr component for numerical safety
        N_new -= np.dot(N_new, T_curr) * T_curr
        norm = np.linalg.norm(N_new)
        return N_new / max(norm, 1e-12) if norm > 1e-12 else N_prev


def _arbitrary_perp(T: np.ndarray) -> np.ndarray:
    """Generate an arbitrary unit vector perpendicular to each row of T."""
    n = len(T)
    N = np.zeros_like(T)
    for i in range(n):
        t = T[i]
        if abs(t[0]) < 0.9:
            helper = np.array([1., 0., 0.])
        else:
            helper = np.array([0., 1., 0.])
        N[i] = np.cross(t, helper)
        N[i] /= max(np.linalg.norm(N[i]), 1e-12)
    return N


# ─────────────────────────────────────────────────────────────────────────────
# Parametric coil generators
# ─────────────────────────────────────────────────────────────────────────────

def generate_solenoid(
    radius: float = 0.5,
    pitch: float = 0.01,
    n_turns: int = 10,
    n_pts_per_turn: int = 100,
    center: np.ndarray = None,
) -> np.ndarray:
    """
    Helical solenoid.

    Parameters
    ----------
    radius         : coil radius (m)
    pitch          : axial advance per turn (m)
    n_turns        : number of complete turns
    n_pts_per_turn : discretisation points per turn
    center         : (3,) center of the solenoid; default origin

    Returns (N, 3) coordinate array.
    """
    if center is None:
        center = np.zeros(3)
    center = np.asarray(center, dtype=np.float64)

    n_total = n_turns * n_pts_per_turn + 1
    theta = np.linspace(0, 2 * np.pi * n_turns, n_total)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = pitch * theta / (2 * np.pi)
    # Centre axially
    z -= z.mean()

    coords = np.column_stack([x, y, z]) + center
    return coords


def generate_racetrack(
    R_end: float = 0.2,
    L_straight: float = 0.5,
    n_turns: int = 1,
    pitch: float = 0.0,
    n_pts: int = 400,
    center: np.ndarray = None,
) -> np.ndarray:
    """
    Racetrack coil: two straight sections connected by semicircular ends.

    Parameters
    ----------
    R_end       : radius of the curved ends (m)
    L_straight  : length of each straight section (m)
    n_turns     : number of turns (stacked axially via pitch)
    pitch       : axial advance per turn (m); 0 = flat
    n_pts       : total discretisation points per turn
    center      : (3,) center; default origin

    Returns (N, 3) coordinate array.
    """
    if center is None:
        center = np.zeros(3)
    center = np.asarray(center, dtype=np.float64)

    pts_per_section = n_pts // 4
    all_coords = []

    for turn in range(n_turns):
        z_off = pitch * turn
        # Bottom straight (along +x)
        xs = np.linspace(-L_straight / 2, L_straight / 2, pts_per_section)
        ys = np.full_like(xs, -R_end)
        all_coords.append(np.column_stack([xs, ys, np.full_like(xs, z_off)]))

        # Right semicircle
        angles = np.linspace(-np.pi / 2, np.pi / 2, pts_per_section)
        xc = L_straight / 2 + R_end * np.sin(angles)
        yc = R_end * (np.cos(angles) - 1)  # shifted so it connects
        # Correct: center of right arc at (L/2, 0)
        xc = L_straight / 2 + R_end * np.sin(angles)
        yc = R_end * np.cos(angles)
        all_coords.append(np.column_stack([xc, yc, np.full_like(xc, z_off)]))

        # Top straight (along -x)
        xs2 = np.linspace(L_straight / 2, -L_straight / 2, pts_per_section)
        ys2 = np.full_like(xs2, R_end)
        all_coords.append(np.column_stack([xs2, ys2, np.full_like(xs2, z_off)]))

        # Left semicircle
        angles2 = np.linspace(np.pi / 2, 3 * np.pi / 2, pts_per_section)
        xc2 = -L_straight / 2 + R_end * np.sin(angles2)
        yc2 = R_end * np.cos(angles2)
        all_coords.append(np.column_stack([xc2, yc2, np.full_like(xc2, z_off)]))

    coords = np.vstack(all_coords)
    # Close the loop
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    # Centre
    coords -= coords.mean(axis=0) - center
    return coords


def generate_d_shape(
    R_inner: float = 0.3,
    R_outer: float = 0.8,
    height: float = 1.0,
    n_pts: int = 400,
    center: np.ndarray = None,
) -> np.ndarray:
    """
    D-shaped coil (tokamak TF coil profile).

    The D-shape is a Princeton-D: inner leg is straight (at R_inner),
    outer contour is an elliptical arc.

    Returns (N, 3) coordinate array in the X-Z plane.
    """
    if center is None:
        center = np.zeros(3)
    center = np.asarray(center, dtype=np.float64)

    half_h = height / 2.0
    n_half = n_pts // 2

    # Inner leg: straight line at x = R_inner, from -half_h to +half_h
    z_inner = np.linspace(-half_h, half_h, n_half)
    x_inner = np.full_like(z_inner, R_inner)

    # Outer arc: ellipse centered at (R_inner, 0) with semi-axes (R_outer - R_inner, half_h)
    a = R_outer - R_inner
    b = half_h
    theta = np.linspace(np.pi / 2, -np.pi / 2, n_half)
    x_outer = R_inner + a * np.cos(theta)
    z_outer = b * np.sin(theta)

    x = np.concatenate([x_inner, x_outer])
    z = np.concatenate([z_inner, z_outer])
    y = np.zeros_like(x)

    coords = np.column_stack([x, y, z])
    # Close
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    coords -= coords.mean(axis=0) - center
    return coords


def generate_saddle_coil(
    radius: float = 0.5,
    length: float = 1.0,
    angle_span: float = 120.0,
    n_pts: int = 400,
    center: np.ndarray = None,
) -> np.ndarray:
    """
    Saddle coil on a cylindrical surface.

    Two azimuthal arcs at each end connected by axial straight runs.
    Common in MRI gradient coils and dipole correctors.

    Parameters
    ----------
    radius     : cylinder radius (m)
    length     : axial extent (m)
    angle_span : azimuthal coverage of each arc (degrees)
    n_pts      : total discretisation points
    center     : (3,) center; default origin

    Returns (N, 3) coordinate array.
    """
    if center is None:
        center = np.zeros(3)
    center = np.asarray(center, dtype=np.float64)

    span_rad = np.radians(angle_span)
    n_sec = n_pts // 4
    half_L = length / 2

    # Bottom arc (z = -half_L)
    phi1 = np.linspace(-span_rad / 2, span_rad / 2, n_sec)
    arc_bottom = np.column_stack([
        radius * np.cos(phi1),
        radius * np.sin(phi1),
        np.full(n_sec, -half_L),
    ])

    # Right axial run (phi = span_rad/2)
    z_up = np.linspace(-half_L, half_L, n_sec)
    run_right = np.column_stack([
        np.full(n_sec, radius * np.cos(span_rad / 2)),
        np.full(n_sec, radius * np.sin(span_rad / 2)),
        z_up,
    ])

    # Top arc (z = +half_L, reversed)
    phi2 = np.linspace(span_rad / 2, -span_rad / 2, n_sec)
    arc_top = np.column_stack([
        radius * np.cos(phi2),
        radius * np.sin(phi2),
        np.full(n_sec, half_L),
    ])

    # Left axial run (phi = -span_rad/2)
    z_down = np.linspace(half_L, -half_L, n_sec)
    run_left = np.column_stack([
        np.full(n_sec, radius * np.cos(-span_rad / 2)),
        np.full(n_sec, radius * np.sin(-span_rad / 2)),
        z_down,
    ])

    coords = np.vstack([arc_bottom, run_right, arc_top, run_left])
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    coords -= coords.mean(axis=0) - center
    return coords


def generate_cct(
    radius: float = 0.3,
    pitch: float = 0.02,
    n_turns: int = 20,
    tilt_angle: float = 30.0,
    n_pts_per_turn: int = 100,
    center: np.ndarray = None,
) -> np.ndarray:
    """
    Canted Cosine Theta (CCT) helical coil.

    A helix on a cylinder with the winding tilted so that the conductor
    path has both helical and sinusoidal components, producing a cos-theta
    field distribution (dipole).

    Parameters
    ----------
    radius         : mandrel radius (m)
    pitch          : axial advance per turn (m)
    n_turns        : number of complete turns
    tilt_angle     : cant angle from the axial direction (degrees)
    n_pts_per_turn : points per turn
    center         : (3,) center; default origin

    Returns (N, 3) coordinate array.
    """
    if center is None:
        center = np.zeros(3)
    center = np.asarray(center, dtype=np.float64)

    tilt = np.radians(tilt_angle)
    n_total = n_turns * n_pts_per_turn + 1
    theta = np.linspace(0, 2 * np.pi * n_turns, n_total)

    # Standard helix with canted modulation
    x = radius * np.cos(theta + np.sin(theta) * np.tan(tilt))
    y = radius * np.sin(theta + np.sin(theta) * np.tan(tilt))
    z = pitch * theta / (2 * np.pi)
    z -= z.mean()

    coords = np.column_stack([x, y, z]) + center
    return coords


# ─────────────────────────────────────────────────────────────────────────────
# STEP/IGES import (optional — requires cadquery)
# ─────────────────────────────────────────────────────────────────────────────

def import_step_centerline(filepath: str, n_discretize: int = 500) -> np.ndarray:
    """
    Extract the longest wire/edge from a STEP or IGES file and discretise
    it into an (N, 3) point array.

    Parameters
    ----------
    filepath      : path to .step / .stp / .iges / .igs file
    n_discretize  : number of points to sample along the wire

    Returns (N, 3) coordinate array.

    Raises ImportError if cadquery is not installed.
    """
    try:
        import cadquery as cq
    except ImportError:
        raise ImportError(
            "cadquery is required for STEP/IGES import.\n"
            "Install with:  pip install cadquery"
        )

    result = cq.importers.importStep(filepath)

    # Extract all edges, pick the longest wire or concatenated edges
    edges = result.edges().vals()
    if not edges:
        raise ValueError(f"No edges found in {filepath}")

    # Try to build a wire from all edges
    try:
        wire = result.wires().val()
        # Discretise the wire
        pts = []
        for i in range(n_discretize + 1):
            t = i / n_discretize
            pt = wire.positionAt(t)
            pts.append([pt.x, pt.y, pt.z])
        return np.array(pts, dtype=np.float64)
    except Exception:
        pass

    # Fallback: find the longest single edge
    best_edge = max(edges, key=lambda e: e.Length())
    pts = []
    for i in range(n_discretize + 1):
        t = i / n_discretize
        pt = best_edge.positionAt(t)
        pts.append([pt.x, pt.y, pt.z])
    return np.array(pts, dtype=np.float64)
