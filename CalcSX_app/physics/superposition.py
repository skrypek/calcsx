# physics/superposition.py
"""
Multi-coil superposition orchestrator.

Manages lightweight CoilAnalysis engines for each coil in the scene and
provides B_ext callbacks that compute the combined external field from all
OTHER coils at any set of query points.  This lets individual CoilAnalysis
instances run their full analysis pipeline (forces, stress, field lines, etc.)
while accounting for the magnetic field contributions of neighbouring coils.
"""

from __future__ import annotations
import numpy as np
from CalcSX_app.physics.physics_utils import CoilAnalysis


class MultiCoilEnvironment:
    """
    Orchestrator for multi-coil superposition.

    Each registered coil gets a *lightweight* CoilAnalysis engine (PCA +
    arc-length only — enough for ``_bfield_vec`` to work).  The environment
    tracks which coils are stale (their analysis results don't account for
    the current set of neighbours).

    Usage from the GUI
    ------------------
    1. ``register_coil(...)`` when a CSV is loaded.
    2. ``make_external_field_func(cid)`` before running analysis — inject
       the returned callable as ``B_ext`` into ``CoilAnalysis.__init__``.
    3. ``mark_fresh(cid)`` after analysis completes.
    4. ``update_coil_coords(cid, ...)`` when a coil is moved.
    5. ``unregister_coil(cid)`` when a coil is deleted.
    """

    def __init__(self):
        self._engines:     dict[str, CoilAnalysis] = {}
        self._coil_params: dict[str, dict]         = {}
        self._stale_set:   set[str]                = set()

    # ── Registration ──────────────────────────────────────────────────────

    def register_coil(
        self,
        coil_id: str,
        coords: np.ndarray,
        winds: int   = 200,
        current: float = 300.0,
        thickness: float = 80.0,
        width: float = 4.0,
        tape_normals: np.ndarray = None,
    ) -> None:
        """Add or replace a coil.  Marks all OTHER coils stale."""
        self._coil_params[coil_id] = dict(
            coords=np.asarray(coords, dtype=np.float64),
            winds=winds,
            current=current,
            thickness=thickness,
            width=width,
            tape_normals=tape_normals,
        )
        self._rebuild_engine(coil_id)
        # Every existing coil's analysis is now outdated (new neighbour)
        for cid in self._engines:
            if cid != coil_id:
                self._stale_set.add(cid)

    def unregister_coil(self, coil_id: str) -> None:
        """Remove a coil.  Marks all remaining coils stale."""
        self._engines.pop(coil_id, None)
        self._coil_params.pop(coil_id, None)
        self._stale_set.discard(coil_id)
        for cid in self._engines:
            self._stale_set.add(cid)

    def update_coil_coords(self, coil_id: str, coords: np.ndarray) -> None:
        """Update world-space coordinates (after a transform).  Marks ALL stale."""
        if coil_id not in self._coil_params:
            return
        self._coil_params[coil_id]['coords'] = np.asarray(coords, dtype=np.float64)
        self._rebuild_engine(coil_id)
        for cid in self._engines:
            self._stale_set.add(cid)

    def update_coil_params(self, coil_id: str, **kwargs) -> None:
        """Update electrical parameters (winds, current, etc.).  Marks ALL stale."""
        if coil_id not in self._coil_params:
            return
        self._coil_params[coil_id].update(kwargs)
        self._rebuild_engine(coil_id)
        for cid in self._engines:
            self._stale_set.add(cid)

    # ── External field closures ───────────────────────────────────────────

    def make_external_field_func(self, exclude_id: str):
        """
        Return a callable ``B_ext(points) -> (M, 3)`` that sums the B-field
        from every coil EXCEPT *exclude_id*.

        The returned closure captures lightweight engine references that are
        safe to pass to a worker thread (read-only after construction).
        Returns None if there are no other coils (single-coil mode).
        """
        others = [eng for cid, eng in self._engines.items() if cid != exclude_id]
        if not others:
            return None  # single-coil — no external field

        def B_ext(points):
            pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
            B = np.zeros((pts.shape[0], 3), dtype=np.float64)
            for eng in others:
                B += np.atleast_2d(eng._bfield_vec(pts))
            return B if np.ndim(points) != 1 else B[0]

        return B_ext

    def make_total_field_func(self):
        """
        Return a callable ``B_total(points) -> (M, 3)`` that sums the B-field
        from ALL coils.  Used for global field lines / cross sections.

        Returns None if no coils are registered.
        """
        all_engines = list(self._engines.values())
        if not all_engines:
            return None

        def B_total(points):
            pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
            B = np.zeros((pts.shape[0], 3), dtype=np.float64)
            for eng in all_engines:
                B += np.atleast_2d(eng._bfield_vec(pts))
            return B if np.ndim(points) != 1 else B[0]

        return B_total

    # ── Staleness ─────────────────────────────────────────────────────────

    def get_stale_coils(self) -> set[str]:
        return set(self._stale_set)

    def mark_fresh(self, coil_id: str) -> None:
        self._stale_set.discard(coil_id)

    def has_coil(self, coil_id: str) -> bool:
        return coil_id in self._engines

    def coil_ids(self) -> list[str]:
        return list(self._engines.keys())

    # ── Internals ─────────────────────────────────────────────────────────

    def get_coil_infos(self) -> list[dict]:
        """Return centroid/radius info for each coil (for global field line seeding)."""
        infos = []
        for cid, eng in self._engines.items():
            if eng.midpoints is not None and eng.mean_point is not None:
                R = float(np.max(np.linalg.norm(eng.midpoints - eng.mean_point, axis=1)))
                infos.append({'centroid': eng.mean_point.copy(), 'radius': max(R, 0.01)})
        return infos

    def compute_mutual_inductance_matrix(self, progress_callback=None) -> dict:
        """
        Compute the full N×N inductance matrix (self + mutual) via Neumann.

        M(i,j) = (µ₀ Ni Nj / 4π) ΣΣ (dl_a · dl_b) / |r_a - r_b|

        Returns dict with:
            'coil_ids' : list of coil IDs (ordering matches matrix rows/cols)
            'L_matrix' : (N, N) ndarray — inductance matrix in Henries
            'energies' : (N,) ndarray — stored energy per coil ½ L_ii I_i²
            'total_energy' : float — total stored magnetic energy (J)
        """
        mu0_4pi = 1e-7
        ids = list(self._engines.keys())
        N = len(ids)
        L = np.zeros((N, N), dtype=np.float64)

        for i in range(N):
            eng_i = self._engines[ids[i]]
            p_i = self._coil_params[ids[i]]
            if eng_i.midpoints is None or eng_i._dl is None:
                continue
            Ni = float(p_i['winds'])
            mid_i = eng_i.midpoints
            dl_i = eng_i._dl

            for j in range(i, N):
                eng_j = self._engines[ids[j]]
                p_j = self._coil_params[ids[j]]
                if eng_j.midpoints is None or eng_j._dl is None:
                    continue
                Nj = float(p_j['winds'])
                mid_j = eng_j.midpoints
                dl_j = eng_j._dl

                # Distance matrix between segment midpoints
                diff = mid_i[:, None, :] - mid_j[None, :, :]
                dist = np.linalg.norm(diff, axis=2)

                if i == j:
                    # Self: regularise diagonal with GMD of cross-section
                    a = eng_i.total_thickness
                    b = eng_i.tape_width
                    gmd = 0.2235 * (a + b)
                    np.fill_diagonal(dist, max(gmd, 1e-6))

                dist = np.maximum(dist, 1e-10)
                dot_m = np.einsum('ik,jk->ij', dl_i, dl_j)
                Mij = mu0_4pi * Ni * Nj * np.sum(dot_m / dist)
                L[i, j] = Mij
                L[j, i] = Mij  # symmetric

            if progress_callback:
                progress_callback(int(100 * (i + 1) / N))

        # Stored energies: E = ½ Σ_ij L_ij I_i I_j
        currents = np.array([
            float(self._coil_params[cid]['current']) for cid in ids
        ])
        energies = 0.5 * np.diag(L) * currents ** 2
        total_E = 0.5 * currents @ L @ currents

        return {
            'coil_ids': ids,
            'L_matrix': L,
            'energies': energies,
            'total_energy': float(total_E),
        }

    def _rebuild_engine(self, coil_id: str) -> None:
        """Create a lightweight CoilAnalysis (PCA + arc only) for _bfield_vec.
        The B_ext closures call _bfield_vec directly (filamentary), so no
        filament grid is needed here — it's only built inside run_analysis
        for the force integration on that coil."""
        p = self._coil_params[coil_id]
        eng = CoilAnalysis(
            p['coords'], p['winds'], p['current'],
            p['thickness'], p['width'],
        )
        eng._compute_pca()
        eng._compute_arc()
        self._engines[coil_id] = eng


# ─────────────────────────────────────────────────────────────────────────────
# Standalone global field line integrator
# ─────────────────────────────────────────────────────────────────────────────

def compute_global_field_lines(
    B_total,
    coil_infos: list[dict],
    n_seeds: int = 24,
    n_steps: int = 600,
    max_radius_factor: float = 4.0,
    progress_callback=None,
):
    """
    Compute 3D magnetic field lines through the superposed B-field of ALL coils.

    Parameters
    ----------
    B_total       : callable (M,3) → (M,3) — total B-field from all coils
    coil_infos    : list of {'centroid': (3,), 'radius': float}
    n_seeds       : total seed count (distributed across all coils)
    n_steps       : RK4 integration steps per direction
    max_radius_factor : kill radius = max(all radii) × factor
    progress_callback : optional int → None

    Returns
    -------
    lines  : list of (N, 3) float32 arrays
    B_mags : list of (N,) float32 arrays
    """
    if not coil_infos or B_total is None:
        return [], []

    golden = (1.0 + 5.0 ** 0.5) / 2.0

    # Distribute seeds across coils proportionally to their radii
    total_R = sum(info['radius'] for info in coil_infos)
    seeds_list = []
    for info in coil_infos:
        n_coil = max(4, int(n_seeds * info['radius'] / max(total_R, 1e-10)))
        centroid = np.asarray(info['centroid'], dtype=np.float64)
        R = info['radius']
        for k in range(n_coil):
            theta = np.arccos(max(-1.0, min(1.0, 1.0 - 2.0 * (k + 0.5) / n_coil)))
            phi = 2.0 * np.pi * k / golden
            seeds_list.append(centroid + 0.6 * R * np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ]))

    seeds = np.array(seeds_list, dtype=np.float64)
    n_total = len(seeds)

    # Global kill radius
    all_centroids = np.array([info['centroid'] for info in coil_infos])
    global_center = all_centroids.mean(axis=0)
    max_dist = max(np.linalg.norm(all_centroids - global_center, axis=1).max(),
                   max(info['radius'] for info in coil_infos))
    max_r = max_dist * max_radius_factor

    step = min(info['radius'] for info in coil_infos) * 0.07

    all_lines = []
    all_B = []

    for d_idx, sign in enumerate((1.0, -1.0)):
        pts = seeds.copy()
        active = np.ones(n_total, dtype=bool)
        trajs = [[seeds[i].copy()] for i in range(n_total)]

        for s in range(n_steps):
            if not active.any():
                break
            ai = np.where(active)[0]
            ap = pts[ai]

            def unit_B(p):
                B = np.atleast_2d(B_total(p))
                mag = np.linalg.norm(B, axis=-1, keepdims=True).clip(1e-30)
                return B / mag

            u1 = unit_B(ap)
            u2 = unit_B(ap + 0.5 * step * u1 * sign)
            u3 = unit_B(ap + 0.5 * step * u2 * sign)
            u4 = unit_B(ap + step * u3 * sign)
            new_ap = ap + sign * (step / 6.0) * (u1 + 2*u2 + 2*u3 + u4)
            pts[ai] = new_ap

            dist = np.linalg.norm(new_ap - global_center, axis=1)
            active[ai[dist >= max_r]] = False
            for j, i in enumerate(ai):
                if active[i]:
                    trajs[i].append(new_ap[j].copy())

            if progress_callback and s % 30 == 0:
                pct = int(100 * (d_idx + s / n_steps) / 2)
                progress_callback(min(pct, 99))

        for i in range(n_total):
            if len(trajs[i]) < 3:
                continue
            pts_arr = np.array(trajs[i], dtype=np.float32)
            B_chunks = []
            chunk = 300
            for ci in range(0, len(pts_arr), chunk):
                Bc = np.atleast_2d(B_total(pts_arr[ci:ci+chunk].astype(np.float64)))
                B_chunks.append(np.linalg.norm(Bc, axis=1).astype(np.float32))
            all_lines.append(pts_arr)
            all_B.append(np.concatenate(B_chunks))

    if progress_callback:
        progress_callback(100)
    return all_lines, all_B
