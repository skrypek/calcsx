# physics_utils.py
import os
import numpy as np
from sklearn.decomposition import PCA

class CoilAnalysis:
    def __init__(self, coords, winds, current, thickness_microns, tape_width_mm,
                 B_ext=None, tape_normals=None):
        # raw inputs
        self.compute_bfield_enabled = False
        self.coords = coords
        self.winds = winds
        self.current = current
        # External B-field callback for multi-coil superposition.
        # Callable[[ndarray], ndarray]: points (M,3) → B (M,3) from other coils.
        # None means single-coil mode (no external contributions).
        self._B_ext = B_ext
        # Supplied tape-face normals from bobbin geometry.
        # (n, 3) array — if provided, used instead of Frenet-Serret inference.
        self._supplied_tape_normals = tape_normals
        # convert units (meters and total thickness)
        self.thickness = thickness_microns * 1e-6 # m per turn
        self.total_thickness = self.thickness * winds # total coil thickness
        self.tape_width = tape_width_mm * 1e-3 # m

        # Gauss-Legendre Quadrature preparation
        nodes16, weights16 = np.polynomial.legendre.leggauss(16)
        self._gauss_ts16 = 0.5 * (nodes16 + 1) # map [-1,1]→[0,1]
        self._gauss_ws16 = weights16 * 0.5

        # placeholders (initializes necessary variables)
        self._dl = None
        self._p0 = None
        self.axis = None
        self.mean_point = None
        self.seg_len = None
        self.arc_mid = None
        self.midpoints = None
        self.total_length = None
        self.F_vecs = None
        self.F_mags = None
        self.total_hoop_force = None
        self.avg_pressure = None
        self.B_total = None
        self.B_magnitude = None
        self.B_axial = None
        self.use_gauss = False
        self._n_grid = 120   # stored so results_page fallback respects user choice
        # Volumetric filament grid (populated by _build_filament_grid)
        self._n_fil       = 1
        self._fil_coords  = None
        self._fil_weights = None
        self._fil_dl      = None
        self._fil_mid     = None
        # Inductance (populated post-analysis)
        self.self_inductance            = None   # H — Neumann filamentary, primary value
        self.self_inductance_volumetric = None   # H — energy-integral, for comparison
        self.stored_energy              = None   # J


    def run_analysis(self, compute_bfield=False, use_gauss=False,
                     n_grid=120, axis_num=200,
                     progress_callback=None, stage_callback=None):
        """
        Full workflow:
          1) PCA -> 10%
          2) Arc-length -> 20%
          3) B-field at centroid -> 30%
          4) Lorentz forces -> 30-80%
          5) Hoop stress -> 90%
          6) Cross-section B-field -> 95%
          7) On-axis B-field -> 97%
          8) Done -> 100%
        """

        def _stage(msg):
            if stage_callback:
                stage_callback(msg)

        self.compute_bfield_enabled = bool(compute_bfield)
        self.use_gauss = bool(use_gauss)
        self._n_grid = int(n_grid)

        # 1) PCA -> 10%
        _stage("Principal component analysis")
        self._compute_pca()
        if progress_callback:
            progress_callback(10)

        # 2) Arc-length -> 20%
        _stage("Arc-length parametrization")
        self._compute_arc()
        if progress_callback:
            progress_callback(20)

        # 2.5) Build volumetric filament grid
        _stage("Building filament grid")
        self._build_filament_grid()

        # 3) B-field at centroid -> 30%
        _stage("Biot-Savart at centroid")
        self._compute_bfield_at_centroid()
        if progress_callback:
            progress_callback(30)

        # 4) Lorentz forces -> 30-80%
        _stage("Lorentz force integration")
        n = len(self.coords) - 1
        self.F_vecs = np.zeros((n, 3), dtype=float)
        # Per-sub-filament force magnitudes for radial gradient viz
        n_fil = getattr(self, '_n_fil', 1)
        if n_fil > 1:
            self._fil_F_mags = [np.zeros(n, dtype=float) for _ in range(n_fil)]
        is_planar = getattr(self, "is_planar", False)
        if is_planar and self.seg_len is None:
            self._compute_arc()
        # Sequential per-segment force integration. Threading was tried
        # but hit a bus error on macOS — the combination of numpy C ops
        # + Qt worker thread + ThreadPoolExecutor inside it is unstable,
        # and the task granularity (~1ms of numpy per segment) is too
        # fine for thread-pool submit overhead to pay off anyway.
        for i in range(n):
            if is_planar:
                core_radius = max(0.5 * self.thickness,
                                   0.3 * float(np.min(self.seg_len)))
                if self.use_gauss:
                    F_i = self._compute_segment_force_gauss_planar(
                        i, order=16, core_radius=core_radius, skip_neighbors=2
                    )
                else:
                    F_i = self._compute_segment_force_simpson_planar(
                        i, panels=4, core_radius=core_radius, skip_neighbors=2
                    )
            else:
                if self.use_gauss:
                    F_i = self._compute_segment_force_gauss(i, order=16)
                else:
                    F_i = self._compute_segment_force_simpson(i, panels=4)
            self.F_vecs[i] = F_i
            if progress_callback:
                pct = 30 + int(50 * (i + 1) / max(n, 1))
                progress_callback(min(pct, 80))
        if progress_callback:
            progress_callback(80)

        # for planar coil, ensure seam smoothing for closed loops
        if is_planar and self._is_closed_loop():
            k = 2
            for offset in range(1, k + 1):
                i1, i2 = offset, -offset
                avg = 0.5 * (self.F_vecs[i1] + self.F_vecs[i2])
                self.F_vecs[i1] = self.F_vecs[i2] = avg
            avg0 = 0.5 * (self.F_vecs[0] + self.F_vecs[-1])
            self.F_vecs[0] = self.F_vecs[-1] = avg0

        # Quick spike check (log only if anomalous)
        _F_mags_dbg = np.linalg.norm(self.F_vecs, axis=1)
        _med = float(np.median(_F_mags_dbg)) if n > 0 else 0
        _max_ratio = float(np.max(_F_mags_dbg) / _med) if _med > 0 else 0
        if _max_ratio > 5:
            import logging as _lg
            _top = int(np.argmax(_F_mags_dbg))
            _lg.warning(f"Force spike: seg {_top} |F|={_F_mags_dbg[_top]:.4e} "
                         f"({_max_ratio:.1f}x median={_med:.4e}) "
                         f"n_fil={getattr(self,'_n_fil',1)}")

        # 4.1) Convert force to density
        self.F_seg_mags = np.linalg.norm(self.F_vecs, axis=1)
        safe = np.where(self.seg_len > 0, self.seg_len, np.inf)
        self.F_mags = self.F_seg_mags / safe

        # 5) Hoop stress -> 90%
        _stage("Hoop stress calculation")
        if is_planar:
            self._compute_hoop_stress_planar()
        else:
            self._compute_hoop_stress()
        if progress_callback:
            progress_callback(90)

        # 6) Cross-section B-field -> 95%
        if self.compute_bfield_enabled:
            _stage("B-field cross-section sampling")
            try:
                self.cross_section_data = self.sample_cross_section(n=self._n_grid)
            except Exception:
                self.cross_section_data = None
        else:
            self.cross_section_data = None
        if progress_callback:
            progress_callback(95)

        # 7) On-axis B-field -> 97%
        _stage("On-axis B-field profile")
        try:
            z_axis, B_axis = self.compute_bfield_along_axis(
                num=axis_num, use_gauss=self.use_gauss
            )
            self.bfield_axis_z = z_axis
            self.bfield_axis_mag = B_axis
        except Exception:
            self.bfield_axis_z = None
            self.bfield_axis_mag = None
        if progress_callback:
            progress_callback(97)

        # 8) Self-inductance & stored energy -> 98%
        _stage("Self-inductance")
        try:
            self._compute_self_inductance()
        except Exception:
            pass
        # Volumetric energy-integral inductance — gated behind a debug flag.
        # Kept in-code for validation use but no longer computed by default
        # (domain/grid under-resolution gave systematically low values vs the
        # Neumann primary). Enable via env var CALCSX_DEBUG_VOL_INDUCTANCE=1.
        if os.environ.get('CALCSX_DEBUG_VOL_INDUCTANCE') == '1':
            def _L_progress(pct: int) -> None:
                if progress_callback:
                    progress_callback(98 + max(0, min(1, pct // 100)))
            try:
                L_vol = self._compute_self_inductance_volumetric(
                    progress_callback=_L_progress,
                )
                if L_vol is not None and np.isfinite(L_vol):
                    self.self_inductance_volumetric = float(L_vol)
            except Exception:
                pass
        if progress_callback:
            progress_callback(99)

        # 9) Done -> 100%
        _stage("Finalizing")
        if progress_callback:
            progress_callback(100)

        # Snapshot the current at analysis time so a later current-only
        # change can be handled by linear rescaling instead of a full
        # Biot-Savart re-sum (see rescale_to_current below).
        self._analysis_current = float(self.current)
        # True if B_ext was active during analysis — in that case the
        # field/force arrays mix a contribution from *other* coils that
        # does NOT scale with this coil's current, so pure linear rescale
        # is unsafe. The MainWindow checks this flag before using rescale.
        self._analyzed_with_ext = self._B_ext is not None

        return self

    # ------------------------------------------------------------------
    # Dynamic current rescaling (no re-analysis)
    # ------------------------------------------------------------------

    def rescale_to_current(self, new_current: float,
                             force: bool = False) -> bool:
        """Linearly rescale all current-dependent results to a new operating
        current, avoiding a full Biot-Savart recompute.

        By default this is only safe when no external B-field was applied
        during analysis (single-coil case). Pass ``force=True`` when the
        caller knows *all* coils in the superposition are being scaled by
        the same ratio (e.g. a circuit-family current change — every
        member's contribution scales identically, so the stored B_ext
        component is correctly rescaled too).

        Returns True if the rescale was applied, False if the caller must
        fall back to a full re-analysis.

        Scaling:
            B          ∝ I    (field magnitudes, on-axis, cross-section)
            F, σ, W    ∝ I²   (forces, hoop stress, stored energy)
            L                  — invariant
        """
        if not force and getattr(self, '_analyzed_with_ext', False):
            # External field mixed in — can't rescale self-contribution alone.
            return False
        # Fall back to self.current if _analysis_current wasn't stored on
        # this engine (e.g., loaded from a .calcsx saved before the
        # attribute was introduced). Using the engine's current field is
        # correct: its stored B / F arrays correspond to that current.
        I_old = getattr(self, '_analysis_current', None)
        if I_old is None:
            I_old = getattr(self, 'current', None)
        if I_old is None:
            return False
        I_old = float(I_old)
        I_new = float(new_current)
        if abs(I_old) < 1e-30:
            return False
        k = I_new / I_old
        k2 = k * k

        # Linear (∝ I) quantities — magnetic field
        if self.B_total is not None:
            self.B_total = self.B_total * k
        if self.B_magnitude is not None:
            self.B_magnitude = float(self.B_magnitude) * abs(k)
        if getattr(self, 'B_axial', None) is not None:
            self.B_axial = float(self.B_axial) * abs(k)
        if getattr(self, 'bfield_axis_mag', None) is not None:
            self.bfield_axis_mag = self.bfield_axis_mag * abs(k)
        cs = getattr(self, 'cross_section_data', None)
        if cs is not None:
            # Expected format: (X, Y, B_plane, e1, e2, center, R) with B_plane
            # a 2D array of |B| values. Rescale only if it looks right.
            try:
                X, Y, B_plane, *rest = cs
                self.cross_section_data = (X, Y, B_plane * abs(k), *rest)
            except Exception:
                pass

        # Quadratic (∝ I²) quantities — forces, stress, stored energy
        if self.F_vecs is not None:
            self.F_vecs = self.F_vecs * k2
        if getattr(self, 'F_seg_mags', None) is not None:
            self.F_seg_mags = self.F_seg_mags * abs(k2)
        if self.F_mags is not None:
            # F_mags is F_seg_mags / segment_length, so same I² scaling.
            self.F_mags = self.F_mags * abs(k2)
        if getattr(self, 'hoop_stress', None) is not None:
            self.hoop_stress = self.hoop_stress * k2
        if self.total_hoop_force is not None:
            self.total_hoop_force = float(self.total_hoop_force) * k2
        if self.avg_pressure is not None:
            self.avg_pressure = float(self.avg_pressure) * k2
        if self.stored_energy is not None:
            self.stored_energy = float(self.stored_energy) * k2
        # self.self_inductance — geometry only, unchanged.

        self.current = I_new
        self._analysis_current = I_new
        return True

    # ------------------------------------------------------------------
    # Core vectorized Biot-Savart kernels
    # ------------------------------------------------------------------

    @staticmethod
    def _bfield_from_source(points, source_coords, current, winds,
                             skip_index=None, core_radius=1e-4,
                             skip_neighbors=0):
        """
        Stateless vectorized Biot-Savart kernel.

        Parameters
        ----------
        points       : (M, 3) observation points (already 2-D)
        source_coords: (n+1, 3) source path vertices
        current, winds: scalar electrical parameters
        skip_index   : int or None — segment to zero out
        core_radius  : regularisation radius
        skip_neighbors : int — also skip this many segments on each side

        Returns (M, 3) B-field array.
        """
        I  = float(current)
        N  = float(winds)
        a2 = float(core_radius) ** 2
        if not np.isfinite(a2) or a2 <= 0.0:
            a2 = 1e-8

        dl  = source_coords[1:] - source_coords[:-1]          # (n, 3)
        mid = 0.5 * (source_coords[:-1] + source_coords[1:])  # (n, 3)
        nseg = len(dl)

        r  = points[:, None, :] - mid[None, :, :]             # (M, n, 3)
        r2 = np.einsum('mni,mni->mn', r, r)                    # (M, n)

        if skip_index is not None:
            si = int(skip_index)
            for off in range(skip_neighbors + 1):
                for idx in (si - off, si + off):
                    if 0 <= idx < nseg:
                        r2[:, idx] = np.inf

        denom = (r2 + a2) ** 1.5
        valid = np.isfinite(denom) & (denom > 0.0)
        inv_d = np.where(valid, 1.0 / np.where(valid, denom, 1.0), 0.0)

        cr = np.cross(dl[None, :, :], r)                      # (M, n, 3)
        del r, r2, denom, valid                                # free large temps
        B  = np.einsum('mn,mni->mi', inv_d, cr)               # (M, 3)
        del inv_d, cr
        B *= 1e-7 * I * N
        return B

    def _bfield_vec(self, points, skip_index=None, core_radius=1e-4,
                     skip_neighbors=0):
        """
        Vectorized Biot-Savart law for one or more observation points.

        points : array-like, shape (3,) or (M, 3)
        skip_index : int or None — segment index whose contribution is zeroed
        core_radius : float — regularisation radius (m) to avoid singularities
        skip_neighbors : int — also skip this many segments on each side of skip_index

        Returns B with the same leading shape as points: (3,) or (M, 3).
        """
        single = (np.ndim(points) == 1)
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        B = self._bfield_from_source(pts, self.coords, self.current, self.winds,
                                      skip_index=skip_index, core_radius=core_radius,
                                      skip_neighbors=skip_neighbors)
        return B[0] if single else B

    def _bfield_vec_planar(self, points, skip_index=None, core_radius=None, skip_neighbors=2):
        """
        Vectorized Biot-Savart for planar coils with periodic neighbor exclusion.

        points : array-like, shape (3,) or (M, 3)
        Returns B with same leading shape as points.
        """
        single = (np.ndim(points) == 1)
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))  # (M, 3)

        I = float(self.current)
        N = float(self.winds)

        if core_radius is None:
            if self.seg_len is None:
                self._compute_arc()
            core_radius = max(0.5 * float(self.thickness),
                              0.3 * float(np.min(self.seg_len)))
        a2 = max(float(core_radius) ** 2, 1e-8)

        closed = self._is_closed_loop()
        coords = np.ascontiguousarray(np.asarray(self.coords, dtype=np.float64))
        nseg   = coords.shape[0] - 1

        # Build extended segment arrays (closed loop adds wraparound segment)
        if closed:
            P_ext  = np.vstack([coords, coords[0]])   # (nseg+2, 3)
            n_use  = nseg + 1
        else:
            P_ext  = coords                            # (nseg+1, 3)
            n_use  = nseg

        dl_all  = P_ext[1:n_use + 1] - P_ext[:n_use]         # (n_use, 3)
        mid_all = 0.5 * (P_ext[:n_use] + P_ext[1:n_use + 1]) # (n_use, 3)

        # Build inclusion mask (exclude neighbors of skip_index)
        if skip_index is not None:
            si  = int(skip_index)
            js  = np.arange(n_use)
            raw = np.abs(js - si)
            if closed:
                dist = np.minimum(raw, nseg - raw)
            else:
                dist = raw
            keep = dist > skip_neighbors
        else:
            keep = np.ones(n_use, dtype=bool)

        dl_use  = dl_all[keep]   # (k, 3)
        mid_use = mid_all[keep]  # (k, 3)

        if len(dl_use) == 0:
            B = np.zeros((len(pts), 3), dtype=np.float64)
            return B[0] if single else B

        r  = pts[:, None, :] - mid_use[None, :, :]   # (M, k, 3)
        r2 = np.einsum('mki,mki->mk', r, r)           # (M, k)
        denom = (r2 + a2) ** 1.5
        valid = np.isfinite(denom) & (denom > 0.0)
        inv_d = np.where(valid, 1.0 / np.where(valid, denom, 1.0), 0.0)

        cr = np.cross(dl_use[None, :, :], r)          # (M, k, 3)
        B  = np.einsum('mk,mki->mi', inv_d, cr)       # (M, 3)
        B *= 1e-7 * I * N

        return B[0] if single else B

    # ------------------------------------------------------------------
    # Volumetric (multi-filament) Biot-Savart
    # ------------------------------------------------------------------

    def _build_filament_grid(self, n_r=0, n_a=0):
        """
        Discretize the rectangular winding-pack cross-section into sub-filaments
        using Gauss-Legendre quadrature.

        REBCO winding: tape wraps around the coil form, each turn adding one
        tape_thickness radially.
          - **radial** extent = total_thickness = winds × tape_thickness
                                (turns stack outward from the form)
          - **axial** extent  = tape_width  (ribbon width along the axis)

        Parameters
        ----------
        n_r : radial subdivisions (0 = auto-scale from winds)
        n_a : axial subdivisions  (0 = auto-scale from geometry)

        After this call, self._fil_coords, self._fil_weights, self._n_fil,
        and self._fil_dl / self._fil_mid are populated.
        """
        # Auto-scale: radial subs from winds (stacked turns), axial from aspect
        # ratio. For winds=1 we collapse to a single centerline (n_fil = 1) —
        # a thin pack with no meaningful radial *or* axial structure is best
        # modeled as a single filament with the full pack as its effective
        # cross-section, which matches Grover's closed-form ``L = µ₀·R·[ln(8R/
        # GMD) − 2]`` exactly by construction. For multi-turn coils we
        # distribute into n_r radial × n_a axial sub-filaments; n_r ≤ 6,
        # n_a ≤ 2 bounds the cold-compute cost at ~12² × 200² ≈ 6M ops per
        # coil pair (well under a second on a laptop).
        if n_r <= 0:
            if self.winds <= 1:
                n_r = 1
            else:
                # Cap at 4: PEEC self-L pair-sum cost scales as n_r² × n_a²
                # × axis_num². Going n_r=4 → 4² = 16 sub-filaments per coil
                # (vs 36 at n_r=6) cuts analysis time by 2.25× with ~1–2 %
                # accuracy loss on the R=0.3 m circle validation set.
                n_r = min(int(self.winds), 4)
        if n_a <= 0:
            if self.winds <= 1:
                n_a = 1
            else:
                ratio = self.tape_width / max(self.total_thickness, 1e-10)
                n_a = max(1, min(2, int(round(ratio))))

        # Remember the grid dimensions so downstream code (distributed-filament
        # inductance) can use the right sub-cell GMD regularisation.
        self._n_r = int(n_r)
        self._n_a = int(n_a)

        if n_r == 1 and n_a == 1:
            # Filamentary mode — single centerline
            self._n_fil = 1
            self._fil_coords  = [self.coords]
            self._fil_weights = np.array([1.0])
            self._fil_dl  = [self._dl]
            self._fil_mid = [self.midpoints]
            return

        # Gauss-Legendre nodes in [-1, 1] → physical offsets
        r_nodes, r_wts = np.polynomial.legendre.leggauss(n_r)
        a_nodes, a_wts = np.polynomial.legendre.leggauss(n_a)

        # REBCO: radial extent = total_thickness, axial extent = tape_width
        half_r = self.total_thickness * 0.5     # radial half-extent
        half_a = self.tape_width * 0.5          # axial half-extent
        r_phys = r_nodes * half_r    # (n_r,) radial offsets in metres
        a_phys = a_nodes * half_a    # (n_a,) axial offsets in metres

        # 2D weights (outer product, normalized so they sum to 1)
        wts_2d = np.outer(r_wts, a_wts).ravel()
        wts_2d /= wts_2d.sum()

        # Local frame per vertex: e_r (radial/normal) and e_w (width direction)
        #
        #   e_r  = tape normal — height of the stack grows along this.
        #          Priority: supplied bobbin normals > Frenet > PCA radial.
        #   e_w  = perpendicular to both the path tangent and e_r.
        #          Width of the tape grows symmetrically along this.

        if (self._supplied_tape_normals is not None
                and len(self._supplied_tape_normals) == len(self.coords)):
            e_r = np.asarray(self._supplied_tape_normals, dtype=np.float64)
            r_mag = np.linalg.norm(e_r, axis=1, keepdims=True).clip(1e-10)
            e_r = e_r / r_mag
        else:
            # Fallback: PCA-based radial direction
            e_ax_g = self.axis / (np.linalg.norm(self.axis) + 1e-30)
            dx = self.coords - self.mean_point
            proj = np.einsum('ij,j->i', dx, e_ax_g)[:, None] * e_ax_g
            radial = dx - proj
            r_mag = np.linalg.norm(radial, axis=1, keepdims=True).clip(1e-10)
            e_r = radial / r_mag

        # Tangent at each vertex (forward difference, wrap for closed coil)
        tangent = np.empty_like(self.coords)
        tangent[:-1] = self.coords[1:] - self.coords[:-1]
        tangent[-1] = tangent[-2] if len(tangent) > 1 else np.array([1, 0, 0])
        t_mag = np.linalg.norm(tangent, axis=1, keepdims=True).clip(1e-10)
        e_t = tangent / t_mag

        # Width direction: perpendicular to both tangent and radial
        e_w = np.cross(e_t, e_r)
        w_mag = np.linalg.norm(e_w, axis=1, keepdims=True).clip(1e-10)
        e_w = e_w / w_mag

        # Build offset paths for each (r_j, a_k) quadrature point
        #   r_phys offsets along e_r  (radial / height of stack)
        #   a_phys offsets along e_w  (width of tape)
        n_fil = n_r * n_a
        fil_coords  = []
        fil_dl      = []
        fil_mid     = []
        for j in range(n_r):
            for k in range(n_a):
                offset = r_phys[j] * e_r + a_phys[k] * e_w       # (n+1, 3)
                fc = self.coords + offset                          # (n+1, 3)
                fc = np.ascontiguousarray(fc, dtype=np.float64)
                fil_coords.append(fc)
                fd = fc[1:] - fc[:-1]
                fil_dl.append(fd)
                fil_mid.append(0.5 * (fc[:-1] + fc[1:]))

        self._n_fil       = n_fil
        self._fil_coords  = fil_coords     # list of (n+1, 3)
        self._fil_weights = wts_2d          # (n_fil,)
        self._fil_dl      = fil_dl          # list of (n, 3)
        self._fil_mid     = fil_mid         # list of (n, 3)

    def _bfield_vec_volumetric(self, points, skip_index=None, core_radius=1e-4,
                                skip_neighbors=0):
        """
        B-field from a finite cross-section coil, computed as a weighted sum
        of sub-filament contributions using Gauss-Legendre quadrature.

        When n_fil == 1, delegates directly to _bfield_vec (zero overhead).

        The core_radius for each sub-filament is raised to at least
        the sub-filament radial spacing, so that adjacent segments'
        outer sub-filaments don't create near-singular fields at
        the centerline evaluation points at high-curvature locations.
        """
        if self._n_fil <= 1:
            return self._bfield_vec(points, skip_index=skip_index,
                                     core_radius=core_radius,
                                     skip_neighbors=skip_neighbors)

        single = (np.ndim(points) == 1)
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        M = pts.shape[0]
        I = float(self.current)
        N = float(self.winds)

        vol_cr = core_radius

        B = np.zeros((M, 3), dtype=np.float64)
        for f in range(self._n_fil):
            B_f = self._bfield_from_source(
                pts, self._fil_coords[f], I, N,
                skip_index=skip_index, core_radius=vol_cr,
                skip_neighbors=skip_neighbors,
            )
            B += self._fil_weights[f] * B_f

        return B[0] if single else B

    # ------------------------------------------------------------------
    # Total B-field wrappers (self-field + external contributions)
    # ------------------------------------------------------------------

    def _total_bfield(self, points, skip_index=None, core_radius=1e-4,
                       skip_neighbors=0):
        """Self-field (volumetric if available) + B_ext from other coils."""
        if self._n_fil > 1:
            B = self._bfield_vec_volumetric(points, skip_index=skip_index,
                                             core_radius=core_radius,
                                             skip_neighbors=skip_neighbors)
        else:
            B = self._bfield_vec(points, skip_index=skip_index,
                                  core_radius=core_radius,
                                  skip_neighbors=skip_neighbors)
        if self._B_ext is not None:
            single = (np.ndim(points) == 1)
            pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
            B_e = np.atleast_2d(self._B_ext(pts))
            B = np.atleast_2d(B) + B_e
            if single:
                B = B[0]
        return B

    def _total_bfield_planar(self, points, skip_index=None,
                              core_radius=None, skip_neighbors=2):
        """Planar self-field + B_ext from other coils (superposition)."""
        B = self._bfield_vec_planar(points, skip_index=skip_index,
                                     core_radius=core_radius,
                                     skip_neighbors=skip_neighbors)
        # For planar coils, add volumetric correction if n_fil > 1
        # (planar kernel handles its own skip logic; volumetric adds the
        #  cross-section spread that the planar kernel doesn't capture)
        if self._B_ext is not None:
            single = (np.ndim(points) == 1)
            pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
            B_e = np.atleast_2d(self._B_ext(pts))
            B = np.atleast_2d(B) + B_e
            if single:
                B = B[0]
        return B

    def _smooth_bfield(self, points):
        """Filamentary self-field + B_ext.  Used for field-line visualization
        to avoid artifacts from multi-filament near-field structure."""
        B = self._bfield_vec(points)
        if self._B_ext is not None:
            single = (np.ndim(points) == 1)
            pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
            B = np.atleast_2d(B) + np.atleast_2d(self._B_ext(pts))
            if single:
                B = B[0]
        return B

    # ------------------------------------------------------------------
    # Legacy scalar wrappers (preserved for API compatibility)
    # ------------------------------------------------------------------

    def _compute_bfield_at_point(self, point, skip_index=None, core_radius=1e-4):
        """Single-point Biot-Savart (thin wrapper around _bfield_vec)."""
        return self._bfield_vec(
            np.asarray(point, dtype=np.float64),
            skip_index=skip_index,
            core_radius=core_radius
        )

    def _compute_bfield_at_point_planar(self, point, skip_index=None,
                                         core_radius=None, skip_neighbors=2):
        """Single-point planar Biot-Savart (thin wrapper around _bfield_vec_planar)."""
        return self._bfield_vec_planar(
            np.asarray(point, dtype=np.float64),
            skip_index=skip_index,
            core_radius=core_radius,
            skip_neighbors=skip_neighbors
        )

    # ------------------------------------------------------------------
    # Force integration — batched evaluation points
    # ------------------------------------------------------------------

    def _compute_segment_force_gauss(self, i, order=16, core_radius=1e-4,
                                      skip_neighbors=0):
        """16-point Gauss-Legendre Lorentz force on segment i.

        For volumetric coils (n_fil > 1), the force is evaluated at
        each sub-filament's own position rather than at the centerline.
        This naturally avoids near-field singularities: each evaluation
        point is at the same radial offset as the nearest source
        sub-filament on adjacent segments, keeping the distance ≈ seg_len.
        """
        if getattr(self, '_n_fil', 1) > 1:
            return self._compute_segment_force_gauss_vol(
                i, order=order, core_radius=core_radius,
                skip_neighbors=skip_neighbors)

        dl_i = self._dl[i]                                      # (3,)
        p0_i = self._p0[i]                                      # (3,)
        pts  = p0_i + self._gauss_ts16[:, None] * dl_i          # (16, 3)
        B_all = self._total_bfield(pts, skip_index=i,
                                    core_radius=core_radius,
                                    skip_neighbors=skip_neighbors)
        crosses = np.cross(dl_i[None], B_all)                   # (16, 3)
        F_i = self._gauss_ws16 @ crosses                        # (3,)
        return self.current * self.winds * F_i

    def _compute_segment_force_gauss_vol(self, i, order=16,
                                          core_radius=1e-4,
                                          skip_neighbors=0):
        """Volumetric Lorentz force: weighted sum over sub-filaments.

        Each sub-filament f evaluates the total B-field at its OWN
        Gauss points (not the centerline), using its own dl.  The
        force F_f = weight_f × I × N × ∫ dl_f × B(pos_f) is summed
        over all sub-filaments to give the total winding-pack force
        on segment i.

        Also stores per-sub-filament force magnitudes in
        self._fil_F_mags[f][i] for radial-gradient visualisation.
        """
        F_total = np.zeros(3, dtype=np.float64)
        I_N = float(self.current * self.winds)
        for f in range(self._n_fil):
            dl_f  = self._fil_dl[f][i]                          # (3,)
            p0_f  = self._fil_coords[f][i]                      # (3,)
            pts_f = p0_f + self._gauss_ts16[:, None] * dl_f     # (16, 3)
            B_f   = self._total_bfield(
                pts_f, skip_index=i,
                core_radius=core_radius,
                skip_neighbors=skip_neighbors)                  # (16, 3)
            crosses = np.cross(dl_f[None, :], B_f)              # (16, 3)
            F_f = self._gauss_ws16 @ crosses                    # (3,)
            F_total += self._fil_weights[f] * F_f
            # Store per-filament force magnitude
            if hasattr(self, '_fil_F_mags'):
                self._fil_F_mags[f][i] = float(np.linalg.norm(
                    I_N * self._fil_weights[f] * F_f))
        return I_N * F_total

    def _compute_segment_force_simpson(self, i, panels=4, core_radius=1e-4,
                                        skip_neighbors=0):
        """Composite Simpson's-rule Lorentz force on segment i (vectorized)."""
        if panels % 2 != 0:
            raise ValueError("Simpson panels must be even")
        p0, p1 = self.coords[i], self.coords[i + 1]
        dl = p1 - p0
        h   = 1.0 / panels
        ts  = np.linspace(0.0, 1.0, panels + 1)               # (p+1,)
        wts = np.ones(panels + 1)
        wts[1:-1:2] = 4
        wts[2:-1:2] = 2
        pts  = p0 + ts[:, None] * dl                           # (p+1, 3)
        B_all = self._total_bfield(pts, skip_index=i,
                                    core_radius=core_radius,
                                    skip_neighbors=skip_neighbors)
        crosses = np.cross(dl[None], B_all)                    # (p+1, 3)
        F_acc = wts @ crosses                                  # (3,)
        return (self.current * self.winds) * (h / 3.0) * F_acc

    def _compute_segment_force_gauss_planar(self, i, order=16,
                                             core_radius=None, skip_neighbors=2):
        """16-point Gauss-Legendre Lorentz force on planar segment i (vectorized)."""
        dl_i = self._dl[i]
        p0_i = self._p0[i]
        pts  = p0_i + self._gauss_ts16[:, None] * dl_i         # (16, 3)
        B_all = self._total_bfield_planar(
            pts, skip_index=i, core_radius=core_radius, skip_neighbors=skip_neighbors
        )                                                        # (16, 3)
        crosses = np.cross(dl_i[None], B_all)                   # (16, 3)
        F_i = self._gauss_ws16 @ crosses                        # (3,)
        return self.current * self.winds * F_i

    def _compute_segment_force_simpson_planar(self, i, panels=4,
                                               core_radius=None, skip_neighbors=2):
        """Composite Simpson's-rule Lorentz force on planar segment i (vectorized)."""
        if panels % 2 != 0:
            raise ValueError("Simpson panels must be even")
        p0, p1 = self.coords[i], self.coords[i + 1]
        dl  = p1 - p0
        h   = 1.0 / panels
        ts  = np.linspace(0.0, 1.0, panels + 1)
        wts = np.ones(panels + 1)
        wts[1:-1:2] = 4
        wts[2:-1:2] = 2
        pts  = p0 + ts[:, None] * dl                           # (p+1, 3)
        B_all = self._total_bfield_planar(
            pts, skip_index=i, core_radius=core_radius, skip_neighbors=skip_neighbors
        )                                                        # (p+1, 3)
        crosses = np.cross(dl[None], B_all)                    # (p+1, 3)
        F_acc   = wts @ crosses                                # (3,)
        return (self.current * self.winds) * (h / 3.0) * F_acc

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _is_closed_loop(self, seam_factor: float = 2.0):
        """Returns True if the point cloud represents a closed loop."""
        P = np.asarray(self.coords, float)
        if P.ndim != 2 or P.shape[0] < 3:
            return False
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        if seg.size == 0:
            return False
        med_dl = float(np.median(seg))
        if med_dl == 0.0 or not np.isfinite(med_dl):
            return False
        chord = float(np.linalg.norm(P[0] - P[-1]))
        return chord <= seam_factor * med_dl

    def _compute_pca(self):
        self.coords = np.ascontiguousarray(np.asarray(self.coords, dtype=np.float64))
        if not np.all(np.isfinite(self.coords)):
            raise ValueError("coords contain NaN/Inf")

        mean_pt  = np.mean(self.coords, axis=0)
        pca      = PCA(n_components=3).fit(self.coords)
        variances = pca.explained_variance_

        tol = 1e-8 * variances[0]
        if variances[2] < tol:
            A, B, C = self.coords[:3]
            v1 = B - A
            v2 = C - A
            axis = np.cross(v1, v2)
            norm = np.linalg.norm(axis)
            if norm < 1e-12:
                axis = pca.components_[-1]
            else:
                axis = axis / norm
            is_planar = True
        else:
            axis = pca.components_[-1]
            is_planar = False

        if np.dot(axis, self.coords[1] - self.coords[0]) < 0:
            axis = -axis

        self.axis       = axis
        self.mean_point = mean_pt
        self.is_planar  = is_planar

    # legacy PCA algorithm, before planar detection implement (SAVED FOR FALLBACK/TESTING)
    def _compute_pca_LEGACY(self):
        pca = PCA(n_components=3).fit(self.coords)
        axis = pca.components_[-1]
        mean_pt = pca.mean_
        if np.dot(axis, self.coords[1] - self.coords[0]) < 0:
            axis = -axis
        self.axis = axis
        self.mean_point = mean_pt

    def _compute_arc(self):
        diffs = np.diff(self.coords, axis=0)
        seg   = np.linalg.norm(diffs, axis=1)

        self._dl = diffs                               # (n, 3)
        self._p0 = self.coords[:-1]                   # (n, 3)

        arc           = np.concatenate(([0], np.cumsum(seg)))
        self.seg_len  = seg
        self.arc_mid  = 0.5 * (arc[:-1] + arc[1:])
        self.midpoints = 0.5 * (self.coords[:-1] + self.coords[1:])
        self.total_length = arc[-1]

    # ------------------------------------------------------------------
    # Field at centroid
    # ------------------------------------------------------------------

    def _compute_bfield_at_centroid(self):
        B = self._total_bfield(self.mean_point)
        self.B_total     = B
        self.B_magnitude = np.linalg.norm(B)
        self.B_axial     = abs(np.dot(B, self.axis))

    # ------------------------------------------------------------------
    # Hoop stress
    # ------------------------------------------------------------------

    def _compute_hoop_stress(self):
        """Fast hoop-stress post-processing using pre-computed values."""
        for name in ("axis", "mean_point", "midpoints", "seg_len", "F_vecs"):
            if getattr(self, name, None) is None:
                raise RuntimeError(
                    f"{name} must be precomputed before calling _compute_hoop_stress. "
                    "Run run_analysis through segment-force computation first."
                )

        try:
            tape_w  = float(self.tape_width)
            t_total = float(self.total_thickness)
        except Exception as e:
            raise TypeError(f"Bad scalar: {e}")
        if not np.isfinite(tape_w) or tape_w <= 0:
            raise ValueError(f"tape_width must be positive; got {tape_w}")
        if not np.isfinite(t_total) or t_total <= 0:
            raise ValueError(f"total_thickness must be positive; got {t_total}")

        axis    = np.asarray(self.axis, dtype=float)
        mean    = np.asarray(self.mean_point, dtype=float)
        mids    = np.asarray(self.midpoints, dtype=float)   # (n, 3)
        seg_len = np.asarray(self.seg_len, dtype=float)     # (n,)
        F       = np.asarray(self.F_vecs, dtype=float)      # (n, 3)

        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            raise ValueError("axis must be a finite 3-vector.")
        if mean.shape != (3,) or not np.all(np.isfinite(mean)):
            raise ValueError("mean_point must be a finite 3-vector.")
        if mids.ndim != 2 or mids.shape[1] != 3 or mids.shape[0] != seg_len.shape[0] or F.shape != mids.shape:
            raise ValueError("Shape mismatch among midpoints/seg_len/F_vecs.")

        axis /= (np.linalg.norm(axis) + 1e-30)

        # Radial direction: prefer supplied normals (bobbin), then Frenet, then radial
        n_seg = len(mids)
        u_r = None
        if self._supplied_tape_normals is not None:
            tn = np.asarray(self._supplied_tape_normals, dtype=float)
            # Normals are per-vertex; average to segment midpoints
            if len(tn) == n_seg + 1:
                u_r = 0.5 * (tn[:-1] + tn[1:])
            elif len(tn) == n_seg:
                u_r = tn
            else:
                u_r = tn[:n_seg] if len(tn) > n_seg else None
        if u_r is None:
            try:
                from CalcSX_app.physics.geometry import compute_frenet_frame
                frame = compute_frenet_frame(self.coords)
                u_r = frame['normal']
            except Exception:
                pass

        if u_r is None:
            dx     = mids - mean
            a      = axis.reshape(3, 1)
            P      = np.eye(3) - a @ a.T
            r_perp = dx @ P.T
            r_mag_raw = np.linalg.norm(r_perp, axis=1)
            u_r = np.divide(r_perp, r_mag_raw[:, None] + 1e-30)

        # Perpendicular distance from axis (for sigma = p * r / t)
        dx = mids - mean
        a_col = axis.reshape(3, 1)
        P = np.eye(3) - a_col @ a_col.T
        r_perp = dx @ P.T
        r_mag = np.linalg.norm(r_perp, axis=1)
        eps = 1e-30
        self.radial_unit = u_r

        F_r          = np.einsum('ij,ij->i', F, u_r)
        self.radial_force = F_r
        force_density = np.where(seg_len > 0, F_r / seg_len, 0.0)
        self.force_density = force_density
        pressure      = force_density / tape_w
        self.pressure = pressure
        sigma_theta   = pressure * r_mag / t_total
        self.hoop_stress      = sigma_theta
        self.stress_membrane  = sigma_theta

        self.total_hoop_force = float(np.sum(F_r))
        L_total  = self.total_length if (getattr(self, "total_length", None) is not None) else float(np.sum(seg_len))
        A_total  = float(L_total) * tape_w
        self.avg_pressure = float(self.total_hoop_force / (A_total + eps))

        return sigma_theta

    def _compute_hoop_stress_planar(self):
        """
        Planar hoop stress on irregular shapes:
          sigma_theta[i] = p_n[i] / (kappa[i] * total_thickness)
        where p_n is pressure from normal force density, kappa is local curvature.
        """
        for name in ("axis", "mean_point", "midpoints", "seg_len", "F_vecs", "_dl"):
            if getattr(self, name, None) is None:
                raise RuntimeError(f"{name} must be computed before planar hoop stress.")

        a       = np.asarray(self.axis, float)
        a      /= (np.linalg.norm(a) + 1e-30)
        dl      = np.asarray(self._dl, float)       # (n, 3)
        seg_len = np.asarray(self.seg_len, float)   # (n,)
        mids    = np.asarray(self.midpoints, float)  # (n, 3)
        F       = np.asarray(self.F_vecs, float)    # (n, 3)
        n       = seg_len.size

        tape_w  = float(self.tape_width)
        t_total = float(self.total_thickness)
        if not (np.isfinite(tape_w) and tape_w > 0 and np.isfinite(t_total) and t_total > 0):
            raise ValueError("tape_width and total_thickness must be positive.")

        t        = dl / (np.linalg.norm(dl, axis=1)[:, None] + 1e-30)
        n_inplane = np.cross(a[None, :], t)
        n_inplane /= (np.linalg.norm(n_inplane, axis=1)[:, None] + 1e-30)

        outward_check = np.einsum('ij,ij->i', mids - self.mean_point, n_inplane)
        if np.mean(outward_check) < 0.0:
            n_inplane = -n_inplane

        F_n = np.einsum('ij,ij->i', F, n_inplane)
        f_n = np.where(seg_len > 0, F_n / seg_len, 0.0)
        p_n = f_n / tape_w

        helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(a, helper); e1 /= (np.linalg.norm(e1) + 1e-30)
        e2 = np.cross(a, e1)

        cos_th = np.einsum('ij,j->i', t, e1)
        sin_th = np.einsum('ij,j->i', t, e2)
        ang    = np.arctan2(sin_th, cos_th)

        def unwrap_delta(phi_next, phi):
            return ((phi_next - phi + np.pi) % (2 * np.pi)) - np.pi

        closed = self._is_closed_loop()

        if closed:
            ang_next = np.roll(ang, -1)
            ds       = 0.5 * (seg_len + np.roll(seg_len, -1))
            dtheta   = unwrap_delta(ang_next, ang)
            kappa    = np.abs(dtheta) / (ds + 1e-30)
        else:
            dtheta = np.zeros_like(ang)
            ds     = np.zeros_like(seg_len)
            if n >= 3:
                dtheta[1:-1] = 0.5 * (unwrap_delta(ang[2:], ang[1:-1]) + unwrap_delta(ang[1:-1], ang[:-2]))
                ds[1:-1]     = 0.5 * (seg_len[1:-1] + seg_len[:-2])
                dtheta[0]    = dtheta[1];  dtheta[-1] = dtheta[-2]
                ds[0]        = ds[1];      ds[-1]     = ds[-2]
            elif n == 2:
                dtheta[:] = unwrap_delta(ang[1], ang[0])
                ds[:]     = seg_len
            else:
                ds[:] = 1.0
            kappa = np.abs(dtheta) / (ds + 1e-30)

        sigma_theta = p_n / (kappa * t_total)

        self.inplane_normal  = n_inplane
        self.radial_unit     = n_inplane
        self.radial_force    = F_n
        self.force_density   = f_n
        self.pressure        = p_n
        self.hoop_stress     = sigma_theta

        self.total_hoop_force = float(np.sum(F_n))
        L_total = self.total_length if (getattr(self, "total_length", None) is not None) else float(np.sum(seg_len))
        self.avg_pressure = float(self.total_hoop_force / (L_total * tape_w + 1e-30))

        if self._is_closed_loop():
            s01  = 0.5 * (self.hoop_stress[0]    + self.hoop_stress[-1])
            p01  = 0.5 * (self.pressure[0]        + self.pressure[-1])
            f01  = 0.5 * (self.force_density[0]   + self.force_density[-1])
            rn01 = 0.5 * (self.radial_force[0]    + self.radial_force[-1])
            self.hoop_stress[0]   = self.hoop_stress[-1]   = s01
            self.pressure[0]      = self.pressure[-1]      = p01
            self.force_density[0] = self.force_density[-1] = f01
            self.radial_force[0]  = self.radial_force[-1]  = rn01

        return sigma_theta

    # ------------------------------------------------------------------
    # Field profiles
    # ------------------------------------------------------------------

    def compute_bfield_along_axis(self, num=200, use_gauss=None):
        """
        Sample |B| along the PCA axis from –L/2 to +L/2.
        All num points are evaluated in a single vectorized Biot-Savart call.
        """
        if self.axis is None or self.mean_point is None or self.total_length is None:
            raise RuntimeError("Must run run_analysis first")

        L   = self.total_length
        zs  = np.linspace(-0.5 * L, 0.5 * L, num)
        pts = self.mean_point + zs[:, None] * self.axis  # (num, 3)

        B_vecs = self._total_bfield(pts)                  # (num, 3)
        Bs     = np.linalg.norm(B_vecs, axis=1)          # (num,)
        return zs, Bs

    def sample_cross_section(self, n=120, margin=1.05):
        """
        Return a planar grid and |B| magnitudes on the plane through the coil
        centroid perpendicular to the PCA axis.

        Uses chunked vectorized Biot-Savart to control memory for large grids.
        """
        # orthonormal basis perpendicular to axis
        a = self.axis / np.linalg.norm(self.axis)
        helper = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(a, helper); e1 /= np.linalg.norm(e1)
        e1 = -e1  # corrects axis to align with PCA plot

        e2 = np.cross(a, e1)

        # radial envelope
        rads        = self.midpoints - self.mean_point
        axial       = (rads @ a)[:, None] * a
        radial_vecs = rads - axial
        radial_dist = np.linalg.norm(radial_vecs, axis=1)
        R = radial_dist.max() * margin

        lin = np.linspace(-R, R, n)
        X, Y = np.meshgrid(lin, lin, indexing='xy')
        mask = (X ** 2 + Y ** 2) <= R ** 2

        # World coordinates of masked points
        P = (self.mean_point
             + X[..., None] * e1[None, None, :]
             + Y[..., None] * e2[None, None, :])

        pts   = P[mask]                                  # (M, 3)
        Bmag  = np.full_like(X, np.nan, dtype=float)

        # Chunked evaluation to keep memory bounded (~60 MB per chunk for n=5000 segments)
        chunk = 500
        mags  = np.empty(len(pts), dtype=np.float64)
        for start in range(0, len(pts), chunk):
            end         = min(start + chunk, len(pts))
            B_batch     = self._total_bfield(pts[start:end])         # (c, 3)
            mags[start:end] = np.linalg.norm(B_batch, axis=1)

        Bmag[mask] = mags
        return X, Y, Bmag, (-R, R, -R, R)

    def compute_bfield_planes(self, n_planes: int = 20, grid_size: int = 50,
                               margin: float = 1.2,
                               progress_callback=None):
        """
        Compute |B| on *n_planes* cross-sections evenly spaced along the PCA axis.

        Each plane is perpendicular to the axis and uses the same orthonormal basis
        as sample_cross_section, so the geometry is consistent.

        Returns
        -------
        positions : (n_planes,)  axis offsets from mean_point (metres)
        X, Y      : (grid_size, grid_size)  in-plane grid coordinates (metres)
        planes    : (n_planes, grid_size, grid_size)  |B| values (T), NaN outside circle
        R         : float  — circle radius (metres)
        e1, e2    : (3,)   — orthonormal in-plane basis vectors
        """
        a = self.axis / np.linalg.norm(self.axis)

        # Orthonormal in-plane basis (consistent with sample_cross_section)
        helper = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(a, helper); e1 /= np.linalg.norm(e1); e1 = -e1
        e2 = np.cross(a, e1)

        # Radial envelope from coil midpoints
        rads        = self.midpoints - self.mean_point
        axial_comp  = (rads @ a)[:, None] * a
        R           = np.linalg.norm(rads - axial_comp, axis=1).max() * margin

        lin = np.linspace(-R, R, grid_size)
        X, Y = np.meshgrid(lin, lin, indexing='xy')  # (gs, gs)
        circ = X ** 2 + Y ** 2 <= R ** 2              # boolean mask

        L         = self.total_length
        positions = np.linspace(-0.5 * L, 0.5 * L, n_planes)
        planes    = np.full((n_planes, grid_size, grid_size), np.nan, dtype=np.float64)

        chunk = 500
        for k, z_pos in enumerate(positions):
            center = self.mean_point + z_pos * a

            # 3-D world coordinates of every grid point on this plane
            P = (center
                 + X[..., None] * e1[None, None, :]
                 + Y[..., None] * e2[None, None, :])   # (gs, gs, 3)

            pts  = P[circ]                              # (M, 3)
            if len(pts) == 0:
                continue

            mags = np.empty(len(pts), dtype=np.float64)
            for start in range(0, len(pts), chunk):
                end = min(start + chunk, len(pts))
                B   = self._total_bfield(pts[start:end])  # (c, 3) — vectorized
                mags[start:end] = np.linalg.norm(B, axis=1)

            planes[k][circ] = mags

            if progress_callback:
                progress_callback(int(100 * (k + 1) / n_planes))

        return positions, X, Y, planes, R, e1, e2

    def compute_bfield_volume(self, n_vox: int = 32, margin: float = 1.4,
                               progress_callback=None):
        """
        Compute |B| on a uniform 3-D Cartesian voxel grid enclosing the coil.

        Parameters
        ----------
        n_vox   : grid points per axis (n_vox³ total evaluation points)
        margin  : grid half-extent = coil_radius × margin

        Returns
        -------
        xs, ys, zs  : 1-D coordinate arrays (voxel centres, metres)
        B_vol       : ndarray (n_vox, n_vox, n_vox), |B| in Tesla
        """
        # Radial envelope from segment midpoints
        R = float(np.max(np.linalg.norm(
            self.midpoints - self.mean_point, axis=1
        ))) * margin
        if R == 0:
            R = 0.1

        c = self.mean_point
        xs = np.linspace(c[0] - R, c[0] + R, n_vox)
        ys = np.linspace(c[1] - R, c[1] + R, n_vox)
        zs = np.linspace(c[2] - R, c[2] + R, n_vox)

        XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')   # (nx, ny, nz)
        pts = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])  # (N, 3)

        N      = len(pts)
        B_flat = np.empty(N, dtype=np.float64)
        # Larger chunks reduce Python loop overhead; each (chunk × n_seg × 3)
        # working array stays well under ~150 MB for typical coil sizes.
        chunk  = 2000
        n_ch   = max(1, (N + chunk - 1) // chunk)

        for ci in range(n_ch):
            sl = slice(ci * chunk, min((ci + 1) * chunk, N))
            B_flat[sl] = np.linalg.norm(self._total_bfield(pts[sl]), axis=1)
            if progress_callback:
                progress_callback(int(100 * (ci + 1) / n_ch))

        return xs, ys, zs, B_flat.reshape(n_vox, n_vox, n_vox)

    def compute_field_lines(self, n_seeds=20, n_steps=600, max_radius_factor=3.5,
                             progress_callback=None):
        """
        Compute 3D magnetic field line traces using batched RK4 integration.

        Uses two complementary seed sets (each ~n_seeds/2):
          - Fibonacci sphere at 0.6×R from centroid — exterior / return-path lines
          - Sunflower disk in the midplane (0→0.85R) — interior / axial bore lines
        Integrates both forward (+1) and backward (−1) from every seed.

        Returns
        -------
        lines  : list of (N, 3) float32 arrays — each is one field line path
        B_mags : list of (N,) float32 arrays  — |B| (Tesla) at each point
        """
        R = float(np.max(np.linalg.norm(self.midpoints - self.mean_point, axis=1)))
        if R == 0:
            R = 0.1

        step  = R * 0.07          # integration step length (metres)
        max_r = R * max_radius_factor

        golden = (1.0 + 5.0 ** 0.5) / 2.0

        # ── Seed set 1: Fibonacci sphere at 0.6×R (exterior / return-path lines) ──
        n_sphere = max(1, n_seeds // 2)
        sphere_seeds = np.zeros((n_sphere, 3), dtype=np.float64)
        for k in range(n_sphere):
            theta = np.arccos(max(-1.0, min(1.0, 1.0 - 2.0 * (k + 0.5) / n_sphere)))
            phi   = 2.0 * np.pi * k / golden
            sphere_seeds[k] = self.mean_point + (R * 0.60) * np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ])

        # ── Seed set 2: cylinder aligned with PCA axis (interior / axial lines) ──
        # Seeds are distributed throughout the full axial extent and radial interior
        # of the coil, not just in a single midplane.  z and r use INDEPENDENT
        # quasi-random sequences so they are not correlated with each other — this
        # avoids the spiral bias that caused visual imbalance for non-circular coils.
        a  = self.axis / (np.linalg.norm(self.axis) + 1e-30)
        helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(a, helper);  e1 /= np.linalg.norm(e1)
        e2 = np.cross(a, e1)

        axial_proj = np.dot(self.midpoints - self.mean_point, a)
        z_lo   = float(axial_proj.min())
        z_hi   = float(axial_proj.max())
        z_span = max(z_hi - z_lo, R * 0.20)   # at least 20 % of R for planar coils

        PHI_FRAC = 1.0 / golden               # ≈ 0.6180 — independent low-discrepancy step

        n_cyl = n_seeds - n_sphere
        cyl_seeds = np.zeros((n_cyl, 3), dtype=np.float64)
        for k in range(n_cyl):
            # r: uniform area density from 0 to 0.80 R  (sqrt of linear sequence)
            t_r = (k + 0.5) / n_cyl
            r   = R * 0.80 * np.sqrt(t_r)
            # z: independent Fibonacci quasi-random sequence — no correlation with r
            t_z = (k * PHI_FRAC) % 1.0
            z   = z_lo + t_z * z_span
            phi = 2.0 * np.pi * k / golden
            cyl_seeds[k] = self.mean_point + z * a + r * (np.cos(phi) * e1 + np.sin(phi) * e2)

        seeds = np.vstack([sphere_seeds, cyl_seeds])

        # Exclude seeds inside the winding pack — they produce non-physical
        # looking traces that visually penetrate the rendered tube.
        conductor_r = max(self.total_thickness, self.tape_width) * 0.6
        if conductor_r > 0:
            min_dist = np.min(
                np.linalg.norm(
                    seeds[:, None, :] - self.midpoints[None, :, :], axis=2
                ),
                axis=1,
            )
            seeds = seeds[min_dist > conductor_r]

        n_total = len(seeds)

        all_lines: list = []
        all_B:     list = []

        for d_idx, sign in enumerate((1.0, -1.0)):
            pts    = seeds.copy()
            active = np.ones(n_total, dtype=bool)
            trajs  = [[seeds[i].copy()] for i in range(n_total)]

            for s in range(n_steps):
                if not active.any():
                    break
                ai  = np.where(active)[0]
                ap  = pts[ai]           # (n_active, 3)

                # RK4 — four batched calls per step.
                # Use _smooth_bfield (filamentary self + B_ext) to avoid
                # artifacts from multi-filament near-field structure.
                def unit_B(p):
                    B   = self._smooth_bfield(p)
                    mag = np.linalg.norm(B, axis=-1, keepdims=True).clip(1e-30)
                    return B / mag

                u1     = unit_B(ap)
                u2     = unit_B(ap + 0.5 * step * u1 * sign)
                u3     = unit_B(ap + 0.5 * step * u2 * sign)
                u4     = unit_B(ap +       step * u3 * sign)
                new_ap = ap + sign * (step / 6.0) * (u1 + 2*u2 + 2*u3 + u4)
                pts[ai] = new_ap

                dist = np.linalg.norm(new_ap - self.mean_point, axis=1)
                active[ai[dist >= max_r]] = False

                # Terminate field lines that enter the winding pack
                if conductor_r > 0:
                    wire_dist = np.min(
                        np.linalg.norm(
                            new_ap[:, None, :] - self.midpoints[None, :, :],
                            axis=2,
                        ),
                        axis=1,
                    )
                    active[ai[wire_dist < conductor_r]] = False

                for j, i in enumerate(ai):
                    if active[i]:
                        trajs[i].append(new_ap[j].copy())

                if progress_callback and s % 30 == 0:
                    pct = int(100 * (d_idx + s / n_steps) / 2)
                    progress_callback(min(pct, 99))

            for i in range(n_total):
                if len(trajs[i]) < 3:
                    continue
                pts_arr  = np.array(trajs[i], dtype=np.float32)
                B_chunks = []
                chunk    = 300
                for ci in range(0, len(pts_arr), chunk):
                    Bc = self._smooth_bfield(pts_arr[ci:ci+chunk].astype(np.float64))
                    B_chunks.append(np.linalg.norm(Bc, axis=1).astype(np.float32))
                all_lines.append(pts_arr)
                all_B.append(np.concatenate(B_chunks))

        if progress_callback:
            progress_callback(100)
        return all_lines, all_B

    def compute_bfield_midplane(self, grid_size=80, margin=1.5,
                                 axis_offset=0.0, progress_callback=None):
        """
        Compute |B| on the 2D cross-sectional plane through the coil centroid,
        perpendicular to the PCA axis.

        Returns
        -------
        X, Y    : (grid_size, grid_size) 2D coordinate grids in the plane basis
        B_plane : (grid_size, grid_size) |B| values in Tesla
        e1, e2  : (3,) orthonormal basis vectors spanning the plane
        center  : (3,) world-space origin of the plane (== mean_point)
        R       : float  — grid half-width in metres
        """
        a      = self.axis / np.linalg.norm(self.axis)
        helper = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([0, 1.0, 0])
        e1     = np.cross(a, helper);  e1 /= np.linalg.norm(e1);  e1 = -e1
        e2     = np.cross(a, e1)

        R_coil = float(np.max(np.linalg.norm(self.midpoints - self.mean_point, axis=1)))
        R      = R_coil * margin

        lin    = np.linspace(-R, R, grid_size)
        X, Y   = np.meshgrid(lin, lin, indexing='ij')  # (gs, gs)

        center = self.mean_point + axis_offset * a
        P    = (center
                + X[..., None] * e1[None, None, :]
                + Y[..., None] * e2[None, None, :])      # (gs, gs, 3)
        pts  = P.reshape(-1, 3)

        N      = len(pts)
        B_flat = np.empty(N, dtype=np.float64)
        chunk  = 2000
        n_ch   = max(1, (N + chunk - 1) // chunk)
        for ci in range(n_ch):
            sl          = slice(ci * chunk, min((ci + 1) * chunk, N))
            B_flat[sl]  = np.linalg.norm(self._total_bfield(pts[sl]), axis=1)
            if progress_callback:
                progress_callback(int(100 * (ci + 1) / n_ch))

        B_plane = B_flat.reshape(grid_size, grid_size)
        return X, Y, B_plane, e1, e2, center, R

    # ------------------------------------------------------------------
    # Self-inductance (PEEC / Neumann formula)
    # ------------------------------------------------------------------

    @staticmethod
    def _grover_bar_self_partial(ell: float, w: float, t: float) -> float:
        """Partial self-inductance of a straight rectangular bar of length
        ℓ and cross-section w × t, returned in μ₀/4π units (i.e. multiply
        the returned value by 1e-7 to get Henries).

        Grover 1946, §11 (rectangular bar formula):
            L = (µ₀·ℓ / 2π) · [ ln(2ℓ/(w+t)) + 0.5 + 0.2235·(w+t)/ℓ ]

        Factoring out µ₀/(4π), that's:
            L / (µ₀/(4π)) = 2ℓ · [ ln(2ℓ/(w+t)) + 0.5 + 0.2235·(w+t)/ℓ ]

        This is the "self-partial" used on the i=j diagonal of the PEEC
        inductance sum — it replaces the crude ℓ²/GMD regularisation
        (which diverges like 1/GMD) with a physically correct thin-bar
        closed form that stays finite for all length/cross-section ratios.
        """
        s = max(float(w) + float(t), 1e-12)
        ell = max(float(ell), 1e-12)
        return 2.0 * ell * (np.log(2.0 * ell / s) + 0.5 + 0.2235 * s / ell)

    @staticmethod
    def _pair_integral(dl_f: np.ndarray, mid_f: np.ndarray,
                        dl_g: np.ndarray, mid_g: np.ndarray,
                        self_pair: bool,
                        w_self: float, t_self: float,
                        same_coil: bool = True) -> float:
        """Hybrid PEEC partial mutual-inductance integral in µ₀/4π units.

        Uses midpoint Neumann ``dl_i · dl_j / r_ij`` as the default kernel
        — accurate for widely separated pairs AND for end-to-end adjacent
        segments on curved filaments (where parallel-coincident assumptions
        fail). Replaces specific singular / near-singular cases with
        closed-form analytical expressions:

          - **i = j on same filament (`self_pair=True`)**: divergent;
            replaced with Grover's partial self-inductance of a straight
            rectangular bar with cross-section ``(w_self, t_self)``.

          - **i = j on DIFFERENT filaments of the SAME coil**
            (``self_pair=False``, ``same_coil=True``): these are spatially
            coincident-longitudinal pairs separated only by the small
            inter-filament distance across the pack. Midpoint Neumann
            gives ``dl²/d`` here, which overcounts when ``d ≪ L``.
            Replaced by the Hoer-Love / Ruehli closed-form for parallel
            coincident filaments:

                M_par(L, d) = 2·[L·arcsinh(L/d) + d − √(L² + d²)]

            exactly reducing to ``L²/d`` for ``d ≫ L`` and staying finite
            for ``d ≪ L``.

          - **i = j between DIFFERENT coils** (``same_coil=False``): same
            segment index carries no "close-pair" meaning — coils are at
            arbitrary positions in space. Midpoint Neumann is retained.

        This hybrid eliminates the distributed-filament overcounting on
        the pack cross-section while preserving midpoint Neumann accuracy
        for genuine inter-coil mutuals.
        """
        ell_f = np.linalg.norm(dl_f, axis=1)                  # (n_f,)
        ell_g = np.linalg.norm(dl_g, axis=1)                  # (n_g,)

        diff = mid_f[:, None, :] - mid_g[None, :, :]          # (n_f, n_g, 3)
        d_arr = np.linalg.norm(diff, axis=2)                   # (n_f, n_g)
        dot   = np.einsum('ik,jk->ij', dl_f, dl_g)             # (n_f, n_g)

        # Standard midpoint Neumann everywhere.
        d_safe = np.maximum(d_arr, 1e-12)
        M_matrix = dot / d_safe

        # Replace the i=j diagonal ONLY when the two filaments are on the
        # same coil (so matching indices means geometrically coincident
        # longitudinal span across the pack cross-section). For different
        # coils, matching indices have no geometric significance and
        # midpoint Neumann is the correct kernel. Fully vectorized — the
        # previous per-index Python loop was the dominant cost (300 × 144
        # scalar numpy calls per coil pair, ~86k iterations for a 2-coil
        # circuit, visibly laggy on macOS).
        n_diag = min(len(ell_f), len(ell_g))
        if not self_pair and same_coil and n_diag > 0:
            diag_d = np.maximum(np.diagonal(d_arr)[:n_diag], 1e-12)      # (n_diag,)
            diag_dot = np.diagonal(dot)[:n_diag]                         # (n_diag,)
            L_diag = 0.5 * (ell_f[:n_diag] + ell_g[:n_diag])              # (n_diag,)
            L_safe = np.maximum(L_diag, 1e-12)
            ell_prod = np.maximum(ell_f[:n_diag] * ell_g[:n_diag], 1e-30)
            cos_th = diag_dot / ell_prod
            M_par = 2.0 * (L_safe * np.arcsinh(L_safe / diag_d)
                           + diag_d - np.sqrt(L_safe**2 + diag_d**2))
            M_diag = cos_th * M_par
            # Write back onto the i=j diagonal
            idx = np.arange(n_diag)
            M_matrix[idx, idx] = M_diag

        if self_pair:
            # Drop the (i=j) diagonal (divergent) and replace with Grover
            # straight-bar self-partial at the filament's cross-section.
            np.fill_diagonal(M_matrix, 0.0)
            s = max(float(w_self) + float(t_self), 1e-12)
            ell_safe = np.maximum(ell_f, 1e-12)
            grover_diag = 2.0 * ell_safe * (
                np.log(np.maximum(2.0 * ell_safe / s, 1e-12))
                + 0.5 + 0.2235 * s / ell_safe
            )
            return float(np.sum(M_matrix)) + float(np.sum(grover_diag))
        return float(np.sum(M_matrix))

    def _compute_self_inductance(self):
        """
        Self-inductance via a proper PEEC formulation:

            L = (µ₀ N² / 4π) · Σ_f Σ_g  α_f · α_g · pair_integral(f, g)

        The pack's rectangular cross-section is discretised into
        ``n_r × n_a`` Gauss-Legendre sub-filaments (``α_f`` are the area
        fractions, Σα = 1). Every filament pair integral uses the
        Hoer-Love / Ruehli parallel-filament closed form on all segment
        pairs — see ``_pair_integral`` docstring. The f==g diagonal
        replaces the divergent i=j self-segment term with Grover's bar
        self-partial at the **sub-cell** cross-section (``t/n_r × w/n_a``);
        the singular behaviour at close filament-to-filament approaches
        is handled exactly by the closed-form pair kernel rather than by
        midpoint Neumann, which was the over-counting mode of the earlier
        distributed attempt.

        For ``n_fil == 1`` (thin pack), the single "filament" uses the
        full pack cross-section for Grover's regularisation — this
        recovers the centerline-based formulation that matches Grover's
        closed-form circular-coil expression ``L = µ₀·R·N²·[ln(8R/GMD) − 2]``.

        Validation targets:
          - Circular loop (R=300mm, N=1, thin tape): 2.22 µH analytical.
          - Circular loop (R=300mm, N=10): 215 µH analytical.
          - 6-turn generator solenoid: ~5.9 µH (Nagaoka).
          - 20-turn deutero pair: 2·L matches 0.858 mH measurement.
        """
        if self.midpoints is None or self._dl is None:
            return

        mu0_4pi = 1e-7
        N = float(self.winds)

        n_fil = getattr(self, '_n_fil', 1)
        fil_dl  = getattr(self, '_fil_dl', None)
        fil_mid = getattr(self, '_fil_mid', None)
        wts     = getattr(self, '_fil_weights', None)

        if n_fil <= 1 or fil_dl is None or fil_mid is None or wts is None:
            # Single-centerline: one "filament" with the full pack cross-section.
            t_cx = float(self.total_thickness)
            w_cx = float(self.tape_width)
            pair = self._pair_integral(
                self._dl, self.midpoints, self._dl, self.midpoints,
                self_pair=True, w_self=w_cx, t_self=t_cx,
            )
            L = mu0_4pi * N * N * pair
            self.self_inductance = L
            self.stored_energy = 0.5 * L * self.current * self.current
            return

        # Distributed-filament PEEC with Hoer-Love pair kernel.
        n_r = max(int(getattr(self, '_n_r', 1)), 1)
        n_a = max(int(getattr(self, '_n_a', 1)), 1)
        t_sub = float(self.total_thickness) / n_r
        w_sub = float(self.tape_width) / n_a
        wts_arr = np.asarray(wts, dtype=np.float64)

        acc = 0.0
        for f in range(n_fil):
            dl_f  = fil_dl[f]
            mid_f = fil_mid[f]
            w_f   = float(wts_arr[f])
            for g in range(n_fil):
                dl_g  = fil_dl[g]
                mid_g = fil_mid[g]
                w_g   = float(wts_arr[g])
                pair = self._pair_integral(
                    dl_f, mid_f, dl_g, mid_g,
                    self_pair=(f == g),
                    w_self=w_sub, t_self=t_sub,
                )
                acc += w_f * w_g * pair

        L = mu0_4pi * N * N * acc
        self.self_inductance = float(L)
        self.stored_energy = 0.5 * L * self.current * self.current

    def _compute_self_inductance_volumetric(self, grid: int = 20,
                                              margin: float = 2.0,
                                              core_radius: float = 1e-3,
                                              progress_callback=None):
        """
        Self-inductance from the volume integral of magnetic energy density.

            L = (1 / (µ₀ I²)) ∫ |B|² dV

        Uses only this coil's own B-field (B_ext from other coils is
        bypassed) over a bounding box sized ``margin × coil extent``.
        Returns L in henries, or None if it can't be evaluated.

        Parameters
        ----------
        grid : int
            Cubical sampling density (grid³ points total). 20 → 8 k pts
            (≈ 2–5 s per coil) balances speed vs. accuracy for
            engineering estimates.
        margin : float
            Box half-extent as a multiple of the coil's own bounding box
            half-width. 2.0 gives a sampling volume 4× wider than the
            coil in each axis — usually enough for |B|² to decay.
        core_radius : float
            Near-segment regularization distance (metres) used to
            prevent the Biot-Savart divergence for grid points that
            fall adjacent to the filaments.
        progress_callback : callable or None
            Called with an integer percentage (0..100) while the grid
            is being swept.
        """
        if self.current == 0 or self.coords is None:
            return None
        mu0 = 4.0 * np.pi * 1e-7

        bb_min = self.coords.min(axis=0)
        bb_max = self.coords.max(axis=0)
        center = 0.5 * (bb_min + bb_max)
        half_extent = 0.5 * (bb_max - bb_min) * margin
        # For near-planar coils give the short axis real depth so the box
        # isn't degenerate; fall back to a scale set by the long axes.
        min_half = 0.5 * float(np.linalg.norm(half_extent))
        half_extent = np.maximum(half_extent, min_half)

        xs = np.linspace(center[0] - half_extent[0],
                         center[0] + half_extent[0], grid)
        ys = np.linspace(center[1] - half_extent[1],
                         center[1] + half_extent[1], grid)
        zs = np.linspace(center[2] - half_extent[2],
                         center[2] + half_extent[2], grid)
        dV = ((xs[1] - xs[0]) * (ys[1] - ys[0]) * (zs[1] - zs[0]))

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        pts_all = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        # Smaller chunks keep peak memory bounded; big chunks spike VRAM
        # when the (M, n_seg, 3) temp in _bfield_from_source gets large.
        chunk = 500
        n_total = len(pts_all)
        n_chunks = (n_total + chunk - 1) // chunk
        sum_B2 = 0.0
        use_vol = self._n_fil > 1
        for i, start in enumerate(range(0, n_total, chunk)):
            end = min(start + chunk, n_total)
            if use_vol:
                B = self._bfield_vec_volumetric(
                    pts_all[start:end], core_radius=core_radius,
                )
            else:
                B = self._bfield_vec(
                    pts_all[start:end], core_radius=core_radius,
                )
            # Drop non-finite contributions (near-segment regularization
            # can still yield NaN/Inf for singular geometries — e.g. a
            # grid point landing exactly on a midpoint).
            B = np.where(np.isfinite(B), B, 0.0)
            sum_B2 += float(np.sum(B * B))
            if progress_callback is not None and (i % 4 == 0 or end == n_total):
                progress_callback(int(100 * end / n_total))

        if not np.isfinite(sum_B2) or sum_B2 <= 0.0:
            return None
        W = sum_B2 * dV / (2.0 * mu0)          # magnetic energy in the box
        L = 2.0 * W / (self.current * self.current)
        return float(L)

    # ------------------------------------------------------------------
    # Field harmonics (cylindrical multipole decomposition)
    # ------------------------------------------------------------------

    def compute_field_harmonics(
        self,
        r_ref: float = None,
        n_phi: int = 64,
        n_max: int = 10,
        z_positions: np.ndarray = None,
    ) -> dict:
        """
        Decompose the B-field into cylindrical multipole harmonics.

        At each axial position z, sample B on a circle of radius r_ref in
        the plane perpendicular to the PCA axis, then Fourier-decompose the
        radial and azimuthal components to extract multipole coefficients
        b_n (normal) and a_n (skew).

        This is the standard method used in accelerator magnet field quality
        assessment (RAT, ROXIE, etc.).

        Parameters
        ----------
        r_ref       : reference radius (m); default = 2/3 of coil bore radius
        n_phi       : azimuthal sampling points per circle
        n_max       : maximum harmonic order
        z_positions : axial positions to sample; default = 5 equally spaced

        Returns
        -------
        dict with:
            'z'    : (nz,) axial positions
            'b_n'  : (nz, n_max) normal harmonics (T) at r_ref
            'a_n'  : (nz, n_max) skew harmonics (T) at r_ref
            'r_ref': float — reference radius used
        """
        if self.mean_point is None or self.axis is None or self.midpoints is None:
            return None

        a = self.axis / np.linalg.norm(self.axis)
        R_coil = float(np.max(np.linalg.norm(
            self.midpoints - self.mean_point, axis=1
        )))

        if r_ref is None:
            r_ref = R_coil * 0.667

        # Build local coordinate system
        if abs(a[2]) < 0.9:
            e1 = np.cross(a, np.array([0., 0., 1.]))
        else:
            e1 = np.cross(a, np.array([1., 0., 0.]))
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(a, e1)

        if z_positions is None:
            L = self.total_length if self.total_length else R_coil * 2
            z_positions = np.linspace(-L * 0.3, L * 0.3, 5)

        nz = len(z_positions)
        phi = np.linspace(0, 2.0 * np.pi, n_phi, endpoint=False)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        b_n = np.zeros((nz, n_max), dtype=np.float64)
        a_n = np.zeros((nz, n_max), dtype=np.float64)

        for iz, z_off in enumerate(z_positions):
            center = self.mean_point + z_off * a
            # Points on the reference circle
            pts = (center[None, :]
                   + r_ref * cos_phi[:, None] * e1[None, :]
                   + r_ref * sin_phi[:, None] * e2[None, :])

            B = self._total_bfield(pts)  # (n_phi, 3)

            # Project B onto radial and azimuthal directions
            B_r = np.zeros(n_phi)
            B_phi_arr = np.zeros(n_phi)
            for ip in range(n_phi):
                e_r = cos_phi[ip] * e1 + sin_phi[ip] * e2
                e_phi_dir = -sin_phi[ip] * e1 + cos_phi[ip] * e2
                B_r[ip] = np.dot(B[ip], e_r)
                B_phi_arr[ip] = np.dot(B[ip], e_phi_dir)

            # Fourier decomposition: B_r(φ) = Σ [b_n cos(nφ) + a_n sin(nφ)]
            for n in range(1, n_max + 1):
                b_n[iz, n - 1] = (2.0 / n_phi) * np.sum(B_r * np.cos(n * phi))
                a_n[iz, n - 1] = (2.0 / n_phi) * np.sum(B_r * np.sin(n * phi))

        return {
            'z': np.asarray(z_positions),
            'b_n': b_n,
            'a_n': a_n,
            'r_ref': r_ref,
        }
