# physics_utils.py
import numpy as np
from sklearn.decomposition import PCA

class CoilAnalysis:
    def __init__(self, coords, winds, current, thickness_microns, tape_width_mm):
        # raw inputs
        self.compute_bfield_enabled = False
        self.coords = coords
        self.winds = winds
        self.current = current
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

        # 3) B-field at centroid -> 30%
        _stage("Biot-Savart at centroid")
        self._compute_bfield_at_centroid()
        if progress_callback:
            progress_callback(30)

        # 4) Lorentz forces -> 30-80%
        _stage("Lorentz force integration")
        n = len(self.coords) - 1
        self.F_vecs = np.zeros((n, 3), dtype=float)
        is_planar = getattr(self, "is_planar", False)
        if is_planar and self.seg_len is None:
            self._compute_arc()
        for i in range(n):
            if is_planar:
                core_radius = max(0.5 * self.thickness, 0.3 * float(np.min(self.seg_len)))
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

        # 8) Done -> 100%
        _stage("Finalizing")
        if progress_callback:
            progress_callback(100)

        return self

    # ------------------------------------------------------------------
    # Core vectorized Biot-Savart kernels
    # ------------------------------------------------------------------

    def _bfield_vec(self, points, skip_index=None, core_radius=1e-4):
        """
        Vectorized Biot-Savart law for one or more observation points.

        points : array-like, shape (3,) or (M, 3)
        skip_index : int or None — segment index whose contribution is zeroed
        core_radius : float — regularisation radius (m) to avoid singularities

        Returns B with the same leading shape as points: (3,) or (M, 3).
        """
        single = (np.ndim(points) == 1)
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))  # (M, 3)

        I = float(self.current)
        N = float(self.winds)
        a2 = float(core_radius) ** 2
        if not np.isfinite(a2) or a2 <= 0.0:
            a2 = 1e-8

        coords = self.coords  # (n+1, 3), float64 contiguous after _compute_pca
        dl  = coords[1:] - coords[:-1]          # (n, 3)
        mid = 0.5 * (coords[:-1] + coords[1:])  # (n, 3)

        # displacement vectors: r[m, j] = pts[m] - mid[j]
        r  = pts[:, None, :] - mid[None, :, :]          # (M, n, 3)
        r2 = np.einsum('mni,mni->mn', r, r)              # (M, n)

        if skip_index is not None:
            r2[:, int(skip_index)] = np.inf  # zeroes this segment's contribution

        denom = (r2 + a2) ** 1.5                         # (M, n)
        valid = np.isfinite(denom) & (denom > 0.0)
        inv_d = np.where(valid, 1.0 / np.where(valid, denom, 1.0), 0.0)

        cr = np.cross(dl[None, :, :], r)                 # (M, n, 3)
        B  = np.einsum('mn,mni->mi', inv_d, cr)          # (M, 3)
        B *= 1e-7 * I * N                                 # mu0/4pi = 1e-7

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

    def _compute_segment_force_gauss(self, i, order=16):
        """16-point Gauss-Legendre Lorentz force on segment i (vectorized)."""
        dl_i = self._dl[i]                                      # (3,)
        p0_i = self._p0[i]                                      # (3,)
        pts  = p0_i + self._gauss_ts16[:, None] * dl_i          # (16, 3)
        B_all = self._bfield_vec(pts, skip_index=i)             # (16, 3)
        crosses = np.cross(dl_i[None], B_all)                   # (16, 3)
        F_i = self._gauss_ws16 @ crosses                        # (3,)
        return self.current * self.winds * F_i

    def _compute_segment_force_simpson(self, i, panels=4):
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
        B_all = self._bfield_vec(pts, skip_index=i)            # (p+1, 3)
        crosses = np.cross(dl[None], B_all)                    # (p+1, 3)
        F_acc = wts @ crosses                                  # (3,)
        return (self.current * self.winds) * (h / 3.0) * F_acc

    def _compute_segment_force_gauss_planar(self, i, order=16,
                                             core_radius=None, skip_neighbors=2):
        """16-point Gauss-Legendre Lorentz force on planar segment i (vectorized)."""
        dl_i = self._dl[i]
        p0_i = self._p0[i]
        pts  = p0_i + self._gauss_ts16[:, None] * dl_i         # (16, 3)
        B_all = self._bfield_vec_planar(
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
        B_all = self._bfield_vec_planar(
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
        B = self._bfield_vec(self.mean_point)
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

        dx     = mids - mean
        a      = axis.reshape(3, 1)
        P      = np.eye(3) - a @ a.T
        r_perp = dx @ P.T
        r_mag  = np.linalg.norm(r_perp, axis=1)
        eps    = 1e-30
        u_r    = np.divide(r_perp, r_mag[:, None] + eps)
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

        B_vecs = self._bfield_vec(pts)                   # (num, 3)
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
            B_batch     = self._bfield_vec(pts[start:end])          # (c, 3)
            mags[start:end] = np.linalg.norm(B_batch, axis=1)

        Bmag[mask] = mags
        return X, Y, Bmag, (-R, R, -R, R)
