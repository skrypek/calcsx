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


    def run_analysis(self, compute_bfield=False, use_gauss=False, progress_callback=None):
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
        
        self.compute_bfield_enabled = bool(compute_bfield)
        self.use_gauss = bool(use_gauss)

        # 1) PCA -> 10%
        self._compute_pca()
        if progress_callback:
            progress_callback(10)

        # 2) Arc-length -> 20%
        self._compute_arc()
        if progress_callback:
            progress_callback(20)

        # 3) B-field at centroid -> 30%
        self._compute_bfield_at_centroid()
        if progress_callback:
            progress_callback(30)

        # 4) Lorentz forces -> 30-80%
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
        if is_planar:
            self._compute_hoop_stress_planar()
        else:
            self._compute_hoop_stress()
        if progress_callback:
            progress_callback(90)

        # 6) Cross-section B-field -> 95%
        if self.compute_bfield_enabled:
            try:
                self.cross_section_data = self.sample_cross_section(n=120)
            except Exception:
                self.cross_section_data = None
        else:
            self.cross_section_data = None
        if progress_callback:
            progress_callback(95)

        # 7) On-axis B-field -> 97%
        try:
            z_axis, B_axis = self.compute_bfield_along_axis(
                num=200, use_gauss=self.use_gauss
            )
            self.bfield_axis_z = z_axis
            self.bfield_axis_mag = B_axis
        except Exception:
            self.bfield_axis_z = None
            self.bfield_axis_mag = None
        if progress_callback:
            progress_callback(97)

        # 8) Done -> 100%
        if progress_callback:
            progress_callback(100)

        return self
    
    def _is_closed_loop(self, seam_factor: float = 2.0):
        """
        Returns True if the point cloud represents a closed loop.
        """
        
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
        
        # find centroid
        mean_pt = np.mean(self.coords, axis=0)
    
        # run PCA on the raw coords (PCA centers internally)
        pca = PCA(n_components=3).fit(self.coords)
        variances = pca.explained_variance_
    
        # planar detection
        tol = 1e-8 * variances[0] # find threshold to compare smallest variance to (i.e., modifier * largest)
        if variances[2] < tol: # if the lowest variance is in fact below the threshold...
            # pick three noncollinear points for a robust normal
            A, B, C = self.coords[:3]
            v1 = B - A
            v2 = C - A
            axis = np.cross(v1, v2)
            norm = np.linalg.norm(axis)
            if norm < 1e-12:
                # fallback to PCA’s (noisy) direction if collinear
                axis = pca.components_[-1]
            else:
                axis = axis / norm
            is_planar = True # mark curve as planar to warn user
        else:
            axis = pca.components_[-1]
            is_planar = False
    
        # ensure axis points “forward” along first segment
        if np.dot(axis, self.coords[1] - self.coords[0]) < 0:
            axis = -axis
    
        # store results
        self.axis = axis
        self.mean_point = mean_pt
        self.is_planar = is_planar 

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
        seg = np.linalg.norm(diffs, axis=1)
        
        self._dl = diffs # shape (n,3)
        self._p0 = self.coords[:-1] # shape (n,3)
        
        arc = np.concatenate(([0], np.cumsum(seg)))
        self.seg_len = seg
        self.arc_mid = 0.5 * (arc[:-1] + arc[1:])
        self.midpoints = 0.5 * (self.coords[:-1] + self.coords[1:])
        self.total_length = arc[-1]
    
    def _compute_bfield_at_point(self, point, skip_index=None, core_radius=1e-4):
        """
        Employs Biot-Savart to iteratively compute B-field at points along curve.
        """
        mu0 = 4.0 * np.pi * 1e-7
    
        # force safe dtypes/strides
        coords = np.ascontiguousarray(np.asarray(self.coords, dtype=np.float64))
        point = np.asarray(point, dtype=np.float64)
        B = np.zeros(3, dtype=np.float64)
    
        # guards
        if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] < 2:
            raise ValueError("coords must be (n,3) with n>=2")
        try:
            I = float(self.current)
            N = float(self.winds)
        except Exception:
            I = float(self.current) if np.isscalar(self.current) else float(np.asarray(self.current).item())
            N = float(self.winds)   if np.isscalar(self.winds)   else float(np.asarray(self.winds).item())
    
        a2 = float(core_radius)**2
        if not np.isfinite(a2) or a2 <= 0.0:
            a2 = 1e-8 # fallback (meters^2)
    
        n = coords.shape[0] - 1
    
        for j in range(n):
            if skip_index is not None and j == int(skip_index):
                continue
    
            p0 = coords[j]
            p1 = coords[j+1]
            dl = p1 - p0 # (3,)
            mid = 0.5 * (p0 + p1)
            r = point - mid # (3,)
    
            # squared distance
            r2 = float(np.dot(r, r))
            if not np.isfinite(r2) or r2 == 0.0:
                continue
    
            denom = (r2 + a2)**1.5
            if not np.isfinite(denom) or denom == 0.0:
                continue
    
            cr = np.cross(dl, r) # (3,)
            if not np.all(np.isfinite(cr)):
                continue
    
            B += cr / denom
    
        return (mu0 / (4.0 * np.pi)) * I * N * B
        
    def _compute_bfield_at_point_planar(self, point, skip_index=None, core_radius=None, skip_neighbors=2):
        
        mu0 = 4.0 * np.pi * 1e-7
    
        coords = np.ascontiguousarray(np.asarray(self.coords, dtype=np.float64))
        point = np.asarray(point, dtype=np.float64)
        B = np.zeros(3, dtype=np.float64)
    
        if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] < 2:
            raise ValueError("coords must be (n,3) with n>=2")
    
        I = float(self.current)
        N = float(self.winds)
    
        if core_radius is None:
            if self.seg_len is None:
                self._compute_arc()
            core_radius = max(0.5 * float(self.thickness), 0.3 * float(np.min(self.seg_len)))
        a2 = float(core_radius)**2
        if not np.isfinite(a2) or a2 <= 0.0:
            a2 = 1e-8
    
        closed = self._is_closed_loop()
        P = coords
        nseg = P.shape[0] - 1
    
        # extend with seam wraparound
        if closed:
            P = np.vstack([P, P[0]])
    
        def _periodic_exclude(j, i, n, skip):
            if not closed:
                return abs(j - i) <= skip
            dist = min(abs(j - i), n - abs(j - i))
            return dist <= skip
    
        si = int(skip_index) if skip_index is not None else None
        for j in range(nseg if not closed else nseg + 1):
            if si is not None and _periodic_exclude(j, si, nseg, skip_neighbors):
                continue
            p0 = P[j]
            p1 = P[j + 1]
            dl = p1 - p0
            mid = 0.5 * (p0 + p1)
            r = point - mid
            r2 = np.dot(r, r)
            if r2 == 0 or not np.isfinite(r2):
                continue
            B += np.cross(dl, r) / (r2 + a2)**1.5
    
        return (mu0 / (4.0 * np.pi)) * I * N * B

    def _compute_segment_force_gauss(self, i, order=16):
        """
        Vectorized 16‑point Gauss Lorentz‐force on segment i.
        """
        dl_i = self._dl[i] # (3,)
        p0_i = self._p0[i] # (3,)
        F_i = np.zeros(3)

        for t, w in zip(self._gauss_ts16, self._gauss_ws16):
            pt = p0_i + t * dl_i
            B_loc = self._compute_bfield_at_point(pt, skip_index=i)
            F_i += np.cross(dl_i, B_loc) * w

        return self.current * self.winds * F_i

    def _compute_segment_force_simpson(self, i, panels=4):
        """
        Composite Simpson’s‐rule Lorentz force on segment i.
        """
        if panels % 2 != 0:
            raise ValueError("Simpson panels must be even")

        # endpoints of segment
        p0, p1 = self.coords[i], self.coords[i+1]
        dl = p1 - p0 # constant vector

        # Simpson parameters
        h = 1.0 / panels
        ts = np.linspace(0.0, 1.0, panels+1) # t = 0,1/p,2/p,…,1
        wts = np.ones(panels+1)
        for k in range(1, panels):
            wts[k] = 4 if (k % 2)==1 else 2

        # accumulate ∫ (dl × B(p(t))) dt
        F_acc = np.zeros(3)
        for t, w in zip(ts, wts):
            pt = p0 + t * dl
            # simple Biot–Savart at pt (midpoint method)
            B_loc = self._compute_bfield_at_point(pt, skip_index=i)
            F_acc += w * np.cross(dl, B_loc)

        # apply Simpson factor h/3 and multiply by I·winds
        return (self.current * self.winds) * (h/3.0) * F_acc

    def _compute_segment_force_gauss_planar(self, i, order=16, core_radius=None, skip_neighbors=2):

        dl_i = self._dl[i]; p0_i = self._p0[i]
        F_i = np.zeros(3)
    
        def ringdist(j, i, n):
            d = abs(j - i)
            return min(d, n - d)
    
        for t, w in zip(self._gauss_ts16, self._gauss_ws16):
            pt = p0_i + t * dl_i
            B_loc = self._compute_bfield_at_point_planar(pt, skip_index=i, core_radius=core_radius, skip_neighbors=skip_neighbors)
            F_i += np.cross(dl_i, B_loc) * w
        return self.current * self.winds * F_i
    
    def _compute_segment_force_simpson_planar(self, i, panels=4, core_radius=None, skip_neighbors=2):

        if panels % 2 != 0:
            raise ValueError("Simpson panels must be even")
    
        p0, p1 = self.coords[i], self.coords[i+1]
        dl = p1 - p0
        h = 1.0 / panels
        ts = np.linspace(0.0, 1.0, panels+1)
        wts = np.ones(panels+1); wts[1:-1:2] = 4; wts[2:-1:2] = 2
        F_acc = np.zeros(3)
        for t, w in zip(ts, wts):
            pt = p0 + t * dl
            B_loc = self._compute_bfield_at_point_planar(pt, skip_index=i, core_radius=core_radius, skip_neighbors=skip_neighbors)
            F_acc += w * np.cross(dl, B_loc)
        return (self.current * self.winds) * (h/3.0) * F_acc

    def _compute_bfield_at_centroid(self):

        B = self._compute_bfield_at_point(
            self.mean_point, skip_index=None
        )
        self.B_total = B
        self.B_magnitude = np.linalg.norm(B)
        self.B_axial = abs(np.dot(B, self.axis))

    def _compute_hoop_stress(self):
        """
        Fast hoop-stress post-processing using pre-computed values
        """
    
        # pre-requisite check
        for name in ("axis", "mean_point", "midpoints", "seg_len", "F_vecs"):
            if getattr(self, name, None) is None:
                raise RuntimeError(
                    f"{name} must be precomputed before calling _compute_hoop_stress. "
                    "Run run_analysis through segment-force computation first."
                )
    
        # scalars
        try:
            tape_w = float(self.tape_width)
            t_total = float(self.total_thickness)
        except Exception as e:
            raise TypeError(f"Bad scalar: {e}")
        if not np.isfinite(tape_w) or tape_w <= 0:
            raise ValueError(f"tape_width must be positive; got {tape_w}")
        if not np.isfinite(t_total) or t_total <= 0:
            raise ValueError(f"total_thickness must be positive; got {t_total}")
    
        # arrays
        axis = np.asarray(self.axis, dtype=float)
        mean = np.asarray(self.mean_point, dtype=float)
        mids = np.asarray(self.midpoints, dtype=float) # (n,3)
        seg_len = np.asarray(self.seg_len, dtype=float) # (n,)
        F = np.asarray(self.F_vecs, dtype=float) # (n,3)
    
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            raise ValueError("axis must be a finite 3-vector.")
        if mean.shape != (3,) or not np.all(np.isfinite(mean)):
            raise ValueError("mean_point must be a finite 3-vector.")
        if mids.ndim != 2 or mids.shape[1] != 3 or mids.shape[0] != seg_len.shape[0] or F.shape != mids.shape:
            raise ValueError("Shape mismatch among midpoints/seg_len/F_vecs.")
    
        # normalize axis
        axis /= (np.linalg.norm(axis) + 1e-30)
    
        # radial geometry, normal vector
        dx = mids - mean # (n,3)
        a = axis.reshape(3, 1)
        P = np.eye(3) - a @ a.T # projector onto plane ⟂ axis
        r_perp = dx @ P.T # (n,3)
        r_mag = np.linalg.norm(r_perp, axis=1) # (n,)
        eps = 1e-30
        u_r = np.divide(r_perp, r_mag[:, None] + eps) # outward radial unit
        self.radial_unit = u_r
    
        # radial load -> density -> pressure -> hoop stress
        F_r = np.einsum('ij,ij->i', F, u_r) # N per segment
        self.radial_force = F_r
        force_density = np.where(seg_len > 0, F_r / seg_len, 0.0) # N/m
        self.force_density = force_density
        pressure = force_density / tape_w # Pa
        self.pressure = pressure
        sigma_theta = pressure * r_mag / t_total # Pa
        self.hoop_stress = sigma_theta
        self.stress_membrane = sigma_theta
    
        # aggregates
        self.total_hoop_force = float(np.sum(F_r)) # N
        L_total = self.total_length if (getattr(self, "total_length", None) is not None) else float(np.sum(seg_len))
        A_total = float(L_total) * tape_w
        self.avg_pressure = float(self.total_hoop_force / (A_total + eps)) # Pa
    
        return sigma_theta

    def _compute_hoop_stress_planar(self):
        """
        Planar hoop stress on irregular shapes:
          sigma_theta[i] = p_n[i] / (kappa[i] * total_thickness)
        where p_n is pressure from normal force density, and kappa is local curvature.
        """
    
        for name in ("axis", "mean_point", "midpoints", "seg_len", "F_vecs", "_dl"):
            if getattr(self, name, None) is None:
                raise RuntimeError(f"{name} must be computed before planar hoop stress.")
    
        a = np.asarray(self.axis, float)
        a /= (np.linalg.norm(a) + 1e-30)
        dl = np.asarray(self._dl, float) # (n,3)
        seg_len = np.asarray(self.seg_len, float) # (n,)
        mids = np.asarray(self.midpoints, float) # (n,3)
        F = np.asarray(self.F_vecs, float) # (n,3)
        n = seg_len.size
    
        tape_w = float(self.tape_width)
        t_total = float(self.total_thickness)
        if not (np.isfinite(tape_w) and tape_w > 0 and np.isfinite(t_total) and t_total > 0):
            raise ValueError("tape_width and total_thickness must be positive.")
    
        # tangent and in‑plane normal (unit)
        t = dl / (np.linalg.norm(dl, axis=1)[:, None] + 1e-30)
        n_inplane = np.cross(a[None, :], t)
        n_inplane /= (np.linalg.norm(n_inplane, axis=1)[:, None] + 1e-30)
    
        # outward normal orientation
        outward_check = np.einsum('ij,ij->i', mids - self.mean_point, n_inplane)
        if np.mean(outward_check) < 0.0:
            n_inplane = -n_inplane
    
        # Normal force -> density -> pressure
        F_n = np.einsum('ij,ij->i', F, n_inplane) # N
        f_n = np.where(seg_len > 0, F_n / seg_len, 0.0) # N/m
        p_n = f_n / tape_w # Pa
    
        # curvature via turning angle in the plane
        helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(a, helper); e1 /= (np.linalg.norm(e1) + 1e-30)
        e2 = np.cross(a, e1)
    
        cos_th = np.einsum('ij,j->i', t, e1)
        sin_th = np.einsum('ij,j->i', t, e2)
        ang = np.arctan2(sin_th, cos_th)
    
        def unwrap_delta(phi_next, phi):
            return ((phi_next - phi + np.pi) % (2*np.pi)) - np.pi
    
        closed = self._is_closed_loop()
    
        if closed:
            ang_next = np.roll(ang, -1)
            ds = 0.5 * (seg_len + np.roll(seg_len, -1))
            dtheta = unwrap_delta(ang_next, ang)
            kappa = np.abs(dtheta) / (ds + 1e-30)
        else:
            # one-sided at ends; centered interior; no wrap
            dtheta = np.zeros_like(ang)
            ds = np.zeros_like(seg_len)
            if n >= 3:
                dtheta[1:-1] = 0.5 * (unwrap_delta(ang[2:], ang[1:-1]) + unwrap_delta(ang[1:-1], ang[:-2]))
                ds[1:-1] = 0.5 * (seg_len[1:-1] + seg_len[:-2])
                # copy neighbors to ends
                dtheta[0] = dtheta[1];  dtheta[-1] = dtheta[-2]
                ds[0] = ds[1]; ds[-1] = ds[-2]
            elif n == 2:
                dtheta[:] = unwrap_delta(ang[1], ang[0])
                ds[:] = seg_len
            else:
                ds[:] = 1.0
            kappa = np.abs(dtheta) / (ds + 1e-30)
    
        # hoop stress
        sigma_theta = p_n / (kappa * t_total)
    
        # store fields (compatible with non‑planar pipeline)
        self.inplane_normal = n_inplane
        self.radial_unit = n_inplane
        self.radial_force = F_n
        self.force_density = f_n
        self.pressure = p_n
        self.hoop_stress = sigma_theta
    
        self.total_hoop_force = float(np.sum(F_n))
        L_total = self.total_length if (getattr(self, "total_length", None) is not None) else float(np.sum(seg_len))
        self.avg_pressure = float(self.total_hoop_force / (L_total * tape_w + 1e-30))
        
        if self._is_closed_loop():
            s01 = 0.5 * (self.hoop_stress[0] + self.hoop_stress[-1])
            p01 = 0.5 * (self.pressure[0]     + self.pressure[-1])
            f01 = 0.5 * (self.force_density[0]+ self.force_density[-1])
            rn01 = 0.5 * (self.radial_force[0] + self.radial_force[-1])
        
            self.hoop_stress[0] = self.hoop_stress[-1] = s01
            self.pressure[0] = self.pressure[-1] = p01
            self.force_density[0] = self.force_density[-1] = f01
            self.radial_force[0] = self.radial_force[-1] = rn01
        
        return sigma_theta

    def compute_bfield_along_axis(self, num=200, use_gauss=None):
        """
        Numerically sample |B| along the PCA axis from –L/2 to +L/2.
        """
        # ensure analysis has been run at least through step 3
        if self.axis is None or self.mean_point is None or self.total_length is None:
            raise RuntimeError("Must run run_analysis(compute_bfield=True) first")

        if use_gauss is None:
            use_gauss = getattr(self, "use_gauss", False)

        L = self.total_length
        zs = np.linspace(-0.5 * L, 0.5 * L, num)
        Bs = np.empty_like(zs)

        for i, z in enumerate(zs):
            P = self.mean_point + z * self.axis
            # midpoint‐Biot–Savart at P:
            B_vec = self._compute_bfield_at_point(P, skip_index=None)
            Bs[i] = np.linalg.norm(B_vec)

        return zs, Bs

    def sample_cross_section(self, n=120, margin=1.05):
        """
        Return a planar grid and |B| magnitudes on the plane through the coil
        centroid perpendicular to the PCA axis.
        """
        # orthonormal basis perpendicular to axis
        a = self.axis / np.linalg.norm(self.axis)
        helper = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(a, helper); e1 /= np.linalg.norm(e1)
        e1 = -e1 # corrects axis to align with PCA plot
        
        e2 = np.cross(a, e1) #TODO! Potentially check if this should be flipped too

        # radial envelope
        rads = self.midpoints - self.mean_point
        axial = (rads @ a)[:, None] * a
        radial_vecs = rads - axial
        radial_dist = np.linalg.norm(radial_vecs, axis=1)
        R = radial_dist.max() * margin

        lin = np.linspace(-R, R, n)
        X, Y = np.meshgrid(lin, lin, indexing='xy')
        mask = (X**2 + Y**2) <= R**2

        # World coordinates
        P = (self.mean_point
             + X[..., None] * e1[None, None, :]
             + Y[..., None] * e2[None, None, :])

        Bmag = np.full_like(X, np.nan, dtype=float)
        pts = P[mask]

        vals = []
        for pt in pts:
            B = self._compute_bfield_at_point(pt)
            vals.append(np.linalg.norm(B))
        Bmag[mask] = np.array(vals)

        return X, Y, Bmag, (-R, R, -R, R)
    