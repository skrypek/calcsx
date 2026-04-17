# physics/bobbin.py
"""
Bobbin/former import system for STEP files.

Extracts winding channels (grooves) from a 3D solid, computes centerline
paths and surface normals at each point.  The surface normal IS the tape-face
normal — this is ground truth for B_perp decomposition in Ic margin analysis.

Supports two backends (tried in order):
  1. cadquery + OCP (OpenCASCADE) — requires Python ≤ 3.12 for wheels
  2. gmsh  (bundles its own OpenCASCADE)  — ``pip install gmsh``
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class BobbinChannel:
    """One winding channel (groove) on the bobbin surface."""
    channel_id: str
    coords: np.ndarray              # (N, 3) centerline points
    normals: np.ndarray             # (N, 3) surface normal at each point
    width_m: float = 0.0           # measured channel width (m)
    depth_m: float = 0.0           # measured channel depth (m)
    _face_ref: object = None       # OCC TopoDS_Face (not serialised)


def _detect_valleys(profile, min_prominence=None):
    """Find valley indices in a 1-D height profile.

    Uses ``scipy.signal.find_peaks`` on the negated profile.
    Falls back to a simple local-minimum scan if scipy is absent.

    Parameters
    ----------
    profile : 1-D array
    min_prominence : float or None
        Minimum prominence.  ``None`` → auto (10 % of profile range).

    Returns
    -------
    valley_indices : ndarray of int
    """
    prange = float(profile.max() - profile.min())
    if prange < 1e-15:
        return np.array([], dtype=int)
    if min_prominence is None:
        min_prominence = prange * 0.10

    try:
        from scipy.signal import find_peaks
        valleys, _ = find_peaks(
            -profile,
            distance=max(3, len(profile) // 30),
            prominence=min_prominence,
        )
        return valleys
    except ImportError:
        pass

    # Fallback: simple local minimum scan
    valleys = []
    for i in range(1, len(profile) - 1):
        if profile[i] < profile[i - 1] and profile[i] < profile[i + 1]:
            if (min(profile[i - 1], profile[i + 1]) - profile[i]
                    >= min_prominence):
                valleys.append(i)
    return np.array(valleys, dtype=int)


def _build_mesh_adjacency(mesh):
    """Pre-compute face adjacency with per-edge dihedral cosines.

    Returns
    -------
    adj : list[list[tuple[int, float]]]
        ``adj[i]`` = list of ``(neighbour_cell, cos_dihedral)`` pairs.
    face_normals : ndarray (n_cells, 3)
    cell_centers : ndarray (n_cells, 3)
    """
    from collections import defaultdict

    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    cell_centers = np.asarray(
        mesh.cell_centers().points, dtype=np.float64)
    n_cells = mesh.n_cells

    faces_flat = np.asarray(mesh.faces)
    edge_to_cells: dict[tuple, list] = defaultdict(list)
    idx = 0
    ci = 0
    while idx < len(faces_flat):
        nv = int(faces_flat[idx])
        verts = faces_flat[idx + 1: idx + 1 + nv]
        for j in range(nv):
            a = int(verts[j])
            b = int(verts[(j + 1) % nv])
            edge_to_cells[(min(a, b), max(a, b))].append(ci)
        idx += nv + 1
        ci += 1

    adj: list[list[tuple[int, float]]] = [[] for _ in range(n_cells)]
    for cids in edge_to_cells.values():
        if len(cids) == 2:
            c0, c1 = cids
            cos_d = float(np.dot(face_normals[c0], face_normals[c1]))
            adj[c0].append((c1, cos_d))
            adj[c1].append((c0, cos_d))

    return adj, face_normals, cell_centers


def _grow_from_seed(seed_cell, adj, face_normals, cell_centers,
                    edge_cos_thresh, seed_cos_thresh,
                    max_path_dist=float('inf')):
    """BFS region-grow from *seed_cell* with three stopping criteria.

    1. **Edge criterion** — the dihedral cosine across the shared edge
       must be ``>= edge_cos_thresh``.  Blocks groove walls and other
       sharp features.
    2. **Seed criterion** — the candidate face normal must satisfy
       ``abs(dot(seed_normal, face_normal)) >= seed_cos_thresh``.
       Prevents the BFS from drifting across large smooth surfaces
       that gradually curve away from the seed orientation.
    3. **Path distance** — the accumulated centroid-to-centroid
       distance along the BFS path must not exceed *max_path_dist*.
       Prevents the BFS from reaching adjacent grooves by going
       around groove walls through smooth terminations.

    Returns a list of cell indices in the grown region.
    """
    seed_n = face_normals[seed_cell]
    visited = {seed_cell}
    region = [seed_cell]
    # Each queue entry: (cell_index, accumulated_path_distance)
    queue = [(seed_cell, 0.0)]
    head = 0

    while head < len(queue):
        face, dist = queue[head]
        head += 1
        fc = cell_centers[face]
        for nb, cos_dihed in adj[face]:
            if nb in visited:
                continue
            visited.add(nb)
            # Sharp edge → stop
            if cos_dihed < edge_cos_thresh:
                continue
            # Drifted too far from seed normal → stop
            if abs(np.dot(seed_n, face_normals[nb])) < seed_cos_thresh:
                continue
            # Accumulated path too long → stop
            step = float(np.linalg.norm(cell_centers[nb] - fc))
            new_dist = dist + step
            if new_dist > max_path_dist:
                continue
            region.append(nb)
            queue.append((nb, new_dist))

    return region


def _grow_groove(seed_cell, adj, face_normals, cell_centers,
                 edge_cos_thresh, seed_cos_thresh,
                 max_width=float('inf')):
    """BFS region-grow with groove-width constraint.

    Like ``_grow_from_seed`` but adds a cross-groove width limit:

    1. Grow a small initial neighbourhood (~30 cells) using only
       dihedral + normal criteria.
    2. PCA on the initial neighbourhood gives groove direction (PC1)
       and cross-groove direction (PC2).
    3. Continue BFS but reject any cell whose projection onto PC2
       would make the region wider than *max_width*.

    PCA is periodically recomputed as the region grows so that the
    groove direction tracks curves (D-shapes, racetrack coils, etc.).

    Parameters
    ----------
    max_width : float
        Maximum allowed extent of the region perpendicular to the
        groove direction, in the same units as the mesh coordinates.
        ``inf`` disables the constraint (legacy behaviour).

    Returns a list of cell indices in the grown region.
    """
    seed_n = face_normals[seed_cell]
    visited = {seed_cell}
    region = [seed_cell]
    queue = [seed_cell]
    head = 0

    # Width constraint state
    cross_dir = None
    cross_lo = 0.0
    cross_hi = 0.0
    cross_center = 0.0
    last_pca_len = 0
    pca_min_cells = 20          # need this many before PCA is meaningful
    use_width = max_width < float('inf')

    def _update_pca():
        nonlocal cross_dir, cross_lo, cross_hi, cross_center, last_pca_len
        pts = cell_centers[region]
        centroid = pts.mean(axis=0)
        _, S, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
        # Only apply width constraint if region is elongated
        if S[0] < 1e-12 or S[1] / S[0] > 0.8:
            cross_dir = None
            return
        cross_dir = Vt[1]      # PC2 = across groove
        proj = (pts - centroid) @ cross_dir
        cross_lo = float(proj.min())
        cross_hi = float(proj.max())
        cross_center = float(centroid @ cross_dir)
        last_pca_len = len(region)

    while head < len(queue):
        face = queue[head]
        head += 1

        for nb, cos_dihed in adj[face]:
            if nb in visited:
                continue
            visited.add(nb)

            # Sharp edge → stop
            if cos_dihed < edge_cos_thresh:
                continue
            # Drifted too far from seed normal → stop
            if abs(np.dot(seed_n, face_normals[nb])) < seed_cos_thresh:
                continue

            # Width constraint
            if use_width and len(region) >= pca_min_cells:
                # Recompute PCA when region has grown 50%
                if cross_dir is None or len(region) > last_pca_len * 1.5:
                    _update_pca()
                if cross_dir is not None:
                    p = float(cell_centers[nb] @ cross_dir) - cross_center
                    new_lo = min(cross_lo, p)
                    new_hi = max(cross_hi, p)
                    if new_hi - new_lo > max_width:
                        continue

            region.append(nb)
            queue.append(nb)

            # Update cross bounds incrementally
            if use_width and cross_dir is not None:
                p = float(cell_centers[nb] @ cross_dir) - cross_center
                cross_lo = min(cross_lo, p)
                cross_hi = max(cross_hi, p)

    return region


def _extract_centerline(pts, normals=None, n_target=500):
    """Order groove-floor vertices into a smooth centerline.

    Detects ring-like (closed coil) vs strip-like shapes and bins by
    angle or PCA projection accordingly.  Works for non-planar 3-D
    paths with torsion because PCA captures the dominant sweep direction.
    """
    if len(pts) < 3:
        return pts, normals

    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)

    ratio = S[1] / max(S[0], 1e-10)
    is_ring = ratio > 0.3          # roughly circular / closed loop

    if is_ring:
        proj2 = centered @ Vt[:2].T
        param = np.arctan2(proj2[:, 1], proj2[:, 0])
        lo, hi = -np.pi, np.pi
    else:
        param = centered @ Vt[0]
        lo, hi = float(param.min()), float(param.max())

    n_bins = min(n_target, max(60, len(pts) // 5))
    edges = np.linspace(lo, hi, n_bins + 1)

    cl_p: list[np.ndarray] = []
    cl_n: list[np.ndarray] = []
    for i in range(n_bins):
        mask = (param >= edges[i]) & (param < edges[i + 1])
        if not mask.any():
            continue
        cl_p.append(pts[mask].mean(axis=0))
        if normals is not None:
            nm = normals[mask].mean(axis=0)
            nm_len = np.linalg.norm(nm)
            if nm_len > 1e-10:
                nm /= nm_len
            cl_n.append(nm)

    cl_p = np.array(cl_p, dtype=np.float64)
    cl_n = (np.array(cl_n, dtype=np.float64)
            if normals is not None and cl_n else None)

    # Smooth with moving average
    if len(cl_p) > 10:
        try:
            from scipy.ndimage import uniform_filter1d
            k = 5
            mode = 'wrap' if is_ring else 'nearest'
            for ax in range(3):
                cl_p[:, ax] = uniform_filter1d(cl_p[:, ax], k, mode=mode)
            if cl_n is not None:
                for ax in range(3):
                    cl_n[:, ax] = uniform_filter1d(
                        cl_n[:, ax], k, mode=mode)
                cl_n /= np.maximum(
                    np.linalg.norm(cl_n, axis=1, keepdims=True), 1e-10)
        except ImportError:
            pass

    return cl_p, cl_n


class BobbinImporter:
    """
    Import and analyse a STEP file representing a winding bobbin.

    Usage
    -----
    imp = BobbinImporter("my_former.step")
    imp.load()                            # tries cadquery, then gmsh
    mesh = imp.get_mesh()                 # PyVista mesh for rendering
    channels = imp.detect_channels()      # auto-detect grooves
    # or
    channels = imp.get_all_faces_as_channels()  # let user pick
    imp.close()                           # release gmsh if active
    """

    def __init__(self, filepath: str, n_discretize: int = 500):
        self._filepath = filepath
        self._n_disc = int(n_discretize)
        self._shape = None        # OCC TopoDS_Shape (cadquery only)
        self._cq_result = None    # cadquery Workplane  (cadquery only)
        self._faces: list[dict] = []
        self._mesh = None         # PyVista PolyData
        self._segmented_mesh = None   # face-tagged mesh with normals (cached)
        self._mesh_adj = None         # adjacency list with dihedral cosines
        self._mesh_fnormals = None    # face normals for adjacency mesh
        self._mesh_ccenters = None    # cell centers for path distance
        self._refined_mesh = None     # curvature-adapted mesh for groove detection
        self._backend: str | None = None   # 'cadquery' | 'gmsh'
        self._gmsh_active = False

    # ── Loading ───────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the STEP file.  Tries cadquery first, then gmsh."""
        # --- cadquery -------------------------------------------------
        try:
            self._load_cadquery()
            self._backend = 'cadquery'
            return
        except ImportError:
            pass
        except Exception as exc:
            # cadquery found but load failed — still try gmsh
            _cq_err = exc

        # --- gmsh -----------------------------------------------------
        try:
            self._load_gmsh()
            self._backend = 'gmsh'
            return
        except ImportError:
            pass

        raise ImportError(
            "STEP import requires cadquery or gmsh.\n\n"
            "Install one of:\n"
            "  pip install gmsh          (recommended)\n"
            "  pip install cadquery       (needs Python ≤ 3.12)\n"
        )

    # ── cadquery backend ──────────────────────────────────────────────────

    def _load_cadquery(self) -> None:
        import cadquery as cq                       # raises ImportError
        self._cq_result = cq.importers.importStep(self._filepath)
        self._shape = self._cq_result.val().wrapped
        self._analyse_faces_occ()

    def _analyse_faces_occ(self) -> None:
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE
        from OCP.BRep import BRep_Tool
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps

        self._faces = []
        explorer = TopExp_Explorer(self._shape, TopAbs_FACE)

        while explorer.More():
            face = explorer.Current()
            explorer.Next()

            props = GProp_GProps()
            BRepGProp.SurfaceProperties(face, props)
            area = props.Mass()

            cog = props.CentreOfMass()
            centroid = np.array([cog.X(), cog.Y(), cog.Z()])

            adaptor = BRepAdaptor_Surface(face)
            u_mid = 0.5 * (adaptor.FirstUParameter() + adaptor.LastUParameter())
            v_mid = 0.5 * (adaptor.FirstVParameter() + adaptor.LastVParameter())

            try:
                from OCP.GeomLProp import GeomLProp_SLProps
                surface = BRep_Tool.Surface_s(face)
                slprops = GeomLProp_SLProps(surface, u_mid, v_mid, 1, 1e-6)
                if slprops.IsNormalDefined():
                    n = slprops.Normal()
                    normal = np.array([n.X(), n.Y(), n.Z()])
                else:
                    normal = np.array([0., 0., 1.])
            except Exception:
                normal = np.array([0., 0., 1.])

            self._faces.append({
                'face': face,
                'adaptor': adaptor,
                'area': area,
                'centroid': centroid,
                'normal': normal,
            })

    def _discretize_face_occ(
        self, finfo: dict, n_pts: int | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        from OCP.BRep import BRep_Tool
        from OCP.GeomLProp import GeomLProp_SLProps

        if n_pts is None:
            n_pts = self._n_disc

        adaptor = finfo['adaptor']
        u0 = adaptor.FirstUParameter()
        u1 = adaptor.LastUParameter()
        v0 = adaptor.FirstVParameter()
        v1 = adaptor.LastVParameter()

        surface = BRep_Tool.Surface_s(finfo['face'])
        coords, normals = [], []

        if (u1 - u0) >= (v1 - v0):
            v_mid = 0.5 * (v0 + v1)
            for u in np.linspace(u0, u1, n_pts):
                pt = surface.Value(u, v_mid)
                coords.append([pt.X(), pt.Y(), pt.Z()])
                try:
                    slp = GeomLProp_SLProps(surface, u, v_mid, 1, 1e-6)
                    if slp.IsNormalDefined():
                        n = slp.Normal()
                        normals.append([n.X(), n.Y(), n.Z()])
                    else:
                        normals.append([0., 0., 1.])
                except Exception:
                    normals.append([0., 0., 1.])
        else:
            u_mid = 0.5 * (u0 + u1)
            for v in np.linspace(v0, v1, n_pts):
                pt = surface.Value(u_mid, v)
                coords.append([pt.X(), pt.Y(), pt.Z()])
                try:
                    slp = GeomLProp_SLProps(surface, u_mid, v, 1, 1e-6)
                    if slp.IsNormalDefined():
                        n = slp.Normal()
                        normals.append([n.X(), n.Y(), n.Z()])
                    else:
                        normals.append([0., 0., 1.])
                except Exception:
                    normals.append([0., 0., 1.])

        if len(coords) < 3:
            return None, None
        return (
            np.array(coords, dtype=np.float64),
            np.array(normals, dtype=np.float64),
        )

    @staticmethod
    def _shape_to_pyvista(shape, linear_deflection: float = 0.05):
        try:
            import pyvista as pv
        except ImportError:
            return None

        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE
        from OCP.BRep import BRep_Tool
        from OCP.TopLoc import TopLoc_Location

        BRepMesh_IncrementalMesh(shape, linear_deflection)
        all_pts, all_faces = [], []
        offset = 0

        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = explorer.Current()
            explorer.Next()
            location = TopLoc_Location()
            tri = BRep_Tool.Triangulation_s(face, location)
            if tri is None:
                continue
            n_nodes = tri.NbNodes()
            n_tris = tri.NbTriangles()
            trsf = location.Transformation()
            for i in range(1, n_nodes + 1):
                pt = tri.Node(i).Transformed(trsf)
                all_pts.append([pt.X(), pt.Y(), pt.Z()])
            for i in range(1, n_tris + 1):
                t = tri.Triangle(i)
                i1, i2, i3 = t.Get()
                all_faces.extend([3, i1 - 1 + offset,
                                  i2 - 1 + offset, i3 - 1 + offset])
            offset += n_nodes

        if not all_pts:
            return pv.PolyData()
        return pv.PolyData(
            np.array(all_pts, dtype=np.float32),
            np.array(all_faces, dtype=np.int64),
        )

    # ── gmsh backend ──────────────────────────────────────────────────────

    def _load_gmsh(self) -> None:
        import gmsh                                  # raises ImportError
        if self._gmsh_active:
            gmsh.finalize()
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(self._filepath)
        self._gmsh_active = True
        self._analyse_faces_gmsh()

    def _analyse_faces_gmsh(self) -> None:
        import gmsh
        surfaces = gmsh.model.getEntities(dim=2)
        self._faces = []

        for dim, tag in surfaces:
            # Area
            try:
                area = gmsh.model.occ.getMass(dim, tag)
            except Exception:
                area = 0.0

            # Parametric bounds
            try:
                bounds = gmsh.model.getParametrizationBounds(dim, tag)
                umin, vmin = bounds[0]
                umax, vmax = bounds[1]
            except Exception:
                continue

            umid = 0.5 * (umin + umax)
            vmid = 0.5 * (vmin + vmax)

            # Centroid (evaluate at parametric center)
            try:
                pt = gmsh.model.getValue(dim, tag, [umid, vmid])
                centroid = np.array(pt[:3], dtype=np.float64)
            except Exception:
                continue

            # Surface normal at centroid
            try:
                nml = gmsh.model.getNormal(tag, [umid, vmid])
                normal = np.array(nml[:3], dtype=np.float64)
                nlen = np.linalg.norm(normal)
                if nlen > 1e-10:
                    normal /= nlen
                else:
                    normal = np.array([0., 0., 1.])
            except Exception:
                normal = np.array([0., 0., 1.])

            self._faces.append({
                'area': area,
                'centroid': centroid,
                'normal': normal,
                'gmsh_tag': tag,
                'gmsh_dim': dim,
                'umin': umin, 'umax': umax,
                'vmin': vmin, 'vmax': vmax,
            })

    def _discretize_face_gmsh(
        self, finfo: dict, n_pts: int | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        import gmsh
        if n_pts is None:
            n_pts = self._n_disc

        tag  = finfo['gmsh_tag']
        dim  = finfo['gmsh_dim']
        umin, umax = finfo['umin'], finfo['umax']
        vmin, vmax = finfo['vmin'], finfo['vmax']

        u_span = umax - umin
        v_span = vmax - vmin

        coords, normals = [], []

        if u_span >= v_span:
            vmid = 0.5 * (vmin + vmax)
            for u in np.linspace(umin, umax, n_pts):
                try:
                    pt  = gmsh.model.getValue(dim, tag, [u, vmid])
                    nml = gmsh.model.getNormal(tag, [u, vmid])
                    coords.append(pt[:3])
                    normals.append(nml[:3])
                except Exception:
                    if coords:
                        coords.append(coords[-1])
                        normals.append(normals[-1])
        else:
            umid = 0.5 * (umin + umax)
            for v in np.linspace(vmin, vmax, n_pts):
                try:
                    pt  = gmsh.model.getValue(dim, tag, [umid, v])
                    nml = gmsh.model.getNormal(tag, [umid, v])
                    coords.append(pt[:3])
                    normals.append(nml[:3])
                except Exception:
                    if coords:
                        coords.append(coords[-1])
                        normals.append(normals[-1])

        if len(coords) < 3:
            return None, None
        return (
            np.array(coords, dtype=np.float64),
            np.array(normals, dtype=np.float64),
        )

    def _gmsh_to_pyvista(self):
        try:
            import pyvista as pv
            import gmsh
        except ImportError:
            return None

        gmsh.model.mesh.generate(2)
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        points = np.array(node_coords).reshape(-1, 3).astype(np.float32)

        tag_to_idx = {}
        for i, t in enumerate(node_tags):
            tag_to_idx[int(t)] = i

        faces_list = []
        elem_types, _, elem_nodes = gmsh.model.mesh.getElements(dim=2)
        for etype, enodes in zip(elem_types, elem_nodes):
            if etype == 2:                           # triangle
                tris = np.array(enodes, dtype=int).reshape(-1, 3)
                for tri in tris:
                    idx = [tag_to_idx.get(int(v)) for v in tri]
                    if None not in idx:
                        faces_list.extend([3, *idx])

        if not faces_list:
            return pv.PolyData()
        return pv.PolyData(points, np.array(faces_list, dtype=np.int64))

    # ── Backend-agnostic methods ──────────────────────────────────────────

    def get_mesh(self) -> 'pv.PolyData':
        """Tessellate the bobbin solid into a PyVista mesh for rendering."""
        if self._mesh is not None:
            return self._mesh
        if self._backend == 'gmsh':
            self._mesh = self._gmsh_to_pyvista()
        else:
            self._mesh = self._shape_to_pyvista(self._shape)
        return self._mesh

    # ── Curvature-refined mesh for groove detection ─────────────────────

    def get_refined_mesh(self, curvature_samples: int = 20):
        """Generate a curvature-adapted mesh with per-surface face tags.

        Each triangle knows which CAD surface it belongs to (stored in
        ``cell_data['SurfaceTag']``).  The segmentation step uses these
        tags as hard barriers — the BFS never crosses an edge between
        triangles from different CAD surfaces.  This is the key to
        isolating individual groove floors.

        Returns a PyVista PolyData with cell/point normals and
        ``SurfaceTag`` cell data.
        """
        if self._refined_mesh is not None:
            return self._refined_mesh
        if self._backend != 'gmsh' or not self._gmsh_active:
            m = self.get_mesh()
            if m is not None and m.n_cells > 0:
                m = m.compute_normals(
                    cell_normals=True, point_normals=True, inplace=False)
            return m

        import gmsh
        try:
            import pyvista as pv
        except ImportError:
            return None

        # Auto-estimate max element size from bounding box
        bb = gmsh.model.getBoundingBox(-1, -1)
        diag = ((bb[3]-bb[0])**2
                + (bb[4]-bb[1])**2
                + (bb[5]-bb[2])**2) ** 0.5
        max_size = diag / 50

        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",
                              curvature_samples)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin",
                              max_size / 10)

        gmsh.model.mesh.generate(2)

        # Build mesh PER SURFACE so we can tag each triangle
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        points = np.array(node_coords).reshape(-1, 3).astype(np.float32)
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

        faces_pv: list[int] = []
        cell_stags: list[int] = []       # CAD surface tag per triangle

        for dim, surf_tag in gmsh.model.getEntities(dim=2):
            etypes, _, enodes = gmsh.model.mesh.getElements(dim, surf_tag)
            for etype, en in zip(etypes, enodes):
                if etype == 2:
                    tris = np.array(en, dtype=int).reshape(-1, 3)
                    for tri in tris:
                        idx = [tag_to_idx.get(int(v)) for v in tri]
                        if None not in idx:
                            faces_pv.extend([3, *idx])
                            cell_stags.append(surf_tag)

        if not faces_pv:
            self._refined_mesh = pv.PolyData()
        else:
            self._refined_mesh = pv.PolyData(
                points, np.array(faces_pv, dtype=np.int64))
            self._refined_mesh.cell_data['SurfaceTag'] = np.array(
                cell_stags, dtype=np.int32)

        if self._refined_mesh.n_cells > 0:
            self._refined_mesh = self._refined_mesh.compute_normals(
                cell_normals=True, point_normals=True, inplace=False)

        # Reset gmsh mesh settings so get_mesh() uses defaults
        gmsh.model.mesh.clear()
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1e22)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0)

        return self._refined_mesh

    def segment_refined_mesh(self, angle_deg: float = 30.0,
                             curvature_multiplier: float = 0.0):
        """Segment the refined mesh using CAD surface boundaries and
        optional curvature-based groove separation.

        Barriers (BFS never crosses):
        1. **CAD surface boundaries** — edges between different
           ``SurfaceTag`` values.
        2. **Dihedral angle** — edges where the angle exceeds
           *angle_deg*.
        3. **Curvature ridges** (when *curvature_multiplier* > 0) —
           cells whose absolute mean curvature exceeds
           ``median + multiplier * IQR`` are marked as barriers.
           Divider ridges between parallel grooves have much higher
           curvature than the groove floors, so this cleanly separates
           adjacent channels that share the same B-spline surface.

        Returns ``(mesh, labels, n_patches)``.
        """
        mesh = self.get_refined_mesh()
        if mesh is None or mesh.n_cells == 0:
            return None, None, 0

        stags = mesh.cell_data.get('SurfaceTag')
        adj, fnormals, _ = _build_mesh_adjacency(mesh)
        n_cells = mesh.n_cells
        cos_thresh = np.cos(np.radians(angle_deg))

        # Curvature barrier mask
        curv_barrier = np.zeros(n_cells, dtype=bool)
        if curvature_multiplier > 0:
            curv_barrier = _compute_curvature_barriers(
                mesh, curvature_multiplier)

        labels = -np.ones(n_cells, dtype=np.int32)
        current_label = 0
        for start in range(n_cells):
            if labels[start] >= 0:
                continue
            # Don't start a region from a barrier cell
            if curv_barrier[start]:
                continue
            labels[start] = current_label
            start_stag = stags[start] if stags is not None else -1
            queue = [start]
            head = 0
            while head < len(queue):
                face = queue[head]
                head += 1
                for nb, cos_d in adj[face]:
                    if labels[nb] >= 0:
                        continue
                    # Curvature barrier — ridge between grooves
                    if curv_barrier[nb]:
                        continue
                    # HARD BARRIER: never cross CAD surface boundaries
                    if stags is not None and stags[nb] != start_stag:
                        continue
                    # Soft barrier: dihedral angle within same surface
                    if cos_d >= cos_thresh:
                        labels[nb] = current_label
                        queue.append(nb)
            current_label += 1

        # Label remaining barrier cells with nearest neighbour patch
        unlabelled = np.where(labels < 0)[0]
        for ci in unlabelled:
            for nb, _ in adj[ci]:
                if labels[nb] >= 0:
                    labels[ci] = labels[nb]
                    break
            if labels[ci] < 0:
                labels[ci] = current_label
                current_label += 1

        return mesh, labels, current_label

    def extract_patch_with_cad_normals(
        self, mesh, labels, patch_ids, n_target: int = 500,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Extract centerline from mesh patches, then evaluate exact
        CAD surface normals via ``gmsh.model.getClosestPoint`` +
        ``gmsh.model.getNormal``.

        The mesh geometry determines WHERE the centerline goes (it
        follows the groove floor), while the CAD parametric surface
        provides the exact normal vectors.
        """
        mask = np.isin(labels, list(patch_ids))
        sub = mesh.extract_cells(np.where(mask)[0])
        if hasattr(sub, 'extract_surface'):
            sub = sub.extract_surface()
        if sub.n_points < 3:
            return None, None

        sub = sub.compute_normals(
            cell_normals=False, point_normals=True, inplace=False)

        pts = np.asarray(sub.points, dtype=np.float64)
        norms = sub.point_data.get('Normals')
        if norms is not None:
            norms = np.asarray(norms, dtype=np.float64)

        coords, mesh_normals = _extract_centerline(pts, norms, n_target)
        if coords is None or len(coords) < 3:
            return None, None

        # Project onto CAD surface for exact normals
        if self._backend == 'gmsh' and self._gmsh_active:
            result = self._project_normals_gmsh(coords)
            if result is not None:
                return result

        # Fallback: mesh-derived normals
        if mesh_normals is not None:
            for i in range(1, len(mesh_normals)):
                if np.dot(mesh_normals[i], mesh_normals[i - 1]) < 0:
                    mesh_normals[i] = -mesh_normals[i]
        return coords, mesh_normals

    # ── Click-seeded region growing ─────────────────────────────────────

    def get_mesh_with_adjacency(self):
        """Return face-tagged mesh with pre-computed adjacency.

        Caches both the tessellated mesh and the adjacency / dihedral
        data.  Call once; reused for all subsequent region-grow clicks.

        Returns ``(mesh, adj, face_normals, cell_centers)`` or four Nones.
        """
        if self._segmented_mesh is None:
            ft = self.get_face_tagged_mesh()
            if ft is None or ft.n_cells == 0:
                return None, None, None, None
            self._segmented_mesh = ft.compute_normals(
                cell_normals=True, point_normals=True, inplace=False)

        if self._mesh_adj is None:
            self._mesh_adj, self._mesh_fnormals, self._mesh_ccenters = \
                _build_mesh_adjacency(self._segmented_mesh)

        return (self._segmented_mesh, self._mesh_adj,
                self._mesh_fnormals, self._mesh_ccenters)

    def mesh_diagonal(self) -> float:
        """Bounding-box diagonal of the cached tessellation."""
        if self._segmented_mesh is None:
            return 1.0
        return float(self._segmented_mesh.length)

    def grow_region_from_cell(
        self, seed_cell: int, edge_angle_deg: float = 20.0,
        seed_angle_deg: float = 55.0,
        max_path_dist: float = float('inf'),
    ) -> list[int]:
        """Region-grow from a mesh triangle.

        Parameters
        ----------
        seed_cell : int
            Triangle index to grow from.
        edge_angle_deg : float
            Max dihedral angle across a shared edge to cross.
        seed_angle_deg : float
            Max angular deviation from the seed face normal.
        max_path_dist : float
            Max accumulated centroid-to-centroid path distance
            from the seed.  Prevents reaching adjacent grooves
            through smooth groove terminations.

        Returns a list of cell indices.
        """
        mesh, adj, fnormals, ccenters = self.get_mesh_with_adjacency()
        if adj is None:
            return [seed_cell]
        edge_cos = np.cos(np.radians(edge_angle_deg))
        seed_cos = np.cos(np.radians(seed_angle_deg))
        return _grow_from_seed(
            seed_cell, adj, fnormals, ccenters,
            edge_cos, seed_cos, max_path_dist)

    def grow_groove_from_cell(
        self, seed_cell: int, edge_angle_deg: float = 20.0,
        seed_angle_deg: float = 55.0,
        max_width: float = float('inf'),
    ) -> list[int]:
        """Width-constrained groove BFS from a mesh triangle.

        Like ``grow_region_from_cell`` but adds a cross-groove width
        limit.  PCA on the growing region determines the groove
        direction; cells that would make the region wider than
        *max_width* perpendicular to the groove are rejected.

        Parameters
        ----------
        max_width : float
            Maximum groove width in mesh coordinate units.
            ``inf`` disables the constraint.
        """
        mesh, adj, fnormals, ccenters = self.get_mesh_with_adjacency()
        if adj is None:
            return [seed_cell]
        edge_cos = np.cos(np.radians(edge_angle_deg))
        seed_cos = np.cos(np.radians(seed_angle_deg))
        return _grow_groove(
            seed_cell, adj, fnormals, ccenters,
            edge_cos, seed_cos, max_width)

    def extract_region_centerline(
        self, mesh, cell_indices, n_target: int = 500,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Extract centerline + normals from a set of mesh triangles.

        Point normals are computed on just the extracted sub-mesh so
        wall / divider-top faces don't contaminate the averaging.
        """
        if not cell_indices:
            return None, None
        idx_arr = np.array(sorted(set(cell_indices)), dtype=int)
        sub = mesh.extract_cells(idx_arr)
        if sub.n_points < 3:
            return None, None

        # extract_cells returns UnstructuredGrid — convert to PolyData
        # so compute_normals is available.
        if hasattr(sub, 'extract_surface'):
            sub = sub.extract_surface()

        sub = sub.compute_normals(
            cell_normals=False, point_normals=True, inplace=False)

        pts = np.asarray(sub.points, dtype=np.float64)
        norms = sub.point_data.get('Normals')
        if norms is not None:
            norms = np.asarray(norms, dtype=np.float64)

        coords, normals = _extract_centerline(pts, norms, n_target)
        if coords is None or len(coords) < 3:
            return None, None

        # Orient normals consistently along the path
        if normals is not None:
            for i in range(1, len(normals)):
                if np.dot(normals[i], normals[i - 1]) < 0:
                    normals[i] = -normals[i]

        return coords, normals

    def extract_region_with_cad_normals(
        self, mesh, cell_indices, n_target: int = 500,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Extract centerline from cell indices with exact CAD normals.

        Uses mesh geometry for the centerline path, then projects onto
        the CAD surface for exact parametric normals via gmsh.
        Falls back to mesh-derived normals if projection fails.
        """
        coords, mesh_normals = self.extract_region_centerline(
            mesh, cell_indices, n_target)
        if coords is None:
            return None, None

        # Try exact CAD projection
        if self._backend == 'gmsh' and self._gmsh_active:
            result = self._project_normals_gmsh(coords)
            if result is not None:
                return result

        return coords, mesh_normals

    def project_and_evaluate_normals(
        self, seed_coords: np.ndarray, n_resample: int = 500,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit a spline through seed points and evaluate exact CAD normals.

        Pipeline
        --------
        1. Fit a smooth spline through the user-placed seed points and
           resample at *n_resample* evenly-spaced points.
        2. Project each resampled point onto the nearest CAD surface
           using ``gmsh.model.getClosestPoint`` (exact geometry, not
           the broken tessellation mesh).
        3. Evaluate the parametric surface normal at each projected
           point via ``gmsh.model.getNormal``.

        Falls back to path-derived normals (parallel transport) when
        gmsh projection is unavailable.

        Returns ``(coords, normals)`` — both ``(M, 3)`` float64.
        """
        try:
            import pyvista as pv
        except ImportError:
            return seed_coords, np.tile([0., 0., 1.], (len(seed_coords), 1))

        if len(seed_coords) < 2:
            return seed_coords, np.tile([0., 0., 1.], (len(seed_coords), 1))

        # Spline fit + resample
        spline = pv.Spline(seed_coords, n_points=n_resample)
        resampled = np.asarray(spline.points, dtype=np.float64)

        # Try exact CAD projection + normals
        if self._backend == 'gmsh' and self._gmsh_active:
            result = self._project_normals_gmsh(resampled)
            if result is not None:
                return result

        # Fallback: parallel-transport normals from path geometry
        normals = self._fallback_normals(resampled)
        return resampled, normals

    def _project_normals_gmsh(self, coords):
        """Project onto CAD surfaces and evaluate exact normals via gmsh."""
        try:
            import gmsh
        except ImportError:
            return None

        surfaces = gmsh.model.getEntities(dim=2)
        if not surfaces:
            return None

        n = len(coords)
        projected = np.zeros_like(coords)
        normals = np.zeros_like(coords)
        last_tag = None       # spatial coherence — try last-best first

        for i in range(n):
            pt = coords[i].tolist()
            best_dist = float('inf')
            best_proj = pt
            best_nml = [0., 0., 1.]

            # Check last-best surface first (usually still closest)
            tags = []
            if last_tag is not None:
                tags.append(last_tag)
            tags.extend(t for _, t in surfaces if t != last_tag)

            for tag in tags:
                try:
                    cp, uv = gmsh.model.getClosestPoint(2, tag, pt)
                    d = ((cp[0]-pt[0])**2
                         + (cp[1]-pt[1])**2
                         + (cp[2]-pt[2])**2) ** 0.5
                    if d < best_dist:
                        best_dist = d
                        best_proj = cp[:3]
                        nml = gmsh.model.getNormal(tag, list(uv))
                        best_nml = nml[:3]
                        last_tag = tag
                        if d < 1e-8:
                            break              # already on surface
                except Exception:
                    continue

            projected[i] = best_proj
            nv = np.array(best_nml, dtype=np.float64)
            nl = np.linalg.norm(nv)
            normals[i] = nv / nl if nl > 1e-10 else [0., 0., 1.]

        # Orient normals consistently along the path
        for i in range(1, n):
            if np.dot(normals[i], normals[i - 1]) < 0:
                normals[i] = -normals[i]

        return projected, normals

    @staticmethod
    def _fallback_normals(coords):
        """Compute normals from path geometry (parallel transport)."""
        try:
            from physics.geometry import compute_frenet_frame
        except ImportError:
            try:
                from .geometry import compute_frenet_frame
            except ImportError:
                return np.tile([0., 0., 1.], (len(coords), 1))
        frame = compute_frenet_frame(coords)
        return frame['normal']

    # ── Per-face discretization (legacy, used by detect_channels) ─────

    def _discretize_face(
        self, finfo: dict, n_pts: int | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self._backend == 'gmsh':
            return self._discretize_face_gmsh(finfo, n_pts)
        return self._discretize_face_occ(finfo, n_pts)

    def detect_channels(
        self,
        area_fraction_max: float = 0.15,
        aspect_ratio_min: float = 3.0,
    ) -> list[BobbinChannel]:
        """
        Auto-detect groove/channel faces, group adjacent faces into
        continuous grooves, and return one BobbinChannel per groove.

        Pipeline
        --------
        1. Identify candidate faces (small area, inward-pointing normal).
        2. Group adjacent candidates into connected grooves (gmsh uses
           shared boundary curves; cadquery falls back to individual faces).
        3. Order faces within each group for path continuity.
        4. Concatenate discretised centerlines + normals.
        5. Merge any remaining close-endpoint channels.
        """
        if not self._faces:
            return []

        # Body centroid (area-weighted average of face centroids)
        areas = np.array([f['area'] for f in self._faces])
        centroids = np.array([f['centroid'] for f in self._faces])
        total_a = areas.sum()
        if total_a > 0:
            body_centroid = (centroids * areas[:, None]).sum(axis=0) / total_a
        else:
            body_centroid = centroids.mean(axis=0)

        max_area = max(f['area'] for f in self._faces)
        area_thresh = area_fraction_max * max_area

        # Step 1 — candidate faces
        candidates = []
        for i, finfo in enumerate(self._faces):
            if finfo['area'] > area_thresh or finfo['area'] < 1e-12:
                continue
            to_center = body_centroid - finfo['centroid']
            to_center_n = to_center / max(np.linalg.norm(to_center), 1e-12)
            if np.dot(finfo['normal'], to_center_n) < 0.1:
                continue
            candidates.append(i)

        if not candidates:
            return []

        # Step 2 — group adjacent faces
        if self._backend == 'gmsh' and self._gmsh_active:
            groups = self._group_faces_gmsh(candidates)
        else:
            groups = [[c] for c in candidates]

        # Step 3+4 — order & concatenate each group
        channels = []
        for group in groups:
            ordered = self._order_group(group)
            c_parts, n_parts = [], []
            for idx in ordered:
                c, n = self._discretize_face(self._faces[idx])
                if c is None:
                    continue
                # Reverse segment if needed for continuity
                if c_parts:
                    prev_end = c_parts[-1][-1]
                    if np.linalg.norm(c[-1] - prev_end) < np.linalg.norm(c[0] - prev_end):
                        c, n = c[::-1], n[::-1]
                c_parts.append(c)
                n_parts.append(n)
            if not c_parts:
                continue
            channels.append(BobbinChannel(
                channel_id=f"groove_{len(channels)+1}",
                coords=np.vstack(c_parts),
                normals=np.vstack(n_parts),
            ))

        # Step 5 — merge channels whose endpoints nearly touch
        channels = self._merge_sequential_channels(channels)
        # Re-number
        for i, ch in enumerate(channels):
            ch.channel_id = f"groove_{i+1}"

        return channels

    def get_all_faces_as_channels(self) -> list[BobbinChannel]:
        """Return every face as a candidate channel for manual selection."""
        channels = []
        for i, finfo in enumerate(self._faces):
            try:
                coords, normals = self._discretize_face(finfo)
                if coords is not None and len(coords) >= 3:
                    channels.append(BobbinChannel(
                        channel_id=f"face_{i+1}",
                        coords=coords,
                        normals=normals,
                        _face_ref=finfo.get('face'),
                    ))
            except Exception:
                continue
        return channels

    # ── Face grouping helpers ─────────────────────────────────────────────

    def _group_faces_gmsh(self, candidate_indices: list[int]) -> list[list[int]]:
        """Group candidate faces that share boundary curves (gmsh)."""
        import gmsh
        # Boundary curves for each candidate face
        face_curves: dict[int, set] = {}
        for idx in candidate_indices:
            tag = self._faces[idx]['gmsh_tag']
            try:
                bounds = gmsh.model.getBoundary(
                    [(2, tag)], oriented=False, recursive=False)
                face_curves[idx] = {abs(b[1]) for b in bounds}
            except Exception:
                face_curves[idx] = set()

        # Adjacency: faces sharing ≥1 curve
        adj: dict[int, set] = {i: set() for i in candidate_indices}
        clist = list(candidate_indices)
        for i in range(len(clist)):
            for j in range(i + 1, len(clist)):
                a, b = clist[i], clist[j]
                if face_curves[a] & face_curves[b]:
                    adj[a].add(b)
                    adj[b].add(a)

        # BFS connected components
        visited: set[int] = set()
        groups: list[list[int]] = []
        for start in candidate_indices:
            if start in visited:
                continue
            comp: list[int] = []
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                comp.append(node)
                queue.extend(nb for nb in adj[node] if nb not in visited)
            groups.append(comp)
        return groups

    def _order_group(self, group: list[int]) -> list[int]:
        """Order faces in a group so the discretised path is continuous."""
        if len(group) <= 1:
            return group

        # Sample 3 points per face to get start/end positions
        eps: dict[int, tuple] = {}
        for idx in group:
            c, _ = self._discretize_face(self._faces[idx], n_pts=5)
            if c is not None and len(c) >= 2:
                eps[idx] = (c[0].copy(), c[-1].copy())
        if not eps:
            return group

        # Greedy chain from an arbitrary start
        ordered = [group[0]]
        remaining = set(group[1:])
        while remaining:
            curr = ordered[-1]
            if curr not in eps:
                break
            curr_end = eps[curr][1]
            best, best_d = None, float('inf')
            for r in remaining:
                if r not in eps:
                    continue
                d = min(np.linalg.norm(curr_end - eps[r][0]),
                        np.linalg.norm(curr_end - eps[r][1]))
                if d < best_d:
                    best_d, best = d, r
            if best is not None:
                ordered.append(best)
                remaining.remove(best)
            else:
                break
        ordered.extend(remaining)
        return ordered

    def _merge_sequential_channels(
        self, channels: list[BobbinChannel],
    ) -> list[BobbinChannel]:
        """Merge channels whose endpoints are close into continuous paths."""
        if len(channels) < 2:
            return channels

        seg_lens = [
            np.linalg.norm(np.diff(ch.coords, axis=0), axis=1).mean()
            for ch in channels if len(ch.coords) > 1
        ]
        merge_tol = np.mean(seg_lens) * 5.0 if seg_lens else 1.0

        merged = list(channels)
        changed = True
        while changed:
            changed = False
            for i in range(len(merged)):
                for j in range(i + 1, len(merged)):
                    ci, cj = merged[i], merged[j]
                    dists = [
                        np.linalg.norm(ci.coords[-1] - cj.coords[0]),
                        np.linalg.norm(ci.coords[-1] - cj.coords[-1]),
                        np.linalg.norm(ci.coords[0]  - cj.coords[0]),
                        np.linalg.norm(ci.coords[0]  - cj.coords[-1]),
                    ]
                    md = min(dists)
                    if md >= merge_tol:
                        continue
                    case = dists.index(md)
                    if case == 0:
                        nc = np.vstack([ci.coords, cj.coords])
                        nn = np.vstack([ci.normals, cj.normals])
                    elif case == 1:
                        nc = np.vstack([ci.coords, cj.coords[::-1]])
                        nn = np.vstack([ci.normals, cj.normals[::-1]])
                    elif case == 2:
                        nc = np.vstack([ci.coords[::-1], cj.coords])
                        nn = np.vstack([ci.normals[::-1], cj.normals])
                    else:
                        nc = np.vstack([cj.coords, ci.coords])
                        nn = np.vstack([cj.normals, ci.normals])
                    merged[i] = BobbinChannel(
                        channel_id=ci.channel_id, coords=nc, normals=nn)
                    merged.pop(j)
                    changed = True
                    break
                if changed:
                    break
        return merged

    # ── Interactive face selection ────────────────────────────────────────

    def get_face_tagged_mesh(self) -> 'pv.PolyData':
        """Return a PyVista mesh with ``FaceId`` cell data.

        Each triangle knows which CAD face (index into ``self._faces``)
        it belongs to.  Used by the interactive face picker.
        """
        if self._backend == 'gmsh':
            return self._gmsh_face_tagged_mesh()
        return self._occ_face_tagged_mesh()

    def _gmsh_face_tagged_mesh(self):
        import gmsh
        try:
            import pyvista as pv
        except ImportError:
            return None
        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        points = np.array(node_coords).reshape(-1, 3).astype(np.float32)
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

        gmsh_tag_to_fi = {}
        for i, f in enumerate(self._faces):
            t = f.get('gmsh_tag')
            if t is not None:
                gmsh_tag_to_fi[t] = i

        faces_pv: list[int] = []
        cell_fids: list[int] = []
        for dim, tag in gmsh.model.getEntities(dim=2):
            fi = gmsh_tag_to_fi.get(tag, -1)
            etypes, _, enodes = gmsh.model.mesh.getElements(dim, tag)
            for etype, en in zip(etypes, enodes):
                if etype == 2:
                    tris = np.array(en, dtype=int).reshape(-1, 3)
                    for tri in tris:
                        idx = [tag_to_idx.get(int(v)) for v in tri]
                        if None not in idx:
                            faces_pv.extend([3, *idx])
                            cell_fids.append(fi)

        mesh = pv.PolyData(points, np.array(faces_pv, dtype=np.int64))
        mesh.cell_data['FaceId'] = np.array(cell_fids, dtype=np.int32)
        return mesh

    def _occ_face_tagged_mesh(self):
        """Face-tagged mesh for the cadquery backend."""
        try:
            import pyvista as pv
        except ImportError:
            return None
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE
        from OCP.BRep import BRep_Tool
        from OCP.TopLoc import TopLoc_Location

        BRepMesh_IncrementalMesh(self._shape, 0.05)
        all_pts, all_faces, all_fids = [], [], []
        offset = 0
        face_idx = 0
        explorer = TopExp_Explorer(self._shape, TopAbs_FACE)
        while explorer.More():
            face = explorer.Current()
            explorer.Next()
            location = TopLoc_Location()
            tri = BRep_Tool.Triangulation_s(face, location)
            if tri is None:
                face_idx += 1
                continue
            trsf = location.Transformation()
            for i in range(1, tri.NbNodes() + 1):
                pt = tri.Node(i).Transformed(trsf)
                all_pts.append([pt.X(), pt.Y(), pt.Z()])
            for i in range(1, tri.NbTriangles() + 1):
                t = tri.Triangle(i)
                i1, i2, i3 = t.Get()
                all_faces.extend([3, i1 - 1 + offset,
                                  i2 - 1 + offset, i3 - 1 + offset])
                all_fids.append(face_idx)
            offset += tri.NbNodes()
            face_idx += 1

        if not all_pts:
            return pv.PolyData()
        mesh = pv.PolyData(
            np.array(all_pts, dtype=np.float32),
            np.array(all_faces, dtype=np.int64))
        mesh.cell_data['FaceId'] = np.array(all_fids, dtype=np.int32)
        return mesh

    def build_adjacency(self) -> None:
        """Pre-compute face adjacency graph (faces sharing an edge)."""
        n = len(self._faces)
        self._adj: dict[int, set[int]] = {i: set() for i in range(n)}
        if self._backend == 'gmsh' and self._gmsh_active:
            self._build_adjacency_gmsh()

    def _build_adjacency_gmsh(self) -> None:
        import gmsh
        from collections import defaultdict
        curve_to_fi: dict[int, list[int]] = defaultdict(list)
        for i, f in enumerate(self._faces):
            tag = f.get('gmsh_tag')
            if tag is None:
                continue
            try:
                bounds = gmsh.model.getBoundary(
                    [(2, tag)], oriented=False, recursive=False)
                for b in bounds:
                    curve_to_fi[abs(b[1])].append(i)
            except Exception:
                pass
        for fis in curve_to_fi.values():
            for a in range(len(fis)):
                for b in range(a + 1, len(fis)):
                    self._adj[fis[a]].add(fis[b])
                    self._adj[fis[b]].add(fis[a])

    def grow_region(self, start_idx: int, angle_deg: float = 20.0) -> list[int]:
        """
        BFS region-growing from a face, stopping at sharp normal
        transitions.  Compares each candidate with its *parent* face
        normal so that gradually curving grooves are followed correctly.

        Default 20° connects adjacent groove-floor patches (which
        differ by ~5–15° on a curved bobbin) while firmly blocking
        groove walls (60–90° transitions) and end transitions.
        """
        if not hasattr(self, '_adj') or start_idx not in self._adj:
            return [start_idx]
        cos_thresh = np.cos(np.radians(angle_deg))

        visited = {start_idx}
        region = [start_idx]
        queue = [(start_idx, self._faces[start_idx]['normal'])]

        while queue:
            fi, parent_n = queue.pop(0)
            for nb in self._adj.get(fi, []):
                if nb in visited:
                    continue
                visited.add(nb)
                nb_n = self._faces[nb]['normal']
                # Use abs() — STEP face orientation flags (FORWARD /
                # REVERSED) can flip normals on adjacent faces of the
                # same smooth surface.  We only care about the angle,
                # not the direction.
                if abs(np.dot(parent_n, nb_n)) >= cos_thresh:
                    region.append(nb)
                    # Propagate a consistently oriented normal
                    oriented_n = nb_n if np.dot(parent_n, nb_n) >= 0 else -nb_n
                    queue.append((nb, oriented_n))
        return region

    def discretize_face_group(
        self, face_indices: list[int],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Order and concatenate discretised centerlines for a face group.

        Normals are oriented consistently along the path (STEP face
        orientation flags can flip them arbitrarily).
        """
        if not face_indices:
            return None, None
        ordered = self._order_group(face_indices)
        c_parts, n_parts = [], []
        for idx in ordered:
            c, n = self._discretize_face(self._faces[idx])
            if c is None:
                continue
            if c_parts:
                prev_end = c_parts[-1][-1]
                if (np.linalg.norm(c[-1] - prev_end)
                        < np.linalg.norm(c[0] - prev_end)):
                    c, n = c[::-1], n[::-1]
            c_parts.append(c)
            n_parts.append(n)
        if not c_parts:
            return None, None

        coords = np.vstack(c_parts)
        normals = np.vstack(n_parts)

        # Orient normals consistently — propagate from the first point.
        # STEP FORWARD/REVERSED flags can arbitrarily flip face normals.
        for i in range(1, len(normals)):
            if np.dot(normals[i], normals[i - 1]) < 0:
                normals[i] = -normals[i]

        return coords, normals

    # ── Parametric groove detection ─────────────────────────────────────

    def detect_grooves_on_surface(
        self, surface_idx: int,
        n_along: int = 300, n_cross: int = 200,
        progress_cb=None,
    ) -> list[BobbinChannel]:
        """Auto-detect parallel grooves on a CAD surface.

        Bypasses the mesh entirely — works directly with the parametric
        surface representation via ``gmsh.model.getValue`` and
        ``gmsh.model.getNormal``.

        Pipeline
        --------
        1. Sample the surface on a dense (u, v) grid.
        2. Compute **local curvature** (magnitude of second derivative
           of position) in each parametric direction.
        3. The cross-groove direction is the one with higher averaged
           curvature — groove walls create curvature spikes.
        4. Find curvature peaks in the averaged cross-groove profile →
           these are groove wall locations.
        5. Between consecutive walls = one groove floor.  The
           centerline is the midpoint between walls.
        6. Trace each centerline along the groove direction, tracking
           the local curvature-minimum at each position.
        7. Evaluate exact 3-D coordinates and surface normals from
           the parametric surface.

        This is purely local — independent of the bobbin's overall
        shape.  A 1 mm groove on a 100 mm D-shaped bobbin is detected
        just as cleanly as on a flat plate.

        Returns one ``BobbinChannel`` per detected groove.
        """
        if self._backend != 'gmsh' or not self._gmsh_active:
            return []
        import gmsh

        finfo = self._faces[surface_idx]
        tag = finfo['gmsh_tag']
        dim = finfo['gmsh_dim']
        umin, umax = finfo['umin'], finfo['umax']
        vmin, vmax = finfo['vmin'], finfo['vmax']

        us = np.linspace(umin, umax, n_along)
        vs = np.linspace(vmin, vmax, n_cross)

        # ── 1. Sample surface on parametric grid ──────────────────────
        grid = np.zeros((n_along, n_cross, 3), dtype=np.float64)
        for i, u in enumerate(us):
            if progress_cb and i % 50 == 0:
                progress_cb(int(30 * i / n_along))
            for j, v in enumerate(vs):
                try:
                    pt = gmsh.model.getValue(dim, tag, [u, v])
                    grid[i, j] = pt[:3]
                except Exception:
                    grid[i, j] = grid[max(i - 1, 0), max(j - 1, 0)]

        # ── 2. Local curvature in each parametric direction ───────────
        # Second derivative magnitude — purely local, independent of
        # overall bobbin shape.  Groove walls are curvature spikes.
        #   axis 0 = u direction, axis 1 = v direction
        d2_u = np.linalg.norm(np.diff(grid, n=2, axis=0), axis=2)
        d2_v = np.linalg.norm(np.diff(grid, n=2, axis=1), axis=2)

        # Average curvature profiles (collapse along the other axis)
        curv_u_profile = d2_u.mean(axis=1)   # (n_along-2,) — curvature in u
        curv_v_profile = d2_v.mean(axis=0)   # (n_cross-2,) — curvature in v

        # ── 3. Cross-groove direction ─────────────────────────────────
        # The cross-groove direction has periodic curvature spikes
        # (groove walls).  Measure the max/median ratio — the
        # direction with sharper spikes is cross-groove.
        def _spikiness(prof):
            med = np.median(prof)
            if med < 1e-15:
                return 0.0
            return float(prof.max() / med)

        u_spiky = _spikiness(curv_u_profile)
        v_spiky = _spikiness(curv_v_profile)

        if v_spiky > u_spiky:
            # v is cross-groove
            curv_cross = d2_v                 # (n_along, n_cross-2)
            cross_profile = curv_v_profile    # (n_cross-2,)
            cross_n = n_cross - 2
            along_n = n_along
            along_vals = us
            cross_vals = 0.5 * (vs[:-2] + vs[2:])   # centers of 2nd-diff
            transposed = False
        else:
            # u is cross-groove
            curv_cross = d2_u.T               # (n_cross, n_along-2) → transpose
            cross_profile = curv_u_profile    # (n_along-2,)
            cross_n = n_along - 2
            along_n = n_cross
            along_vals = vs
            cross_vals = 0.5 * (us[:-2] + us[2:])
            transposed = True

        if progress_cb:
            progress_cb(40)

        # ── 4. Find groove walls (curvature peaks) ────────────────────
        walls = _detect_valleys(-cross_profile)   # peaks in curvature
        if len(walls) < 2:
            # Fewer than 2 walls → can't define a groove between them
            return []

        # ── 5. Groove centers = midpoints between consecutive walls ───
        groove_centers = []
        for k in range(len(walls) - 1):
            mid = (walls[k] + walls[k + 1]) // 2
            groove_centers.append(mid)

        hw = max(3, cross_n // 40)

        channels: list[BobbinChannel] = []
        for gi, c_idx in enumerate(groove_centers):
            coords_list: list[list[float]] = []
            normals_list: list[list[float]] = []

            # Wall boundaries for this groove
            left_wall = walls[gi]
            right_wall = walls[gi + 1]
            search_lo = max(0, left_wall + 1)
            search_hi = min(cross_n, right_wall)

            for a_idx in range(along_n):
                # Find local curvature MINIMUM within this groove's
                # wall boundaries — the floor of the groove
                if transposed:
                    local_curv = curv_cross[a_idx, search_lo:search_hi]
                else:
                    local_curv = curv_cross[a_idx, search_lo:search_hi]

                if len(local_curv) == 0:
                    continue
                local_min = search_lo + int(np.argmin(local_curv))

                if transposed:
                    u_val, v_val = cross_vals[local_min], along_vals[a_idx]
                else:
                    u_val, v_val = along_vals[a_idx], cross_vals[local_min]

                try:
                    pt = gmsh.model.getValue(dim, tag, [u_val, v_val])
                    nml = gmsh.model.getNormal(tag, [u_val, v_val])
                    coords_list.append(list(pt[:3]))
                    normals_list.append(list(nml[:3]))
                except Exception:
                    if coords_list:
                        coords_list.append(coords_list[-1])
                        normals_list.append(normals_list[-1])

            if len(coords_list) < 3:
                continue

            c = np.array(coords_list, dtype=np.float64)
            n = np.array(normals_list, dtype=np.float64)

            # Normalise
            nlen = np.linalg.norm(n, axis=1, keepdims=True)
            n = n / np.maximum(nlen, 1e-10)

            # Orient normals consistently along path
            for k in range(1, len(n)):
                if np.dot(n[k], n[k - 1]) < 0:
                    n[k] = -n[k]

            # Smooth with moving average
            if len(c) > 10:
                try:
                    from scipy.ndimage import uniform_filter1d
                    kern = 5
                    for ax in range(3):
                        c[:, ax] = uniform_filter1d(
                            c[:, ax], kern, mode='nearest')
                        n[:, ax] = uniform_filter1d(
                            n[:, ax], kern, mode='nearest')
                    n /= np.maximum(
                        np.linalg.norm(n, axis=1, keepdims=True), 1e-10)
                except ImportError:
                    pass

            channels.append(BobbinChannel(
                channel_id=f"groove_{gi + 1}",
                coords=c,
                normals=n,
            ))

            if progress_cb:
                progress_cb(40 + int(50 * (gi + 1) / len(groove_centers)))

        if progress_cb:
            progress_cb(100)
        return channels

    def detect_all_grooves(self, progress_cb=None) -> list[BobbinChannel]:
        """Auto-detect grooves across all CAD surfaces.

        Runs ``detect_grooves_on_surface`` on every surface that is
        large enough to plausibly contain a groove.  Returns all
        detected channels.
        """
        if not self._faces:
            return []

        max_area = max(f['area'] for f in self._faces)
        # Only consider surfaces with significant area
        min_area = max_area * 0.02

        all_channels: list[BobbinChannel] = []
        candidates = [i for i, f in enumerate(self._faces)
                      if f['area'] >= min_area]

        for ci, idx in enumerate(candidates):
            def sub_progress(p):
                if progress_cb:
                    base = int(100 * ci / len(candidates))
                    span = int(100 / len(candidates))
                    progress_cb(base + int(span * p / 100))

            channels = self.detect_grooves_on_surface(
                idx, progress_cb=sub_progress)
            all_channels.extend(channels)

        # Re-number
        for i, ch in enumerate(all_channels):
            ch.channel_id = f"groove_{i + 1}"

        return all_channels

    # ── Face splitting (OCC kernel) ──────────────────────────────────────

    def _determine_cross_direction(
        self, face_idx: int, n_samples: int = 30,
    ) -> str:
        """Return ``'u'`` or ``'v'`` — the cross-groove parametric direction.

        Samples the surface on a coarse grid, computes the second-
        derivative magnitude in each direction, and picks the one with
        higher curvature spikiness (max / median ratio).  The cross-
        groove direction has sharp spikes at the divider ridges.
        """
        if self._backend != 'gmsh' or not self._gmsh_active:
            return 'v'
        import gmsh

        finfo = self._faces[face_idx]
        tag, dim = finfo['gmsh_tag'], finfo['gmsh_dim']
        umin, umax = finfo['umin'], finfo['umax']
        vmin, vmax = finfo['vmin'], finfo['vmax']

        us = np.linspace(umin, umax, n_samples)
        vs = np.linspace(vmin, vmax, n_samples)

        grid = np.zeros((n_samples, n_samples, 3), dtype=np.float64)
        for i, u in enumerate(us):
            for j, v in enumerate(vs):
                try:
                    pt = gmsh.model.getValue(dim, tag, [u, v])
                    grid[i, j] = pt[:3]
                except Exception:
                    grid[i, j] = grid[max(i - 1, 0), max(j - 1, 0)]

        d2_u = np.linalg.norm(np.diff(grid, n=2, axis=0), axis=2)
        d2_v = np.linalg.norm(np.diff(grid, n=2, axis=1), axis=2)

        def _spikiness(arr):
            flat = arr.flatten()
            med = np.median(flat)
            return float(flat.max() / med) if med > 1e-15 else 0.0

        return 'v' if _spikiness(d2_v) > _spikiness(d2_u) else 'u'

    def _eval_iso_curve(
        self, face_idx: int, param_value: float,
        cross_direction: str, n_pts: int = 50,
    ) -> np.ndarray | None:
        """Evaluate 3D points along an iso-parametric curve on a face.

        If *cross_direction* is ``'v'``, the curve is at constant
        ``v = param_value``, sweeping u from umin to umax (and
        vice versa).

        Returns ``(n_pts, 3)`` array or ``None``.
        """
        if self._backend != 'gmsh' or not self._gmsh_active:
            return None
        import gmsh

        finfo = self._faces[face_idx]
        tag, dim = finfo['gmsh_tag'], finfo['gmsh_dim']
        umin, umax = finfo['umin'], finfo['umax']
        vmin, vmax = finfo['vmin'], finfo['vmax']

        pts: list[list[float]] = []
        if cross_direction == 'v':
            # Constant v, sweep u
            for u in np.linspace(umin, umax, n_pts):
                try:
                    p = gmsh.model.getValue(dim, tag, [u, param_value])
                    pts.append(list(p[:3]))
                except Exception:
                    if pts:
                        pts.append(pts[-1])
        else:
            # Constant u, sweep v
            for v in np.linspace(vmin, vmax, n_pts):
                try:
                    p = gmsh.model.getValue(dim, tag, [param_value, v])
                    pts.append(list(p[:3]))
                except Exception:
                    if pts:
                        pts.append(pts[-1])

        if len(pts) < 3:
            return None
        return np.array(pts, dtype=np.float64)

    def split_face_at_params(
        self, face_idx: int,
        split_values: list[float],
        cross_direction: str,
        n_curve_pts: int = 50,
    ) -> list[int]:
        """Split a CAD face along iso-parametric lines.

        Creates B-spline curves on the surface at each *split_value*
        in the *cross_direction*, then uses ``gmsh.model.occ.fragment``
        to cut the face.  Rebuilds ``self._faces`` and invalidates
        all mesh caches.

        Returns the indices (into ``self._faces``) of the new faces.
        """
        if self._backend != 'gmsh' or not self._gmsh_active:
            raise RuntimeError("Face splitting requires the gmsh backend.")
        import gmsh

        finfo = self._faces[face_idx]
        face_tag = finfo['gmsh_tag']
        dim = finfo['gmsh_dim']

        # Sort split values so faces come out in order
        split_values = sorted(split_values)

        # ── Create splitting curves ───────────────────────────────────
        curve_tags: list[int] = []
        for sv in split_values:
            curve_pts = self._eval_iso_curve(
                face_idx, sv, cross_direction, n_curve_pts)
            if curve_pts is None or len(curve_pts) < 3:
                continue

            pt_tags = []
            for p in curve_pts:
                t = gmsh.model.occ.addPoint(
                    float(p[0]), float(p[1]), float(p[2]))
                pt_tags.append(t)

            try:
                ct = gmsh.model.occ.addBSpline(pt_tags)
            except Exception:
                # Fallback: polyline through the points
                try:
                    ct = gmsh.model.occ.addSpline(pt_tags)
                except Exception:
                    continue
            curve_tags.append(ct)

        if not curve_tags:
            raise RuntimeError(
                "Could not create any splitting curves.")

        # ── Fragment ──────────────────────────────────────────────────
        tool_dimtags = [(1, ct) for ct in curve_tags]
        result, result_map = gmsh.model.occ.fragment(
            [(2, face_tag)], tool_dimtags)
        gmsh.model.occ.synchronize()

        # New face tags from the first entry in result_map
        # (corresponds to the input face)
        new_face_tags = [
            tag for d, tag in result_map[0] if d == 2]

        if not new_face_tags:
            # fragment didn't produce new faces — try result directly
            new_face_tags = [tag for d, tag in result if d == 2]

        if not new_face_tags:
            raise RuntimeError(
                "Fragment did not produce any new faces.")

        # ── Rebuild self._faces ───────────────────────────────────────
        # Remove old face
        self._faces.pop(face_idx)

        # Analyse each new face
        new_indices: list[int] = []
        insert_at = face_idx
        for nt in sorted(new_face_tags):
            try:
                area = gmsh.model.occ.getMass(2, nt)
            except Exception:
                area = 0.0
            try:
                bounds = gmsh.model.getParametrizationBounds(2, nt)
                umin, vmin = bounds[0]
                umax, vmax = bounds[1]
            except Exception:
                continue
            umid = 0.5 * (umin + umax)
            vmid = 0.5 * (vmin + vmax)
            try:
                pt = gmsh.model.getValue(2, nt, [umid, vmid])
                centroid = np.array(pt[:3], dtype=np.float64)
            except Exception:
                continue
            try:
                nml = gmsh.model.getNormal(nt, [umid, vmid])
                normal = np.array(nml[:3], dtype=np.float64)
                nlen = np.linalg.norm(normal)
                if nlen > 1e-10:
                    normal /= nlen
                else:
                    normal = np.array([0., 0., 1.])
            except Exception:
                normal = np.array([0., 0., 1.])

            new_face = {
                'area': area,
                'centroid': centroid,
                'normal': normal,
                'gmsh_tag': nt,
                'gmsh_dim': 2,
                'umin': umin, 'umax': umax,
                'vmin': vmin, 'vmax': vmax,
            }
            self._faces.insert(insert_at, new_face)
            new_indices.append(insert_at)
            insert_at += 1

        # ── Invalidate caches ─────────────────────────────────────────
        self._mesh = None
        self._segmented_mesh = None
        self._mesh_adj = None
        self._mesh_fnormals = None
        self._mesh_ccenters = None
        self._refined_mesh = None

        return new_indices

    def split_face_equal(
        self, face_idx: int, n_parts: int,
        cross_direction: str | None = None,
    ) -> list[int]:
        """Split a face into *n_parts* equal strips.

        Auto-detects the cross-groove direction if not given.
        Computes ``n_parts - 1`` evenly spaced parametric values and
        delegates to ``split_face_at_params``.
        """
        if n_parts < 2:
            return [face_idx]

        if cross_direction is None:
            cross_direction = self._determine_cross_direction(face_idx)

        finfo = self._faces[face_idx]
        if cross_direction == 'v':
            lo, hi = finfo['vmin'], finfo['vmax']
        else:
            lo, hi = finfo['umin'], finfo['umax']

        # n_parts - 1 interior split lines, evenly spaced
        split_vals = [
            lo + (hi - lo) * k / n_parts
            for k in range(1, n_parts)
        ]
        return self.split_face_at_params(
            face_idx, split_vals, cross_direction)

    def get_face_cross_span(
        self, face_idx: int, cross_direction: str | None = None,
    ) -> float:
        """Return the parametric span of a face in the cross-groove direction."""
        if cross_direction is None:
            cross_direction = self._determine_cross_direction(face_idx)
        finfo = self._faces[face_idx]
        if cross_direction == 'v':
            return finfo['vmax'] - finfo['vmin']
        return finfo['umax'] - finfo['umin']

    def extract_channel_strip(
        self, combined_face_idx: int,
        reference_face_idx: int,
        cross_direction: str | None = None,
    ) -> tuple[list[int], int | None]:
        """Extract a single-channel strip from a combined multi-groove face.

        Uses the *reference_face_idx* (a properly isolated single-
        channel face) to determine the groove width and alignment.
        Projects the reference centerline endpoint onto the combined
        face to find the strip center, then splits with two iso-
        parametric curves at ``center ± ref_width / 2``.

        Returns ``(all_new_face_indices, strip_face_index)``.
        *strip_face_index* is the new face that corresponds to the
        channel (the one closest to the reference).  ``None`` if
        identification failed.
        """
        if self._backend != 'gmsh' or not self._gmsh_active:
            raise RuntimeError("Requires the gmsh backend.")
        import gmsh

        if cross_direction is None:
            cross_direction = self._determine_cross_direction(
                combined_face_idx)

        # Reference channel width in the cross-groove direction
        ref_span = self.get_face_cross_span(
            reference_face_idx, cross_direction)

        # Get the reference channel centerline endpoint closest
        # to the combined face
        ref_coords, _ = self._discretize_face(
            self._faces[reference_face_idx], n_pts=100)
        if ref_coords is None or len(ref_coords) < 3:
            raise RuntimeError("Could not discretize reference face.")

        comb_finfo = self._faces[combined_face_idx]
        comb_tag = comb_finfo['gmsh_tag']
        comb_centroid = comb_finfo['centroid']

        # Which endpoint of the reference is closer to the combined face?
        d_start = float(np.linalg.norm(ref_coords[0] - comb_centroid))
        d_end = float(np.linalg.norm(ref_coords[-1] - comb_centroid))
        endpoint = ref_coords[0] if d_start < d_end else ref_coords[-1]

        # Project onto the combined face
        try:
            cp, uv = gmsh.model.getClosestPoint(
                2, comb_tag, endpoint.tolist())
        except Exception as exc:
            raise RuntimeError(
                f"Could not project onto combined face: {exc}")

        # Strip center in the cross-groove direction
        if cross_direction == 'v':
            center = uv[1]
            lo_bound = comb_finfo['vmin']
            hi_bound = comb_finfo['vmax']
        else:
            center = uv[0]
            lo_bound = comb_finfo['umin']
            hi_bound = comb_finfo['umax']

        half = ref_span / 2.0
        cut_lo = center - half
        cut_hi = center + half

        # Clamp to face bounds (don't create cuts outside the face)
        split_vals = []
        if cut_lo > lo_bound + ref_span * 0.05:
            split_vals.append(cut_lo)
        if cut_hi < hi_bound - ref_span * 0.05:
            split_vals.append(cut_hi)

        if not split_vals:
            # Strip spans the entire face — nothing to cut
            return [combined_face_idx], combined_face_idx

        new_indices = self.split_face_at_params(
            combined_face_idx, split_vals, cross_direction)

        # Identify which new face is the strip — the one whose
        # centroid is closest to the projected endpoint
        ep_3d = np.array(cp[:3], dtype=np.float64)
        best_idx = None
        best_dist = float('inf')
        for ni in new_indices:
            c = self._faces[ni]['centroid']
            d = float(np.linalg.norm(c - ep_3d))
            if d < best_dist:
                best_dist = d
                best_idx = ni

        return new_indices, best_idx

    # ── Cleanup ───────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release gmsh resources if active."""
        if self._gmsh_active:
            try:
                import gmsh
                gmsh.finalize()
            except Exception:
                pass
            self._gmsh_active = False

    def __del__(self):
        self.close()
