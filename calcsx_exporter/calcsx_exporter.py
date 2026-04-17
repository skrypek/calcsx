# CalcSX Bobbin Exporter for Fusion 360

import adsk.core
import adsk.fusion
import traceback
import json
import os


# ── Utility helpers ───────────────────────────────────────────────

def _linspace(a, b, n):
    if n < 2: return [0.5*(a+b)]
    return [a+(b-a)*i/(n-1) for i in range(n)]

def _dist3d(a, b):
    return sum((x-y)**2 for x,y in zip(a,b))**0.5

def _arc_len(ev, u0, u1, v0, v1, sweep_u, n=20):
    L, prev = 0.0, None
    if sweep_u:
        vm = 0.5*(v0+v1)
        for u in _linspace(u0,u1,n):
            ok,pt = ev.getPointAtParameter(adsk.core.Point2D.create(u,vm))
            if ok:
                if prev: L+=_dist3d((pt.x,pt.y,pt.z),(prev.x,prev.y,prev.z))
                prev=pt
    else:
        um = 0.5*(u0+u1)
        for v in _linspace(v0,v1,n):
            ok,pt = ev.getPointAtParameter(adsk.core.Point2D.create(um,v))
            if ok:
                if prev: L+=_dist3d((pt.x,pt.y,pt.z),(prev.x,prev.y,prev.z))
                prev=pt
    return L

def _face_info(face):
    """Classify a face: which parametric direction is 'along' the groove
    (longer arc) and which is 'cross' (shorter arc)."""
    ev = face.evaluator
    r = ev.parametricRange()
    u0,u1 = r.minPoint.x, r.maxPoint.x
    v0,v1 = r.minPoint.y, r.maxPoint.y
    lu = _arc_len(ev,u0,u1,v0,v1,True)
    lv = _arc_len(ev,u0,u1,v0,v1,False)
    if lu >= lv:
        return dict(u0=u0,u1=u1,v0=v0,v1=v1,along='u',cross='v',cw=lv,al=lu)
    return dict(u0=u0,u1=u1,v0=v0,v1=v1,along='v',cross='u',cw=lu,al=lv)

def _cross_mid(info):
    """Midpoint of the cross-groove parameter range."""
    return 0.5*(info['v0']+info['v1']) if info['cross']=='v' else 0.5*(info['u0']+info['u1'])

def _sample(face, info, cross_val, n):
    """Sample n points + normals along the groove at a fixed cross value."""
    ev = face.evaluator
    pts, nms = [], []
    if info['along']=='u':
        for u in _linspace(info['u0'],info['u1'],n):
            ok1,pt = ev.getPointAtParameter(adsk.core.Point2D.create(u,cross_val))
            ok2,nm = ev.getNormalAtParameter(adsk.core.Point2D.create(u,cross_val))
            if ok1 and ok2:
                pts.append((pt.x,pt.y,pt.z)); nms.append((nm.x,nm.y,nm.z))
            elif pts: pts.append(pts[-1]); nms.append(nms[-1])
    else:
        for v in _linspace(info['v0'],info['v1'],n):
            ok1,pt = ev.getPointAtParameter(adsk.core.Point2D.create(cross_val,v))
            ok2,nm = ev.getNormalAtParameter(adsk.core.Point2D.create(cross_val,v))
            if ok1 and ok2:
                pts.append((pt.x,pt.y,pt.z)); nms.append((nm.x,nm.y,nm.z))
            elif pts: pts.append(pts[-1]); nms.append(nms[-1])
    return pts, nms


# ── Edge / adjacency helpers ─────────────────────────────────────

def _edge_ids(face):
    return {face.edges.item(i).tempId for i in range(face.edges.count)}

def _shared_edge_objs(fa, fb):
    b_ids = _edge_ids(fb)
    out = []
    for i in range(fa.edges.count):
        e = fa.edges.item(i)
        if e.tempId in b_ids: out.append(e)
    return out

def _edge_midpoint_3d(edge):
    ev = edge.evaluator
    ok,sp,ep = ev.getParameterExtents()
    if not ok: return None
    ok2,pt = ev.getPointAtParameter(0.5*(sp+ep))
    return (pt.x,pt.y,pt.z) if ok2 else None


# ── Parametric projection ────────────────────────────────────────

def _project_point_to_face(face, info, pt3d):
    """Project a 3D point onto a face's parametric space.

    Returns (cross_val, along_val, distance).
    *distance* is the 3D gap between *pt3d* and the closest point
    found on the surface (0.0 when the point lies exactly on it).
    Uses Fusion's exact getParameterAtPoint first; falls back to
    a grid scan if the exact call fails (e.g. point is off-surface).
    """
    ev = face.evaluator
    pt = adsk.core.Point3D.create(pt3d[0], pt3d[1], pt3d[2])
    ok, param = ev.getParameterAtPoint(pt)
    if ok:
        ok2, spt = ev.getPointAtParameter(param)
        dist = _dist3d(pt3d, (spt.x, spt.y, spt.z)) if ok2 else 0.0
        if info['along'] == 'u':
            return param.y, param.x, dist    # cross=v, along=u
        else:
            return param.x, param.y, dist    # cross=u, along=v

    # Fallback: grid scan
    n_cross, n_along = 80, 20
    if info['cross'] == 'v':
        cross_vals = _linspace(info['v0'], info['v1'], n_cross)
        along_vals = _linspace(info['u0'], info['u1'], n_along)
    else:
        cross_vals = _linspace(info['u0'], info['u1'], n_cross)
        along_vals = _linspace(info['v0'], info['v1'], n_along)

    best_cv, best_av, best_d = cross_vals[0], along_vals[0], float('inf')
    for cv in cross_vals:
        for av in along_vals:
            if info['cross'] == 'v':
                p2 = adsk.core.Point2D.create(av, cv)
            else:
                p2 = adsk.core.Point2D.create(cv, av)
            ok2, spt = ev.getPointAtParameter(p2)
            if ok2:
                d = _dist3d(pt3d, (spt.x, spt.y, spt.z))
                if d < best_d:
                    best_d, best_cv, best_av = d, cv, av
    return best_cv, best_av, best_d


# ── Face ordering within a chain ─────────────────────────────────

def _order_faces(faces, infos, group):
    """Order faces within a narrow-chain end-to-end by endpoint proximity."""
    if len(group) <= 1: return group
    eps = {}
    for fi in group:
        p, _ = _sample(faces[fi], infos[fi], _cross_mid(infos[fi]), 5)
        if len(p) >= 2: eps[fi] = (p[0], p[-1])
    ordered = [group[0]]
    rem = set(group[1:])
    while rem:
        curr = ordered[-1]
        if curr not in eps: break
        ce = eps[curr][1]
        best, bd = None, float('inf')
        for r in rem:
            if r not in eps: continue
            for ep in eps[r]:
                d = _dist3d(ce, ep)
                if d < bd: bd, best = d, r
        if best: ordered.append(best); rem.remove(best)
        else: break
    ordered.extend(rem)
    return ordered


# ── Segment chaining ─────────────────────────────────────────────

def _chain_segments(segments):
    """Chain segments end-to-end by nearest-neighbour.

    Each segment is (pts, nms, face_idx).
    Returns (all_pts, all_nms).
    """
    if len(segments) == 1:
        return list(segments[0][0]), list(segments[0][1])

    n = len(segments)
    chain_pts = list(segments[0][0])
    chain_nms = list(segments[0][1])
    used = {0}

    for _ in range(n - 1):
        chain_end = chain_pts[-1]
        best_j, best_d, best_flip = -1, float('inf'), False

        for j in range(n):
            if j in used:
                continue
            p = segments[j][0]
            ds = _dist3d(chain_end, p[0])
            de = _dist3d(chain_end, p[-1])
            if ds <= de:
                if ds < best_d:
                    best_d, best_j, best_flip = ds, j, False
            else:
                if de < best_d:
                    best_d, best_j, best_flip = de, j, True

        if best_j < 0:
            break

        used.add(best_j)
        pts = list(segments[best_j][0])
        nms = list(segments[best_j][1])
        if best_flip:
            pts.reverse(); nms.reverse()
        chain_pts.extend(pts)
        chain_nms.extend(nms)

    return chain_pts, chain_nms


def _consistent_normals(nms):
    """Propagate normals so consecutive vectors point in the same
    hemisphere.  Flips any normal whose dot product with its
    predecessor is negative."""
    if len(nms) < 2:
        return nms
    out = [nms[0]]
    for i in range(1, len(nms)):
        prev = out[-1]
        curr = nms[i]
        dot = sum(a * b for a, b in zip(prev, curr))
        if dot < 0:
            out.append(tuple(-c for c in curr))
        else:
            out.append(curr)
    return out


# ── Main channel builder ─────────────────────────────────────────

def _build_channels(faces, n_pts):
    """Build circumferential channels from a mix of narrow (single-groove)
    and wide (multi-groove) faces.

    Algorithm
    ---------
    1. Classify faces as narrow / wide by cross-groove width.
    2. BFS-group connected narrow faces into chains (edge-length gated
       so adjacent grooves sharing a sharp edge are NOT merged).
    3. For each wide face determine which chains enter from each side
       (left / right in the along-groove direction).  Match chains
       from opposite sides by cross-groove rank  →  Union-Find merge.
    4. After merging, each connected component = one groove.
    5. Sample narrow centerlines and wide-face strips, chain per groove.

    Returns list of (pts, nms) tuples, one per groove.
    """
    ui = adsk.core.Application.get().userInterface
    dbg = []

    infos = [_face_info(f) for f in faces]
    cws   = [info['cw'] for info in infos]
    pos   = [w for w in cws if w > 0]
    if not pos:
        return []
    min_cw = min(pos)

    # ── 1. Classify ──────────────────────────────────────────────
    narrow, wide = [], []
    for i in range(len(faces)):
        (wide if cws[i] > min_cw * 1.8 else narrow).append(i)
    narrow_set = set(narrow)

    dbg.append(f"{len(faces)} faces: {len(narrow)} narrow, {len(wide)} wide")
    dbg.append(f"Min cross-width: {min_cw:.4f}")
    for i in range(len(faces)):
        tag = "WIDE" if i in set(wide) else "narrow"
        dbg.append(f"  f{i}: cw={cws[i]:.4f} along={infos[i]['along']} [{tag}]")

    # All faces same type → one channel per face
    if not wide or not narrow:
        channels = []
        for i in range(len(faces)):
            p, n = _sample(faces[i], infos[i], _cross_mid(infos[i]), n_pts)
            if len(p) >= 3:
                channels.append((p, n))
        dbg.append(f"No narrow/wide mix → {len(channels)} channels")
        ui.messageBox('\n'.join(dbg), 'CalcSX Export')
        return channels

    # ── 2. Face adjacency ────────────────────────────────────────
    esets = [_edge_ids(f) for f in faces]
    face_adj = {i: set() for i in range(len(faces))}
    for a in range(len(faces)):
        for b in range(a+1, len(faces)):
            if esets[a] & esets[b]:
                face_adj[a].add(b)
                face_adj[b].add(a)

    # ── 3. BFS-group narrow faces into chains ────────────────────
    # Gate: only merge through SHORT shared edges (sequential faces
    # of the same groove).  Long edges connect adjacent grooves.
    visited = set()
    chains = []
    for s in narrow:
        if s in visited:
            continue
        comp, q = [], [s]
        while q:
            nd = q.pop(0)
            if nd in visited:
                continue
            visited.add(nd)
            comp.append(nd)
            for nb in face_adj[nd]:
                if nb not in narrow_set or nb in visited:
                    continue
                shared = _shared_edge_objs(faces[nd], faces[nb])
                total_len = sum(e.length for e in shared)
                if total_len < min_cw * 2.5:
                    q.append(nb)
        chains.append(comp)

    dbg.append(f"{len(chains)} narrow chains:")
    for ci, ch in enumerate(chains):
        dbg.append(f"  chain {ci}: faces {ch}")

    # ── 4. Merge chains across wide faces (Union-Find) ───────────
    #
    # getParameterAtPoint projects onto the UNBOUNDED underlying
    # surface, so dist=0 for every face on the same body.  Instead
    # we sample the physical 3D boundary edges of each wide face
    # and match narrow-chain endpoints to the NEAREST boundary in
    # real 3D space.  "start" boundary = along_min edge, "end" =
    # along_max edge.
    parent = list(range(len(chains)))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x, y):
        rx, ry = _find(x), _find(y)
        if rx != ry:
            parent[rx] = ry

    # Sample start / end 3D points for each chain's centerline
    chain_eps = {}                              # ci → (start_pt, end_pt)
    for ci, chain in enumerate(chains):
        ordered = _order_faces(faces, infos, chain)
        p0, _ = _sample(faces[ordered[0]],  infos[ordered[0]],
                         _cross_mid(infos[ordered[0]]),  5)
        p1, _ = _sample(faces[ordered[-1]], infos[ordered[-1]],
                         _cross_mid(infos[ordered[-1]]), 5)
        if len(p0) >= 2 and len(p1) >= 2:
            chain_eps[ci] = (p0[0], p1[-1])
        elif len(p0) >= 2:
            chain_eps[ci] = (p0[0], p0[-1])

    # Sample 3D boundary points for each wide face
    n_bdy = 30
    wide_bdy = {}                # wi → (start_3d_pts, end_3d_pts)
    for wi in wide:
        w_info = infos[wi]
        ev = faces[wi].evaluator
        s_pts, e_pts = [], []
        if w_info['along'] == 'v':
            for u in _linspace(w_info['u0'], w_info['u1'], n_bdy):
                ok, pt = ev.getPointAtParameter(
                    adsk.core.Point2D.create(u, w_info['v0']))
                if ok: s_pts.append((pt.x, pt.y, pt.z))
                ok, pt = ev.getPointAtParameter(
                    adsk.core.Point2D.create(u, w_info['v1']))
                if ok: e_pts.append((pt.x, pt.y, pt.z))
        else:
            for v in _linspace(w_info['v0'], w_info['v1'], n_bdy):
                ok, pt = ev.getPointAtParameter(
                    adsk.core.Point2D.create(w_info['u0'], v))
                if ok: s_pts.append((pt.x, pt.y, pt.z))
                ok, pt = ev.getPointAtParameter(
                    adsk.core.Point2D.create(w_info['u1'], v))
                if ok: e_pts.append((pt.x, pt.y, pt.z))
        wide_bdy[wi] = (s_pts, e_pts)

    chain_wide_cv = {}          # (chain_idx, wide_idx) → cross_val
    tol = min_cw * 2            # 3D proximity threshold

    # Collect all detected connections: (ci, wi, side, cv)
    all_entries = []

    for wi in wide:
        s_bdy, e_bdy = wide_bdy[wi]

        for ci in range(len(chains)):
            if ci not in chain_eps:
                continue
            for ep in chain_eps[ci]:
                d_s = min((_dist3d(ep, bp) for bp in s_bdy),
                          default=float('inf'))
                d_e = min((_dist3d(ep, bp) for bp in e_bdy),
                          default=float('inf'))
                d_min = min(d_s, d_e)
                if d_min >= tol:
                    continue

                cv, _, _ = _project_point_to_face(
                    faces[wi], infos[wi], ep)

                key = (ci, wi)
                if key not in chain_wide_cv:
                    chain_wide_cv[key] = cv

                side = 'L' if d_s <= d_e else 'R'
                all_entries.append((ci, wi, side, cv))
                dbg.append(
                    f"  chain {ci} ep→wide f{wi}: "
                    f"d_s={d_s:.4f} d_e={d_e:.4f} cv={cv:.4f} [{side}]")

    # ── Phase A: per-face L↔R matching ──
    for wi in wide:
        left_d  = {}
        right_d = {}
        for ci, w, s, cv in all_entries:
            if w != wi: continue
            target = left_d if s == 'L' else right_d
            if ci not in target:
                target[ci] = cv

        left_sorted  = sorted(left_d.items(),  key=lambda x: x[1])
        right_sorted = sorted(right_d.items(), key=lambda x: x[1])

        dbg.append(
            f"  Wide f{wi}:  L={[(c,f'{v:.4f}') for c,v in left_sorted]}"
            f"  R={[(c,f'{v:.4f}') for c,v in right_sorted]}")

        n_match = min(len(left_sorted), len(right_sorted))
        for i in range(n_match):
            dbg.append(
                f"  MERGE chain {left_sorted[i][0]} ↔ "
                f"chain {right_sorted[i][0]} "
                f"(cv {left_sorted[i][1]:.4f} / {right_sorted[i][1]:.4f})")
            _union(left_sorted[i][0], right_sorted[i][0])

    # ── Phase B: cross-face matching via far-end proximity ──
    # For each entry, evaluate the 3D point at the OPPOSITE
    # boundary of the wide face (the end the groove exits toward
    # the next section).  Entries on different wide faces whose
    # far-ends are close in 3D are the same groove.
    far_ends = {}               # (ci, wi) → 3D point
    far_side = {}               # (ci, wi) → side tag

    for ci, wi, side, cv in all_entries:
        key = (ci, wi)
        if key in far_ends:
            continue
        w_info = infos[wi]
        ev = faces[wi].evaluator

        if w_info['along'] == 'v':
            # chain at START (L) → far end at v_max;
            # chain at END   (R) → far end at v_min
            far_v = w_info['v1'] if side == 'L' else w_info['v0']
            param = adsk.core.Point2D.create(cv, far_v)
        else:
            far_u = w_info['u1'] if side == 'L' else w_info['u0']
            param = adsk.core.Point2D.create(far_u, cv)

        ok, pt = ev.getPointAtParameter(param)
        if ok:
            far_ends[key] = (pt.x, pt.y, pt.z)
            far_side[key] = side

    # Greedy 1:1 nearest-neighbour across different wide faces.
    # Build all cross-face pairs, sort by distance, accept each
    # pair only if neither chain has been matched yet.
    cross_pairs = []
    keys = list(far_ends.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ci_a, wi_a = keys[i]
            ci_b, wi_b = keys[j]
            if wi_a == wi_b:
                continue
            d = _dist3d(far_ends[keys[i]], far_ends[keys[j]])
            cross_pairs.append((d, ci_a, wi_a, ci_b, wi_b))
    cross_pairs.sort()                          # shortest first

    xmatched = set()
    for d, ci_a, wi_a, ci_b, wi_b in cross_pairs:
        if ci_a in xmatched or ci_b in xmatched:
            continue
        if _find(ci_a) == _find(ci_b):
            continue
        xmatched.add(ci_a)
        xmatched.add(ci_b)
        _union(ci_a, ci_b)
        dbg.append(
            f"  XMERGE chain {ci_a}(f{wi_a}) ↔ "
            f"chain {ci_b}(f{wi_b}) "
            f"[far-end d={d:.4f}]")

    # Build merged groove groups
    groove_map = {}
    for ci in range(len(chains)):
        root = _find(ci)
        groove_map.setdefault(root, []).append(ci)
    grooves = list(groove_map.values())
    n_grooves = len(grooves)

    dbg.append(f"After merge: {n_grooves} grooves")
    for gi, g in enumerate(grooves):
        all_fi = [fi for ci in g for fi in chains[ci]]
        dbg.append(f"  groove {gi}: chains {g} → narrow faces {all_fi}")

    # ── 5. Collect segments (narrow + wide strips) ───────────────
    groove_segments = [[] for _ in range(n_grooves)]

    # Narrow segments
    for gi, groove_chains in enumerate(grooves):
        for ci in groove_chains:
            ordered = _order_faces(faces, infos, chains[ci])
            for fi in ordered:
                p, n = _sample(faces[fi], infos[fi],
                               _cross_mid(infos[fi]), n_pts)
                if len(p) >= 3:
                    groove_segments[gi].append((p, n, fi))

    # Wide-face strips
    for wi in wide:
        w_info = infos[wi]
        used_cvs = set()                      # avoid duplicate strips

        for gi, groove_chains in enumerate(grooves):
            cv = None
            for ci in groove_chains:
                key = (ci, wi)
                if key in chain_wide_cv:
                    cv = chain_wide_cv[key]
                    break
            if cv is None:
                continue

            # Deduplicate: merged chains may map to the same cv
            cv_key = round(cv, 6)
            if cv_key in used_cvs:
                continue
            used_cvs.add(cv_key)

            strip_p, strip_n = _sample(faces[wi], w_info, cv, n_pts)
            if len(strip_p) >= 3:
                groove_segments[gi].append((strip_p, strip_n, wi))
                dbg.append(f"  groove {gi} strip on wide f{wi}: "
                           f"cv={cv:.4f}, {len(strip_p)} pts")

    # ── 6. Chain segments per groove ─────────────────────────────
    import math
    channels = []
    for gi in range(n_grooves):
        segs = groove_segments[gi]
        if not segs:
            continue
        pts, nms = _chain_segments(segs)
        nms = _consistent_normals(nms)
        # Ensure normals point OUTWARD (away from the path centroid).
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        cz = sum(p[2] for p in pts) / len(pts)
        dot_sum = sum((pts[i][0]-cx)*nms[i][0] +
                      (pts[i][1]-cy)*nms[i][1] +
                      (pts[i][2]-cz)*nms[i][2]
                      for i in range(len(pts)))
        if dot_sum < 0:
            nms = [tuple(-c for c in nm) for nm in nms]
        # Remove near-duplicate junction points only
        if len(pts) > 1:
            clean_p, clean_n = [pts[0]], [nms[0]]
            for i in range(1, len(pts)):
                if _dist3d(pts[i], clean_p[-1]) > 0.005:
                    clean_p.append(pts[i])
                    clean_n.append(nms[i])
            pts, nms = clean_p, clean_n
        if len(pts) >= 3:
            channels.append((pts, nms))
            dbg.append(f"Channel {gi}: {len(pts)} pts from {len(segs)} segments")

            # ── Junction diagnostics ──
            # Find segment-length anomalies (the root cause of
            # Biot-Savart force spikes is a segment that is much
            # shorter or longer than its neighbors).
            seg_lens = [_dist3d(pts[i], pts[i+1])
                        for i in range(len(pts)-1)]
            if seg_lens:
                median_sl = sorted(seg_lens)[len(seg_lens)//2]
                anomalies = []
                for i, sl in enumerate(seg_lens):
                    ratio = sl / median_sl if median_sl > 0 else 0
                    if ratio < 0.3 or ratio > 3.0:
                        # Also compute tangent angle at this point
                        if 0 < i < len(seg_lens) - 1:
                            t1 = tuple(pts[i][k]-pts[i-1][k] for k in range(3))
                            t2 = tuple(pts[i+1][k]-pts[i][k] for k in range(3))
                            m1 = max(sum(x*x for x in t1)**0.5, 1e-12)
                            m2 = max(sum(x*x for x in t2)**0.5, 1e-12)
                            dt = sum(a*b for a,b in zip(t1,t2))/(m1*m2)
                            dt = max(-1.0, min(1.0, dt))
                            ang = math.degrees(math.acos(dt))
                        else:
                            ang = 0.0
                        anomalies.append((i, sl, ratio, ang))
                dbg.append(f"  median seg_len={median_sl:.5f}")
                if anomalies:
                    dbg.append(f"  {len(anomalies)} seg-length anomalies:")
                    for idx, sl, ratio, ang in anomalies[:15]:
                        p = pts[idx]
                        dbg.append(
                            f"    [{idx}] len={sl:.5f} "
                            f"ratio={ratio:.2f}× Δ={ang:.1f}° "
                            f"({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})")
                else:
                    dbg.append(f"  No seg-length anomalies")

            # Normal-angle discontinuities (root cause of
            # volumetric filament grid shifts → force spikes)
            n_angles = []
            for i in range(1, len(nms)):
                dn = sum(nms[i][d]*nms[i-1][d] for d in range(3))
                dn = max(-1.0, min(1.0, dn))
                na = math.degrees(math.acos(dn))
                n_angles.append((na, i))
            n_angles.sort(reverse=True)
            dbg.append(f"  Top normal breaks:")
            for na, idx in n_angles[:8]:
                p = pts[idx]
                dbg.append(
                    f"    pt {idx}/{len(pts)}: normal Δ={na:.2f}° "
                    f"({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})")

            turns = []
            for i in range(1, len(pts) - 1):
                t1 = tuple(pts[i][k] - pts[i-1][k] for k in range(3))
                t2 = tuple(pts[i+1][k] - pts[i][k] for k in range(3))
                m1 = max(sum(x*x for x in t1)**0.5, 1e-12)
                m2 = max(sum(x*x for x in t2)**0.5, 1e-12)
                dot = sum(a*b for a,b in zip(t1,t2)) / (m1*m2)
                dot = max(-1.0, min(1.0, dot))
                angle = math.degrees(math.acos(dot))
                turns.append((angle, i))
            turns.sort(reverse=True)
            dbg.append(f"  Top tangent breaks:")
            for angle, idx in turns[:5]:
                gap = _dist3d(pts[idx], pts[idx+1])
                p = pts[idx]
                dbg.append(f"    pt {idx}/{len(pts)}: Δ={angle:.1f}° "
                           f"gap={gap:.4f} pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})")

    ui.messageBox('\n'.join(dbg), 'CalcSX Export Debug')
    return channels


# ── Bobbin mesh export ────────────────────────────────────────────

def _export_bobbin_mesh(faces):
    if not faces: return None
    try:
        comp = faces[0].body.parentComponent
        vv, tt, off = [], [], 0
        for bi in range(comp.bRepBodies.count):
            body = comp.bRepBodies.item(bi)
            if not body.isVisible: continue
            c = body.meshManager.createMeshCalculator()
            c.setQuality(
                adsk.fusion.TriangleMeshQualityOptions.NormalQualityTriangleMesh)
            m = c.calculate()
            for p in m.nodeCoordinates: vv.append([p.x, p.y, p.z])
            idx = m.nodeIndices
            for i in range(0, len(idx), 3):
                tt.append([idx[i]+off, idx[i+1]+off, idx[i+2]+off])
            off += len(m.nodeCoordinates)
        return {"vertices": vv, "faces": tt} if vv else None
    except Exception:
        return None


# ── Fusion 360 command plumbing ───────────────────────────────────

_handlers = []

class _Created(adsk.core.CommandCreatedEventHandler):
    def notify(self, args):
        try:
            inp = args.command.commandInputs
            sel = inp.addSelectionInput(
                'faces', 'Groove Faces',
                'Select all groove + full-face surfaces')
            sel.addSelectionFilter('Faces')
            sel.setSelectionLimits(1, 0)
            inp.addIntegerSpinnerCommandInput(
                'npts', 'Points per segment', 100, 2000, 50, 500)
            h = _Execute()
            args.command.execute.add(h)
            _handlers.append(h)
        except Exception:
            adsk.core.Application.get().userInterface.messageBox(
                traceback.format_exc())


class _Execute(adsk.core.CommandEventHandler):
    def notify(self, args):
        try:
            ui  = adsk.core.Application.get().userInterface
            inp = args.command.commandInputs
            sel = inp.itemById('faces')
            n_pts = inp.itemById('npts').value
            faces = [sel.selection(i).entity
                     for i in range(sel.selectionCount)
                     if isinstance(sel.selection(i).entity,
                                   adsk.fusion.BRepFace)]
            if not faces:
                ui.messageBox('No faces selected.')
                return

            dlg = ui.createFileDialog()
            dlg.title = 'Save .bobsx'
            dlg.filter = 'CalcSX Bobbin (*.bobsx)'
            dlg.isMultiSelectEnabled = False
            if dlg.showSave() != adsk.core.DialogResults.DialogOK:
                return
            fp = dlg.filename
            if not fp.endswith('.bobsx'):
                fp += '.bobsx'

            chains = _build_channels(faces, n_pts)
            chs = [{"name": f"groove_{i+1}",
                    "points": [{"x": p[0], "y": p[1], "z": p[2],
                                "nx": n[0], "ny": n[1], "nz": n[2]}
                               for p, n in zip(pts, nms)]}
                   for i, (pts, nms) in enumerate(chains)]
            if not chs:
                ui.messageBox('No channels extracted.')
                return

            data = {"version": 1, "unit": "cm", "channels": chs}
            bm = _export_bobbin_mesh(faces)
            if bm:
                data["bobbin_mesh"] = bm
            with open(fp, 'w') as f:
                json.dump(data, f, indent=2)

            # Dump raw per-point CSV for channel 0 (debug)
            import math as _m
            csv_path = fp.replace('.bobsx', '_debug_ch0.csv')
            if chains:
                p0, n0 = chains[0]
                with open(csv_path, 'w') as cf:
                    cf.write('idx,x,y,z,nx,ny,nz,seg_len,tang_deg,curv,norm_deg\n')
                    for i in range(len(p0)):
                        sl = _dist3d(p0[i], p0[i+1]) if i < len(p0)-1 else 0.0
                        # tangent angle
                        if 0 < i < len(p0)-1:
                            t1 = tuple(p0[i][d]-p0[i-1][d] for d in range(3))
                            t2 = tuple(p0[i+1][d]-p0[i][d] for d in range(3))
                            m1 = max(sum(x*x for x in t1)**0.5, 1e-12)
                            m2 = max(sum(x*x for x in t2)**0.5, 1e-12)
                            td = max(-1, min(1, sum(a*b for a,b in zip(t1,t2))/(m1*m2)))
                            ta = _m.degrees(_m.acos(td))
                            # curvature ≈ angle / seg_len
                            curv = _m.radians(ta) / max(sl, 1e-12)
                        else:
                            ta, curv = 0.0, 0.0
                        # normal angle
                        if i > 0:
                            nd = max(-1, min(1, sum(n0[i][d]*n0[i-1][d] for d in range(3))))
                            na = _m.degrees(_m.acos(nd))
                        else:
                            na = 0.0
                        cf.write(f'{i},{p0[i][0]:.6f},{p0[i][1]:.6f},{p0[i][2]:.6f},'
                                 f'{n0[i][0]:.6f},{n0[i][1]:.6f},{n0[i][2]:.6f},'
                                 f'{sl:.6f},{ta:.4f},{curv:.4f},{na:.4f}\n')

            ui.messageBox(f'Exported {len(chs)} channel(s) to:\n{fp}\n'
                          f'Debug CSV: {csv_path}')
        except Exception:
            adsk.core.Application.get().userInterface.messageBox(
                traceback.format_exc())


def run(context):
    try:
        ui = adsk.core.Application.get().userInterface
        cd = ui.commandDefinitions.addButtonDefinition(
            'calcsx_export_cmd', 'CalcSX Bobbin Export',
            'Export groove channels')
        h = _Created()
        cd.commandCreated.add(h)
        _handlers.append(h)
        cd.execute()
        adsk.autoTerminate(False)
    except Exception:
        adsk.core.Application.get().userInterface.messageBox(
            traceback.format_exc())


def stop(context):
    try:
        cd = adsk.core.Application.get().userInterface \
                .commandDefinitions.itemById('calcsx_export_cmd')
        if cd: cd.deleteMe()
    except Exception:
        pass
