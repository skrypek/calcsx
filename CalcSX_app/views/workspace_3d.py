# views/workspace_3d.py
"""
Workspace3DView — PyVista/VTK 3-D workspace with a named layer system.

Layer names (analysis)
----------------------
  'Forces'        — Lorentz-force arrow glyphs, plasma-coloured by magnitude
  'Stress'        — hoop-stress point cloud (YlOrRd, MPa)
  'B Axis'        — on-axis |B| scatter along PCA axis (cool cmap)
  'Field Lines'   — 3D magnetic field line traces, coloured by log₁₀|B|
  'Cross Section' — 2D mid-plane |B| heatmap, coloured by log₁₀|B|

Coils are tracked separately in _coil_entries (supports multiple coils).
Gizmo interaction is handled via a Qt event filter (_GizmoEventFilter) installed
on the plotter widget — this intercepts mouse events before VTK sees them, giving
reliable camera freeze without fighting VTK's C++ vtable.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel,
)
from PyQt5.QtCore import Qt, QObject, QEvent

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    _HAS_PYVISTA = True
except ImportError:
    _HAS_PYVISTA = False

from gui.gui_utils import THEME


def _ptp(arr: np.ndarray, axis=None) -> np.ndarray:
    """np.ptp was removed in NumPy 2.0 — use max - min."""
    return arr.max(axis=axis) - arr.min(axis=axis)


@dataclass
class _Layer:
    name:       str
    actors:     list = field(default_factory=list)
    visible:    bool = True
    scalar_bar: object = None


def _set_visible(actor, visible: bool) -> None:
    try:
        actor.SetVisibility(int(visible))
        return
    except AttributeError:
        pass
    try:
        actor.visibility = visible
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Interactive transform gizmo
# ─────────────────────────────────────────────────────────────────────────────

class _TransformGizmo:
    """
    SolidWorks-style interactive transform gizmo.

    3 translation arrows (X=red / Y=green / Z=blue) + 3 rotation arcs.
    Geometry is built at world origin and repositioned via a single shared
    vtkTransform so dragging costs only a matrix update per frame.

    Hover highlighting: white edge outline + 10% scale-up.
    """

    _AXIS_COLORS = [
        (1.00, 0.20, 0.20),   # X — red
        (0.20, 1.00, 0.20),   # Y — green
        (0.20, 0.55, 1.00),   # Z — blue
    ]
    _AXIS_DIRS = [
        np.array([1., 0., 0.]),
        np.array([0., 1., 0.]),
        np.array([0., 0., 1.]),
    ]

    def __init__(self, renderer, callback):
        self._renderer = renderer   # dedicated VTK renderer for gizmo actors
        self._cb       = callback
        self._actors   = []
        self._t_actors = []
        self._r_actors = []
        self._visible  = False

        # addr → (axis_idx, 'T'|'R')
        self._actor_tags:       dict = {}
        self._actor_by_addr:    dict = {}
        self._actor_orig_color: dict = {}   # addr → (r, g, b)
        self._actor_peers:      dict = {}   # addr → [peer_addr, ...] for co-highlighting
        self._hovered_addr: str | None = None
        self._cell_picker = None            # vtkCellPicker, created lazily

        self._orig_centroid = np.zeros(3)
        self._cumul = [0., 0., 0., 0., 0., 0.]
        self._scale = 1.0

        try:
            import vtk as _vtk
            self._pos_xfm = _vtk.vtkTransform()
            self._pos_xfm.PostMultiply()
        except Exception:
            self._pos_xfm = None

        self._drag_mode           = None
        self._drag_axis           = -1
        self._drag_start_cumul    = None
        self._drag_start_t        = 0.
        self._drag_start_angle    = 0.
        self._drag_start_centroid = np.zeros(3)
        self._drag_dot_actor      = None

    # ── Public ────────────────────────────────────────────────────────────────

    def load(self, centroid: np.ndarray, scale: float) -> None:
        self._orig_centroid = np.asarray(centroid, float).copy()
        self._cumul = [0., 0., 0., 0., 0., 0.]
        self._scale = max(float(scale), 1e-4)
        self._remove_all()
        self._visible = False

    def show(self, mode: str) -> None:
        self._remove_all()
        self._build_actors(mode)
        self._sync_pos()
        self._visible = True

    def hide(self) -> None:
        self._drag_mode    = None
        self._drag_axis    = -1
        self._hovered_addr = None
        self._remove_dot()
        self._remove_all()
        self._visible = False

    def reset(self) -> None:
        self._cumul = [0., 0., 0., 0., 0., 0.]
        self._drag_mode = None
        self._drag_axis = -1
        self._sync_pos()

    # ── Drag entry points ─────────────────────────────────────────────────────

    def _start_drag(self, mode: str, axis_idx: int, x: int, y: int) -> None:
        self._clear_hover()
        self._drag_mode = mode
        self._drag_axis = axis_idx
        self._drag_start_cumul    = list(self._cumul)
        self._drag_start_centroid = self._current_centroid().copy()
        if mode == 'T':
            self._drag_start_t = self._axis_t(x, y, axis_idx, self._drag_start_centroid)
        else:
            self._drag_start_angle = self._screen_angle(x, y, self._drag_start_centroid)
            self._update_dot(x, y, axis_idx)

    def _update_drag(self, x: int, y: int) -> None:
        if self._drag_mode is None:
            return
        c0 = self._drag_start_centroid
        new_cumul = list(self._drag_start_cumul)
        if self._drag_mode == 'T':
            # Always measure t relative to the drag-start centroid so the
            # reference stays fixed and the translation doesn't snap/drift.
            delta = self._axis_t(x, y, self._drag_axis, c0) - self._drag_start_t
            new_cumul[self._drag_axis] += delta
        else:
            delta = self._screen_angle(x, y, c0) - self._drag_start_angle
            while delta >  180.: delta -= 360.
            while delta < -180.: delta += 360.
            new_cumul[3 + self._drag_axis] += delta
            self._update_dot(x, y, self._drag_axis)
        self._cumul = new_cumul
        self._sync_pos()
        self._cb(*self._cumul)

    def _end_drag(self) -> None:
        self._drag_mode = None
        self._drag_axis = -1
        self._remove_dot()

    # ── Pick + hover ──────────────────────────────────────────────────────────

    def _get_picker(self):
        """Return a cached vtkCellPicker (created once)."""
        if self._cell_picker is None:
            try:
                import vtk as _vtk
                self._cell_picker = _vtk.vtkCellPicker()
                # ~20 physical pixels on a 2×-Retina display (window diagonal ≈ 2490px).
                # Large enough to catch near-misses on thin shafts; small enough to
                # avoid picking the wrong actor across the gizmo.
                self._cell_picker.SetTolerance(0.008)
            except Exception:
                pass
        return self._cell_picker

    def _pick_actor(self, x: int, y: int):
        """Return (axis_idx, 'T'|'R') or (-1, None). x,y are VTK physical display coords."""
        try:
            picker = self._get_picker()
            if picker is None:
                return -1, None
            picker.Pick(x, y, 0, self._renderer)
            hit = picker.GetActor()
            if hit is None:
                return -1, None
            addr = hit.GetAddressAsString('')
            if addr in self._actor_tags:
                return self._actor_tags[addr]
        except Exception:
            pass
        return -1, None

    def _update_hover(self, x: int, y: int) -> bool:
        """Highlight handle under cursor. Returns True if appearance changed."""
        addr = None
        try:
            picker = self._get_picker()
            if picker is not None:
                picker.Pick(x, y, 0, self._renderer)
                hit = picker.GetActor()
                if hit is not None:
                    candidate = hit.GetAddressAsString('')
                    if candidate in self._actor_tags:
                        addr = candidate
        except Exception:
            pass

        if addr == self._hovered_addr:
            return False

        self._dehighlight(self._hovered_addr)
        self._hovered_addr = addr
        self._highlight(addr)
        return True

    def _clear_hover(self) -> None:
        self._dehighlight(self._hovered_addr)
        self._hovered_addr = None

    def _highlight_one(self, addr: str) -> None:
        if addr not in self._actor_by_addr:
            return
        a = self._actor_by_addr[addr]
        p = a.GetProperty()
        p.SetAmbient(1.0)
        p.SetDiffuse(1.0)
        p.SetSpecular(0.4)
        p.SetOpacity(1.0)
        try:
            p.EdgeVisibilityOn()
            p.SetEdgeColor(1.0, 1.0, 1.0)
            p.SetLineWidth(1.5)
        except Exception:
            pass
        a.SetUserTransform(self._make_hover_xfm())

    def _dehighlight_one(self, addr: str) -> None:
        if addr not in self._actor_by_addr:
            return
        a = self._actor_by_addr[addr]
        p = a.GetProperty()
        orig = self._actor_orig_color.get(addr, (1., 1., 1.))
        p.SetColor(*orig)
        p.SetAmbient(0.5)
        p.SetDiffuse(0.5)
        p.SetSpecular(0.0)
        try:
            p.EdgeVisibilityOff()
        except Exception:
            pass
        if addr in self._actor_tags and self._actor_tags[addr][1] == 'R':
            p.SetOpacity(0.75)
        else:
            p.SetOpacity(1.0)
        a.SetUserTransform(self._pos_xfm)

    def _highlight(self, addr: str | None) -> None:
        if not addr:
            return
        self._highlight_one(addr)
        for peer in self._actor_peers.get(addr, []):
            self._highlight_one(peer)

    def _dehighlight(self, addr: str | None) -> None:
        if not addr:
            return
        self._dehighlight_one(addr)
        for peer in self._actor_peers.get(addr, []):
            self._dehighlight_one(peer)

    def _make_hover_xfm(self):
        """1.1× scale at world origin then translate to centroid."""
        try:
            import vtk as _vtk
            c = self._current_centroid()
            xfm = _vtk.vtkTransform()
            xfm.PostMultiply()
            xfm.Scale(1.1, 1.1, 1.1)
            xfm.Translate(float(c[0]), float(c[1]), float(c[2]))
            return xfm
        except Exception:
            return self._pos_xfm

    # ── Geometry ──────────────────────────────────────────────────────────────

    def _current_centroid(self) -> np.ndarray:
        return self._orig_centroid + np.array(self._cumul[:3])

    def _sync_pos(self) -> None:
        if self._pos_xfm is None:
            return
        c = self._current_centroid()
        self._pos_xfm.Identity()
        self._pos_xfm.Translate(float(c[0]), float(c[1]), float(c[2]))

    def _remove_all(self) -> None:
        renderer = self._renderer
        for actor in self._actors:
            try:
                renderer.RemoveActor(actor)
            except Exception:
                pass
        self._actors.clear()
        self._t_actors.clear()
        self._r_actors.clear()
        self._actor_tags.clear()
        self._actor_by_addr.clear()
        self._actor_orig_color.clear()
        self._actor_peers.clear()
        self._hovered_addr = None

    @staticmethod
    def _rot_y_to(target: np.ndarray):
        y = np.array([0., 1., 0.])
        t = np.asarray(target, float) / np.linalg.norm(target)
        cross = np.cross(y, t)
        sin_a = np.linalg.norm(cross)
        cos_a = float(np.dot(y, t))
        angle = float(np.degrees(np.arctan2(sin_a, cos_a)))
        if sin_a < 1e-6:
            return 0., [0., 0., 1.]
        return angle, (cross / sin_a).tolist()

    @staticmethod
    def _perp(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, float)
        return np.cross(v, [1., 0., 0.]) if abs(v[0]) < 0.9 else np.cross(v, [0., 1., 0.])

    def _build_actors(self, mode: str = 'both') -> None:
        """Build gizmo geometry at world origin. mode: 'T' | 'R' | 'both'."""
        try:
            import vtk as _vtk
        except ImportError:
            return
        if self._pos_xfm is None:
            return

        s        = self._scale
        renderer = self._renderer

        for color, adir in zip(self._AXIS_COLORS, self._AXIS_DIRS):

            # ── Translation arrow (cylinder shaft + cone tip as separate actors) ─
            if mode in ('T', 'both'):
                cyl = _vtk.vtkCylinderSource()
                cyl.SetRadius(s * 0.022)
                cyl.SetHeight(s * 0.70)
                cyl.SetResolution(10)
                cyl.Update()

                ang, rax = self._rot_y_to(adir)
                cxfm = _vtk.vtkTransform()
                cxfm.Translate(*(adir * s * 0.35).tolist())
                if abs(ang) > 0.01:
                    cxfm.RotateWXYZ(ang, *rax)

                cyl_f = _vtk.vtkTransformPolyDataFilter()
                cyl_f.SetInputConnection(cyl.GetOutputPort())
                cyl_f.SetTransform(cxfm)

                cyl_m = _vtk.vtkPolyDataMapper()
                cyl_m.SetInputConnection(cyl_f.GetOutputPort())
                cyl_a = _vtk.vtkActor()
                cyl_a.SetMapper(cyl_m)
                cyl_a.GetProperty().SetColor(*color)
                cyl_a.GetProperty().SetAmbient(0.5)
                cyl_a.GetProperty().SetDiffuse(0.5)
                cyl_a.SetUserTransform(self._pos_xfm)
                renderer.AddActor(cyl_a)

                cone = _vtk.vtkConeSource()
                cone.SetRadius(s * 0.075)
                cone.SetHeight(s * 0.30)
                cone.SetResolution(14)
                cone.SetDirection(*adir.tolist())
                cone.SetCenter(*(adir * s * 0.885).tolist())
                cone.Update()

                cone_m = _vtk.vtkPolyDataMapper()
                cone_m.SetInputConnection(cone.GetOutputPort())
                cone_a = _vtk.vtkActor()
                cone_a.SetMapper(cone_m)
                cone_a.GetProperty().SetColor(*color)
                cone_a.GetProperty().SetAmbient(0.5)
                cone_a.GetProperty().SetDiffuse(0.5)
                cone_a.SetUserTransform(self._pos_xfm)
                renderer.AddActor(cone_a)

                idx = len(self._t_actors)
                cyl_addr  = cyl_a.GetAddressAsString('')
                cone_addr = cone_a.GetAddressAsString('')
                for a, addr in ((cyl_a, cyl_addr), (cone_a, cone_addr)):
                    self._actor_tags[addr]       = (idx, 'T')
                    self._actor_by_addr[addr]    = a
                    self._actor_orig_color[addr] = color
                    self._actors.append(a)
                # peer links so hovering either part highlights both
                self._actor_peers[cyl_addr]  = [cone_addr]
                self._actor_peers[cone_addr] = [cyl_addr]
                self._t_actors.append(cyl_a)

            # ── Rotation arc ──────────────────────────────────────────────────
            if mode in ('R', 'both'):
                perp1 = self._perp(adir)
                perp2 = np.cross(adir, perp1)
                perp1 /= np.linalg.norm(perp1)
                perp2 /= np.linalg.norm(perp2)

                arc_r  = s * 1.30
                n_pts  = 44
                thetas = np.linspace(0., 1.5 * np.pi, n_pts)
                arc_pts = np.array([
                    arc_r * (np.cos(t) * perp1 + np.sin(t) * perp2)
                    for t in thetas
                ], dtype=np.float32)

                arc_pd = _vtk.vtkPolyData()
                pts = _vtk.vtkPoints()
                for p in arc_pts:
                    pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
                arc_pd.SetPoints(pts)

                cells = _vtk.vtkCellArray()
                for j in range(n_pts - 1):
                    cells.InsertNextCell(2)
                    cells.InsertCellPoint(j)
                    cells.InsertCellPoint(j + 1)
                arc_pd.SetLines(cells)

                tube = _vtk.vtkTubeFilter()
                tube.SetInputData(arc_pd)
                tube.SetRadius(s * 0.026)
                tube.SetNumberOfSides(10)

                am = _vtk.vtkPolyDataMapper()
                am.SetInputConnection(tube.GetOutputPort())

                aa = _vtk.vtkActor()
                aa.SetMapper(am)
                aa.GetProperty().SetColor(*color)
                aa.GetProperty().SetOpacity(0.75)
                aa.GetProperty().SetAmbient(0.5)
                aa.GetProperty().SetDiffuse(0.5)
                aa.SetUserTransform(self._pos_xfm)
                renderer.AddActor(aa)

                aaddr = aa.GetAddressAsString('')
                ridx  = len(self._r_actors)
                self._actor_tags[aaddr]       = (ridx, 'R')
                self._actor_by_addr[aaddr]    = aa
                self._actor_orig_color[aaddr] = color
                self._r_actors.append(aa)
                self._actors.append(aa)

    # ── Drag math ─────────────────────────────────────────────────────────────

    def _get_ray(self, x: int, y: int):
        renderer = self._renderer
        renderer.SetDisplayPoint(float(x), float(y), 0.0)
        renderer.DisplayToWorld()
        w0 = np.array(renderer.GetWorldPoint()[:3])
        renderer.SetDisplayPoint(float(x), float(y), 1.0)
        renderer.DisplayToWorld()
        w1 = np.array(renderer.GetWorldPoint()[:3])
        d  = w1 - w0
        n  = np.linalg.norm(d)
        return w0, (d / n if n > 1e-12 else np.array([0., 0., 1.]))

    def _axis_t(self, x: int, y: int, axis_idx: int,
                centroid: np.ndarray | None = None) -> float:
        ray_o, ray_d = self._get_ray(x, y)
        c  = centroid if centroid is not None else self._current_centroid()
        ad = self._AXIS_DIRS[axis_idx]
        w0 = ray_o - c
        b  = float(np.dot(ray_d, ad))
        d  = float(np.dot(ray_d, w0))
        e  = float(np.dot(ad,    w0))
        den = 1. - b * b
        return (e - b * d) / den if abs(den) > 1e-10 else 0.

    def _screen_angle(self, x: int, y: int,
                      centroid: np.ndarray | None = None) -> float:
        renderer = self._renderer
        c = centroid if centroid is not None else self._current_centroid()
        renderer.SetWorldPoint(float(c[0]), float(c[1]), float(c[2]), 1.0)
        renderer.WorldToDisplay()
        dp = renderer.GetDisplayPoint()
        return float(np.degrees(np.arctan2(y - dp[1], x - dp[0])))

    # ── Arc indicator dot ─────────────────────────────────────────────────────

    def _arc_world_point(self, x: int, y: int, axis_idx: int) -> np.ndarray | None:
        """Project pick ray onto rotation plane; return closest point on arc circle."""
        ray_o, ray_d = self._get_ray(x, y)
        c  = self._drag_start_centroid
        ad = self._AXIS_DIRS[axis_idx]
        denom = float(np.dot(ray_d, ad))
        if abs(denom) < 1e-8:
            return None
        t   = float(np.dot(c - ray_o, ad)) / denom
        hit = ray_o + t * ray_d
        v   = hit - c
        v  -= float(np.dot(v, ad)) * ad           # remove axis component
        r   = np.linalg.norm(v)
        if r < 1e-8:
            return None
        return c + (v / r) * (self._scale * 1.30) # project onto arc radius

    def _update_dot(self, x: int, y: int, axis_idx: int) -> None:
        pt = self._arc_world_point(x, y, axis_idx)
        if pt is None:
            return
        try:
            import vtk as _vtk
            if self._drag_dot_actor is None:
                src = _vtk.vtkSphereSource()
                src.SetRadius(self._scale * 0.055)
                src.SetPhiResolution(10)
                src.SetThetaResolution(10)
                src.Update()
                mapper = _vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(src.GetOutputPort())
                actor = _vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(1.0, 1.0, 0.2)   # bright yellow
                actor.GetProperty().SetAmbient(1.0)
                actor.GetProperty().SetDiffuse(0.0)
                self._renderer.AddActor(actor)
                self._drag_dot_actor = actor
            xfm = _vtk.vtkTransform()
            xfm.Translate(float(pt[0]), float(pt[1]), float(pt[2]))
            self._drag_dot_actor.SetUserTransform(xfm)
        except Exception:
            pass

    def _remove_dot(self) -> None:
        if self._drag_dot_actor is not None:
            try:
                self._renderer.RemoveActor(self._drag_dot_actor)
            except Exception:
                pass
            self._drag_dot_actor = None


# ─────────────────────────────────────────────────────────────────────────────
# Qt event filter — intercepts mouse events BEFORE VTK sees them
# ─────────────────────────────────────────────────────────────────────────────

class _GizmoEventFilter(QObject):
    """
    Installed on the pyvistaqt plotter widget via installEventFilter().

    When the gizmo is visible this filter intercepts left-button press/move/
    release events at the Qt level — before they reach VTK.  Returning True
    consumes the event so VTK never enters its camera-rotate state.

    Right-button (dolly) and middle-button (pan) events are never touched.
    Mouse-move events are always forwarded to VTK after gizmo hover processing
    (so VTK can update its cursor / internal state).

    Coordinate note
    ---------------
    Qt:  (0, 0) is top-left, y increases downward.
    VTK: (0, 0) is bottom-left, y increases upward.
    Conversion: vtk_y = widget_height - qt_y - 1
    """

    def __init__(self, gizmo: _TransformGizmo, plotter):
        super().__init__()
        self._gizmo    = gizmo
        self._plotter  = plotter
        self._dragging = False
        self._panning  = False
        self._pan_last: tuple | None = None

    def _to_vtk(self, widget, qt_x: int, qt_y: int):
        # VTK's display coordinate system matches the PHYSICAL framebuffer
        # (confirmed: WorldToDisplay returns physical-pixel coords on Retina).
        # Qt event coords are logical pixels — scale by DPR to get physical.
        try:
            ratio = widget.devicePixelRatioF()
        except AttributeError:
            ratio = 1.0
        return (
            int(round(qt_x * ratio)),
            int(round((widget.height() - qt_y - 1) * ratio)),
        )

    def eventFilter(self, obj, event) -> bool:
        g = self._gizmo
        t = event.type()

        # ── Right button — camera pan (two-finger click+drag on Mac) ────────
        if t == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
            self._panning = True
            self._pan_last = (event.x(), event.y())
            return True

        if t == QEvent.MouseButtonRelease and event.button() == Qt.RightButton:
            self._panning = False
            self._pan_last = None
            return True

        # ── Left button press ─────────────────────────────────────────────────
        if t == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            if g._visible:
                x, y = self._to_vtk(obj, event.x(), event.y())
                axis_idx, mode = g._pick_actor(x, y)
                if axis_idx >= 0:
                    g._start_drag(mode, axis_idx, x, y)
                    self._dragging = True
                    return True   # consume — VTK never enters rotate state
                # Gizmo visible but no handle under cursor — freeze camera
                return True

        # ── Mouse move ────────────────────────────────────────────────────────
        elif t == QEvent.MouseMove:
            if self._panning and self._pan_last is not None:
                dx = event.x() - self._pan_last[0]
                dy = event.y() - self._pan_last[1]
                self._pan_last = (event.x(), event.y())
                try:
                    cam  = self._plotter.renderer.GetActiveCamera()
                    pos  = np.array(cam.GetPosition())
                    foc  = np.array(cam.GetFocalPoint())
                    dist = max(np.linalg.norm(foc - pos), 1e-10)
                    view = (foc - pos) / dist
                    up   = np.array(cam.GetViewUp())
                    right = np.cross(view, up)
                    rn = np.linalg.norm(right)
                    if rn > 1e-10:
                        right /= rn
                    scale  = dist * 0.002
                    pan_3d = (-dx * right + dy * up) * scale
                    cam.SetPosition(*(pos + pan_3d).tolist())
                    cam.SetFocalPoint(*(foc + pan_3d).tolist())
                    self._plotter.render()
                except Exception:
                    pass
                return True

            x, y = self._to_vtk(obj, event.x(), event.y())
            if self._dragging and g._drag_mode is not None:
                g._update_drag(x, y)
                # apply_coil_transform (called via _cb inside _update_drag) already renders
                return True       # consume during active drag
            elif g._visible:
                if g._update_hover(x, y):
                    try:
                        self._plotter.render()
                    except Exception:
                        pass
            return False          # let VTK see mouse moves when not dragging

        # ── Left button release ───────────────────────────────────────────────
        elif t == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            if self._dragging:
                g._end_drag()
                self._dragging = False
                return True

        return False




# ─────────────────────────────────────────────────────────────────────────────
# 3-D workspace
# ─────────────────────────────────────────────────────────────────────────────

class Workspace3DView(QWidget):
    """Full-bleed PyVista 3-D workspace embedded in a Qt widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layers:        dict[tuple, _Layer] = {}   # analysis layers keyed by (coil_id, name)
        self._coil_entries:  dict[str, dict]   = {}   # coil_id → {actors, coords, color}
        self._active_coil_id: str | None = None
        self._floor_actors: list = []
        self._gizmo:        _TransformGizmo | None = None
        self._event_filter: _GizmoEventFilter | None = None
        self._hint_label    = None
        self._stale_callback = None   # called when a coil is transformed after analysis

        if not _HAS_PYVISTA:
            lbl = QLabel(
                "PyVista not installed.\nRun:  pip install pyvista pyvistaqt", self,
            )
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                f"color:{THEME['warning']}; font-size:11pt; background:transparent;"
            )
            QVBoxLayout(self).addWidget(lbl)
            self._plotter = None
            return

        self._plotter = QtInteractor(self)
        self._plotter.set_background(THEME['bg'])
        try:
            self._plotter.enable_anti_aliasing('ssaa')
        except Exception:
            try:
                self._plotter.enable_anti_aliasing()
            except Exception:
                pass
        self._plotter.hide_axes()

        # Gizmo overlay renderer — sits on VTK layer 1 above the main scene.
        # By preserving the color buffer (keeps layer-0 scene intact) but using
        # a FRESH depth buffer, gizmo actors are never occluded by scene geometry
        # (field lines, cross sections, etc.).  Picking against this renderer
        # also avoids accidentally hitting dense scene meshes.
        _gizmo_rend = self._plotter.renderer   # safe fallback
        try:
            import vtk as _vtk_ov
            _gr = _vtk_ov.vtkRenderer()
            _gr.SetLayer(1)
            _gr.SetInteractive(0)
            try:
                _gr.PreserveColorBufferOn()   # don't erase layer-0 colors
                _gr.PreserveDepthBufferOff()  # fresh depth → gizmo always in front
            except AttributeError:
                _gr.EraseOff()               # older VTK fallback
            _gr.SetActiveCamera(self._plotter.renderer.GetActiveCamera())
            try:
                self._plotter.ren_win.SetNumberOfLayers(2)
                self._plotter.ren_win.AddRenderer(_gr)
            except AttributeError:
                self._plotter.render_window.SetNumberOfLayers(2)
                self._plotter.render_window.AddRenderer(_gr)
            _gizmo_rend = _gr
        except Exception:
            pass
        self._gizmo_renderer = _gizmo_rend

        # Gizmo + Qt event filter (intercepts mouse before VTK acts on them).
        # On macOS/VTK 9.6, Python virtual method overrides on VTK style classes
        # do not fire — the Qt event filter is the only reliable mechanism.
        self._gizmo        = _TransformGizmo(self._gizmo_renderer, self._on_gizmo_transform)
        self._event_filter = _GizmoEventFilter(self._gizmo, self._plotter)
        self._plotter.installEventFilter(self._event_filter)

        # Navigation ViewCube (Fusion 360-style orientation widget)
        self._viewcube = None
        self._setup_viewcube()

        self._hint_label = QLabel("Load a coil CSV to begin", self._plotter)
        self._hint_label.setAlignment(Qt.AlignCenter)
        self._hint_label.setStyleSheet(
            f"color:{THEME['text_dim']}; font-size:14pt; background:transparent;"
        )
        self._hint_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._hint_label.resize(self._plotter.size())

        self._toolbar = self._make_toolbar()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._plotter, stretch=1)
        lay.addWidget(self._toolbar)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._plotter is not None and self._hint_label is not None:
            self._hint_label.resize(self._plotter.size())

    # ── Theme ─────────────────────────────────────────────────────────────────

    def apply_theme(self) -> None:
        """Update 3D viewport colours to match the current THEME."""
        if self._plotter is None:
            return
        self._plotter.set_background(THEME['bg'])
        # Update floor grid colour
        for a in self._floor_actors:
            try:
                h = THEME.get('floor', '#606060').lstrip('#')
                r, g, b = (int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
                a.GetProperty().SetColor(r, g, b)
            except Exception:
                pass
        # Recreate ViewCube so it picks up the new background contrast
        self._setup_viewcube()
        # Update scalar bar text colours on all existing layers
        sb_hex = THEME.get('sb_text', '#ffffff').lstrip('#')
        sb_r = int(sb_hex[0:2], 16) / 255.0
        sb_g = int(sb_hex[2:4], 16) / 255.0
        sb_b = int(sb_hex[4:6], 16) / 255.0
        for layer in self._layers.values():
            if layer.scalar_bar is not None:
                try:
                    for prop in (layer.scalar_bar.GetTitleTextProperty(),
                                 layer.scalar_bar.GetLabelTextProperty()):
                        prop.SetColor(sb_r, sb_g, sb_b)
                except Exception:
                    pass
        # Update hint label if still visible
        if self._hint_label is not None:
            self._hint_label.setStyleSheet(
                f"color:{THEME['text_dim']}; font-size:14pt; background:transparent;"
            )
        # Update toolbar bar + buttons
        self._toolbar.setStyleSheet(
            f"background:{THEME['panel']}; border-top:1px solid {THEME['border']};"
        )
        _sty = self._toolbar_btn_style()
        self._btn_reset_view.setStyleSheet(_sty)
        self._btn_save.setStyleSheet(_sty)
        self._plotter.render()

    def _setup_viewcube(self) -> None:
        """Create (or recreate) the Fusion 360-style navigation cube, themed."""
        if self._plotter is None:
            return
        if self._viewcube is not None:
            try:
                self._viewcube.Off()
            except Exception:
                pass
            self._viewcube = None
        try:
            import vtk as _vtk_vc
            vc = _vtk_vc.vtkCameraOrientationWidget()
            vc.SetParentRenderer(self._plotter.renderer)

            rep = vc.GetRepresentation()
            is_light = THEME.get('mode') == 'light'

            # Cube body
            cp = rep.GetContainerProperty()
            if is_light:
                cp.SetColor(0.88, 0.88, 0.90)
                cp.SetOpacity(0.85)
                cp.SetEdgeColor(0.5, 0.5, 0.55)
            else:
                cp.SetColor(0.22, 0.22, 0.24)
                cp.SetOpacity(0.85)
                cp.SetEdgeColor(0.45, 0.45, 0.50)
            try:
                cp.EdgeVisibilityOn()
            except Exception:
                pass

            # Face labels
            lbl_color = (0.15, 0.15, 0.18) if is_light else (0.92, 0.92, 0.92)
            for getter in ('GetXPlusLabelProperty', 'GetXMinusLabelProperty',
                           'GetYPlusLabelProperty', 'GetYMinusLabelProperty',
                           'GetZPlusLabelProperty', 'GetZMinusLabelProperty'):
                try:
                    lp = getattr(rep, getter)()
                    lp.SetColor(*lbl_color)
                    lp.SetFontSize(14)
                    lp.BoldOn()
                except Exception:
                    pass

            # Axis handle colours (subtle tints)
            if is_light:
                rep.SetXAxisColor(0.75, 0.25, 0.25)
                rep.SetYAxisColor(0.25, 0.65, 0.25)
                rep.SetZAxisColor(0.25, 0.40, 0.80)
            else:
                rep.SetXAxisColor(0.90, 0.30, 0.30)
                rep.SetYAxisColor(0.30, 0.85, 0.30)
                rep.SetZAxisColor(0.35, 0.55, 1.00)

            rep.SetContainerVisibility(True)
            vc.On()
            self._viewcube = vc
        except Exception:
            pass

    # ── Gizmo callback ────────────────────────────────────────────────────────

    def _on_gizmo_transform(self, tx, ty, tz, rx, ry, rz) -> None:
        self.apply_coil_transform(tx, ty, tz, rx, ry, rz)

    # ── Floor grid ────────────────────────────────────────────────────────────

    def _rebuild_floor(self, coords: np.ndarray) -> None:
        if self._plotter is None:
            return
        for a in self._floor_actors:
            try:
                self._plotter.remove_actor(a, render=False)
            except Exception:
                pass
        self._floor_actors.clear()

        z_min = float(coords[:, 2].min())
        diam  = float(max(_ptp(coords[:, :2], axis=0).max(), 0.01))
        z_fl  = z_min - 0.25 * diam
        size  = max(diam * 10.0, 1.0)
        cx    = float(coords[:, 0].mean())
        cy    = float(coords[:, 1].mean())

        plane = pv.Plane(
            center=(cx, cy, z_fl),
            direction=(0., 0., 1.),
            i_size=size,
            j_size=size,
            i_resolution=24,
            j_resolution=24,
        )
        a = self._plotter.add_mesh(
            plane,
            style='wireframe',
            color=THEME.get('floor', '#606060'),
            line_width=1.1,
            opacity=0.70,
            render=False,
        )
        self._floor_actors.append(a)

    # ── Multi-coil API ────────────────────────────────────────────────────────

    @staticmethod
    def _build_tube_mesh(coords: np.ndarray, total_thickness: float,
                          tape_width: float) -> 'pv.PolyData | None':
        """Build a swept rectangular cross-section mesh along *coords*.

        Returns a pv.PolyData with quad faces, or None on failure.

        REBCO winding: tape wraps around the coil form, each turn adding
        one tape_thickness radially.  Therefore:
          - **radial** extent = total_thickness = winds × tape_thickness
                                (turns stack outward from the coil form)
          - **axial** extent  = tape_width  (ribbon width runs along the axis)
        """
        try:
            c = np.asarray(coords, dtype=np.float64)
            n = len(c)
            if n < 3:
                return None

            # Local frame per vertex
            centroid = c.mean(axis=0)
            tangents = np.empty_like(c)
            tangents[1:-1] = c[2:] - c[:-2]                     # central diff
            tangents[0]    = c[1] - c[0]
            tangents[-1]   = c[-1] - c[-2]
            t_mag = np.linalg.norm(tangents, axis=1, keepdims=True).clip(1e-12)
            tangents /= t_mag

            # Radial: outward from centroid, perpendicular to tangent
            dx     = c - centroid
            proj   = np.einsum('ij,ij->i', dx, tangents)[:, None] * tangents
            radial = dx - proj
            r_mag  = np.linalg.norm(radial, axis=1, keepdims=True).clip(1e-12)
            e_r    = radial / r_mag                               # (n, 3)
            # Binormal = tangent × radial (≈ axial direction)
            e_b    = np.cross(tangents, e_r)
            b_mag  = np.linalg.norm(e_b, axis=1, keepdims=True).clip(1e-12)
            e_b   /= b_mag

            # REBCO: total_thickness is radial, tape_width is axial
            hr = total_thickness * 0.5    # half radial extent
            ha = tape_width * 0.5         # half axial extent

            # 4 corner offsets: (radial, axial)
            corners = [
                (-hr, -ha),
                ( hr, -ha),
                ( hr,  ha),
                (-hr,  ha),
            ]

            # Build vertex array: n_verts × 4 corners
            all_pts = np.empty((n * 4, 3), dtype=np.float32)
            for ci, (dr, db) in enumerate(corners):
                all_pts[ci::4] = (c + dr * e_r + db * e_b).astype(np.float32)

            # Build quad faces connecting adjacent cross-sections
            faces = []
            for i in range(n - 1):
                base0 = i * 4
                base1 = (i + 1) * 4
                for j in range(4):
                    j1 = (j + 1) % 4
                    faces.extend([4, base0 + j, base0 + j1, base1 + j1, base1 + j])

            mesh = pv.PolyData(all_pts, np.array(faces, dtype=np.int32))
            mesh.compute_normals(inplace=True, auto_orient_normals=True)
            return mesh
        except Exception:
            return None

    def add_coil(self, coords: np.ndarray, coil_id: str, color: str = None,
                  total_thickness: float = 0.0, tape_width: float = 0.0) -> None:
        """Add (or replace) a coil in the scene without clearing other coils.

        If total_thickness and tape_width are > 0, the coil renders as a
        swept rectangular tube (winding-pack cross-section).  Otherwise it
        falls back to a thin spline wire.
        """
        if self._plotter is None:
            return

        coil_color = color or THEME['accent']
        c = np.asarray(coords, dtype=float)

        if coil_id in self._coil_entries:
            for actor in self._coil_entries[coil_id]['actors']:
                try:
                    self._plotter.remove_actor(actor, render=False)
                except Exception:
                    pass

        if self._hint_label is not None:
            self._hint_label.hide()
            self._hint_label = None

        actors = []
        tube_ok = total_thickness > 0 and tape_width > 0
        if tube_ok:
            tube_mesh = self._build_tube_mesh(c, total_thickness, tape_width)
            if tube_mesh is not None:
                a_tube = self._plotter.add_mesh(
                    tube_mesh, color=coil_color, opacity=0.55,
                    show_edges=True, edge_color=THEME.get('edge', '#404040'),
                    reset_camera=False, render=False,
                )
                actors.append(a_tube)
        # Always add a centerline wire (thin if tube present, thicker if not)
        n_pts  = max(len(c) * 3, 400)
        spline = pv.Spline(c, n_points=n_pts)
        a_wire = self._plotter.add_mesh(
            spline,
            color=coil_color,
            line_width=1.5 if tube_ok else 3.0,
            render=False,
        )
        actors.append(a_wire)

        self._coil_entries[coil_id] = {
            'actors':            actors,
            'coords':            c,
            'color':             coil_color,
            'xfm_params':        None,  # (tx,ty,tz,rx,ry,rz,cx,cy,cz) when transformed
            'analysis_xfm_params': None,  # snapshot of xfm_params at last analysis time
        }
        self._active_coil_id = coil_id

        if self._gizmo is not None:
            centroid = c.mean(axis=0)
            bbox     = float(_ptp(c, axis=0).max())
            self._gizmo.load(centroid, max(bbox * 0.18, 0.05))

        all_coords = np.vstack([e['coords'] for e in self._coil_entries.values()])
        self._rebuild_floor(all_coords)

        if len(self._coil_entries) == 1:
            self._plotter.reset_camera()
        self._plotter.render()

    def remove_coil(self, coil_id: str) -> None:
        if coil_id not in self._coil_entries or self._plotter is None:
            return
        for actor in self._coil_entries[coil_id]['actors']:
            try:
                self._plotter.remove_actor(actor, render=False)
            except Exception:
                pass
        # Also remove analysis layers belonging to this coil
        keys_to_del = [k for k in self._layers if k[0] == coil_id]
        for k in keys_to_del:
            self._remove_layer(k[0], k[1])
        del self._coil_entries[coil_id]

        if self._active_coil_id == coil_id:
            if self._coil_entries:
                self._active_coil_id = next(iter(self._coil_entries))
                entry = self._coil_entries[self._active_coil_id]
                if self._gizmo is not None:
                    c    = entry['coords']
                    bbox = float(_ptp(c, axis=0).max())
                    self._gizmo.load(c.mean(axis=0), max(bbox * 0.18, 0.05))
            else:
                self._active_coil_id = None
                if self._gizmo is not None:
                    self._gizmo.hide()

        if self._coil_entries:
            all_coords = np.vstack([e['coords'] for e in self._coil_entries.values()])
            self._rebuild_floor(all_coords)

        self._plotter.render()

    def set_active_coil(self, coil_id: str) -> None:
        if coil_id not in self._coil_entries:
            return
        self._active_coil_id = coil_id
        entry = self._coil_entries[coil_id]
        if self._gizmo is not None:
            c    = entry['coords']
            bbox = float(_ptp(c, axis=0).max())
            self._gizmo.load(c.mean(axis=0), max(bbox * 0.18, 0.05))
            # Restore any previously applied transform so gizmo matches coil position
            params = entry.get('xfm_params')
            if params is not None:
                tx, ty, tz, rx, ry, rz = params[:6]
                self._gizmo._cumul = [tx, ty, tz, rx, ry, rz]
                self._gizmo._sync_pos()

    def update_coil_mesh(self, coil_id: str,
                          total_thickness: float, tape_width: float) -> None:
        """Rebuild a coil's visual mesh (tube + wire) after parameter changes,
        preserving transforms and analysis layers."""
        entry = self._coil_entries.get(coil_id)
        if entry is None or self._plotter is None:
            return

        c          = entry['coords']
        coil_color = entry['color']
        xfm        = entry.get('xfm_params')

        # Remove old actors
        for actor in entry['actors']:
            try:
                self._plotter.remove_actor(actor, render=False)
            except Exception:
                pass

        # Rebuild mesh
        actors = []
        tube_ok = total_thickness > 0 and tape_width > 0
        if tube_ok:
            tube_mesh = self._build_tube_mesh(c, total_thickness, tape_width)
            if tube_mesh is not None:
                a_tube = self._plotter.add_mesh(
                    tube_mesh, color=coil_color, opacity=0.55,
                    show_edges=True, edge_color=THEME.get('edge', '#404040'),
                    reset_camera=False, render=False,
                )
                actors.append(a_tube)
        n_pts  = max(len(c) * 3, 400)
        spline = pv.Spline(c, n_points=n_pts)
        a_wire = self._plotter.add_mesh(
            spline, color=coil_color,
            line_width=1.5 if tube_ok else 3.0,
            reset_camera=False, render=False,
        )
        actors.append(a_wire)

        entry['actors'] = actors

        # Re-apply existing transform to new actors
        if xfm is not None:
            try:
                try:
                    from vtkmodules.vtkCommonTransforms import vtkTransform
                except ImportError:
                    import vtk as _v
                    vtkTransform = _v.vtkTransform
                tx, ty, tz, rx, ry, rz, cx, cy, cz = xfm
                t = vtkTransform()
                t.PostMultiply()
                t.Translate(-cx, -cy, -cz)
                t.RotateX(rx); t.RotateY(ry); t.RotateZ(rz)
                t.Translate(cx + tx, cy + ty, cz + tz)
                for actor in actors:
                    actor.SetUserTransform(t)
            except Exception:
                pass

        self._plotter.render()

    def set_coil_color(self, coil_id: str, color: str) -> None:
        """Change the wire color of a coil actor."""
        if coil_id not in self._coil_entries or self._plotter is None:
            return
        try:
            h = color.lstrip('#')
            r, g, b = (int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        except Exception:
            return
        self._coil_entries[coil_id]['color'] = color
        for actor in self._coil_entries[coil_id]['actors']:
            try:
                actor.GetProperty().SetColor(r, g, b)
            except Exception:
                pass
        self._plotter.render()

    def set_coil_visible(self, coil_id: str, visible: bool) -> None:
        if coil_id not in self._coil_entries or self._plotter is None:
            return
        for actor in self._coil_entries[coil_id]['actors']:
            _set_visible(actor, visible)
        self._plotter.render()

    def set_stale_callback(self, cb) -> None:
        """Register a zero-arg callable fired whenever the active coil is transformed."""
        self._stale_callback = cb

    # ── Gizmo control ─────────────────────────────────────────────────────────

    def show_gizmo(self, mode: str) -> None:
        """Show transform handles. The Qt event filter handles all interaction."""
        if self._plotter is None or self._gizmo is None:
            return
        self._gizmo.show(mode)
        self._plotter.render()

    def hide_gizmo(self) -> None:
        if self._plotter is None or self._gizmo is None:
            return
        self._gizmo.hide()
        # Clear stale drag state so next show() starts fresh.
        if self._event_filter is not None:
            self._event_filter._dragging = False
        self._plotter.render()

    def apply_coil_transform(
        self,
        tx: float = 0., ty: float = 0., tz: float = 0.,
        rx: float = 0., ry: float = 0., rz: float = 0.,
    ) -> None:
        if self._active_coil_id not in self._coil_entries or self._plotter is None:
            return
        try:
            try:
                from vtkmodules.vtkCommonTransforms import vtkTransform
            except ImportError:
                import vtk as _v
                vtkTransform = _v.vtkTransform

            cx = cy = cz = 0.0
            if self._gizmo is not None:
                cx, cy, cz = map(float, self._gizmo._orig_centroid)

            t = vtkTransform()
            t.PostMultiply()
            t.Translate(-cx, -cy, -cz)
            t.RotateX(rx)
            t.RotateY(ry)
            t.RotateZ(rz)
            t.Translate(cx + tx, cy + ty, cz + tz)

            for actor in self._coil_entries[self._active_coil_id]['actors']:
                actor.SetUserTransform(t)

            # Move analysis layers using a DELTA transform so their already-world-space
            # geometry moves correctly.  The geometry was placed at analysis-time world
            # positions (T_analysis applied to original coords).  We need:
            #   delta = T_new ∘ T_analysis⁻¹
            # so that applying delta to analysis-time positions gives new positions.
            entry = self._coil_entries[self._active_coil_id]
            ap = entry.get('analysis_xfm_params')  # transform snapshot at analysis time

            delta = vtkTransform()
            delta.PostMultiply()
            if ap is not None:
                tx_a, ty_a, tz_a, rx_a, ry_a, rz_a, cx_a, cy_a, cz_a = ap
                # Undo T_analysis (reverse order, negated)
                delta.Translate(-(cx_a + tx_a), -(cy_a + ty_a), -(cz_a + tz_a))
                delta.RotateZ(-rz_a)
                delta.RotateY(-ry_a)
                delta.RotateX(-rx_a)
                delta.Translate(cx_a, cy_a, cz_a)
            # Apply T_new
            delta.Translate(-cx, -cy, -cz)
            delta.RotateX(rx)
            delta.RotateY(ry)
            delta.RotateZ(rz)
            delta.Translate(cx + tx, cy + ty, cz + tz)

            for (cid, _nm), layer in self._layers.items():
                if cid == self._active_coil_id:
                    for actor in layer.actors:
                        actor.SetUserTransform(delta)

            # Persist transform params so physics can use world-space coords
            self._coil_entries[self._active_coil_id]['xfm_params'] = (
                tx, ty, tz, rx, ry, rz, cx, cy, cz
            )
        except Exception:
            pass
        if self._stale_callback is not None:
            try:
                self._stale_callback()
            except Exception:
                pass
        self._plotter.render()

    def reset_coil_transform(self) -> None:
        if self._active_coil_id not in self._coil_entries or self._plotter is None:
            return
        entry = self._coil_entries[self._active_coil_id]
        try:
            # Coil actors: remove transform (back to original CSV geometry)
            for actor in entry['actors']:
                actor.SetUserTransform(None)

            # Layer actors: their geometry is at analysis-time world positions.
            # If analysis was done after a move, we must apply T_analysis⁻¹ so the
            # layers snap back to the original position alongside the coil.
            ap = entry.get('analysis_xfm_params')
            if ap is not None:
                try:
                    from vtkmodules.vtkCommonTransforms import vtkTransform
                except ImportError:
                    import vtk as _v
                    vtkTransform = _v.vtkTransform
                tx_a, ty_a, tz_a, rx_a, ry_a, rz_a, cx_a, cy_a, cz_a = ap
                inv = vtkTransform()
                inv.PostMultiply()
                inv.Translate(-(cx_a + tx_a), -(cy_a + ty_a), -(cz_a + tz_a))
                inv.RotateZ(-rz_a)
                inv.RotateY(-ry_a)
                inv.RotateX(-rx_a)
                inv.Translate(cx_a, cy_a, cz_a)
                for (cid, _nm), layer in self._layers.items():
                    if cid == self._active_coil_id:
                        for actor in layer.actors:
                            actor.SetUserTransform(inv)
            else:
                # Analysis was at original position — identity is correct
                for (cid, _nm), layer in self._layers.items():
                    if cid == self._active_coil_id:
                        for actor in layer.actors:
                            actor.SetUserTransform(None)
            entry['xfm_params'] = None
        except Exception:
            pass
        if self._gizmo is not None:
            self._gizmo.reset()
        self._plotter.render()

    def get_transformed_coords(self, coil_id: str) -> 'np.ndarray | None':
        """Return coil coords in world space after applying the stored gizmo transform."""
        if coil_id not in self._coil_entries:
            return None
        entry = self._coil_entries[coil_id]
        coords = entry['coords'].astype(float)
        params = entry.get('xfm_params')
        if params is None:
            return coords
        tx, ty, tz, rx, ry, rz, cx, cy, cz = params
        try:
            from scipy.spatial.transform import Rotation
            shifted  = coords - np.array([cx, cy, cz])
            rotated  = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True).apply(shifted)
            return rotated + np.array([cx + tx, cy + ty, cz + tz])
        except ImportError:
            # Fallback: translation only (no rotation decomposition without scipy)
            return coords + np.array([tx, ty, tz])

    def mark_analysis_transform(self, coil_id: str) -> None:
        """Snapshot current xfm_params as the reference for analysis layer delta math."""
        if coil_id in self._coil_entries:
            entry = self._coil_entries[coil_id]
            entry['analysis_xfm_params'] = entry.get('xfm_params')

    def reapply_coil_transform(self, coil_id: str) -> None:
        """Re-apply the stored delta transform to all existing layers for a coil.
        Call this after replacing a layer (e.g. normalize toggle) so the new actor
        respects any translation/rotation that was applied since analysis."""
        entry = self._coil_entries.get(coil_id)
        if entry is None or self._plotter is None:
            return
        xfm_params = entry.get('xfm_params')
        if xfm_params is None:
            return  # coil never moved — layers already at correct position
        ap = entry.get('analysis_xfm_params')
        try:
            try:
                from vtkmodules.vtkCommonTransforms import vtkTransform
            except ImportError:
                import vtk as _v
                vtkTransform = _v.vtkTransform
            tx, ty, tz, rx, ry, rz, cx, cy, cz = xfm_params
            delta = vtkTransform()
            delta.PostMultiply()
            if ap is not None:
                tx_a, ty_a, tz_a, rx_a, ry_a, rz_a, cx_a, cy_a, cz_a = ap
                delta.Translate(-(cx_a + tx_a), -(cy_a + ty_a), -(cz_a + tz_a))
                delta.RotateZ(-rz_a)
                delta.RotateY(-ry_a)
                delta.RotateX(-rx_a)
                delta.Translate(cx_a, cy_a, cz_a)
            delta.Translate(-cx, -cy, -cz)
            delta.RotateX(rx)
            delta.RotateY(ry)
            delta.RotateZ(rz)
            delta.Translate(cx + tx, cy + ty, cz + tz)
            for (cid_key, _nm), layer in self._layers.items():
                if cid_key == coil_id:
                    for actor in layer.actors:
                        actor.SetUserTransform(delta)
        except Exception:
            pass
        self._plotter.render()

    # ── Analysis layer API ────────────────────────────────────────────────────

    def add_force_layer(self, engine, coil_id: str, normalized: bool = False) -> None:
        if self._plotter is None:
            return
        name = 'Forces'
        self._remove_layer(coil_id, name)

        midpoints = np.asarray(engine.midpoints, dtype=float)
        F_vecs    = np.asarray(engine.F_vecs,    dtype=float)
        coords    = np.asarray(engine.coords,    dtype=float)
        bbox      = float(_ptp(coords, axis=0).max())
        arrow_len = bbox * 0.07

        mags    = np.linalg.norm(F_vecs, axis=1)
        max_mag = max(float(mags.max()), 1e-30)

        n    = len(midpoints)
        step = max(1, n // 300)
        mp   = np.ascontiguousarray(midpoints[::step], dtype=np.float32)
        ms   = np.ascontiguousarray(mags[::step],      dtype=np.float32)

        unit_vecs = F_vecs / mags[:, None].clip(1e-30)
        uv = np.ascontiguousarray(unit_vecs[::step], dtype=np.float32)

        cloud = pv.PolyData(mp)
        cloud['vectors']   = uv
        cloud['magnitude'] = ms
        cloud['mag_norm']  = (ms / max_mag).astype(np.float32)
        cloud.set_active_vectors('vectors')

        try:
            if normalized:
                glyphs = cloud.glyph(orient='vectors', scale=False,
                                     factor=arrow_len, geom=pv.Arrow())
            else:
                glyphs = cloud.glyph(orient='vectors', scale='mag_norm',
                                     factor=arrow_len, geom=pv.Arrow())
            a = self._plotter.add_mesh(
                glyphs, scalars='magnitude', cmap=THEME.get('cmap_forces', 'plasma'),
                clim=[float(mags.min()), float(mags.max())],
                show_scalar_bar=False, reset_camera=False, render=False,
            )
        except Exception:
            a = self._plotter.add_mesh(
                cloud, scalars='magnitude', cmap=THEME.get('cmap_forces', 'plasma'),
                point_size=6, show_scalar_bar=False,
                reset_camera=False, render=False,
            )

        self._layers[(coil_id, name)] = _Layer(name, [a])
        self._plotter.render()

    def add_stress_layer(self, engine, coil_id: str) -> None:
        if self._plotter is None:
            return
        name = 'Stress'
        self._remove_layer(coil_id, name)

        midpoints = np.asarray(engine.midpoints, dtype=float)
        stress    = np.asarray(engine.hoop_stress, dtype=float) / 1e6

        cloud = pv.PolyData(np.ascontiguousarray(midpoints, dtype=np.float32))
        cloud['stress_MPa'] = stress.astype(np.float32)

        a = self._plotter.add_mesh(
            cloud, scalars='stress_MPa', cmap=THEME.get('cmap_stress', 'YlOrRd'),
            point_size=8, render_points_as_spheres=True,
            show_scalar_bar=False, reset_camera=False, render=False,
        )
        self._layers[(coil_id, name)] = _Layer(name, [a])
        self._plotter.render()

    def add_axis_layer(self, engine, coil_id: str) -> None:
        if self._plotter is None:
            return
        name = 'B Axis'
        self._remove_layer(coil_id, name)

        if engine.bfield_axis_z is None or engine.bfield_axis_mag is None:
            return

        z_vals = np.asarray(engine.bfield_axis_z,  dtype=float)
        b_vals = np.asarray(engine.bfield_axis_mag, dtype=float)
        a_dir  = np.asarray(engine.axis,            dtype=float)
        m_pt   = np.asarray(engine.mean_point,      dtype=float)
        pts    = m_pt[np.newaxis, :] + np.outer(z_vals, a_dir)

        cloud = pv.PolyData(np.ascontiguousarray(pts, dtype=np.float32))
        cloud['B_mag'] = b_vals.astype(np.float32)

        a = self._plotter.add_mesh(
            cloud, scalars='B_mag', cmap=THEME.get('cmap_field', 'cool'),
            point_size=6, render_points_as_spheres=True,
            show_scalar_bar=False, reset_camera=False, render=False,
        )
        self._layers[(coil_id, name)] = _Layer(name, [a])
        self._plotter.render()

    def add_field_lines_layer(self, lines: list, B_mags: list, coil_id: str) -> None:
        if self._plotter is None or not lines:
            return
        name = 'Field Lines'
        self._remove_layer(coil_id, name)

        all_pts  = np.vstack(lines).astype(np.float32)
        all_B    = np.concatenate(B_mags).astype(np.float32)
        fl_log_B = np.log10(np.maximum(all_B, 1e-15)).astype(np.float32)

        cells  = []
        offset = 0
        for seg in lines:
            n = len(seg)
            cells.append(n)
            cells.extend(range(offset, offset + n))
            offset += n

        mesh = pv.PolyData()
        mesh.points = all_pts
        mesh.lines  = np.array(cells, dtype=np.int_)
        mesh.point_data['fl_log_B'] = fl_log_B
        mesh.point_data['B']        = all_B

        a = self._plotter.add_mesh(
            mesh, scalars='fl_log_B', cmap=THEME.get('cmap_field', 'cool'),
            line_width=1.2, show_scalar_bar=False,
            reset_camera=False, render=False,
        )
        sb = self._make_scalar_bar(a, 'log\u2081\u2080|B| (T)')
        self._layers[(coil_id, name)] = _Layer(name, [a], scalar_bar=sb)
        self._reposition_scalar_bars()
        self._plotter.render()

    def add_cross_section_layer(
        self,
        X: np.ndarray, Y: np.ndarray, B_plane: np.ndarray,
        e1: np.ndarray, e2: np.ndarray, center: np.ndarray, R: float,
        coil_id: str = '',
    ) -> None:
        if self._plotter is None:
            return
        name = 'Cross Section'
        self._remove_layer(coil_id, name)

        gs = X.shape[0]
        P  = (center[np.newaxis, np.newaxis, :]
              + X[..., np.newaxis] * e1[np.newaxis, np.newaxis, :]
              + Y[..., np.newaxis] * e2[np.newaxis, np.newaxis, :])

        sgrid = pv.StructuredGrid()
        sgrid.points     = P.reshape(-1, 3).astype(np.float32)
        sgrid.dimensions = (gs, gs, 1)

        B_flat   = B_plane.ravel().astype(np.float32)
        cs_log_B = np.log10(np.maximum(B_flat, 1e-15)).astype(np.float32)
        sgrid.point_data['B']        = B_flat
        sgrid.point_data['cs_log_B'] = cs_log_B

        a = self._plotter.add_mesh(
            sgrid, scalars='cs_log_B', cmap=THEME.get('cmap_section', 'inferno'),
            opacity=0.85, show_scalar_bar=False,
            reset_camera=False, render=False,
        )
        sb = self._make_scalar_bar(a, 'log\u2081\u2080|B| (T)')
        self._layers[(coil_id, name)] = _Layer(name, [a], scalar_bar=sb)
        self._reposition_scalar_bars()
        self._plotter.render()

    def set_layer_visible(self, coil_id: str, name: str, visible: bool) -> None:
        key = (coil_id, name)
        if key not in self._layers or self._plotter is None:
            return
        layer = self._layers[key]
        layer.visible = visible
        for actor in layer.actors:
            _set_visible(actor, visible)
        if layer.scalar_bar is not None:
            try:
                layer.scalar_bar.SetVisibility(int(visible))
            except Exception:
                pass
        self._reposition_scalar_bars()
        self._plotter.render()

    def clear_analysis_layers(self, coil_id: str) -> None:
        for nm in ('Forces', 'Stress', 'B Axis', 'Field Lines', 'Cross Section'):
            self._remove_layer(coil_id, nm)
        if self._plotter:
            self._plotter.render()

    def clear_inspect_layers(self, coil_id: str) -> None:
        for nm in ('Field Lines', 'Cross Section'):
            self._remove_layer(coil_id, nm)
        if self._plotter:
            self._plotter.render()

    def clear_field_lines_layer(self, coil_id: str) -> None:
        self._remove_layer(coil_id, 'Field Lines')
        if self._plotter:
            self._plotter.render()

    def clear_cross_section_layer(self, coil_id: str) -> None:
        self._remove_layer(coil_id, 'Cross Section')
        if self._plotter:
            self._plotter.render()

    def has_layer(self, coil_id: str, name: str) -> bool:
        return (coil_id, name) in self._layers

    # ── Private ───────────────────────────────────────────────────────────────

    def _remove_layer(self, coil_id: str, name: str) -> None:
        key = (coil_id, name)
        if key not in self._layers or self._plotter is None:
            return
        layer = self._layers[key]
        for actor in layer.actors:
            try:
                self._plotter.remove_actor(actor, render=False)
            except Exception:
                pass
        if layer.scalar_bar is not None:
            try:
                self._plotter.renderer.RemoveActor2D(layer.scalar_bar)
            except Exception:
                try:
                    self._plotter.remove_actor(layer.scalar_bar, render=False)
                except Exception:
                    pass
        del self._layers[key]

    def _make_scalar_bar(self, mesh_actor, title: str) -> object:
        try:
            import vtk as _vtk
            lut = mesh_actor.GetMapper().GetLookupTable()
            sb  = _vtk.vtkScalarBarActor()
            sb.SetLookupTable(lut)
            sb.SetTitle(title)
            sb.SetNumberOfLabels(5)
            sb.UnconstrainedFontSizeOff()
            coord = sb.GetPositionCoordinate()
            coord.SetCoordinateSystemToNormalizedViewport()
            # Placeholder position — _reposition_scalar_bars will fix it
            sb.SetPosition(0.85, 0.05)
            sb.SetWidth(0.12)
            sb.SetHeight(0.40)
            sb_hex = THEME.get('sb_text', '#ffffff').lstrip('#')
            sb_r = int(sb_hex[0:2], 16) / 255.0
            sb_g = int(sb_hex[2:4], 16) / 255.0
            sb_b = int(sb_hex[4:6], 16) / 255.0
            for prop in (sb.GetTitleTextProperty(), sb.GetLabelTextProperty()):
                prop.SetColor(sb_r, sb_g, sb_b)
                prop.BoldOff()
                prop.ItalicOff()
                prop.ShadowOff()
            sb.GetTitleTextProperty().SetFontSize(10)
            sb.GetLabelTextProperty().SetFontSize(9)
            self._plotter.renderer.AddActor2D(sb)
            return sb
        except Exception:
            return None

    def _reposition_scalar_bars(self) -> None:
        """Stack visible scalar bars vertically so they don't overlap."""
        visible = []
        for layer in self._layers.values():
            if layer.scalar_bar is not None and layer.visible:
                visible.append(layer.scalar_bar)
        n = len(visible)
        if n == 0:
            return
        bar_h   = min(0.40, 0.88 / max(n, 1))
        spacing = 0.02
        for i, sb in enumerate(visible):
            y = 0.05 + i * (bar_h + spacing)
            try:
                sb.SetPosition(0.85, y)
                sb.SetHeight(bar_h)
            except Exception:
                pass

    @staticmethod
    def _toolbar_btn_style() -> str:
        return (
            f"QPushButton {{ background:{THEME['panel']}; border:1px solid {THEME['border']};"
            f" border-radius:3px; padding:2px 8px;"
            f" color:{THEME['text_dim']}; font-size:8pt; }}"
            f"QPushButton:hover {{ background:{THEME['input']};"
            f" border-color:{THEME['accent']}; color:{THEME['text']}; }}"
        )

    def _make_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setStyleSheet(
            f"background:{THEME['panel']}; border-top:1px solid {THEME['border']};"
        )
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(4)
        lay.addStretch()

        _sty = self._toolbar_btn_style()

        self._btn_reset_view = QPushButton("⌖ Reset View")
        self._btn_reset_view.setStyleSheet(_sty)
        self._btn_reset_view.clicked.connect(self._reset_view)

        self._btn_save = QPushButton("💾 Save")
        self._btn_save.setStyleSheet(_sty)
        self._btn_save.clicked.connect(self._save)

        lay.addWidget(self._btn_reset_view)
        lay.addWidget(self._btn_save)
        return bar

    def _reset_view(self) -> None:
        if self._plotter:
            self._plotter.reset_camera()
            self._plotter.render()

    def _save(self) -> None:
        if self._plotter is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "calcsx_3d.png",
            "PNG image (*.png);;JPEG (*.jpg)",
        )
        if path:
            self._plotter.screenshot(path)
