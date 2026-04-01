# views/workspace_3d.py
"""
Workspace3DView — PyVista/VTK 3-D workspace with a named layer system.

Layer names
-----------
  'Coil'          — wire spline (floor grid is separate, not layer-controlled)
  'Forces'        — Lorentz-force arrow glyphs, plasma-coloured by magnitude
  'Stress'        — hoop-stress point cloud (YlOrRd, MPa)
  'B Axis'        — on-axis |B| scatter along PCA axis (cool cmap)
  'Field Lines'   — 3D magnetic field line traces, coloured by log₁₀|B|
  'Cross Section' — 2D mid-plane |B| heatmap, coloured by log₁₀|B|
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel,
)
from PyQt5.QtCore import Qt

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
    scalar_bar: object = None   # vtkScalarBarActor, if the layer has one


def _set_visible(actor, visible: bool) -> None:
    """Toggle actor visibility across PyVista versions and actor types."""
    try:
        actor.SetVisibility(int(visible))
        return
    except AttributeError:
        pass
    try:
        actor.visibility = visible
    except Exception:
        pass


class Workspace3DView(QWidget):
    """Full-bleed PyVista 3-D workspace embedded in a Qt widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layers: dict[str, _Layer] = {}
        self._floor_actors: list = []   # world-space floor — not layer-controlled
        self._hint_label = None

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

        # Terrain interactor: camera up-axis is locked to Z so the camera
        # never rolls — the floor plane always sits at the bottom of the
        # viewport giving the Fusion 360 "coil spins, floor stays fixed" feel.
        try:
            import vtk as _vtk
            _terrain = _vtk.vtkInteractorStyleTerrain()
            self._plotter.iren.SetInteractorStyle(_terrain)
        except Exception:
            pass

        self._hint_label = QLabel("Load a coil CSV to begin", self._plotter)
        self._hint_label.setAlignment(Qt.AlignCenter)
        self._hint_label.setStyleSheet(
            f"color:{THEME['text_dim']}; font-size:14pt; background:transparent;"
        )
        self._hint_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._hint_label.resize(self._plotter.size())

        toolbar = self._make_toolbar()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._plotter, stretch=1)
        lay.addWidget(toolbar)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._hint_label is not None and self._plotter is not None:
            self._hint_label.resize(self._plotter.size())

    # ── Public layer API ──────────────────────────────────────────────────────

    def load_coil(self, coords: np.ndarray, fname: str) -> None:
        """Clear scene, draw floor grid + coil spline, register 'Coil' layer."""
        if self._plotter is None:
            return

        self._plotter.clear()
        self._layers.clear()
        self._floor_actors.clear()
        if self._hint_label is not None:
            self._hint_label.hide()
            self._hint_label = None

        c     = np.asarray(coords, dtype=float)
        spans = _ptp(c, axis=0)
        max_s = max(float(spans.max()), 1e-6)
        cx    = float(c[:, 0].mean())
        cy    = float(c[:, 1].mean())
        z_bot = float(c[:, 2].min()) - max_s * 0.25

        # ── World-space floor reference grid (NOT part of any layer) ──────────
        # Large enough to fill the background and read as a fixed ground plane.
        # Not stored in _layers so the browser eye-icon cannot hide it.
        half  = max_s * 5.0
        plane = pv.Plane(
            center=(cx, cy, z_bot),
            direction=(0, 0, 1),
            i_size=half * 2,
            j_size=half * 2,
            i_resolution=24,
            j_resolution=24,
        )
        fa = self._plotter.add_mesh(
            plane,
            color='#606060',
            style='wireframe',
            line_width=1.1,
            opacity=0.70,
            render=False,
        )
        self._floor_actors.append(fa)

        # ── Coil wire (smooth spline) ─────────────────────────────────────────
        n_pts  = max(len(c) * 3, 400)
        spline = pv.Spline(c, n_points=n_pts)
        a = self._plotter.add_mesh(
            spline,
            color=THEME['accent'],
            line_width=3.0,
            render=False,
        )
        self._layers['Coil'] = _Layer('Coil', [a])
        self._plotter.reset_camera()
        self._plotter.render()

    def add_force_layer(self, engine, normalized: bool = False) -> None:
        """Lorentz force quiver arrows coloured by |F| (plasma)."""
        if self._plotter is None:
            return
        name = 'Forces'
        self._remove_layer(name)

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
                glyphs, scalars='magnitude', cmap='plasma',
                clim=[float(mags.min()), float(mags.max())],
                show_scalar_bar=False, reset_camera=False, render=False,
            )
        except Exception:
            a = self._plotter.add_mesh(
                cloud, scalars='magnitude', cmap='plasma',
                point_size=6, show_scalar_bar=False,
                reset_camera=False, render=False,
            )

        self._layers[name] = _Layer(name, [a])
        self._plotter.render()

    def add_stress_layer(self, engine) -> None:
        """Hoop stress point cloud on midpoints (YlOrRd, MPa)."""
        if self._plotter is None:
            return
        name = 'Stress'
        self._remove_layer(name)

        midpoints = np.asarray(engine.midpoints, dtype=float)
        stress    = np.asarray(engine.hoop_stress, dtype=float) / 1e6

        cloud = pv.PolyData(np.ascontiguousarray(midpoints, dtype=np.float32))
        cloud['stress_MPa'] = stress.astype(np.float32)

        a = self._plotter.add_mesh(
            cloud, scalars='stress_MPa', cmap='YlOrRd',
            point_size=8, render_points_as_spheres=True,
            show_scalar_bar=False, render=False,
        )
        self._layers[name] = _Layer(name, [a])
        self._plotter.render()

    def add_axis_layer(self, engine) -> None:
        """On-axis |B| point cloud along PCA axis (cool cmap)."""
        if self._plotter is None:
            return
        name = 'B Axis'
        self._remove_layer(name)

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
            cloud, scalars='B_mag', cmap='cool',
            point_size=6, render_points_as_spheres=True,
            show_scalar_bar=False, render=False,
        )
        self._layers[name] = _Layer(name, [a])
        self._plotter.render()

    def add_field_lines_layer(self, lines: list, B_mags: list) -> None:
        """
        Render 3D magnetic field line traces.

        lines  — list of (N, 3) float32 arrays (one per field line)
        B_mags — list of (N,) float32 arrays (|B| at each point)

        Coloured by log10(|B|) so near-coil strong field and far weak field
        are both distinguishable.
        """
        if self._plotter is None or not lines:
            return
        name = 'Field Lines'
        self._remove_layer(name)

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
        cells_arr = np.array(cells, dtype=np.int_)

        mesh = pv.PolyData()
        mesh.points = all_pts
        mesh.lines  = cells_arr
        mesh.point_data['fl_log_B'] = fl_log_B
        mesh.point_data['B']        = all_B

        a = self._plotter.add_mesh(
            mesh,
            scalars='fl_log_B',
            cmap='cool',
            line_width=1.2,
            show_scalar_bar=False,
            reset_camera=False,
            render=False,
        )
        sb = self._make_scalar_bar(a, 'log\u2081\u2080|B| (T)')
        layer = _Layer(name, [a], scalar_bar=sb)
        self._layers[name] = layer
        self._plotter.render()

    def add_cross_section_layer(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        B_plane: np.ndarray,
        e1: np.ndarray,
        e2: np.ndarray,
        center: np.ndarray,
        R: float,
    ) -> None:
        """
        Render the mid-plane |B| cross-section as a coloured 2D surface in 3D.

        X, Y     — (gs, gs) 2D coordinate grids in the plane basis
        B_plane  — (gs, gs) |B| values
        e1, e2   — orthonormal basis vectors
        center   — world-space origin of the plane
        R        — half-width of the grid (for display only)
        """
        if self._plotter is None:
            return
        name = 'Cross Section'
        self._remove_layer(name)

        gs = X.shape[0]
        # Build 3D world positions: P[i,j] = center + X[i,j]*e1 + Y[i,j]*e2
        P = (center[np.newaxis, np.newaxis, :]
             + X[..., np.newaxis] * e1[np.newaxis, np.newaxis, :]
             + Y[..., np.newaxis] * e2[np.newaxis, np.newaxis, :])  # (gs,gs,3)

        # Flatten to (gs^2, 3) for PolyData, then build a StructuredGrid
        sgrid = pv.StructuredGrid()
        sgrid.points = P.reshape(-1, 3).astype(np.float32)
        sgrid.dimensions = (gs, gs, 1)

        B_flat    = B_plane.ravel().astype(np.float32)
        cs_log_B  = np.log10(np.maximum(B_flat, 1e-15)).astype(np.float32)
        sgrid.point_data['B']        = B_flat
        sgrid.point_data['cs_log_B'] = cs_log_B

        a = self._plotter.add_mesh(
            sgrid,
            scalars='cs_log_B',
            cmap='inferno',
            opacity=0.85,
            show_scalar_bar=False,
            reset_camera=False,
            render=False,
        )
        sb = self._make_scalar_bar(a, 'log\u2081\u2080|B| (T)')
        layer = _Layer(name, [a], scalar_bar=sb)
        self._layers[name] = layer
        self._plotter.reset_camera()
        self._plotter.render()

    def set_layer_visible(self, name: str, visible: bool) -> None:
        if name not in self._layers or self._plotter is None:
            return
        layer = self._layers[name]
        layer.visible = visible
        for actor in layer.actors:
            _set_visible(actor, visible)
        if layer.scalar_bar is not None:
            try:
                layer.scalar_bar.SetVisibility(int(visible))
            except Exception:
                pass
        self._plotter.render()

    def clear_analysis_layers(self) -> None:
        """Remove all analysis and inspect layers; keep Coil and floor grid."""
        for nm in ('Forces', 'Stress', 'B Axis', 'Field Lines', 'Cross Section'):
            self._remove_layer(nm)
        if self._plotter:
            self._plotter.render()

    def clear_inspect_layers(self) -> None:
        """Remove Field Lines and Cross Section layers only."""
        for nm in ('Field Lines', 'Cross Section'):
            self._remove_layer(nm)
        if self._plotter:
            self._plotter.render()

    def clear_field_lines_layer(self) -> None:
        self._remove_layer('Field Lines')
        if self._plotter:
            self._plotter.render()

    def clear_cross_section_layer(self) -> None:
        self._remove_layer('Cross Section')
        if self._plotter:
            self._plotter.render()

    def has_layer(self, name: str) -> bool:
        return name in self._layers

    # ── Private ───────────────────────────────────────────────────────────────

    def _remove_layer(self, name: str) -> None:
        if name not in self._layers or self._plotter is None:
            return
        layer = self._layers[name]
        for actor in layer.actors:
            try:
                self._plotter.remove_actor(actor, render=False)
            except Exception:
                pass
        # Remove manually-added scalar bar from the renderer's 2D actor list
        if layer.scalar_bar is not None:
            try:
                self._plotter.renderer.RemoveActor2D(layer.scalar_bar)
            except Exception:
                try:
                    self._plotter.remove_actor(layer.scalar_bar, render=False)
                except Exception:
                    pass
        del self._layers[name]

    def _make_scalar_bar(self, mesh_actor, title: str) -> object:
        """
        Create a vtkScalarBarActor linked to mesh_actor's lookup table and
        add it directly to the renderer.  Returns the actor (or None on failure).
        Bypasses PyVista's ScalarBars dict so we always hold a direct reference.
        """
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
            sb.SetPosition(0.85, 0.25)
            sb.SetWidth(0.13)
            sb.SetHeight(0.50)
            for prop in (sb.GetTitleTextProperty(), sb.GetLabelTextProperty()):
                prop.SetColor(1.0, 1.0, 1.0)
                prop.BoldOff()
                prop.ItalicOff()
                prop.ShadowOff()
            sb.GetTitleTextProperty().SetFontSize(11)
            sb.GetLabelTextProperty().SetFontSize(10)
            self._plotter.renderer.AddActor2D(sb)
            return sb
        except Exception:
            return None

    def _make_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setStyleSheet(
            f"background:{THEME['panel']}; border-top:1px solid {THEME['border']};"
        )
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(4)
        lay.addStretch()

        _sty = (
            f"QPushButton {{ background:{THEME['panel']}; border:1px solid {THEME['border']};"
            f" border-radius:3px; padding:2px 8px;"
            f" color:{THEME['text_dim']}; font-size:8pt; }}"
            f"QPushButton:hover {{ background:{THEME['input']};"
            f" border-color:{THEME['accent']}; color:{THEME['text']}; }}"
        )

        btn_reset = QPushButton("⌖ Reset View")
        btn_reset.setStyleSheet(_sty)
        btn_reset.clicked.connect(self._reset_view)

        btn_save = QPushButton("💾 Save")
        btn_save.setStyleSheet(_sty)
        btn_save.clicked.connect(self._save)

        lay.addWidget(btn_reset)
        lay.addWidget(btn_save)
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
