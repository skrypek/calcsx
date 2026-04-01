# tabs/tab_defs.py
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QSplitter,
)
from PyQt5.QtCore import Qt
from gui.gui_utils import make_canvas, make_canvas_with_toolbar, PlotToolbar, THEME
from plots.plot_defs import (
    plot_force_vs_arc,
    plot_bfield_vs_axis_distance,
    plot_lorentz_vec,
    plot_lorentz_normalized,
)

_STAT_STYLE = (
    f"color:{THEME['text']}; font-size:8pt; "
    f"background:{THEME['panel']}; "
    f"padding:4px 12px; border-top:1px solid {THEME['border']};"
)

_CTRL_STYLE = (
    f"background:{THEME['panel']}; border-bottom:1px solid {THEME['border']};"
)


def _stat_label(html: str) -> QLabel:
    lbl = QLabel(html)
    lbl.setTextFormat(Qt.RichText)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet(_STAT_STYLE)
    return lbl


def _dim(txt: str) -> str:
    return f"<span style='color:{THEME['text_dim']}'>{txt}</span>"


def _accent(txt: str) -> str:
    return f"<b style='color:{THEME['accent']}'>{txt}</b>"


# ─────────────────────────────────────────────────────────────
# Forces view: 3D quiver (left) | Force vs Arc 2D (right)
# Controls: Normalize vectors, Show coil
# ─────────────────────────────────────────────────────────────

def make_forces_view(ctx) -> QWidget:
    """
    Split view combining Lorentz 3D and Force vs Arc.
    Left pane: 3D quiver with normalize + coil toggles.
    Right pane: Force density vs arc length with toolbar.
    Bottom: key force/stress statistics.
    """
    w = QWidget()
    outer = QVBoxLayout(w)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)

    # ── Controls row ────────────────────────────────────────────
    ctrl = QWidget()
    ctrl.setFixedHeight(34)
    ctrl.setStyleSheet(_CTRL_STYLE)
    cl = QHBoxLayout(ctrl)
    cl.setContentsMargins(12, 0, 12, 0)
    cl.setSpacing(20)

    chk_norm = QCheckBox("Normalize vectors")
    chk_coil = QCheckBox("Show coil")
    chk_coil.setChecked(True)
    for chk in (chk_norm, chk_coil):
        chk.setStyleSheet(f"color:{THEME['text']}; font-size:8pt;")

    N_seg   = len(ctx.F_vecs)
    F_total = ctx.total_hoop_force
    p_avg   = ctx.avg_pressure
    stat_html = (
        _dim("Segments: ") + _accent(f"{N_seg}") + "&nbsp;&nbsp;&nbsp;"
        + _dim("Total Lorentz Force: ") + _accent(f"{F_total/1000:.2f} kN") + "&nbsp;&nbsp;&nbsp;"
        + _dim("Internal Pressure: ") + _accent(f"{p_avg/1e6:.2f} MPa")
    )
    stat_lbl = QLabel(stat_html)
    stat_lbl.setTextFormat(Qt.RichText)
    stat_lbl.setStyleSheet(f"color:{THEME['text']}; font-size:8pt;")

    cl.addWidget(chk_norm)
    cl.addWidget(chk_coil)
    cl.addStretch()
    cl.addWidget(stat_lbl)
    outer.addWidget(ctrl)

    # ── Split view ──────────────────────────────────────────────
    splitter = QSplitter(Qt.Horizontal)
    splitter.setStyleSheet(
        f"QSplitter::handle {{ background:{THEME['border']}; width:2px; }}"
    )

    # Left: 3D quiver canvas + toolbar
    canvas_3d = make_canvas(plot_lorentz_vec, ctx, figsize=(7, 6), projection='3d')
    fig_3d    = canvas_3d.figure
    ax_3d_ref = {'ax': fig_3d.axes[0]}
    view_state = {'azim': fig_3d.axes[0].azim, 'elev': fig_3d.axes[0].elev}

    def _save_view(event):
        a = ax_3d_ref['ax']
        view_state['azim'] = a.azim
        view_state['elev'] = a.elev

    canvas_3d.mpl_connect('button_release_event', _save_view)

    left_w = QWidget()
    left_l = QVBoxLayout(left_w)
    left_l.setContentsMargins(0, 0, 0, 0)
    left_l.setSpacing(0)
    left_l.addWidget(canvas_3d, stretch=1)
    left_l.addWidget(PlotToolbar(canvas_3d))

    # Right: force vs arc with toolbar
    right_w = make_canvas_with_toolbar(plot_force_vs_arc, ctx)

    splitter.addWidget(left_w)
    splitter.addWidget(right_w)
    splitter.setSizes([520, 480])

    outer.addWidget(splitter, stretch=1)

    # ── Redraw on toggle ────────────────────────────────────────
    def _redraw(_):
        norm_on = chk_norm.isChecked()
        coil_on = chk_coil.isChecked()

        fig_3d.clf()
        new_ax = fig_3d.add_subplot(111, projection='3d')
        ax_3d_ref['ax'] = new_ax

        if norm_on:
            plot_lorentz_normalized(new_ax, ctx, show_coil=coil_on)
        else:
            plot_lorentz_vec(new_ax, ctx, show_coil=coil_on)

        if view_state.get('azim') is not None:
            new_ax.view_init(elev=view_state['elev'], azim=view_state['azim'])

        canvas_3d.mpl_connect('button_release_event', _save_view)
        canvas_3d.draw()

    chk_norm.stateChanged.connect(_redraw)
    chk_coil.stateChanged.connect(_redraw)

    return w


# ─────────────────────────────────────────────────────────────
# Magnetics view: on-axis B-field
# ─────────────────────────────────────────────────────────────

def make_bfield_calcs_tab(ctx) -> QWidget:
    ctx._compute_bfield_at_centroid()

    B_mag   = ctx.B_magnitude
    B_axial = ctx.B_axial

    txt = (
        _dim("Total |B|: ") + _accent(f"{B_mag:.5f} T") + "&nbsp;&nbsp;&nbsp;"
        + _dim("Axial |B|: ") + _accent(f"{B_axial:.5f} T")
    )

    w = QWidget()
    v = QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(0)
    v.addWidget(make_canvas_with_toolbar(plot_bfield_vs_axis_distance, ctx), stretch=1)
    v.addWidget(_stat_label(txt))
    return w
