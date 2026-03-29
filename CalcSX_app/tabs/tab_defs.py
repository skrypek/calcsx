# tab_defs.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QCheckBox
from PyQt5.QtCore import Qt
from gui.gui_utils import make_canvas
from plots.plot_defs import (
    plot_force_vs_arc,
    plot_bfield_vs_axis_distance,
    plot_lorentz_vec,
    plot_lorentz_normalized,
    plot_bfield_cross_section
)

def make_force_vs_arc_tab(ctx):
    """
    Returns a QWidget containing:
      - The Force vs Arc matplotlib canvas
      - A QLabel underneath with N_seg, total hoop‑force, and average pressure
    """
    # 1) plot Force vs Arc
    canvas = make_canvas(plot_force_vs_arc, ctx)

    # 2) pull in totals from the analysis engine
    N_seg   = len(ctx.F_vecs)
    F_total = ctx.total_hoop_force
    p_avg   = ctx.avg_pressure

    # 3) format stats
    txt = (
        f"<b>Sample segments:</b> {N_seg}<br>"
        f"<b>Total Lorentz Force:</b> {F_total/1000:.2f} kN<br>"
        f"<b>Internal Pressure:</b> {p_avg/1e6:.2f} MPa"
    )
    label = QLabel(txt)
    label.setTextFormat(Qt.RichText)
    label.setAlignment(Qt.AlignCenter)

    # 4) pack into widget
    w = QWidget()
    v = QVBoxLayout(w)
    v.addWidget(canvas, stretch=1)
    v.addWidget(label,  stretch=0)
    return w

def make_bfield_calcs_tab(ctx):
    """
    Returns a QWidget containing:
      - The B‑field vs axis‐distance canvas
      - A QLabel with total and axial B‑field at coil centroid
    """
    # ensure centroid field is up to date
    ctx._compute_bfield_at_centroid()

    # 1) plot on‐axis B‑field
    canvas = make_canvas(plot_bfield_vs_axis_distance, ctx)

    # 2) totals
    B_mag   = ctx.B_magnitude
    B_axial = ctx.B_axial

    # 3) format stats
    txt = (
        f"<b>Total |B|:</b> {B_mag:.5f} T<br>"
        f"<b>Axial |B|:</b> {B_axial:.5f} T"
    )
    label = QLabel(txt)
    label.setTextFormat(Qt.RichText)
    label.setAlignment(Qt.AlignCenter)

    # 4) pack into widget
    w = QWidget()
    v = QVBoxLayout(w)
    v.addWidget(canvas, stretch=1)
    v.addWidget(label,  stretch=0)
    return w

def make_lorentz_toggle_tab(ctx):
    """
    Returns a QWidget allowing toggle between raw and normalized Lorentz vectors,
    clearing the figure each time to avoid duplicate legends/colorbars.
    """
    w = QWidget()
    v = QVBoxLayout(w)

    chk = QCheckBox("Show normalized vectors")
    v.addWidget(chk)

    # Initial canvas and axes
    canvas = make_canvas(plot_lorentz_vec, ctx, projection='3d')
    fig    = canvas.figure
    ax     = fig.axes[0]

    # Remember view angles
    view = {'azim': ax.azim, 'elev': ax.elev}

    # Capture view changes
    def on_release(event):
        view['azim'], view['elev'] = ax.azim, ax.elev
    canvas.mpl_connect('button_release_event', on_release)

    def redraw(_state):
        # 1) clear entire figure (removes old axes, legends, colorbars)
        fig.clf()
        # 2) create a fresh 3D subplot
        new_ax = fig.add_subplot(111, projection='3d')
        # 3) plot either normalized or raw
        if chk.isChecked():
            plot_lorentz_normalized(new_ax, ctx)
        else:
            plot_lorentz_vec(new_ax, ctx)
        # 4) restore camera view
        if view['azim'] is not None and view['elev'] is not None:
            new_ax.view_init(elev=view['elev'], azim=view['azim'])
        # 5) rebind release event on the new axis
        def on_rel(event):
            view['azim'], view['elev'] = new_ax.azim, new_ax.elev
        canvas.mpl_connect('button_release_event', on_rel)
        # 6) redraw the canvas
        canvas.draw()

    chk.stateChanged.connect(redraw)

    v.addWidget(canvas, stretch=1)
    return w

def make_bfield_cross_section_tab(ctx):
    canvas = make_canvas(plot_bfield_cross_section, ctx)
    w = QWidget()
    v = QVBoxLayout(w)
    v.addWidget(canvas)
    return w
