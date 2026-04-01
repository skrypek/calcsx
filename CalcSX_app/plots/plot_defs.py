# plot_defs.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator
import mplcursors

from gui.gui_utils import THEME


# ─────────────────────────────────────────────────────────────
# CAD-style 3-D axis helper
# ─────────────────────────────────────────────────────────────

def _style_3d_cad(ax, fig=None):
    """
    Fusion 360-style floating-model 3-D appearance.

    • Pane faces and all border/spine lines fully transparent → no bounding box.
    • No tick marks or labels.
    • Grid lines completely off (we draw our own floor grid in load_coil).
    • Model floats in open dark space.
    """
    if fig is None:
        fig = ax.get_figure()
    fig.patch.set_facecolor(THEME['bg'])
    ax.set_facecolor(THEME['bg'])

    # ── Pane faces + pane border lines → invisible ─────────────────────────────
    transparent = (0.0, 0.0, 0.0, 0.0)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_facecolor(transparent)
        pane.set_edgecolor(transparent)

    # ── Axis spine lines (the box-edge lines matplotlib draws) → invisible ─────
    try:
        ax.xaxis.line.set_color(transparent)
        ax.yaxis.line.set_color(transparent)
        ax.zaxis.line.set_color(transparent)
    except Exception:
        pass

    # ── Grid lines off; ticks off ──────────────────────────────────────────────
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis._axinfo['grid']['color']          = transparent
            axis._axinfo['tick']['inward_factor']  = 0.0
            axis._axinfo['tick']['outward_factor'] = 0.0
        except (KeyError, AttributeError):
            pass

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')

    ax.title.set_color(THEME['text'])
    ax.title.set_fontsize(10)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _cursor_style(ann):
    """Apply dark annotation style to an mplcursors annotation."""
    bb = ann.get_bbox_patch()
    if bb:
        bb.set(facecolor=THEME['panel'], edgecolor=THEME['accent'], alpha=0.90)
    ann.set_fontsize(9)
    ann.set_color(THEME['text'])


# ─────────────────────────────────────────────────────────────
# Planar-axis warning (shown once per session)
# ─────────────────────────────────────────────────────────────

_warned_planar_axes = False

def _warn_planar_axes_once():
    global _warned_planar_axes
    if _warned_planar_axes:
        return
    try:
        from PyQt5.QtWidgets import QMessageBox, QStyle, QLayout
        box = QMessageBox()
        box.setWindowTitle("Planar Coil Warning")
        box.setIcon(QMessageBox.Warning)
        box.setText("Planar coil detected!")
        box.setInformativeText(
            "Pay close attention to axis scaling on the plots. "
            "Output ranges may make a nearly-constant curve look non-constant."
        )
        box.setStyleSheet(
            "QLabel#qt_msgbox_label, QLabel#qt_msgbox_informativelabel { min-width: 280px; }"
        )
        box.setStandardButtons(QMessageBox.Ok)
        box.exec_()
        _warned_planar_axes = True
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# Equal-axis scaling for 3-D plots
# ─────────────────────────────────────────────────────────────

def _set_equal_3d_axes(ax, coords):
    try:
        coords = np.asarray(coords)
        if coords.size == 0:
            return
        ranges = coords.max(axis=0) - coords.min(axis=0)
        ranges = np.where(ranges == 0, 1.0, ranges)
        max_r   = ranges.max()
        centers = (coords.min(axis=0) + coords.max(axis=0)) / 2
        half    = (max_r + max_r * 0.10) / 2
        ax.set_xlim(centers[0] - half, centers[0] + half)
        ax.set_ylim(centers[1] - half, centers[1] + half)
        ax.set_zlim(centers[2] - half, centers[2] + half)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# 3-D plots
# ─────────────────────────────────────────────────────────────

def plot_3d_filament(ax, ctx):
    if getattr(ctx, "is_planar", False):
        _warn_planar_axes_once()

    c, m, a = ctx.coords, ctx.mean_point, ctx.axis
    ax.plot(c[:, 0], c[:, 1], c[:, 2],
            color=THEME['accent'], lw=1.6, label='Filament')

    L = ctx.total_length * 0.5
    ax.quiver(m[0], m[1], m[2], a[0], a[1], a[2], length=L * 0.05,
              normalize=True, color='#ff7043', arrow_length_ratio=0.5,
              lw=0.9, label='PCA axis')

    a_n = a / (np.linalg.norm(a) + 1e-30)
    seg_dir = c[1] - c[0]
    e1 = seg_dir - np.dot(seg_dir, a_n) * a_n
    if np.linalg.norm(e1) < 1e-12:
        e1 = np.array([1., 0., 0.])
        if abs(np.dot(e1, a_n)) > 0.9:
            e1 = np.array([0., 1., 0.])
        e1 = e1 - np.dot(e1, a_n) * a_n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a_n, e1) / np.linalg.norm(np.cross(a_n, e1) + 1e-30)

    blen = L * 0.03
    ax.quiver(m[0], m[1], m[2], *(e1 * blen),
              color='#64b5f6', normalize=False,
              arrow_length_ratio=0.25, lw=0.9, label=r'$\vec{x}_1$')
    ax.quiver(m[0], m[1], m[2], *(e2 * blen),
              color='#ce93d8', normalize=False,
              arrow_length_ratio=0.25, lw=0.9, label=r'$\vec{x}_2$')

    ax.scatter(*c[0], color=THEME['success'], s=40, label='Arc=0')
    idx = max(int(len(c) * 0.1), 1)
    dv  = c[idx] - c[0]
    ax.quiver(c[0, 0], c[0, 1], c[0, 2], dv[0], dv[1], dv[2],
              length=np.linalg.norm(dv) * 0.5, normalize=True,
              color=THEME['success'], arrow_length_ratio=0.3, lw=1,
              label='Param dir')

    _set_equal_3d_axes(ax, c)
    _style_3d_cad(ax)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(loc='upper left', fontsize=7, borderaxespad=0,
              facecolor=THEME['panel'], edgecolor=THEME['border'],
              labelcolor=THEME['text'])
    ax.set_title('Filament Geometry  |  PCA Axis & Basis', fontsize=10)


def plot_lorentz_vec(ax, ctx, show_coil: bool = True):
    c, md, vec = ctx.coords, ctx.midpoints, ctx.F_vecs
    mags = ctx.F_mags

    if show_coil:
        ax.plot(c[:, 0], c[:, 1], c[:, 2],
                color=THEME['accent'], lw=1.2, label='Coil', alpha=0.7)

    n    = len(md)
    step = max(1, n // 300)
    idx  = np.arange(0, n, step)
    pts  = md[idx]; vecs = vec[idx]; mags_s = mags[idx]

    mags_full = np.linalg.norm(vecs, axis=1)
    max_mag   = mags_full.max() if mags_full.size else 1.0
    pseudo    = vecs / (max_mag + 1e-30) * 0.08

    norm   = Normalize(mags.min(), mags.max())
    colors = plt.cm.plasma(norm(mags_s))
    ax.quiver(pts[:, 0], pts[:, 1], pts[:, 2],
              pseudo[:, 0], pseudo[:, 1], pseudo[:, 2],
              length=1.0, normalize=False, colors=colors, linewidths=0.8)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap='plasma')
    mappable.set_array(mags_s)
    cbar = ax.get_figure().colorbar(mappable, ax=ax, label='N/m', shrink=0.70)
    cbar.ax.yaxis.label.set_color(THEME['text'])
    cbar.ax.tick_params(colors=THEME['text'])

    _set_equal_3d_axes(ax, c)
    _style_3d_cad(ax)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Lorentz Force Density  (N/m)', fontsize=10)


def plot_lorentz_normalized(ax, ctx, show_coil: bool = True):
    c, md, vec = ctx.coords, ctx.midpoints, ctx.F_vecs
    mags = ctx.F_mags

    if show_coil:
        ax.plot(c[:, 0], c[:, 1], c[:, 2],
                color=THEME['accent'], lw=1.2, label='Coil', alpha=0.7)

    n    = len(md)
    step = max(1, n // 300)
    idx  = np.arange(0, n, step)
    pts  = md[idx]; vecs = vec[idx]; mags_s = mags[idx]

    coil_dia  = np.linalg.norm(c.max(axis=0) - c.min(axis=0))
    arrow_len = coil_dia * 0.03
    norm      = Normalize(mags.min(), mags.max())
    colors    = plt.cm.plasma(norm(mags_s))
    ax.quiver(pts[:, 0], pts[:, 1], pts[:, 2],
              vecs[:, 0], vecs[:, 1], vecs[:, 2],
              length=arrow_len, normalize=True, colors=colors, linewidths=0.8)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap='plasma')
    mappable.set_array(mags_s)
    cbar = ax.get_figure().colorbar(mappable, ax=ax, label='N/m', shrink=0.70)
    cbar.ax.yaxis.label.set_color(THEME['text'])
    cbar.ax.tick_params(colors=THEME['text'])

    _set_equal_3d_axes(ax, c)
    _style_3d_cad(ax)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Lorentz Force Density  —  Normalized Vectors', fontsize=10)


# ─────────────────────────────────────────────────────────────
# 2-D plots  (rcParams already set dark by apply_mpl_dark_theme)
# ─────────────────────────────────────────────────────────────

def plot_force_vs_arc(ax, ctx):
    line, = ax.plot(ctx.arc_mid, ctx.F_mags,
                    color=THEME['accent'], linewidth=1.6)
    line.set_picker(True); line.set_pickradius(20)

    cursor = mplcursors.cursor(line, hover=True, highlight=False)
    @cursor.connect("add")
    def _(sel):
        sel.annotation.set_text(
            f"s = {sel.target[0]:.3f} m\nf = {sel.target[1]:.5g} N/m")
        _cursor_style(sel.annotation)

    ax.set_xlabel('Arc length (m)')
    ax.set_ylabel('Force density (N/m)')
    ax.set_title('Lorentz Force Density vs Arc Length')
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(10))
    ax.yaxis.set_major_locator(MaxNLocator(10))


def plot_stress_vs_arc(ax, ctx):
    line, = ax.plot(ctx.arc_mid, ctx.hoop_stress / 1e6,
                    color=THEME['accent2'], linewidth=1.6)
    line.set_picker(True); line.set_pickradius(20)

    cursor = mplcursors.cursor(line, hover=True, highlight=False)
    @cursor.connect("add")
    def _(sel):
        sel.annotation.set_text(
            f"s = {sel.target[0]:.3f} m\nStress = {sel.target[1]:.5g} MPa")
        _cursor_style(sel.annotation)

    ax.set_xlabel('Arc length (m)')
    ax.set_ylabel('Stress (MPa)')
    ax.set_title('Hoop Stress vs Arc Length')
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(10))
    ax.yaxis.set_major_locator(MaxNLocator(10))


def plot_bfield_vs_axis_distance(ax, ctx):
    if ctx.bfield_axis_z is not None and ctx.bfield_axis_mag is not None:
        zs, mags = ctx.bfield_axis_z, ctx.bfield_axis_mag
    else:
        zs, mags = ctx.compute_bfield_along_axis()

    line, = ax.plot(zs, mags, color=THEME['success'], linewidth=1.6)
    line.set_picker(True); line.set_pickradius(20)

    cursor = mplcursors.cursor(line, hover=True, highlight=False)
    @cursor.connect("add")
    def _(sel):
        sel.annotation.set_text(
            f"x = {sel.target[0]:.3f} m\n|B| = {sel.target[1]:.5g} T")
        _cursor_style(sel.annotation)

    ax.set_xlabel('Distance along axis (m)')
    ax.set_ylabel('|B| (T)')
    ax.set_title('|B| vs. Distance Along Axis of Symmetry')
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(10))
    ax.yaxis.set_major_locator(MaxNLocator(10))


def plot_bfield_cross_section(ax, ctx, n: int = 120,
                               log_scale: bool = True,
                               threshold: float | None = None):
    """
    |B| on the centroid plane: viridis data, masked outside, black above threshold.
    """
    if getattr(ctx, "cross_section_data", None) is not None:
        X, Y, Bmag, extent = ctx.cross_section_data
    else:
        X, Y, Bmag, extent = ctx.sample_cross_section(n=n)

    if threshold is None:
        threshold = getattr(ctx, "cross_section_threshold", None)

    outside     = np.isnan(Bmag)
    thresh_mask = (Bmag >= threshold) & ~outside if threshold is not None else np.zeros_like(Bmag, bool)
    Bmask       = np.ma.masked_array(Bmag, mask=outside)

    cmap = plt.cm.get_cmap('viridis').copy()
    cmap.set_bad((0, 0, 0, 1))
    cmap.set_over((0.925, 0.855, 0.663, 0.3))

    vmin = vmax = norm = None
    if log_scale:
        valid = Bmask.compressed()
        valid = valid[valid > 0]
        if valid.size:
            vmin, vmax = valid.min(), valid.max()
            if vmin == vmax:
                vmin *= 0.5; vmax *= 1.5
            norm = LogNorm(vmin=vmin, vmax=vmax)

    if norm is None:
        if threshold is not None:
            vmax = threshold
        im = ax.imshow(Bmask, origin='lower', extent=extent,
                       cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        if threshold is not None:
            norm.vmax = threshold
        im = ax.imshow(Bmask, origin='lower', extent=extent,
                       cmap=cmap, norm=norm)

    ax.set_facecolor(THEME['bg'])
    ax.set_aspect('equal')
    ax.set_xlabel(r"$\vec{v}_1$ (m)")
    ax.set_ylabel(r"$\vec{v}_2$ (m)")
    title = "|B| Cross-Section at Centroid"
    if threshold is not None:
        title += f"  (|B| < {threshold:.2f} T)"
    ax.set_title(title)

    cbar = ax.figure.colorbar(im, ax=ax, label='|B| (T)')
    cbar.ax.yaxis.label.set_color(THEME['text'])
    cbar.ax.tick_params(colors=THEME['text'])
    cbar.minorticks_off()
    if vmin and vmax:
        lo = vmin; hi = threshold if threshold else vmax
        try:
            ticks = np.logspace(np.log10(lo), np.log10(hi), 6)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{t:.2g}" for t in ticks])
        except Exception:
            pass

    cursor = mplcursors.cursor([im], hover=True, highlight=False)
    @cursor.connect("add")
    def _show(sel):
        x, y = sel.target
        j    = int(np.argmin(np.abs(X[0] - x)))
        i    = int(np.argmin(np.abs(Y[:, 0] - y)))
        if i >= Bmag.shape[0] or j >= Bmag.shape[1]:
            sel.annotation.set_visible(False); return
        if outside[i, j] or thresh_mask[i, j]:
            sel.annotation.set_visible(False); return
        sel.annotation.set_text(
            rf"$\vec{{v}}_1$ = {x:7.3f}" + "\n"
            rf"$\vec{{v}}_2$ = {y:7.3f}" + "\n"
            rf"$|B|$ = {Bmag[i,j]:7.4f} T"
        )
        _cursor_style(sel.annotation)
