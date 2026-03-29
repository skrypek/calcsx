# plot_defs.py
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator, LogLocator, FuncFormatter
import mpl_toolkits.mplot3d as mplot3d
import mplcursors

# --- Planar axes warning (popup once) ---------------------------------------
_warned_planar_axes = False

def _warn_planar_axes_once():
    global _warned_planar_axes
    if _warned_planar_axes:
        return
    try:
        try:
            from PyQt5.QtWidgets import QMessageBox, QStyle, QLayout
        except Exception:
            from PySide6.QtWidgets import QMessageBox, QStyle, QLayout
    
        box = QMessageBox()
        box.setWindowTitle("Planar Coil Warning")
        box.setIcon(QMessageBox.Warning)
    
        # Keep main text short; move details to informative text (more compact layout)
        box.setText("Planar coil detected!")
        box.setInformativeText(
            "Pay close attention to axis scaling on the plots. "
            "Output ranges may make a nearly-constant curve look non-constant."
        )
    
        # Smaller icon helps reduce height
        try:
            px = box.style().standardIcon(QStyle.SP_MessageBoxWarning).pixmap(36, 36)
            box.setIconPixmap(px)
        except Exception:
            pass
    
        box.setStyleSheet(
            "QLabel#qt_msgbox_label, QLabel#qt_msgbox_informativelabel { min-width: 280px; }"
        )
        box.layout().setSizeConstraint(QLayout.SetFixedSize)
    
        box.setStandardButtons(QMessageBox.Ok)
        box.exec_()
        _warned_planar_axes = True
        return
    except Exception:
        pass

def _set_equal_3d_axes(ax, coords):
    """
    Set equal axis scaling for 3D plots based on coordinate data.
    Makes all axes have the same scale to show real-world proportions.
    """
    try:
        if coords is None or len(coords) == 0:
            # Fallback to standard box aspect
            try:
                ax.set_box_aspect((1,1,1))
            except:
                pass
            return
        
        # Convert to numpy array if needed
        coords = np.asarray(coords)
        
        # Get min/max for each axis
        x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
        y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
        z_min, z_max = np.min(coords[:, 2]), np.max(coords[:, 2])
        
        # Calculate ranges
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        # Handle zero ranges (degenerate cases)
        if x_range == 0:
            x_range = 1.0
        if y_range == 0:
            y_range = 1.0
        if z_range == 0:
            z_range = 1.0
        
        # Find the maximum range
        max_range = max(x_range, y_range, z_range)
        
        # Calculate centers
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        
        # Add 10% padding
        padding = max_range * 0.1
        half_range = (max_range + padding) / 2
        
        # Set equal limits for all axes
        ax.set_xlim(x_center - half_range, x_center + half_range)
        ax.set_ylim(y_center - half_range, y_center + half_range)
        ax.set_zlim(z_center - half_range, z_center + half_range)
        
        # Try to set box aspect as well
        try:
            ax.set_box_aspect((1,1,1))
        except:
            pass
            
    except Exception as e:
        print(f"Warning: Could not set equal axes: {e}")
        # Fallback to standard box aspect
        try:
            ax.set_box_aspect((1,1,1))
        except:
            pass

def plot_3d_filament(ax, ctx):
    if getattr(ctx, "is_planar", False):
        _warn_planar_axes_once()
    
    c, m, a = ctx.coords, ctx.mean_point, ctx.axis
    ax.plot(c[:,0], c[:,1], c[:,2], color='black', lw=1, label='Filament')
    L = ctx.total_length * 0.5
    ax.quiver(m[0], m[1], m[2], a[0], a[1], a[2], length=L*0.05,
              normalize=True, color='red', arrow_length_ratio=0.5,
              lw=0.75, label='PCA axis')
    
    # --- orthonormal in‑plane basis (e1,e2) perpendicular to axis ---
    a = a / np.linalg.norm(a)

    # First segment direction
    seg_dir = c[1] - c[0]
    # Remove any axial component
    e1 = seg_dir - np.dot(seg_dir, a) * a
    if np.linalg.norm(e1) < 1e-12:          # degenerate fallback
        e1 = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(e1, a)) > 0.9:
            e1 = np.array([0.0, 1.0, 0.0])
        e1 = e1 - np.dot(e1, a) * a
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    e2 /= np.linalg.norm(e2)

    basis_len = L * 0.03          # one common length
    
    ax.quiver(m[0], m[1], m[2],
              *(e1 * basis_len),
              color='blue', normalize=False,
              arrow_length_ratio=0.25, lw=0.9, label=r'$\vec{x}_1$')
    
    ax.quiver(m[0], m[1], m[2],
              *(e2 * basis_len),
              color='purple', normalize=False,
              arrow_length_ratio=0.25, lw=0.9, label=r'$\vec{x}_2$')
    
    ax.scatter(*c[0], color='green', s=40, label='Arc=0')
    idx = max(int(len(c)*0.1), 1)
    dv  = c[idx] - c[0]
    ax.quiver(c[0,0], c[0,1], c[0,2], dv[0], dv[1], dv[2],
              length=np.linalg.norm(dv)*0.5,
              normalize=True, color='green', arrow_length_ratio=0.3, lw=1,
              label='Param dir')
    
    # Apply equal axis scaling
    _set_equal_3d_axes(ax, c)
    
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(7))
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(loc='upper left', fontsize=8, borderaxespad=0)
    ax.set_title('Filament Curve (PCA & Direction)', fontsize=10)

def plot_lorentz_vec(ax, ctx):
    c, md, vec = ctx.coords, ctx.midpoints, ctx.F_vecs
    mags = ctx.F_mags
    ax.plot(c[:,0], c[:,1], c[:,2], color='black', lw=1)
    max_arrows = 300
    n = len(md)
    step = max(1, n//max_arrows)
    inds = np.arange(0, n, step)
    pts  = md[inds]
    vecs = vec[inds]
    mags_s = mags[inds]
    mags_full = np.linalg.norm(vecs, axis=1)
    max_mag = mags_full.max() if mags_full.size else 1.0
    pseudo_vecs = vecs / max_mag * 0.08
    norm = plt.Normalize(mags.min(), mags.max())
    colors = plt.cm.viridis(norm(mags_s))
    ax.quiver(pts[:,0], pts[:,1], pts[:,2],
              pseudo_vecs[:,0], pseudo_vecs[:,1], pseudo_vecs[:,2],
              length=1.0, normalize=False, colors=colors, linewidths=0.75)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    mappable.set_array(mags_s)
    ax.get_figure().colorbar(mappable, ax=ax, label='N/m')
    
    # Apply equal axis scaling
    _set_equal_3d_axes(ax, c)
    
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(7))
    ax.tick_params(which='major', labelsize=8)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Lorentz Force Density Vectors', fontsize=10)

def plot_lorentz_normalized(ax, ctx):
    c, md, vec = ctx.coords, ctx.midpoints, ctx.F_vecs
    mags = ctx.F_mags
    ax.plot(c[:,0], c[:,1], c[:,2], color='black', lw=1)
    max_arrows = 300
    n = len(md)
    step = max(1, n//max_arrows)
    inds = np.arange(0, n, step)
    pts  = md[inds]
    vecs = vec[inds]
    mags_s = mags[inds]
    extents = np.ptp(c, axis=0)
    coil_dia = np.linalg.norm(extents)
    arrow_len = coil_dia * 0.03
    norm = plt.Normalize(mags.min(), mags.max())
    colors = plt.cm.viridis(norm(mags_s))
    ax.quiver(pts[:,0], pts[:,1], pts[:,2],
              vecs[:,0], vecs[:,1], vecs[:,2],
              length=arrow_len, normalize=True, colors=colors, linewidths=0.75)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    mappable.set_array(mags_s)
    ax.get_figure().colorbar(mappable, ax=ax, label='N/m')
    
    # Apply equal axis scaling
    _set_equal_3d_axes(ax, c)
    
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(7))
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Normalized Lorentz Force Density Vectors', fontsize=10)

def plot_force_vs_arc(ax, ctx):
    """
    Plots force density (N/m) vs arc length.
    """
    line, = ax.plot(ctx.arc_mid, ctx.F_mags, linewidth=1.5)
    line.set_picker(True)
    line.set_pickradius(20)

    cursor = mplcursors.cursor(line, hover=True, highlight=False)
    @cursor.connect("add")
    def _(sel):
        s = sel.target[0]
        f = sel.target[1]
        sel.annotation.set_text(f"s = {s:.3f} m\nf = {f:.5} N/m")
        bb = sel.annotation.get_bbox_patch()
        bb.set(fc="lightblue", ec="gray", alpha=0.75)
        sel.annotation.set_fontsize(9)

    ax.set_xlabel('Arc length (m)')
    ax.set_ylabel('Force density (N/m)')
    ax.set_title('Lorentz Force Density vs Arc Length')
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(10))
    ax.yaxis.set_major_locator(MaxNLocator(10))

def plot_stress_vs_arc(ax, ctx):
    line, = ax.plot(ctx.arc_mid, ctx.hoop_stress/1e6, linewidth=1.5)
    line.set_picker(True); line.set_pickradius(20)
    cursor = mplcursors.cursor(line, hover=True, highlight=False)
    @cursor.connect("add")
    def _(sel):
        sel.annotation.set_text(
            f"s = {sel.target[0]:.3f} m\nStress = {sel.target[1]:.5} MPa"
        )
        bb = sel.annotation.get_bbox_patch()
        bb.set(fc="lightblue", ec="gray", alpha=0.75)
        sel.annotation.set_fontsize(9)
    ax.set_xlabel('Arc length (m)')
    ax.set_ylabel('Stress (MPa)')
    ax.set_title('Hoop Stress vs Arc Length')
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(10))
    ax.yaxis.set_major_locator(MaxNLocator(10))

def plot_bfield_vs_axis_distance(ax, ctx, num=200):
    """
    Plot |B| vs distance along the coil's symmetry axis,
    sampling with midpoint–Biot–Savart.
    """
    # get the on‐axis curve
    zs, mags = ctx.compute_bfield_along_axis(num=num, use_gauss=ctx.use_gauss)

    # plot
    line, = ax.plot(zs, mags, linewidth=1.5)
    line.set_picker(True)
    line.set_pickradius(20)

    # interactive hover labels
    cursor = mplcursors.cursor(line, hover=True, highlight=False)
    @cursor.connect("add")
    def _(sel):
        z_val, B_val = sel.target
        sel.annotation.set_text(f"x = {z_val:.3f} m\n|B| = {B_val:.5} T")
        bb = sel.annotation.get_bbox_patch()
        bb.set(fc="lightblue", ec="gray", alpha=0.75)
        sel.annotation.set_fontsize(9)

    # axes
    ax.set_xlabel('Distance along axis (m)')
    ax.set_ylabel('|B| (T)')
    ax.set_title('|B| vs. distance along axis of symmetry')
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(10))
    ax.yaxis.set_major_locator(MaxNLocator(10))

def plot_bfield_cross_section(ax,
                              ctx,
                              n: int = 120,
                              log_scale: bool = True,
                              threshold: float | None = None):
    """
    |B| on the centroid plane, with:
      • data        → viridis
      • ≥ threshold → black
      • outside     → white
    """

    # ------------------------------------------------------------------ #
    # Data grid
    # ------------------------------------------------------------------ #
    if getattr(ctx, "cross_section_data", None) is not None:
        X, Y, Bmag, extent = ctx.cross_section_data
    else:
        X, Y, Bmag, extent = ctx.sample_cross_section(n=n)

    if threshold is None:
        threshold = getattr(ctx, "cross_section_threshold", None)

    outside = np.isnan(Bmag)
    thresh_mask = (Bmag >= threshold) & ~outside if threshold is not None else np.zeros_like(Bmag, dtype=bool)

    # Mask only the outside area; leave ≥threshold so they can take "over" colour.
    Bmask = np.ma.masked_array(Bmag, mask=outside)

    # ------------------------------------------------------------------ #
    # Colormap & norm
    # ------------------------------------------------------------------ #
    cmap = plt.cm.get_cmap('viridis').copy()
    cmap.set_bad((0,0,0,1))       # outside
    cmap.set_over((0.925, 0.855, 0.663, 0.3))      # ≥ threshold

    vmin = vmax = None
    norm = None
    if log_scale:
        valid = Bmask.compressed()
        valid = valid[valid > 0]
        if valid.size:
            vmin, vmax = valid.min(), valid.max()
            if vmin == vmax:
                vmin *= 0.5
                vmax *= 1.5
            norm = LogNorm(vmin=vmin, vmax=vmax)

    # ------------------------------------------------------------------ #
    # Image
    # ------------------------------------------------------------------ #
    if norm is None:
        if threshold is not None:
            vmax = threshold
        im = ax.imshow(Bmask, origin='lower', extent=extent,
                       cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        if threshold is not None:
            norm.vmax = threshold      # values ≥ threshold → cmap.set_over
        im = ax.imshow(Bmask, origin='lower', extent=extent,
                       cmap=cmap, norm=norm)

    ax.set_aspect('equal')
    ax.set_xlabel(r"$\vec{v}_1$ (m)")
    ax.set_ylabel(r"$\vec{v}_2$ (m)")

    title = "|B| Cross‑Section at Centroid"
    if threshold is not None:
        title += f" (|B| < {threshold:.2f} T)"

    ax.set_title(title)

    # ------------------------------------------------------------------ #
    # Colour‑bar
    # ------------------------------------------------------------------ #
    cbar = ax.figure.colorbar(im, ax=ax, label='|B| (T)')
    
    cbar.minorticks_off()

    lo = vmin
    hi = threshold if threshold is not None else vmax
    ticks = np.logspace(np.log10(lo), np.log10(hi), 6)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2g}" for t in ticks])

    # ------------------------------------------------------------------ #
    # Hover cursor
    # ------------------------------------------------------------------ #
    cursor = mplcursors.cursor([im], hover=True, highlight=False)

    @cursor.connect("add")
    def _show(sel):
        # Data coords from mplcursors
        x, y = sel.target
        j = int(np.argmin(np.abs(X[0] - x)))
        i = int(np.argmin(np.abs(Y[:, 0] - y)))

        if i >= Bmag.shape[0] or j >= Bmag.shape[1]:
            sel.annotation.set_visible(False); return
        if outside[i, j] or thresh_mask[i, j]:
            sel.annotation.set_visible(False); return

        bval = Bmag[i, j]
        sel.annotation.set_text(
            rf"$\vec{{v}}_1$ = {x:7.3f}" + "\n"
            rf"$\vec{{v}}_2$ = {y:7.3f}" + "\n"
            rf"$|B|$ = {bval:7.4f} T"
        )
        sel.annotation.get_bbox_patch().set(fc="lightblue", ec="gray", alpha=0.75)
        sel.annotation.set_fontsize(9)