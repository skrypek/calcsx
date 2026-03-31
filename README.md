# CalcSX™
### A Magnetostatics Simulation
**Version:** 1.1.0 &nbsp;|&nbsp; **Author:** Alexander Skrypek &nbsp;|&nbsp; **License:** MIT

Developed with assistance from the Columbia Fusion Research Center (CFRC).

---

## Overview

CalcSX™ is a desktop GUI application for simulating and analyzing the electromagnetic and mechanical behavior of superconducting coils. Given a coil geometry defined as a 3D coordinate path, CalcSX™ computes:

- **B-field** magnitude at the coil centroid and along the axis of symmetry
- **Lorentz force density** vectors at each coil segment (J×B)
- **Hoop stress** via membrane decomposition
- **B-field cross-section** heat map on the centroid plane (optional)

The coil geometry is analyzed using Principal Component Analysis (PCA) to automatically determine the axis of symmetry and detect planar coil configurations.

---

## What's New in v1.1.0

| Change | Detail |
|---|---|
| **Vectorized Biot-Savart** | Core field kernel fully NumPy-vectorized — no Python loops over segments. Typical speedup: 10–50× for force integration, 100×+ for cross-section sampling |
| **Chunked cross-section** | Large grids computed in 500-point memory-safe batches; no RAM spikes at high resolution |
| **Adaptive grid resolution** | Cross-section grid size is user-configurable (32–512 pts/axis); no longer fixed at 120 |
| **Configurable on-axis samples** | On-axis B-field profile sample count configurable (50–1000 pts) |
| **Save Results** | Export arc data (force density, hoop stress) and on-axis B-field to CSV directly from the results view |
| **Stage-aware loading screen** | Progress dialog shows the active computation step alongside quippy messages |

---

## Features

| Feature | Description |
|---|---|
| Filament Visualization | 3D parametric plot of coil geometry with PCA axis and direction of parametrization |
| B-Field Analysis | On-axis magnitude profile (configurable sample count) and optional on-plane heat map |
| Lorentz Force Density | Segment-wise J×B vector field with viridis colormap |
| Hoop Stress | Arc-length-resolved membrane hoop stress (MPa) |
| Integration Methods | Simpson's Rule (default) or Gaussian Quadrature (higher accuracy) |
| Interactive Plots | Hover cursor on all 2D plots for precise numerical readout |
| Planar Coil Support | Automatic planar detection with adapted force/stress algorithms (beta) |
| Result Export | Save computed arrays to CSV for downstream analysis |

---

## Requirements

### Python Version
Python **3.9** or later is recommended.

### Required Libraries

| Library | Purpose |
|---|---|
| `numpy` | Numerical arrays, vectorized Biot-Savart kernel |
| `pandas` | CSV loading, data handling, result export |
| `matplotlib` | All plotting (2D and 3D) |
| `PyQt5` | GUI framework |
| `mplcursors` | Interactive hover cursors on 2D plots |
| `scikit-learn` | PCA for coil axis detection |

Install all dependencies at once:

```bash
pip install numpy pandas matplotlib PyQt5 mplcursors scikit-learn
```

Or with a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
.venv\Scripts\activate          # Windows

pip install numpy pandas matplotlib PyQt5 mplcursors scikit-learn
```

---

## Getting Started

### 1. Prepare Your Coil CSV

The input CSV must contain the 3D coordinates of the coil centerline. Columns should be labeled `x`, `y`, `z` (case-sensitive). If no headers are present, the first three columns are used.

```
x,y,z
0.1,0.0,0.0
0.0707,0.0707,0.001
...
```

The curve does **not** need to be manually closed — CalcSX™ will close open loops automatically.

### 2. Run the Application

```bash
python -m CalcSX_app.main
```

Or, if running from inside the `CalcSX_app` directory:

```bash
python main.py
```

### 3. Workflow

1. Click **Load CSV...** and select your coil geometry file.
2. Click **Preview Curve** to inspect the 3D coil shape.
3. Set coil parameters:
   - **Number of Winds** — total winding count
   - **Current (A)** — operating current
   - **Tape Thickness (µm)** — per-layer tape thickness (default: 80 µm, standard ReBCO)
   - **Tape Width (mm)** — tape width (default: 4.00 mm)
   - **On-Axis Samples** — number of points on the B-field profile (default: 200)
4. *(Optional)* Check **Calculate B-Field Cross-section** to compute a heat map on the centroid plane.
   - Set the **max |B|** filter threshold for display fidelity.
   - Set the **Cross-section Grid** resolution (32–512 pts/axis; higher = more detail, slower).
5. *(Optional)* Check **Use Gaussian Quadrature** for higher-accuracy force integration (not recommended for coils with more than ~300 points).
6. Click **Generate** to run the full analysis.
7. On the results page, click **Save Results…** to export computed arrays to CSV.

---

## Output Plots

After generation, CalcSX™ displays the following in a tabbed results view:

- **Filament Curve** — 3D plot with PCA axis, in-plane basis vectors, and parametrization direction
- **Lorentz Force Density Vectors** — 3D quiver plot colored by magnitude (N/m); toggle normalized arrows via checkbox
- **Force Density vs Arc Length** — 2D line plot with interactive cursor
- **Hoop Stress vs Arc Length** — 2D line plot with interactive cursor (MPa)
- **|B| vs Axis Distance** — On-axis B-field profile with interactive cursor
- **|B| Cross-Section** *(if enabled)* — Log-scale heat map of field magnitude on the centroid plane

---

## Notes

- The **?** button in the top-right corner of the main window opens the in-app help dialog.
- Zooming is not currently supported on any generated plot.
- 3D plots support left-drag rotation.
- Planar coil detection is available but considered **beta** — axis scaling on plots may appear non-intuitive for near-constant fields.

---

## License

MIT License. © 2025 Alexander Skrypek. CalcSX™ is a pending trademark of Alexander Skrypek.

For support: as7168@columbia.edu
