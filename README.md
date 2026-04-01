# CalcSX™
### A Magnetostatics Simulation
**Version:** 2.0.0 &nbsp;|&nbsp; **Author:** Alexander Skrypek &nbsp;|&nbsp; **License:** MIT

Developed with experimental validation assistance from the Columbia Fusion Research Center (CFRC).

---

## Overview

CalcSX™ is a desktop GUI for simulating and analyzing the electromagnetic and mechanical behavior of superconducting coils. Given a coil geometry (CSV of 3D coordinates), CalcSX™ computes:

- **B-field** magnitude at the coil centroid and along the axis of symmetry
- **Lorentz force density** vectors at each segment (J×B)
- **Hoop stress** via membrane decomposition
- **3-D magnetic field lines** traced via batched RK4 integration
- **Cross-sectional B-field maps** — interactive 2D heatmap sliced through any plane along the axis of symmetry

The coil geometry is analyzed with PCA to determine the axis of symmetry automatically.

---

## Version 2.0.0 has arrived!

### INSPECT Tools — Field Visualization Overhaul

| Change | Detail |
|---|---|
| **Magnetic field line tracing** | 3-D streamlines seeded on a Fibonacci sphere around the coil, integrated forward and backward via batched RK4 |
| **Slidable cross-section plane** | 2-D B-field heatmap sliced perpendicular to the PCA symmetry axis; position adjustable via "Section Pos. (m)" spinbox in the Properties panel |
| **log₁₀ colormap scaling** | Dynamic range compression for both INSPECT layers — near-coil strong-field and far-field weak-field are simultaneously readable |
| **Independent scalar bars** | Each INSPECT layer carries its own legend; visibility follows the browser eye-icon toggle |
| **Non-interfering layers** | Field Lines and Cross Section coexist — running one does not clear the other |

### Workspace and UI

| Change | Detail |
|---|---|
| **Removed B-Field Volume** | Legacy feature was never pushed and ultimately scrapped in favor of more practical visualization tools |

### Physics Core

| Change | Detail |
|---|---|
| **Batched RK4 field-line integrator** | All active seed lines advanced simultaneously in NumPy — no Python loop over individual lines |
| **Chunked midplane evaluation** | Cross-section B-field computed in 2 000-point chunks for memory efficiency |
| **Fibonacci sphere seeding** | Uniform seed distribution on the unit sphere; configurable seed count (8–60) in the Properties panel |

---

## Version History

| Version | Highlights |
|---|---|
| **2.0.0** | INSPECT tools introduced; UI adjustments |
| 1.3.0 | Full PyVista/VTK 3-D backend; Fusion 360 workbench (ribbon, layer browser, properties panel); B-field volume (GPU + point-cloud); normalize-forces toggle |
| 1.2.0 | Major UI overhaul: two-panel workbench, dark theme throughout; B-field slice viewer |
| 1.1.0 | Vectorized Biot-Savart kernel (10–50× speedup); chunked cross-section; Save Results |
| 1.0.0 | Initial release: CSV loading, Biot-Savart, PyQt5 GUI, force/stress/axis plots |

---

## Features

| Feature | Description |
|---|---|
| Coil Visualization | GPU-rendered smooth spline wire over a floor-grid reference plane |
| Lorentz Force Layer | Per-segment J×B arrow glyphs, plasma-coloured by magnitude; normalizable |
| Hoop Stress Layer | Midpoint point cloud coloured by hoop stress (MPa, YlOrRd) |
| On-Axis B-Field Layer | Point cloud along the PCA axis coloured by \|B\| (cool cmap) |
| Field Lines (INSPECT) | 3-D magnetic streamlines, batched RK4, log₁₀\|B\| coloured (cool) |
| Cross Section (INSPECT) | 2-D B-field heatmap in any plane along the axis; slidable offset |
| Layer Browser | Eye-icon toggles per layer and per group; legend visibility follows |
| Ribbon Toolbar | SIMULATION / INSPECT / CONSTRUCT / UTILITIES tabs |
| Integration Methods | Vectorized Biot-Savart; optional Gaussian Quadrature |
| Dark Theme | VS-Code-inspired palette throughout UI |

---

## Requirements

### Python Version
Python **3.9** or later is recommended.

### Required Libraries

| Library | Purpose |
|---|---|
| `numpy` | Numerical arrays, vectorized Biot-Savart kernel |
| `scipy` | Scientific utilities |
| `pandas` | CSV loading and data handling |
| `matplotlib` | 2-D legacy plots |
| `PyQt5` | GUI framework |
| `mplcursors` | Interactive hover cursors on 2D plots |
| `scikit-learn` | PCA for coil axis detection |
| `pyvista` | 3-D mesh/volume data structures (VTK wrapper) |
| `pyvistaqt` | PyVista Qt-embedded interactor widget |

Install all dependencies:

```bash
pip install numpy scipy pandas matplotlib PyQt5 mplcursors scikit-learn pyvista pyvistaqt
```

With a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows

pip install numpy scipy pandas matplotlib PyQt5 mplcursors scikit-learn pyvista pyvistaqt
```

> **Note — Apple Silicon (M1/M2/M3):** VTK is available via pip on arm64 since VTK 9.2. If `pip install pyvista pyvistaqt` fails, try `conda install -c conda-forge pyvista pyvistaqt`.

---

## Getting Started

### 1. Prepare Your Coil CSV

The input CSV must contain the 3-D coordinates of the coil centerline with columns `x`, `y`, `z` (case-sensitive). If no headers are present, the first three columns are used.

```
x,y,z
0.1,0.0,0.0
0.0707,0.0707,0.001
...
```

The curve does **not** need to be manually closed — CalcSX™ closes open loops automatically.

### 2. Run the Application

```bash
python -m CalcSX_app.main
```

### 3. Workflow

The interface uses a **Fusion 360-style workbench**: ribbon at top, layer browser + properties on the left, 3-D viewport filling the rest.

1. Click **▲ Load CSV** in the ribbon FILE group — the coil wire appears in the viewport and "Coil" is added to the browser under **Coils**.
2. Set coil parameters in the Properties panel (winds, current, tape dimensions, etc.).
3. Click **▶ Run Analysis** — Forces, Stress, and B Axis layers appear; results summary populates.
4. Switch to the **INSPECT** tab in the ribbon:
   - **∿ Field Lines** — traces 3-D magnetic field lines (set seed count in Properties).
   - **⊡ Cross Section** — renders a 2-D B-field heatmap; use "Section Pos. (m)" to slide the plane along the symmetry axis.
5. Toggle any layer using the eye icons (●/○) in the browser panel; legends hide/show with their layer.

### 3-D Navigation

| Action | Result |
|---|---|
| Left-click drag | Orbital rotate (Z-axis locked — ground stays fixed) |
| Right-click drag | Zoom |
| Scroll wheel | Zoom |
| Middle-click drag | Pan |
| **⌖ Reset View** button | Fit all to camera |

---

## Future Directions

The following capabilities are planned for future versions of CalcSX™:

### High Priority
- **GPU acceleration** — offload the Biot-Savart kernel to CUDA/Metal via CuPy or PyTorch so that large coil geometries (>10 000 segments) and fine field-line integration become interactive-speed.
- **Real conductor cross-sections** — replace the filamentary wire model with a finite cross-section discretization (e.g., a stack of ReBCO tapes or a cable bundle); compute B and force distributions within the winding pack rather than just at the centerline.

### Physics Extensions
- **Quench simulation** — thermal runaway model propagating through the winding pack; track normal-zone growth, hot-spot temperature, and energy dump into a protection circuit. Visualize the propagating quench front in the 3-D viewport.
- **Field line topology tools** — Poincaré section maps, separatrix detection, and field-line connection-length plots for confinement analysis.
- **Multi-coil superposition** — load multiple CSVs and sum their B-fields; layer browser groups each coil independently; useful for solenoid pairs, Helmholtz coils, and tokamak PF-coil sets.

### Design and CAD
- **In-app coil prototyping** — parametric coil geometry builder (solenoid, saddle, helical, pancake) with real-time preview; export to CSV or STEP for manufacturing.
- **Click-to-inspect** — click any point on the coil wire to query local B, J×B, and stress; scroll the sidebar to the corresponding layer.
- **Undo / redo** — full action stack so CSV reloads and analysis re-runs can be stepped back.

### Output and Integration
- **VTK / ParaView export** — save computed field volumes and surface meshes for downstream post-processing.
- **2-D detail panels** — floating windows per layer with arc-length plots (force vs arc, stress vs arc, on-axis B-field profile).
- **Persistent settings** — save last-used coil parameters, seed count, and section offset to a JSON config file.
- **Briefcase packaging** — signed, distributable macOS and Windows app bundles.

---

## License

MIT License. © 2026 Alexander Skrypek. CalcSX™ is a pending trademark of Alexander Skrypek.

For support: as7168@columbia.edu
