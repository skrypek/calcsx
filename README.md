# CalcSX™
### A Magnetostatics Simulation
**Version:** 2.1.0 &nbsp;|&nbsp; **Author:** Alexander Skrypek &nbsp;|&nbsp; **License:** MIT

Developed with experimental validation assistance from the Columbia Fusion Research Center (CFRC).

---

## Overview

CalcSX™ is a desktop GUI for simulating and analyzing the electromagnetic and mechanical behavior of superconducting coils. Given a coil geometry (CSV of 3D coordinates), CalcSX™ computes:

- **B-field** magnitude at the coil centroid and along the axis of symmetry
- **Lorentz force density** vectors at each segment (J×B), with multi-coil superposition
- **Hoop stress** via membrane decomposition, including radial stress profiles
- **3-D magnetic field lines** traced via batched RK4 integration through the superposed field
- **Cross-sectional B-field maps** — interactive 2D heatmap sliced through any plane along the axis of symmetry
- **Multi-coil interactions** — superposition of B-fields from all coils; forces account for neighboring coil contributions
- **Volumetric current carrying** — finite winding-pack cross-section via Gauss-Legendre sub-filament discretization

The coil geometry is analyzed with PCA to determine the axis of symmetry automatically.

---

## Version 2.1.0 — The Global Update

### Multi-Coil Superposition

| Change | Detail |
|---|---|
| **Physics-aware environment** | `MultiCoilEnvironment` orchestrator tracks all coils; B-field from each coil includes contributions from every other coil via superposition |
| **External field injection** | Each `CoilAnalysis` engine receives a `B_ext` callback that sums Biot-Savart from all neighboring coils — forces, stresses, and field lines see the full multi-coil field |
| **Per-coil analysis** | Each coil stores its own engine; analyses are independent — adding or re-analyzing one coil doesn't destroy another's results |
| **Staleness tracking** | Adding, removing, or moving a coil marks all others stale (browser warning icon); **Re-analyze All** button processes the queue sequentially |
| **Global field lines** | Toggle in INSPECT tab traces field lines through the superposed B-field of all coils simultaneously; mutually exclusive with per-coil field lines |

### Volumetric REBCO Coil Model

| Change | Detail |
|---|---|
| **Finite cross-section** | Winding pack discretized into sub-filaments via 2D Gauss-Legendre quadrature over the rectangular REBCO cross-section |
| **REBCO orientation** | Tape wraps around the coil form: radial build = winds × tape thickness; axial extent = tape width |
| **Auto-scaling** | Sub-filament count scales with winding count and aspect ratio (1×1 for single-turn coils, up to 4×4 for thick packs) |
| **Tube rendering** | Coils render as swept rectangular cross-section meshes showing the actual winding-pack geometry; updates live as parameters change |
| **Smooth field lines** | Field-line visualization uses the filamentary macroscopic field to avoid near-field artifacts from multi-filament structure; conductor exclusion prevents lines from passing through the winding pack |

### Dark and Light Themes

| Change | Detail |
|---|---|
| **Light mode** | Viridis-inspired palette: teal accent, deep text, white backgrounds; colormaps switch to viridis for field lines, forces, and cross sections |
| **Dark mode** | VS-Code-inspired palette (unchanged from v2.0.0): cyan accent, warm orange secondary, dark backgrounds |
| **Live switching** | Settings dialog in UTILITIES tab; all UI elements update instantly — ribbon, browser, properties, scalar bars, floor grid, navigation cube, toolbar buttons, layer swatches |
| **Theme-aware colormaps** | Force arrows, field lines, cross sections, and scalar bar text all adapt to the active theme |

### UI Enhancements

| Change | Detail |
|---|---|
| **Navigation ViewCube** | Classic CAD-style orientation cube in top-right corner; click faces for preset views, drag corners for quick orbit; themes with the active mode |
| **Per-coil parameters** | Winds, current, tape thickness, and tape width are stored independently per coil; switching selection in the browser loads that coil's parameters into the Properties panel |
| **Seamless gizmo switching** | Selecting a different coil while Translate or Rotate is active moves the gizmo to the new coil's centroid automatically |
| **Coil re-coloring** | Click any coil's color swatch in the browser to open a color picker |
| **Non-overlapping scalar bars** | Multiple legends stack vertically and resize dynamically |
| **Parameter-driven staleness** | Changing any coil parameter (winds, current, etc.) marks all analyses stale and updates the tube rendering in real time |
| **Normalize Forces** | Moved to UTILITIES ribbon tab as a toggle button |

---

## Version History

| Version | Highlights |
|---|---|
| **2.1.0** | Multi-coil superposition; volumetric REBCO current carrying; dark/light themes; navigation ViewCube; per-coil parameters; global field lines |
| 2.0.0 | INSPECT tools (field lines, cross section); interactive transform gizmo |
| 1.3.0 | Full PyVista/VTK 3-D backend; CAD style workbench; B-field volume |
| 1.2.0 | Two-panel workbench; dark theme throughout; B-field slice viewer |
| 1.1.0 | Vectorized Biot-Savart kernel (10–50× speedup); chunked cross-section |
| 1.0.0 | Initial release: CSV loading, Biot-Savart, PyQt5 GUI |

---

## Features

| Feature | Description |
|---|---|
| Coil Visualization | Swept rectangular tube showing winding-pack cross-section, with centerline wire overlay |
| Multi-Coil Superposition | B-fields from all coils are summed; forces on each coil account for the full environment |
| Volumetric Current | Gauss-Legendre sub-filament discretization of the REBCO winding pack |
| Lorentz Force Layer | Per-segment J×B arrow glyphs, colored by magnitude |
| Hoop Stress Layer | Midpoint point cloud colored by hoop stress (MPa) |
| On-Axis B-Field Layer | Point cloud along the PCA axis colored by \|B\| |
| Field Lines (INSPECT) | 3-D magnetic streamlines with conductor exclusion; per-coil or global mode |
| Cross Section (INSPECT) | 2-D B-field heatmap in any plane along the axis |
| Global Field Lines | Superposed field topology across all coils via toggle in INSPECT tab |
| Layer Browser | Eye-icon toggles per layer; color swatches; rename and recolor coils |
| Ribbon Toolbar | SIMULATION / INSPECT / CONSTRUCT / UTILITIES tabs |
| Navigation Cube | CAD-style ViewCube with click-to-orient and drag-to-orbit |
| Dark / Light Themes | Switchable via Settings in UTILITIES; viridis light mode, VS-Code dark mode |
| Transform Gizmo | SolidWorks-style translate/rotate handles; seamless coil switching |

---

## Requirements

### Python Version
Python **3.9** or later is recommended.

### Required Libraries

| Library | Purpose |
|---|---|
| `numpy` | Numerical arrays, vectorized Biot-Savart kernel |
| `scipy` | Rotation transforms for coil positioning |
| `pandas` | CSV loading and data handling |
| `matplotlib` | Matplotlib backend for legacy plots |
| `PyQt5` | GUI framework |
| `scikit-learn` | PCA for coil axis detection |
| `pyvista` | 3-D mesh data structures (VTK wrapper) |
| `pyvistaqt` | PyVista Qt-embedded interactor widget |

Install all dependencies:

```bash
pip install numpy scipy pandas matplotlib PyQt5 scikit-learn pyvista pyvistaqt
```

With a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows

pip install numpy scipy pandas matplotlib PyQt5 scikit-learn pyvista pyvistaqt
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

The interface uses a **CAD-style workbench**: ribbon at top, layer browser + properties on the left, 3-D viewport filling the rest.

1. Click **Load CSV** in the ribbon — the coil appears as a rendered tube with its winding-pack cross-section visible.
2. Set coil parameters in the Properties panel (winds, current, tape dimensions). Each coil stores its own parameters independently.
3. Click **Run Analysis** — Forces, Stress, and B Axis layers appear under the coil in the browser.
4. Load additional coils — each gets its own color; existing analyses are marked stale. Use **Re-analyze All** to update.
5. Switch to the **INSPECT** tab:
   - **Field Lines** — traces 3-D field lines through the superposed B-field of all coils.
   - **Cross Section** — renders a 2-D B-field heatmap.
   - **Global Field** — toggle to visualize the combined field topology across all coils.
6. Use the **CONSTRUCT** tab to translate/rotate coils; selecting a different coil while the gizmo is active seamlessly moves the handles.
7. Toggle any layer using the eye icons in the browser. Switch between dark and light themes via **Settings** in the UTILITIES tab.

### 3-D Navigation

| Action | Result |
|---|---|
| Left-click drag | Orbital rotate |
| Right-click drag / Two-finger drag | Pan |
| Scroll wheel / Two-finger scroll | Zoom |
| Click ViewCube face | Snap to preset orientation (Front, Back, Top, Bottom, Left, Right) |
| Drag ViewCube corner | Quick orbit |
| **Reset View** button | Fit all to camera |

---

## Future Directions

### High Priority
- **GPU acceleration** — offload the Biot-Savart kernel to CUDA/Metal via CuPy or PyTorch for interactive-speed computation on large coil geometries.
- **Full volumetric force integration** — integrate J×B over the conductor cross-section (not just at the centerline) with adaptive near/far splitting for performance.
- **Radial stress profiles** — compute stress at each radial layer within the winding pack; identify peak stress at the innermost turn.

### Physics Extensions
- **Quench simulation** — thermal runaway propagation with animated quench-front visualization.
- **Field line topology tools** — Poincare section maps, separatrix detection, connection-length plots.
- **Global cross-section mode** — user-positioned plane showing total |B| from all coils.

### Design and CAD
- **In-app coil prototyping** — parametric builder (solenoid, saddle, helical, pancake); export to CSV/STEP.
- **Click-to-inspect** — query local B, J×B, and stress at any point on the coil.
- **Undo / redo** — full action stack.

### Output and Integration
- **VTK / ParaView export** — save field volumes and meshes for downstream post-processing.
- **Persistent settings** — JSON config for last-used parameters and theme preference.
- **Briefcase packaging** — signed macOS and Windows app bundles.

---

## License

MIT License. © 2026 Alexander Skrypek. CalcSX™ is a pending trademark of Alexander Skrypek.

For support: as7168@columbia.edu
