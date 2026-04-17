# CalcSX™
### A Magnetostatics Simulation
**Version:** 2.2.0 &nbsp;|&nbsp; **Author:** Alexander Skrypek &nbsp;|&nbsp; **License:** MIT

Developed with experimental validation assistance from the Columbia Fusion Research Center (CFRC).

---

## Overview

CalcSX™ is a desktop GUI for simulating the electromagnetic and mechanical behavior of superconducting coils. It computes Lorentz forces, hoop stress, B-field profiles, and 3-D magnetic field lines using a vectorized Biot-Savart kernel with multi-coil superposition and volumetric current carrying.

---

## Features

| Feature | Description |
|---|---|
| Volumetric Biot-Savart | Gauss-Legendre sub-filament discretization of the REBCO winding pack |
| Multi-Coil Superposition | Forces, stresses, and field lines account for contributions from all coils |
| Per-Vertex Force Gradient | Continuous \|J×B\| color gradient on the tape-stack mesh surface |
| Hoop Stress | Membrane-decomposed hoop stress at each segment midpoint |
| Field Lines | 3-D magnetic streamlines via RK4; per-coil or global mode with conductor exclusion |
| Cross-Section B-Field | 2-D \|B\| heatmap sliced along the axis of symmetry |
| Coil Generators | Solenoid, Princeton Dee, Saddle, CCT — parametric generators with live preview |
| Bobbin Import | `.bobsx` groove channel import with surface normals; Fusion 360 exporter included |
| Session Save/Load | `.csx` files preserve the full coil arrangement, parameters, and transforms |
| Transform Gizmo | Translate/rotate handles with seamless coil switching |
| Dark / Light Themes | Switchable palettes; colormaps, scalar bars, and UI adapt to the active theme |
| VTK Export | `.vtp` export for ParaView post-processing |
| Web Layer Export | Dark + light glTF layers for interactive web demos |

---

## Requirements

Python **3.9+**

```bash
pip install numpy scipy pandas PyQt5 scikit-learn pyvista pyvistaqt
```

> **Apple Silicon:** If `pip install pyvista pyvistaqt` fails, use `conda install -c conda-forge pyvista pyvistaqt`.

---

## Getting Started

```bash
python -m CalcSX_app.main
```

1. **Load** a coil CSV (or generate one via CONSTRUCT tab)
2. Set **parameters** (winds, current, tape thickness/width)
3. **Run Analysis** — Forces, Stress, and B Axis layers appear
4. **INSPECT** tab — Field Lines, Cross Section, Global Field
5. **CONSTRUCT** tab — Translate, Rotate, Generate Coil
6. **UTILITIES** tab — Save/Load Session, Normalize Forces, Settings

---

## File Formats

| Extension | Description |
|---|---|
| `.csx` | CalcSX session — coil coordinates, parameters, transforms, bobbin meshes |
| `.bobsx` | Bobbin groove geometry — centerline, normals, optional mesh (Fusion 360 export) |
| `.csv` | Coil centerline coordinates (`x,y,z` columns) |
| `.vtp` | VTK PolyData export for ParaView |

---

## Version History

| Version | Highlights |
|---|---|
| **2.2.0** | Bobbin import; per-vertex force gradient; `.csx` sessions; VTK/web export; Princeton Dee generator; legacy cleanup |
| 2.1.0 | Multi-coil superposition; volumetric REBCO model; dark/light themes; global field lines |
| 2.0.0 | Field lines; cross section; transform gizmo |
| 1.3.0 | PyVista 3-D backend; CAD workbench |
| 1.0.0 | Initial release |

---

## License

MIT License. © 2026 Alexander Skrypek. CalcSX™ is a pending trademark of Alexander Skrypek.

For support: as7168@columbia.edu
