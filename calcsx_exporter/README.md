# CalcSX Bobbin Exporter for Fusion 360

Exports groove channel geometry from a Fusion 360 bobbin model into a `.bobsx` file that CalcSX can import directly. Each selected groove face becomes an individual channel with exact centerline coordinates and surface normals sampled from Fusion's CAD kernel.

## Installation

1. Download or clone this folder (`calcsx_exporter/`)
2. Open Fusion 360
3. Go to **UTILITIES > Scripts and Add-Ins**
4. Click the green **+** next to "My Scripts"
5. Browse to the `calcsx_exporter` folder and select it
6. The script appears in your Scripts list

## Usage

1. Open your bobbin model in Fusion 360
2. Go to **UTILITIES > Scripts and Add-Ins > My Scripts > calcsx_exporter > Run**
3. A command dialog appears — select all groove floor faces:
   - Click a groove face to select it
   - Hold **Ctrl** (Windows) or **Cmd** (Mac) to select additional faces
   - Each face becomes one channel in the output
4. Set **Points per channel** (default 500 — higher = smoother centerline)
5. Click **OK**
6. Choose a save location — the file is saved as `.bobsx`

## In CalcSX

1. Click **Import Bobbin** (⬡) in the SIMULATION ribbon
2. Select your `.bobsx` file
3. Each channel appears as a separate coil that can be:
   - Shown/hidden via the eye toggle in the browser panel
   - Deleted individually
   - Analyzed independently or as part of a multi-coil superposition

## What Gets Exported

- **Channels**: One per selected face. Each contains 3D centerline points with exact surface normals evaluated from the parametric CAD surface (not the tessellation mesh).
- **Bobbin mesh** (optional): A triangulated display mesh of the bobbin body for visual context in CalcSX. Not used for physics calculations.

## File Format

The `.bobsx` file is JSON:

```json
{
  "version": 1,
  "unit": "cm",
  "channels": [
    {
      "name": "groove_1",
      "points": [
        {"x": 1.0, "y": 2.0, "z": 3.0, "nx": 0.0, "ny": 0.0, "nz": 1.0}
      ]
    }
  ],
  "bobbin_mesh": {
    "vertices": [[x, y, z], ...],
    "faces": [[i, j, k], ...]
  }
}
```

Coordinates are in centimeters (Fusion's internal unit). CalcSX converts to metres on import.

## Requirements

- Fusion 360 (any recent version)
- No additional Python packages needed — the script uses only Fusion's built-in API
