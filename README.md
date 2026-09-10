# Coyote Driver Demand Table Validator

Validates Ford Coyote driver demand torque tables against the indicated
torque table, logged engine max, and drivability criteria.

## Why

The driver demand table maps (pedal position, RPM) -> desired torque (N m).
It is the **input** to the PCM's torque system. The stock Coyote table is
intentionally non-linear (aggressive tip-in, WOT plateau) and the PCM's
CLIP/ADD system handles it without drivability issues.

Problems appear when **aftermarket power adders** (superchargers, turbos,
nitrous) or **cam swaps** change the engine's torque characteristics and
the driver demand table is re-tuned incorrectly:

1. **Demand exceeds engine capacity** -- if the table asks for more torque
   than the engine can produce at a given RPM, the PCM's torque model gets
   confused. It tries to hit a load target that's physically impossible.
   This also contaminates the Torque Tuning Tool's log-based corrections:
   requested torque would always exceed actual, producing false correction
   values.

2. **Non-smooth table** -- abrupt gradient changes introduced during
   re-tuning cause jerkiness during pedal transitions. The driver feels
   surging or hesitation.

3. **Inaccurate low-pedal cells** -- if the lowest pedal positions request
   too much torque at low RPM after modifications, it affects idle quality
   and tip-in behavior.

The validation thresholds are calibrated so a **bone stock table passes
cleanly**. The tool catches problems introduced by modifications.

## Features

- Paste-in driver demand tables (Normal, Sport, Fault/FMEM, OSS modifier)
- Paste-in indicated torque table (for max-torque comparison)
- Load WOT log (CSV or HPL) to extract the engine's actual max torque curve
- Validation checks:
  - **Max torque vs indicated table** (CRITICAL) -- demand must not exceed
    what the indicated torque table says the engine can produce
  - **Max torque vs logged engine max** (CRITICAL) -- demand must not
    exceed what a WOT log pull shows the engine actually produces. This
    catches modified engines where the indicated table no longer reflects
    real output.
  - **Monotonicity** (WARNING) -- torque should increase with pedal above
    the decel region
  - **Gradient smoothness** (WARNING) -- flags abrupt gradient changes
  - **Low-pedal / idle** (INFO) -- flags excessive torque at low pedal/RPM
  - **RPM coverage** (INFO) -- checks axis spans idle to redline
  - **Pedal coverage** (INFO) -- checks axis spans closed to WOT
  - **OSS modifier** (WARNING) -- checks multipliers are within bounds of
    neutral (1.0)
  - **OSS effective demand** (WARNING) -- checks that base demand x OSS
    modifier does not exceed indicated max, and flags large deltas
  - **Cross-table comparison** (WARNING) -- Sport >= Normal, Fault <= Normal
- Color-coded issue highlighting on the table grid
- Plots: 3D surface, demand vs indicated max, gradient heatmap
- Export to CSV

## Quick Start

```bash
python driver_demand_tool.py
```

The tool loads with default tables pre-populated. Click **Run Validation**
to see the report.

## Using Your Own Tables

1. Copy your driver demand table from HP Tuners (Ctrl+C in the table editor)
2. In the tool, click **Paste Demand from Clipboard**
3. Copy your indicated torque table from HP Tuners
4. Click **Paste Indicated from Clipboard**
5. Click **Run Validation**

## Loading a WOT Log

To compare driver demand against the engine's actual output:

1. Log a WOT pull through the full RPM range (3rd gear, flat road, safe
   conditions). Log these PIDs at minimum:
   - RPM
   - Accelerator Pedal Position (preferred for WOT detection)
   - Engine Brake Torque (or Air Load as a fallback)
2. In the tool, click **Load WOT Log**
3. Select the CSV or HPL file
4. Click **Run Validation**

The WOT log filter uses accelerator pedal position (default threshold 80%)
to identify WOT samples. If the log has no pedal channel, it falls back to
throttle position. Samples are binned by RPM (250 RPM bins) and the max
torque in each bin becomes the engine's max curve.

The logged-max check is separate from the indicated-table check, so issues
in one report do not contaminate the other.

## Table Format

The tool accepts tab-delimited grids with a trailing axis-label token:

```text
N m    600  1150 1500 1950 2500 3200 3900 4600 5300 6850 rpm
15     30   -1   -6   -11  -16  -22  -30  -35  -43  -55
25     55   38   25   20   12   0    -10  -17  -27  -40
...
```

- First row: column axis (RPM) with a trailing label token
- Each subsequent row: pedal breakpoint, then torque values
- Negative values at low pedal are expected (engine braking / decel torque)

## Multiple Tables

Use the **Demand Table** dropdown to switch between:

- **Normal** -- primary driving mode
- **Sport** -- performance mode (should be >= Normal at high pedal)
- **Fault** -- limp mode (should be <= Normal everywhere)
- **OSS** -- output shaft speed modifier (multipliers, not torque)

Paste each table while the corresponding name is selected.

The OSS modifier table has different axes (pedal x OSS RPM) and stores
multipliers (1.0 = neutral), not torque values. It is validated separately:

- **OSS self-check**: flags multipliers that deviate more than 20% from 1.0
- **OSS effective demand**: computes base demand x OSS multiplier and checks
  the result against the indicated torque table

These checks are separate from the base demand table validation. OSS
issues do not contaminate the demand table report, and vice versa.

## Validation Settings

On the **Validation Report** tab, adjust:

- **Max torque margin** -- tolerance for demand exceeding indicated (N m)
- **Monotonicity tolerance** -- allowed torque decrease between pedal rows
- **Gradient change threshold** -- fraction of average gradient
- **Idle max torque** -- threshold for low-pedal torque at low RPM
- **Idle max RPM** -- RPM below which idle check applies

## Dependencies

- Python 3.10+
- numpy
- matplotlib (for plots, optional)

## Files

| File | Description |
| ---- | ----------- |
| `driver_demand_tool.py` | Main GUI application |
| `table_model.py` | Table data model and interpolation |
| `validator.py` | Validation logic |
| `log_loader.py` | Log loading and WOT max extraction |

## PCM Background

The driver demand table is processed by the PCM's `app_driver_demand_calc`
function, which applies VCT corrections, enrichment blending, lambda
tracking, and clamping before producing the final desired torque. The
CLIP/ADD system (mode 3, stock) then adapts the table over time so
requested torques match actual outputs.

The OSS modifier is applied as a multiplier on the base driver demand
torque, scaling the request based on output shaft speed (vehicle speed).
This is separate from the normal/sport/fault mode tables.

This tool validates the **calibration table** as written, not the
adapted runtime table. The CLIP/ADD system can mask small table errors,
but large errors or non-smooth tables will cause drivability issues that
adaptation cannot fully correct.
