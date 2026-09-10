"""Driver Demand table data model.

A DriverDemandTable is a 2D calibration table:
  * Rows: pedal position (ADC counts or %)
  * Cols: RPM
  * Values: desired torque in N m

The driver demand table maps (pedal, RPM) -> desired brake torque.
This is the INPUT to the PCM's torque system — it tells the PCM how
much torque the driver is requesting. The PCM then uses the inverse
torque table to convert that request into a target load, and actuates
the throttle to hit it.

Key relationships:
  - Driver demand max torque should NOT exceed the indicated torque
    table's max at any RPM. If it does, the PCM will try to hit a load
    target that the indicated table says is impossible, throwing the
    torque model off.
  - The table should be smooth (monotonic in pedal above the decel
    region, no abrupt gradient changes) to avoid drivability issues.
  - Low-pedal cells should produce small or negative torque to avoid
    affecting idle quality.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DriverDemandTable:
    """A 2D driver demand table: (pedal, RPM) -> desired torque (N m)."""

    row_axis: np.ndarray          # 1D, length = n_rows (pedal ADC or %)
    col_axis: np.ndarray          # 1D, length = n_cols (RPM)
    values: np.ndarray            # 2D, shape (n_rows, n_cols), N m
    row_label: str = "Pedal"
    col_label: str = "RPM"
    val_label: str = "N m"
    table_name: str = "Normal"    # Normal, Sport, Fault/FMEM, etc.

    def __post_init__(self) -> None:
        self.row_axis = np.asarray(self.row_axis, dtype=float)
        self.col_axis = np.asarray(self.col_axis, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        if self.values.shape != (len(self.row_axis), len(self.col_axis)):
            raise ValueError(
                f"values shape {self.values.shape} != "
                f"({len(self.row_axis)}, {len(self.col_axis)})"
            )

    @property
    def n_rows(self) -> int:
        return len(self.row_axis)

    @property
    def n_cols(self) -> int:
        return len(self.col_axis)

    def copy(self) -> "DriverDemandTable":
        return DriverDemandTable(
            self.row_axis.copy(), self.col_axis.copy(), self.values.copy(),
            self.row_label, self.col_label, self.val_label, self.table_name,
        )

    # ---- interpolation -------------------------------------------------
    def interp(self, pedal: float | np.ndarray,
               rpm: float | np.ndarray) -> np.ndarray:
        """Bilinear interpolation of the table at (pedal, rpm)."""
        return _bilinear(self.row_axis, self.col_axis, self.values, pedal, rpm)

    def max_torque_per_rpm(self) -> np.ndarray:
        """Max torque across all pedal positions, per RPM column."""
        return self.values.max(axis=0)

    def max_torque_at_rpm(self, rpm: float) -> float:
        """Interpolated max torque at a given RPM."""
        max_per_col = self.max_torque_per_rpm()
        return float(np.interp(rpm, self.col_axis, max_per_col))

    # ---- I/O -----------------------------------------------------------
    @classmethod
    def from_grid_text(cls, text: str, table_name: str = "Normal",
                       row_label: str = "Pedal",
                       col_label: str = "RPM",
                       val_label: str = "N m") -> "DriverDemandTable":
        """Parse a paste-in grid where the first row is the column axis
        (with a trailing axis-name token) and each following row is
        row_breakpoint followed by values."""
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        if not lines:
            raise ValueError("empty table text")
        if len(lines) > 200:
            raise ValueError(
                f"File has {len(lines)} rows — this doesn't look like a "
                f"driver demand table (max ~50 rows)."
            )
        header = _split_numbers_and_label(lines[0])
        col_axis = np.array(header["numbers"], dtype=float)
        if len(col_axis) > 200:
            raise ValueError(
                f"First row has {len(col_axis)} columns — this doesn't "
                f"look like a driver demand table."
            )
        if len(col_axis) == 0:
            raise ValueError("No column breakpoints found in first row")
        rows, row_axis = [], []
        for ln in lines[1:]:
            parts = _split_numbers_and_label(ln)
            if not parts["numbers"]:
                continue
            row_axis.append(parts["numbers"][0])
            vals = parts["numbers"][1:]
            if len(vals) != len(col_axis):
                vals = vals[:len(col_axis)]
            rows.append(vals)
        if not rows:
            raise ValueError("No data rows found")
        return cls(
            np.array(row_axis, dtype=float),
            col_axis,
            np.array(rows, dtype=float),
            row_label, col_label, val_label, table_name,
        )

    def to_grid_text(self) -> str:
        """Render as a paste-in grid with a trailing axis-label token."""
        lines = []
        header = "\t" + "\t".join(_fmt(c) for c in self.col_axis) + "\t" + self.col_label
        lines.append(header)
        for r, rv in enumerate(self.row_axis):
            row = _fmt(rv) + "\t" + "\t".join(_fmt(v) for v in self.values[r])
            lines.append(row)
        return "\n".join(lines)

    def to_csv(self) -> str:
        """Render as CSV (no trailing label token, leading cell empty)."""
        lines = []
        header = "," + ",".join(_fmt(c) for c in self.col_axis) + "," + self.col_label
        lines.append(header)
        for r, rv in enumerate(self.row_axis):
            row = _fmt(rv) + "," + ",".join(_fmt(v) for v in self.values[r])
            lines.append(row)
        return "\n".join(lines)


@dataclass
class IndicatedTorqueTable:
    """A 2D indicated torque table: (load, RPM) -> indicated torque (N m).
    Used to check that driver demand doesn't exceed what the engine can
    actually produce."""

    row_axis: np.ndarray          # 1D, length = n_rows (load, g/cyl)
    col_axis: np.ndarray          # 1D, length = n_cols (RPM)
    values: np.ndarray            # 2D, shape (n_rows, n_cols), N m
    row_label: str = "Load"
    col_label: str = "RPM"
    val_label: str = "N m"

    def __post_init__(self) -> None:
        self.row_axis = np.asarray(self.row_axis, dtype=float)
        self.col_axis = np.asarray(self.col_axis, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        if self.values.shape != (len(self.row_axis), len(self.col_axis)):
            raise ValueError(
                f"values shape {self.values.shape} != "
                f"({len(self.row_axis)}, {len(self.col_axis)})"
            )

    @property
    def n_rows(self) -> int:
        return len(self.row_axis)

    @property
    def n_cols(self) -> int:
        return len(self.col_axis)

    def max_torque_per_rpm(self) -> np.ndarray:
        """Max indicated torque across all load rows, per RPM column."""
        return self.values.max(axis=0)

    def max_torque_at_rpm(self, rpm: float) -> float:
        """Interpolated max indicated torque at a given RPM."""
        max_per_col = self.max_torque_per_rpm()
        return float(np.interp(rpm, self.col_axis, max_per_col))

    @classmethod
    def from_grid_text(cls, text: str, row_label: str = "Load",
                       col_label: str = "RPM",
                       val_label: str = "N m") -> "IndicatedTorqueTable":
        """Parse a paste-in grid."""
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        if not lines:
            raise ValueError("empty table text")
        if len(lines) > 200:
            raise ValueError(
                f"File has {len(lines)} rows — this doesn't look like a "
                f"torque table (max ~50 rows)."
            )
        header = _split_numbers_and_label(lines[0])
        col_axis = np.array(header["numbers"], dtype=float)
        if len(col_axis) == 0:
            raise ValueError("No column breakpoints found in first row")
        rows, row_axis = [], []
        for ln in lines[1:]:
            parts = _split_numbers_and_label(ln)
            if not parts["numbers"]:
                continue
            row_axis.append(parts["numbers"][0])
            vals = parts["numbers"][1:]
            if len(vals) != len(col_axis):
                vals = vals[:len(col_axis)]
            rows.append(vals)
        if not rows:
            raise ValueError("No data rows found")
        return cls(
            np.array(row_axis, dtype=float),
            col_axis,
            np.array(rows, dtype=float),
            row_label, col_label, val_label,
        )


# ---- helpers ---------------------------------------------------------------

def _bilinear(xa: np.ndarray, ya: np.ndarray, Z: np.ndarray,
              x: float | np.ndarray, y: float | np.ndarray) -> np.ndarray:
    """Bilinear interpolation. xa is row axis (x), ya is col axis (y),
    Z has shape (len(xa), len(ya)). Clamps to axis bounds (no extrapolation),
    matching the PCM's cal_interp_2d behavior."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    scalar = x.ndim == 0 and y.ndim == 0
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    xc = np.clip(x, xa[0], xa[-1])
    yc = np.clip(y, ya[0], ya[-1])
    i0 = np.searchsorted(xa, xc, side="right") - 1
    i0 = np.clip(i0, 0, len(xa) - 2)
    j0 = np.searchsorted(ya, yc, side="right") - 1
    j0 = np.clip(j0, 0, len(ya) - 2)
    x0 = xa[i0]; x1 = xa[i0 + 1]
    y0 = ya[j0]; y1 = ya[j0 + 1]
    tx = (xc - x0) / (x1 - x0)
    ty = (yc - y0) / (y1 - y0)
    z00 = Z[i0, j0]; z01 = Z[i0, j0 + 1]
    z10 = Z[i0 + 1, j0]; z11 = Z[i0 + 1, j0 + 1]
    out = (z00 * (1 - tx) * (1 - ty) + z01 * (1 - tx) * ty
           + z10 * tx * (1 - ty) + z11 * tx * ty)
    return float(out[0]) if scalar else out


def _fmt(v: float) -> str:
    """Compact number formatting that avoids scientific notation."""
    if v == 0:
        return "0"
    if abs(v) >= 1000:
        return f"{v:.1f}"
    if abs(v) >= 100:
        return f"{v:.2f}"
    if abs(v) >= 10:
        return f"{v:.3f}"
    if abs(v) >= 1:
        return f"{v:.4f}"
    return f"{v:.6f}".rstrip("0").rstrip(".")


def _split_numbers_and_label(line: str) -> dict:
    """Split a tab/comma separated line into numeric tokens and a trailing
    non-numeric label token (e.g. 'rpm', 'N m')."""
    raw = line.replace(",", "\t").split("\t")
    raw = [c.strip() for c in raw if c.strip() != ""]
    numbers: list[float] = []
    label = ""
    for tok in raw:
        try:
            numbers.append(float(tok))
        except ValueError:
            label = tok
    return {"numbers": numbers, "label": label}
