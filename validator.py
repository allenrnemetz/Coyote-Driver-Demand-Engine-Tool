"""Driver Demand table validation logic.

Validation checks for driver demand torque tables and OSS modifiers.
  1. Max torque vs indicated table -- demand must not exceed what the
     engine can actually produce at any RPM.
  2. Monotonicity -- torque should increase with pedal above the decel
     region (negative torque at low pedal is expected for engine braking).
  3. Gradient smoothness -- abrupt gradient changes cause jerkiness.
  4. Low-pedal / idle accuracy -- excessive torque at low pedal affects
     idle quality.
  5. RPM axis coverage -- should span idle to redline.
  6. Pedal axis coverage -- should span closed to WOT.
  7. OSS modifier (optional) -- should be within reasonable bounds.
  8. Cross-table comparison (Normal vs Sport vs Fault).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from table_model import DriverDemandTable, IndicatedTorqueTable
from log_loader import WotMaxCurve


class Severity(Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ValidationIssue:
    """A single validation finding."""
    check: str           # which check produced this
    severity: Severity
    message: str
    row_idx: Optional[int] = None    # cell row index (if applicable)
    col_idx: Optional[int] = None    # cell col index (if applicable)
    value: Optional[float] = None    # the offending value
    limit: Optional[float] = None    # the limit it violated
    rpm: Optional[float] = None      # RPM at this column
    pedal: Optional[float] = None    # pedal at this row


@dataclass
class ValidationConfig:
    """Tuning knobs for the validation checks.

    Defaults are calibrated so a BONE STOCK table passes cleanly. The
    stock Coyote driver demand table has intentional non-linearity
    (aggressive tip-in, plateau at WOT) that the PCM's CLIP/ADD system
    handles without drivability issues. These thresholds are set to
    catch problems introduced by aftermarket power adders (superchargers,
    turbos, nitrous) or cam swaps that change the engine's torque
    characteristics and require re-tuning the driver demand table.
    """
    # Max torque check: flag cells where demand exceeds indicated max
    # by more than this margin (N m). This is the most important check
    # for modified tunes -- if demand exceeds what the engine can
    # produce, the torque model gets confused.
    max_torque_margin: float = 5.0
    # Monotonicity: allow torque to decrease by this much between
    # adjacent pedal rows without flagging (N m). Stock tables have
    # small non-monotonic regions by design.
    monotonicity_tolerance: float = 10.0
    # Gradient smoothness: flag if gradient changes by more than this
    # fraction of the average gradient in the column. Stock tables have
    # intentional non-linear pedal mapping (aggressive tip-in, WOT
    # plateau), so this threshold is set high to avoid false positives
    # on stock tunes. Modified tunes with jagged pedal mapping will
    # still trip this.
    gradient_change_threshold: float = 2.0
    # Low-pedal / idle: flag if the lowest pedal row requests more than
    # this much torque at low RPM (N m). Stock tables have moderate
    # torque at low pedal for tip-in response; only flag if it's
    # clearly excessive.
    idle_max_torque: float = 150.0
    idle_max_rpm: float = 1500.0   # RPM below which idle check applies
    # RPM axis: expected range
    rpm_min_expected: float = 500.0
    rpm_max_expected: float = 6500.0
    # Pedal axis: expected range (ADC counts)
    pedal_min_expected: float = 0.0
    pedal_max_expected: float = 542.0
    # OSS modifier: max allowed deviation from base demand (fraction)
    oss_max_deviation: float = 0.20


@dataclass
class ValidationResult:
    """Full result of validating one or more driver demand tables."""
    issues: list[ValidationIssue] = field(default_factory=list)
    n_critical: int = 0
    n_warning: int = 0
    n_info: int = 0
    table_name: str = ""

    def add(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        if issue.severity == Severity.CRITICAL:
            self.n_critical += 1
        elif issue.severity == Severity.WARNING:
            self.n_warning += 1
        else:
            self.n_info += 1

    @property
    def passed(self) -> bool:
        return self.n_critical == 0


def validate_demand_table(
    demand: DriverDemandTable,
    indicated: Optional[IndicatedTorqueTable] = None,
    cfg: ValidationConfig = None,
) -> ValidationResult:
    """Run all validation checks on a single driver demand table.

    If *indicated* is provided, checks that demand doesn't exceed the
    indicated table's max torque. If not provided, skips that check.
    """
    if cfg is None:
        cfg = ValidationConfig()
    result = ValidationResult(table_name=demand.table_name)

    if indicated is not None:
        check_max_torque(demand, indicated, cfg, result)
    check_monotonicity(demand, cfg, result)
    check_gradient_smoothness(demand, cfg, result)
    check_idle_region(demand, cfg, result)
    check_rpm_coverage(demand, cfg, result)
    check_pedal_coverage(demand, cfg, result)

    return result


# ---- individual checks -----------------------------------------------------

def check_max_torque(
    demand: DriverDemandTable,
    indicated: IndicatedTorqueTable,
    cfg: ValidationConfig,
    result: ValidationResult,
) -> None:
    """Check 1: demand torque must not exceed indicated table max at any RPM."""
    ind_max_per_rpm = indicated.max_torque_per_rpm()
    for j, rpm in enumerate(demand.col_axis):
        # Interpolate indicated max at this demand-table RPM
        ind_max = float(np.interp(rpm, indicated.col_axis, ind_max_per_rpm))
        for i, pedal in enumerate(demand.row_axis):
            val = float(demand.values[i, j])
            if val > ind_max + cfg.max_torque_margin:
                result.add(ValidationIssue(
                    check="max_torque",
                    severity=Severity.CRITICAL,
                    message=(
                        f"Demand {val:.1f} N m exceeds indicated max "
                        f"{ind_max:.1f} N m at {rpm:.0f} RPM, pedal {pedal:.0f}"
                    ),
                    row_idx=i, col_idx=j,
                    value=val, limit=ind_max,
                    rpm=float(rpm), pedal=float(pedal),
                ))


def check_monotonicity(
    demand: DriverDemandTable,
    cfg: ValidationConfig,
    result: ValidationResult,
) -> None:
    """Check 2: torque should increase monotonically with pedal above the
    decel region."""
    for j in range(demand.n_cols):
        col = demand.values[:, j]
        rpm = demand.col_axis[j]
        # Find where torque crosses zero (transition from decel to accel)
        # If all negative, skip (pure decel column)
        positive_indices = np.where(col > 0)[0]
        if len(positive_indices) == 0:
            continue
        accel_start = positive_indices[0]
        # Check monotonicity above the decel region
        for i in range(accel_start, demand.n_rows - 1):
            diff = col[i + 1] - col[i]
            if diff < -cfg.monotonicity_tolerance:
                result.add(ValidationIssue(
                    check="monotonicity",
                    severity=Severity.WARNING,
                    message=(
                        f"Torque decreases {diff:.1f} N m from pedal "
                        f"{demand.row_axis[i]:.0f} to {demand.row_axis[i+1]:.0f} "
                        f"at {rpm:.0f} RPM (non-monotonic in accel region)"
                    ),
                    row_idx=i, col_idx=j,
                    value=float(col[i + 1]), limit=float(col[i]),
                    rpm=float(rpm),
                ))


def check_gradient_smoothness(
    demand: DriverDemandTable,
    cfg: ValidationConfig,
    result: ValidationResult,
) -> None:
    """Check 3: flag abrupt gradient changes that cause jerkiness."""
    for j in range(demand.n_cols):
        col = demand.values[:, j]
        rpm = demand.col_axis[j]
        if demand.n_rows < 3:
            continue
        # Calculate gradient between adjacent pedal rows
        gradients = np.diff(col)
        # Only check in the accel region (positive gradients)
        positive_grads = gradients[gradients > 0]
        if len(positive_grads) < 2:
            continue
        avg_grad = float(np.mean(positive_grads))
        if avg_grad <= 0:
            continue
        for i in range(len(gradients) - 1):
            g1 = gradients[i]
            g2 = gradients[i + 1]
            # Only flag if both are in the accel region
            if g1 <= 0 or g2 <= 0:
                continue
            change = abs(g2 - g1) / avg_grad
            if change > cfg.gradient_change_threshold:
                result.add(ValidationIssue(
                    check="gradient_smoothness",
                    severity=Severity.WARNING,
                    message=(
                        f"Abrupt gradient change at pedal "
                        f"{demand.row_axis[i+1]:.0f}, {rpm:.0f} RPM: "
                        f"gradient {g1:.1f} -> {g2:.1f} N m/pedal "
                        f"(avg {avg_grad:.1f}, change {change:.0%})"
                    ),
                    row_idx=i + 1, col_idx=j,
                    value=float(g2), limit=float(avg_grad),
                    rpm=float(rpm),
                ))


def check_idle_region(
    demand: DriverDemandTable,
    cfg: ValidationConfig,
    result: ValidationResult,
) -> None:
    """Check 4: low-pedal cells should produce small torque at low RPM."""
    for j, rpm in enumerate(demand.col_axis):
        if rpm > cfg.idle_max_rpm:
            continue
        for i in range(min(3, demand.n_rows)):
            val = float(demand.values[i, j])
            pedal = demand.row_axis[i]
            if val > cfg.idle_max_torque:
                result.add(ValidationIssue(
                    check="idle_region",
                    severity=Severity.INFO,
                    message=(
                        f"Low-pedal cell requests {val:.1f} N m at "
                        f"{rpm:.0f} RPM, pedal {pedal:.0f} "
                        f"(exceeds idle threshold {cfg.idle_max_torque:.0f} N m)"
                    ),
                    row_idx=i, col_idx=j,
                    value=val, limit=cfg.idle_max_torque,
                    rpm=float(rpm), pedal=float(pedal),
                ))


def check_rpm_coverage(
    demand: DriverDemandTable,
    cfg: ValidationConfig,
    result: ValidationResult,
) -> None:
    """Check 5: RPM axis should span idle to redline."""
    rpm_min = float(demand.col_axis[0])
    rpm_max = float(demand.col_axis[-1])
    if rpm_min > cfg.rpm_min_expected:
        result.add(ValidationIssue(
            check="rpm_coverage",
            severity=Severity.INFO,
            message=(
                f"RPM axis starts at {rpm_min:.0f}, expected <= "
                f"{cfg.rpm_min_expected:.0f} (may miss idle region)"
            ),
        ))
    if rpm_max < cfg.rpm_max_expected:
        result.add(ValidationIssue(
            check="rpm_coverage",
            severity=Severity.INFO,
            message=(
                f"RPM axis ends at {rpm_max:.0f}, expected >= "
                f"{cfg.rpm_max_expected:.0f} (may miss redline)"
            ),
        ))
    # Check for large gaps
    if demand.n_cols >= 2:
        gaps = np.diff(demand.col_axis)
        max_gap = float(np.max(gaps))
        avg_gap = float(np.mean(gaps))
        if max_gap > 2 * avg_gap:
            result.add(ValidationIssue(
                check="rpm_coverage",
                severity=Severity.INFO,
                message=(
                    f"Large RPM gap: {max_gap:.0f} RPM (avg {avg_gap:.0f}). "
                    f"Interpolation may be coarse in this region."
                ),
            ))


def check_pedal_coverage(
    demand: DriverDemandTable,
    cfg: ValidationConfig,
    result: ValidationResult,
) -> None:
    """Check 6: pedal axis should span closed to WOT."""
    pedal_min = float(demand.row_axis[0])
    pedal_max = float(demand.row_axis[-1])
    if pedal_min > cfg.pedal_min_expected + 20:
        result.add(ValidationIssue(
            check="pedal_coverage",
            severity=Severity.INFO,
            message=(
                f"Pedal axis starts at {pedal_min:.0f}, expected near "
                f"{cfg.pedal_min_expected:.0f} (may miss closed-throttle region)"
            ),
        ))
    if pedal_max < cfg.pedal_max_expected - 50:
        result.add(ValidationIssue(
            check="pedal_coverage",
            severity=Severity.INFO,
            message=(
                f"Pedal axis ends at {pedal_max:.0f}, expected near "
                f"{cfg.pedal_max_expected:.0f} (may miss WOT region)"
            ),
        ))


def check_oss_modifier(
    oss_table: DriverDemandTable,
    cfg: ValidationConfig,
    result: ValidationResult,
) -> None:
    """Check 7: OSS modifier multipliers should be within reasonable bounds.

    The OSS modifier table stores multipliers (1.0 = neutral), not torque
    values. We check that each multiplier stays within [1 - oss_max_deviation,
    1 + oss_max_deviation] of neutral. Values far from 1.0 can produce
    effective demand that exceeds or falls far below the base table.
    """
    for i in range(oss_table.n_rows):
        for j in range(oss_table.n_cols):
            mult = float(oss_table.values[i, j])
            deviation = mult - 1.0
            if abs(deviation) > cfg.oss_max_deviation:
                result.add(ValidationIssue(
                    check="oss_modifier",
                    severity=Severity.WARNING,
                    message=(
                        f"OSS multiplier {mult:.3f} deviates {deviation:+.0%} "
                        f"from neutral at pedal {oss_table.row_axis[i]:.0f}, "
                        f"{oss_table.col_axis[j]:.0f} OSS RPM"
                    ),
                    row_idx=i, col_idx=j,
                    value=mult, limit=1.0,
                ))


def apply_oss_modifier(
    base_demand: DriverDemandTable,
    oss_table: DriverDemandTable,
) -> DriverDemandTable:
    """Compute effective demand = base_demand × oss_multiplier.

    The OSS modifier table has axes (pedal, OSS_RPM) and stores multipliers.
    The base demand table has axes (pedal, engine_RPM) and stores torque.

    The PCM applies the OSS modifier by interpolating the multiplier at the
    current (pedal, OSS_RPM) operating point and multiplying the base demand
    torque by it.  OSS RPM is related to engine RPM via the final drive ratio,
    but for validation purposes we map engine RPM directly onto the OSS RPM
    axis (clamped to the OSS axis range) -- this is conservative because it
    checks the worst-case multiplier at each RPM.

    Returns a NEW DriverDemandTable with the effective torque values.  The
    source tables are not modified.
    """
    effective = base_demand.copy()
    effective.table_name = f"{base_demand.table_name} + OSS"
    for i, pedal in enumerate(base_demand.row_axis):
        for j, rpm in enumerate(base_demand.col_axis):
            # Interpolate OSS multiplier at (pedal, rpm).
            # Clamp rpm to OSS axis range (no extrapolation).
            oss_rpm = float(np.clip(rpm, oss_table.col_axis[0], oss_table.col_axis[-1]))
            oss_pedal = float(np.clip(pedal, oss_table.row_axis[0], oss_table.row_axis[-1]))
            mult = float(oss_table.interp(oss_pedal, oss_rpm))
            effective.values[i, j] = base_demand.values[i, j] * mult
    return effective


def validate_with_oss(
    base_demand: DriverDemandTable,
    oss_table: DriverDemandTable,
    indicated: Optional[IndicatedTorqueTable],
    cfg: ValidationConfig,
) -> ValidationResult:
    """Validate the EFFECTIVE demand (base × OSS modifier).

    This runs the max-torque check on the effective demand table -- the
    torque the PCM will actually request after the OSS modifier is applied.
    The result is separate from the base-table validation so issues in the
    OSS-applied demand do not contaminate the base-table report, and vice
    versa.
    """
    result = ValidationResult(
        table_name=f"{base_demand.table_name} + OSS (effective)")
    effective = apply_oss_modifier(base_demand, oss_table)
    if indicated is not None:
        check_max_torque(effective, indicated, cfg, result)
    # Also flag cells where the OSS modifier pushes effective demand
    # significantly above or below the base -- these are drivability risks
    # even if they don't exceed the indicated table.
    for i in range(base_demand.n_rows):
        for j in range(base_demand.n_cols):
            base_val = float(base_demand.values[i, j])
            eff_val = float(effective.values[i, j])
            if abs(base_val) < 1.0:
                continue
            delta = eff_val - base_val
            if abs(delta) > cfg.max_torque_margin * 5:
                result.add(ValidationIssue(
                    check="oss_effective_delta",
                    severity=Severity.WARNING,
                    message=(
                        f"OSS modifier changes demand by {delta:+.1f} N m "
                        f"at pedal {base_demand.row_axis[i]:.0f}, "
                        f"{base_demand.col_axis[j]:.0f} RPM "
                        f"({base_val:.1f} -> {eff_val:.1f})"
                    ),
                    row_idx=i, col_idx=j,
                    value=eff_val, limit=base_val,
                ))
    return result


def validate_vs_logged_max(
    demand: DriverDemandTable,
    wot_curve: WotMaxCurve,
    cfg: ValidationConfig,
) -> ValidationResult:
    """Check: driver demand vs logged engine max.

    Compares every driver demand cell against the engine's actual max
    torque (from a WOT log pull), interpolated at the cell's RPM. This
    catches the case where driver demand requests more torque than the
    engine can produce -- which would contaminate the Torque Tuning
    Tool's log-based corrections (requested would always exceed actual).

    This check is separate from the indicated-table max check so that
    logged-max issues don't contaminate the indicated-table report.
    """
    result = ValidationResult(
        table_name=f"{demand.table_name} vs logged max")
    for j, rpm in enumerate(demand.col_axis):
        logged_max = float(wot_curve.torque_at_rpm(rpm))
        for i, pedal in enumerate(demand.row_axis):
            val = float(demand.values[i, j])
            if val > logged_max + cfg.max_torque_margin:
                result.add(ValidationIssue(
                    check="logged_max",
                    severity=Severity.CRITICAL,
                    message=(
                        f"Demand {val:.1f} exceeds logged engine max "
                        f"{logged_max:.1f} at {rpm:.0f} RPM, pedal {pedal:.0f}"
                    ),
                    row_idx=i, col_idx=j,
                    value=val, limit=logged_max,
                    rpm=float(rpm), pedal=float(pedal),
                ))
    return result


def compare_tables(
    tables: dict[str, DriverDemandTable],
    cfg: ValidationConfig,
) -> ValidationResult:
    """Check 8: cross-table comparison (Normal vs Sport vs Fault).

    Expected relationships:
      - Sport >= Normal at high pedal
      - Fault <= Normal everywhere
    """
    result = ValidationResult(table_name="cross-table")
    normal = tables.get("Normal")
    sport = tables.get("Sport")
    fault = tables.get("Fault") or tables.get("FMEM")

    if normal is None:
        result.add(ValidationIssue(
            check="cross_table",
            severity=Severity.INFO,
            message="No 'Normal' table found for cross-table comparison.",
        ))
        return result

    if sport is not None and _shapes_match(normal, sport):
        for i in range(normal.n_rows):
            for j in range(normal.n_cols):
                # Sport should be >= Normal at high pedal
                if normal.row_axis[i] > normal.row_axis[normal.n_rows // 2]:
                    if sport.values[i, j] < normal.values[i, j] - cfg.monotonicity_tolerance:
                        result.add(ValidationIssue(
                            check="cross_table",
                            severity=Severity.WARNING,
                            message=(
                                f"Sport table is LOWER than Normal at high pedal "
                                f"({normal.row_axis[i]:.0f}), "
                                f"{normal.col_axis[j]:.0f} RPM: "
                                f"{sport.values[i, j]:.1f} < {normal.values[i, j]:.1f}"
                            ),
                            row_idx=i, col_idx=j,
                        ))

    if fault is not None and _shapes_match(normal, fault):
        for i in range(normal.n_rows):
            for j in range(normal.n_cols):
                if fault.values[i, j] > normal.values[i, j] + cfg.max_torque_margin:
                    result.add(ValidationIssue(
                        check="cross_table",
                        severity=Severity.WARNING,
                        message=(
                            f"Fault/FMEM table is HIGHER than Normal at "
                            f"pedal {normal.row_axis[i]:.0f}, "
                            f"{normal.col_axis[j]:.0f} RPM: "
                            f"{fault.values[i, j]:.1f} > {normal.values[i, j]:.1f}"
                        ),
                        row_idx=i, col_idx=j,
                    ))
    return result


def _shapes_match(a: DriverDemandTable, b: DriverDemandTable) -> bool:
    return (a.n_rows == b.n_rows and a.n_cols == b.n_cols
            and np.allclose(a.row_axis, b.row_axis)
            and np.allclose(a.col_axis, b.col_axis))


# ---- reporting -------------------------------------------------------------

def format_report(result: ValidationResult) -> str:
    """Format a ValidationResult as human-readable text."""
    lines = []
    header = f"=== Validation Report: {result.table_name} ==="
    lines.append(header)
    status = "PASS" if result.passed else "FAIL"
    lines.append(f"Status: {status}  |  "
                 f"{result.n_critical} critical, "
                 f"{result.n_warning} warnings, "
                 f"{result.n_info} info")
    lines.append("")
    if not result.issues:
        lines.append("No issues found.")
        return "\n".join(lines)
    # Group by check
    by_check: dict[str, list[ValidationIssue]] = {}
    for issue in result.issues:
        by_check.setdefault(issue.check, []).append(issue)
    for check_name, issues in by_check.items():
        lines.append(f"--- {check_name} ({len(issues)} issues) ---")
        for issue in issues:
            sev = issue.severity.value
            lines.append(f"  [{sev}] {issue.message}")
        lines.append("")
    return "\n".join(lines)
