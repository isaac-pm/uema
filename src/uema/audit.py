"""Per-station, per-sensor data coverage audit.

Builds a coverage table (station x sensor -> start, end, expected/actual row
counts, %missing) and flags gaps, so downstream analysis knows which stations
have enough clean history before any method is run on them.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from uema.io import SAMPLING_INTERVAL, RawFile, discover_raw_files, load_raw_series


@dataclass(frozen=True)
class SensorCoverage:
    station: str
    feature: str
    start: pd.Timestamp
    end: pd.Timestamp
    span_days: float
    expected_rows: int
    actual_rows: int
    missing_rows: int
    pct_missing: float
    n_gaps: int
    max_gap: pd.Timedelta


def _expected_rows(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return int((end - start) / SAMPLING_INTERVAL) + 1


def _gap_stats(index: pd.DatetimeIndex) -> tuple[int, pd.Timedelta]:
    """Count gaps (missed bins) and the largest single gap."""
    diffs = index.to_series().diff().dropna()
    gaps = diffs[diffs > SAMPLING_INTERVAL]
    if gaps.empty:
        return 0, pd.Timedelta(0)
    return len(gaps), gaps.max()


def audit_sensor(raw_file: RawFile) -> SensorCoverage:
    series = load_raw_series(raw_file)
    start, end = series.index.min(), series.index.max()
    expected = _expected_rows(start, end)
    actual = len(series)
    n_gaps, max_gap = _gap_stats(series.index)
    return SensorCoverage(
        station=raw_file.station,
        feature=raw_file.feature,
        start=start,
        end=end,
        span_days=(end - start).total_seconds() / 86400,
        expected_rows=expected,
        actual_rows=actual,
        missing_rows=expected - actual,
        pct_missing=100 * (expected - actual) / expected,
        n_gaps=n_gaps,
        max_gap=max_gap,
    )


def build_coverage_table() -> pd.DataFrame:
    """Coverage table for every discovered raw station/sensor CSV."""
    rows = [audit_sensor(f) for f in discover_raw_files()]
    df = pd.DataFrame(rows)
    return df.sort_values(["station", "feature"]).reset_index(drop=True)


def coverage_segments(raw_file: RawFile) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Contiguous (start, end) segments of actual data, split wherever a gap occurs."""
    series = load_raw_series(raw_file)
    idx = series.index
    if len(idx) == 0:
        return []
    idx_series = idx.to_series()
    segment_id = (idx_series.diff() > SAMPLING_INTERVAL).cumsum()
    return [(g.iloc[0], g.iloc[-1]) for _, g in idx_series.groupby(segment_id)]


# Go/no-go thresholds for this analysis phase. Deliberately about raw
# coverage only (span + row completeness) — not modeling-readiness
# (window counts, label alignment, etc.), which is out of scope here.
MIN_SPAN_DAYS = 90
PCT_MISSING_DEGRADED = 25.0
PCT_MISSING_ABSENT = 90.0


def classify_sensor(row: pd.Series) -> str:
    """Tier a single station/sensor coverage row: ok / degraded / short-history / absent."""
    if row["actual_rows"] == 0 or row["pct_missing"] >= PCT_MISSING_ABSENT:
        return "absent"
    if row["span_days"] < MIN_SPAN_DAYS:
        return "short-history"
    if row["pct_missing"] > PCT_MISSING_DEGRADED:
        return "degraded"
    return "ok"


# Manual overrides for this analysis phase — analyst judgment calls that
# override the numeric-threshold decision above (e.g. a station that clears
# the thresholds on paper but is known/observed to be unreliable, or vice
# versa). Edit this dict as the assessment changes; each entry is
# station -> (decision, reason). Keep reasons specific enough to audit later.
MANUAL_OVERRIDES: dict[str, tuple[str, str]] = {
    "sede-caribe_limon": ("NO-GO", "flagged by manual review despite clearing numeric thresholds"),
}


def station_recommendation(coverage: pd.DataFrame) -> pd.DataFrame:
    """Per-station go/no-go for this analysis phase.

    Rolls up per-sensor tiers (classify_sensor) into one row per station:
    GO (all sensors clean), CONDITIONAL-GO (usable but specific sensors need
    a bounded exclusion — see rationale), or NO-GO (pressure — the sensor
    every station is expected to have — is absent or too short a history to
    use). Purely a function of the computed tiers, no per-station
    special-casing, except for the explicit MANUAL_OVERRIDES table above —
    this is an explicit decision, not a silent filter applied later in a
    pipeline.
    """
    coverage = coverage.copy()
    coverage["tier"] = coverage.apply(classify_sensor, axis=1)

    rows = []
    for station, group in coverage.groupby("station"):
        tiers = dict(zip(group["feature"], group["tier"]))
        pressure_tier = tiers.get("pressure")

        if pressure_tier == "absent":
            decision = "NO-GO"
            rationale = "pressure sensor effectively absent"
        elif pressure_tier == "short-history":
            decision = "NO-GO"
            rationale = f"pressure sensor span under {MIN_SPAN_DAYS} days — insufficient history to use"
        elif any(t != "ok" for t in tiers.values()):
            decision = "CONDITIONAL-GO"
            flagged = [f for f, t in tiers.items() if t != "ok"]
            rationale = f"usable with caveats on: {', '.join(flagged)}"
        else:
            decision = "GO"
            rationale = "all sensors within coverage thresholds"

        if station in MANUAL_OVERRIDES:
            decision, override_reason = MANUAL_OVERRIDES[station]
            rationale = f"manual override: {override_reason}"

        rows.append(
            {"station": station, **tiers, "decision": decision, "rationale": rationale}
        )

    return pd.DataFrame(rows).set_index("station").sort_index()


def station_availability(coverage: pd.DataFrame) -> pd.Series:
    """Per-station overall data availability (actual/expected rows, summed
    across sensors) — used to sort/rank stations by how much data they have,
    independent of the go/no-go decision."""
    totals = coverage.groupby("station")[["actual_rows", "expected_rows"]].sum()
    return (totals["actual_rows"] / totals["expected_rows"]).rename("availability")
