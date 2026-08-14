"""Per-station, per-sensor data coverage audit.

Builds a coverage table (station x sensor -> start, end, expected/actual row
counts, %missing) and flags gaps, so downstream analysis knows which stations
have enough clean history before any method is run on them. See
data/stations/raw/README.md for the caveats this module surfaces
(finca-2 luminous outage, recinto-guapiles known issues, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from uema.io import SAMPLING_INTERVAL, RawFile, discover_raw_files, load_raw_series

# Known, documented caveats (data/stations/raw/README.md) applied as
# annotations during the audit — not silently patched into the data.
FINCA2_LUX_UNRELIABLE_BEFORE = pd.Timestamp("2025-05-20 18:20:00")
KNOWN_BAD_STATIONS = {"recinto-guapiles"}


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
    notes: str


def _expected_rows(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return int((end - start) / SAMPLING_INTERVAL) + 1


def _gap_stats(index: pd.DatetimeIndex) -> tuple[int, pd.Timedelta]:
    """Count gaps (missed bins) and the largest single gap."""
    diffs = index.to_series().diff().dropna()
    gaps = diffs[diffs > SAMPLING_INTERVAL]
    if gaps.empty:
        return 0, pd.Timedelta(0)
    return len(gaps), gaps.max()


def _notes_for(raw_file: RawFile) -> str:
    notes = []
    if raw_file.station in KNOWN_BAD_STATIONS:
        notes.append("known data-quality issues across all sensors (see raw README)")
    if raw_file.feature == "luminous_intensity" and raw_file.station == "sede-central_finca-2":
        notes.append(f"unreliable before {FINCA2_LUX_UNRELIABLE_BEFORE}")
    return "; ".join(notes)


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
        notes=_notes_for(raw_file),
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


def station_recommendation(coverage: pd.DataFrame) -> pd.DataFrame:
    """Per-station go/no-go for this analysis phase.

    Rolls up per-sensor tiers (classify_sensor) into one row per station:
    GO (all sensors clean), CONDITIONAL-GO (usable but specific sensors need
    a bounded exclusion — see rationale), or NO-GO (documented station-wide
    issue, or the pressure sensor is effectively absent). This is an
    explicit decision, not a silent filter applied later in the pipeline.
    """
    coverage = coverage.copy()
    coverage["tier"] = coverage.apply(classify_sensor, axis=1)

    rows = []
    for station, group in coverage.groupby("station"):
        tiers = dict(zip(group["feature"], group["tier"]))
        notes = "; ".join(n for n in group["notes"] if n)
        known_bad = station in KNOWN_BAD_STATIONS

        if known_bad or tiers.get("pressure") == "absent":
            decision = "NO-GO"
            rationale = (
                "documented station-wide data-quality issue"
                if known_bad
                else "pressure sensor effectively absent"
            )
        elif any(t != "ok" for t in tiers.values()):
            decision = "CONDITIONAL-GO"
            flagged = [f for f, t in tiers.items() if t != "ok"]
            rationale = f"usable with caveats on: {', '.join(flagged)}"
        else:
            decision = "GO"
            rationale = "all sensors within coverage thresholds"

        if notes:
            rationale = f"{rationale} ({notes})"

        rows.append({"station": station, **tiers, "decision": decision, "rationale": rationale})

    return pd.DataFrame(rows).set_index("station").sort_index()
