"""Per-station, per-sensor data coverage audit.

Builds a coverage table (station x sensor -> start, end, expected/actual row
counts, %missing) and flags gaps, so downstream analysis knows which stations
have enough clean history before any method is run on them.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from uema.io import FEATURES, SAMPLING_INTERVAL, RawFile, discover_raw_files, load_raw_series


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


def _rolling_pct_missing(
    series: pd.Series, full_index: pd.DatetimeIndex, window_days: int
) -> pd.Series:
    """Trailing rolling %missing of `series` against a full sampling-interval grid."""
    present = pd.Series(0.0, index=full_index)
    present.loc[present.index.isin(series.index)] = 1.0
    rolling_mean = present.rolling(f"{window_days}D", min_periods=1).mean()
    return 100 * (1 - rolling_mean)


def _contiguous_true_stretches(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(start, end) of each contiguous run of True values in a boolean Series."""
    if not mask.any():
        return []
    group_id = (mask != mask.shift()).cumsum()
    return [
        (g.index[0], g.index[-1])
        for _, g in mask.groupby(group_id)
        if g.iloc[0]
    ]


WINDOW_DAYS = 30


def find_best_common_window(
    raw_files: list[RawFile],
    station: str,
    sensors: tuple[str, ...] = FEATURES,
    window_days: int = WINDOW_DAYS,
    pct_missing_threshold: float = PCT_MISSING_DEGRADED,
    min_span_days: float = MIN_SPAN_DAYS,
) -> tuple[pd.Timestamp, pd.Timestamp, float] | None:
    """Longest stretch where all `sensors` are simultaneously clean at `station`.

    Complements the whole-span decision in `station_recommendation`: a station
    can fail whole-span thresholds while still containing a long, clean
    sub-period worth using on its own. For each sensor, computes a trailing
    `window_days`-day rolling %missing over a full sampling-interval grid,
    marks bins where that stays under `pct_missing_threshold`, and intersects
    across `sensors`. Returns (window_start, window_end, window_days) for the
    longest resulting contiguous stretch, or None if nothing clears
    `min_span_days`.
    """
    station_files = {f.feature: f for f in raw_files if f.station == station}
    if any(s not in station_files for s in sensors):
        return None

    series_by_sensor = {s: load_raw_series(station_files[s]) for s in sensors}
    # Intersection of sensor spans, not union: a bin before a sensor's own
    # recorded start isn't a real gap, it's the sensor not existing yet, and
    # counting it as missing would poison the trailing rolling window for the
    # first window_days after that sensor's real data begins.
    full_start = max(s.index.min() for s in series_by_sensor.values())
    full_end = min(s.index.max() for s in series_by_sensor.values())
    if full_start >= full_end:
        return None
    full_index = pd.date_range(full_start, full_end, freq=SAMPLING_INTERVAL)

    combined_good = pd.Series(True, index=full_index)
    for s in sensors:
        pct_missing = _rolling_pct_missing(series_by_sensor[s], full_index, window_days)
        combined_good &= pct_missing < pct_missing_threshold

    stretches = _contiguous_true_stretches(combined_good)
    if not stretches:
        return None

    window_start, window_end = max(stretches, key=lambda se: se[1] - se[0])
    window_span_days = (window_end - window_start).total_seconds() / 86400
    if window_span_days < min_span_days:
        return None
    return window_start, window_end, window_span_days


def station_recommendation(
    coverage: pd.DataFrame, raw_files: list[RawFile] | None = None
) -> pd.DataFrame:
    """Per-station go/no-go for this analysis phase.

    Rolls up per-sensor tiers (classify_sensor) into one row per station:
    GO (all sensors clean), CONDITIONAL-GO (usable but specific sensors need
    a bounded exclusion — see rationale), or NO-GO (pressure — the sensor
    every station is expected to have — is absent or too short a history to
    use). Purely a function of the computed tiers, no per-station
    special-casing, except for the explicit MANUAL_OVERRIDES table above —
    this is an explicit decision, not a silent filter applied later in a
    pipeline.

    Also runs `find_best_common_window` per station and derives
    `windowed_decision`: GO stations keep "GO" (whole span already applies);
    everything else becomes "WINDOWED-GO" if a valid common window exists,
    else "NO-GO". This is independent of MANUAL_OVERRIDES on `decision` — an
    overridden station's window (if any) still shows up here, so both facts
    stay visible rather than one silently masking the other.
    """
    if raw_files is None:
        raw_files = discover_raw_files()

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

        window = find_best_common_window(raw_files, station)
        if window is not None:
            best_window_start, best_window_end, best_window_days = window
        else:
            best_window_start = best_window_end = best_window_days = None

        if decision == "GO":
            windowed_decision = "GO"
        elif best_window_days is not None:
            windowed_decision = "WINDOWED-GO"
        else:
            windowed_decision = "NO-GO"

        rows.append(
            {
                "station": station,
                **tiers,
                "decision": decision,
                "rationale": rationale,
                "best_window_start": best_window_start,
                "best_window_end": best_window_end,
                "best_window_days": best_window_days,
                "windowed_decision": windowed_decision,
            }
        )

    return pd.DataFrame(rows).set_index("station").sort_index()


def go_station_date_ranges(coverage: pd.DataFrame, recommendation: pd.DataFrame) -> pd.DataFrame:
    """Per-GO-station date range where all three sensors are simultaneously present.

    max() of each sensor's start and min() of each sensor's end — not any
    single sensor's own range — since any analysis using this station needs
    all three sensors available at once.
    """
    go_stations = recommendation.index[recommendation["decision"] == "GO"]
    rows = []
    for station in go_stations:
        group = coverage[coverage["station"] == station]
        start_date = group["start"].max()
        end_date = group["end"].min()
        span_days = (end_date - start_date).total_seconds() / 86400
        rows.append(
            {"station": station, "start_date": start_date, "end_date": end_date, "span_days": span_days}
        )
    return pd.DataFrame(rows).sort_values("station").reset_index(drop=True)


def station_availability(coverage: pd.DataFrame) -> pd.Series:
    """Per-station overall data availability (actual/expected rows, summed
    across sensors) — used to sort/rank stations by how much data they have,
    independent of the go/no-go decision."""
    totals = coverage.groupby("station")[["actual_rows", "expected_rows"]].sum()
    return (totals["actual_rows"] / totals["expected_rows"]).rename("availability")
