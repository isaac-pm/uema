"""Per-station, per-sensor data coverage audit.

Builds a coverage table (station x sensor -> start, end, expected/actual row
counts, %missing) and flags gaps, so downstream analysis knows which stations
have enough clean history before any method is run on them.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from uema.correlation import season_bucket
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
# A single gap longer than this is a genuine extended field outage (not a
# routine few-day connectivity blip) — long enough to break any window/lag
# feature crossing it regardless of how low the sensor's aggregate %missing
# looks. Tiering on %missing alone would miss a single week-long hole
# sitting inside an otherwise-clean history.
MAX_GAP_DAYS = 7.0
# Minimum calendar days of each Costa Rica season (see
# uema.correlation.season_bucket) required within a station's decision
# window for a GO station to be trusted for dry/wet-conditioned correlation
# (Step 1) without a caveat — a window that clears every completeness
# threshold can still sit almost entirely inside one season.
MIN_SEASON_DAYS = 30


def classify_sensor(stats: dict) -> str:
    """Tier a single sensor's coverage stats: ok / degraded / short-history / absent.

    `stats` must provide `actual_rows`, `pct_missing`, `span_days`, and
    `max_gap_days` — computed by `_window_coverage` over the window actually
    being scored (the station's joint sensor-overlap window in
    `station_recommendation`, not necessarily the sensor's own full recorded
    span).
    """
    if stats["actual_rows"] == 0 or stats["pct_missing"] >= PCT_MISSING_ABSENT:
        return "absent"
    if stats["span_days"] < MIN_SPAN_DAYS:
        return "short-history"
    if stats["pct_missing"] > PCT_MISSING_DEGRADED:
        return "degraded"
    if stats["max_gap_days"] > MAX_GAP_DAYS:
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


def _window_coverage(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Coverage stats for `series` restricted to [start, end].

    This is the window actually used for the go/no-go decision — the
    station's joint sensor-overlap span — not the sensor's own full recorded
    history, which can run much longer than what's usable for joint analysis
    (see MAX_GAP_DAYS / MIN_SEASON_DAYS docstrings above for why that
    distinction matters).
    """
    windowed = series.loc[start:end]
    expected = _expected_rows(start, end)
    actual = len(windowed)
    n_gaps, max_gap = _gap_stats(windowed.index)
    # _gap_stats only diffs consecutive *present* timestamps inside the
    # window, so an outage straddling the window boundary (last real reading
    # before `start`, or the run from the last reading up to `end`) isn't
    # caught by it — add those two edge gaps explicitly.
    if actual:
        max_gap = max(max_gap, windowed.index.min() - start, end - windowed.index.max())
    else:
        max_gap = end - start
    return {
        "span_days": (end - start).total_seconds() / 86400,
        "expected_rows": expected,
        "actual_rows": actual,
        "pct_missing": 100 * (expected - actual) / expected if expected else 100.0,
        "n_gaps": n_gaps,
        "max_gap_days": max_gap.total_seconds() / 86400,
    }


def _station_window_stats(
    raw_files: list[RawFile],
    station: str,
    sensors: tuple[str, ...],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, dict]:
    """`_window_coverage` for every sensor at `station` over [start, end].

    A sensor with no raw file at all for this station scores as fully
    absent rather than being silently dropped from the roll-up.
    """
    station_files = {f.feature: f for f in raw_files if f.station == station}
    out = {}
    for s in sensors:
        if s not in station_files:
            out[s] = {
                "span_days": (end - start).total_seconds() / 86400,
                "expected_rows": _expected_rows(start, end),
                "actual_rows": 0,
                "pct_missing": 100.0,
                "n_gaps": 0,
                "max_gap_days": (end - start).total_seconds() / 86400,
            }
            continue
        out[s] = _window_coverage(load_raw_series(station_files[s]), start, end)
    return out


def _season_day_counts(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    """Calendar days of each Costa Rica season within [start, end].

    Daily resolution is enough here — this is a coarse "is there enough of
    each season to trust a seasonal split" check (MIN_SEASON_DAYS), not a
    precise accounting. See uema.correlation.season_bucket for the
    dry/wet-month convention.
    """
    days = pd.date_range(start.normalize(), end.normalize(), freq="D")
    if len(days) == 0:
        return {"dry": 0.0, "wet": 0.0}
    counts = season_bucket(days).value_counts()
    return {"dry": float(counts.get("dry", 0)), "wet": float(counts.get("wet", 0))}


def station_recommendation(
    coverage: pd.DataFrame, raw_files: list[RawFile] | None = None
) -> pd.DataFrame:
    """Per-station go/no-go for this analysis phase.

    Rolls up per-sensor tiers into one row per station: GO (all sensors
    clean), CONDITIONAL-GO (usable but specific sensors/seasons need a
    bounded caveat — see rationale), or NO-GO (sensors don't overlap at all,
    or pressure — the sensor every station is expected to have — is absent
    or too short a history to use). Purely a function of the computed
    tiers, no per-station special-casing, except for the explicit
    MANUAL_OVERRIDES table above — this is an explicit decision, not a
    silent filter applied later in a pipeline.

    Tiers are computed on each station's *joint* sensor-overlap window
    (`joint_span_start`/`joint_span_end` — the intersection of all three
    sensors' recorded spans), not on each sensor's own individual span.
    Scoring a sensor over its own full history would let a sensor with a
    long, mostly-clean history but a short overlap with its sibling sensors
    pass a tier it doesn't deserve for *joint* analysis — the extra history
    outside the overlap can't be used for anything that needs all three
    sensors together, so it shouldn't count toward "clean." Per-sensor tiers
    also fail (`degraded`) if the single longest gap within that window
    exceeds MAX_GAP_DAYS, even when aggregate %missing is low — an
    unbridgeable multi-day outage can hide behind an otherwise-good average.

    A GO station is further checked for season coverage
    (`_season_day_counts`): if the joint window has fewer than
    MIN_SEASON_DAYS of either Costa Rica season, it's downgraded to
    CONDITIONAL-GO with a rationale calling that out, since Step 1
    (`uema.correlation`) conditions some correlations on dry/wet season and
    a window sitting almost entirely in one season would silently bias that
    split.

    `decision` is the only go/no-go signal this function produces — deliberately.
    An earlier version also ran a second "windowed" layer that searched each
    station for its own optimal rolling clean sub-window and used that to
    rescue stations that failed the whole-span decision. That's closer to
    breakpoint/homogeneity analysis than standard coverage gating, and it let
    a station that fails on its own numbers get quietly re-admitted the
    moment a new data pull happened to contain a rescuing stretch. Standard
    meteorological practice is simpler: accept the station's actual joint
    window as-is, interpolate short gaps downstream (`uema.silver.fill_gaps`),
    leave longer gaps as missing (handled by pairwise exclusion in
    `uema.correlation`), and gate on a straightforward completeness threshold
    over that one window — which is what `decision` already is.
    """
    if raw_files is None:
        raw_files = discover_raw_files()

    rows = []
    for station, group in coverage.groupby("station"):
        # Raw intersection of all three sensors' recorded spans. No quality
        # filter here (contrast with the tiering below, which is quality-
        # filtered but restricted to this same window) — and only valid if
        # every sensor actually has a file for this station.
        has_all_sensors = set(group["feature"]) == set(FEATURES)
        joint_span_start = group["start"].max()
        joint_span_end = group["end"].min()
        has_joint_span = has_all_sensors and joint_span_start < joint_span_end
        if has_joint_span:
            joint_span_days = (joint_span_end - joint_span_start).total_seconds() / 86400
        else:
            joint_span_start = joint_span_end = joint_span_days = None

        if has_joint_span:
            window_stats = _station_window_stats(
                raw_files, station, FEATURES, joint_span_start, joint_span_end
            )
            tiers = {f: classify_sensor(stats) for f, stats in window_stats.items()}
        else:
            tiers = {f: "absent" for f in FEATURES}

        pressure_tier = tiers.get("pressure")

        if not has_joint_span:
            decision = "NO-GO"
            rationale = "sensors have no overlapping recorded period"
        elif pressure_tier == "absent":
            decision = "NO-GO"
            rationale = "pressure sensor effectively absent within the joint window"
        elif pressure_tier == "short-history":
            decision = "NO-GO"
            rationale = f"joint sensor overlap under {MIN_SPAN_DAYS} days — insufficient history to use"
        elif any(t != "ok" for t in tiers.values()):
            decision = "CONDITIONAL-GO"
            flagged = [f for f, t in tiers.items() if t != "ok"]
            rationale = f"usable with caveats on: {', '.join(flagged)}"
        else:
            decision = "GO"
            rationale = "all sensors within coverage thresholds over the joint window"

        season_days = _season_day_counts(joint_span_start, joint_span_end) if has_joint_span else {
            "dry": None,
            "wet": None,
        }
        if decision == "GO" and min(season_days["dry"], season_days["wet"]) < MIN_SEASON_DAYS:
            thin = [s for s, d in season_days.items() if d < MIN_SEASON_DAYS]
            decision = "CONDITIONAL-GO"
            rationale = (
                f"all sensors clean, but thin coverage for season(s): {', '.join(thin)} "
                f"(<{MIN_SEASON_DAYS:.0f}d) — caution with seasonal conditioning"
            )

        if station in MANUAL_OVERRIDES:
            decision, override_reason = MANUAL_OVERRIDES[station]
            rationale = f"manual override: {override_reason}"

        rows.append(
            {
                "station": station,
                **tiers,
                "decision": decision,
                "rationale": rationale,
                "dry_days": season_days["dry"],
                "wet_days": season_days["wet"],
                "joint_span_start": joint_span_start,
                "joint_span_end": joint_span_end,
                "joint_span_days": joint_span_days,
            }
        )

    return pd.DataFrame(rows).set_index("station").sort_index()


def station_availability(coverage: pd.DataFrame) -> pd.Series:
    """Per-station overall data availability (actual/expected rows, summed
    across sensors) — used to sort/rank stations by how much data they have,
    independent of the go/no-go decision."""
    totals = coverage.groupby("station")[["actual_rows", "expected_rows"]].sum()
    return (totals["actual_rows"] / totals["expected_rows"]).rename("availability")
