"""Consolidation of in-scope stations' raw sensor CSVs into analysis frames.

Step 0 (`uema.audit`) decided which stations have enough raw coverage to be
worth analyzing at all, and over what date range. This module does the next,
separate job: turning an in-scope station's three still-separate sensor CSVs
into one clean, jointly-usable DataFrame — same time grid, trimmed to the
range Step 0 recommends, small gaps bridged so short-lived pandas operations
that trip on any NaN (raw plotting, naive corr calls) don't have to
special-case them.
"""

from __future__ import annotations

import pandas as pd

from uema.io import FEATURES, discover_raw_files, load_raw_series


def analysis_stations(recommendation: pd.DataFrame) -> pd.DataFrame:
    """Stations in scope for downstream per-sensor analysis, with the date
    range each should be trimmed to.

    Derived entirely from Step 0's `decision` — no station names hardcoded
    here, so a station that newly clears GO or CONDITIONAL-GO on a future
    audit run is picked up automatically, no code change needed. Every
    in-scope station is used over its full `joint_span` (the intersection of
    all three sensors' recorded spans) — no per-station rescue-window search;
    short gaps within that span are linearly bridged downstream
    (`fill_gaps`) and longer gaps are left as `NaN` for pairwise exclusion,
    the standard way to handle blocky missingness rather than hunting for an
    optimal clean sub-window. NO-GO stations are excluded. Returns a
    DataFrame indexed by station with `window_start`, `window_end`,
    `decision`.
    """
    in_scope = recommendation[recommendation["decision"].isin(["GO", "CONDITIONAL-GO"])].copy()
    in_scope["window_start"] = in_scope["joint_span_start"]
    in_scope["window_end"] = in_scope["joint_span_end"]
    return in_scope[["window_start", "window_end", "decision"]]


def consolidate_stations(recommendation: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Load and join each in-scope station's three sensors into one DataFrame.

    Per station (see `analysis_stations`): loads pressure/precipitation/
    luminous_intensity as separate series, concatenates them column-wise on
    their own combined time index, and trims to that station's recommended
    window. Column names are the bare feature names (`pressure`,
    `precipitation`, `luminous_intensity`).
    """
    raw_files = discover_raw_files()
    scope = analysis_stations(recommendation)
    result: dict[str, pd.DataFrame] = {}

    for station, row in scope.iterrows():
        station_files = {f.feature: f for f in raw_files if f.station == station}
        missing = [f for f in FEATURES if f not in station_files]
        if missing:
            raise ValueError(f"{station}: missing raw files for {missing}")

        series_by_feature = {
            feature: load_raw_series(station_files[feature]).rename(feature) for feature in FEATURES
        }

        df = pd.concat(series_by_feature.values(), axis=1)
        df = df.loc[row["window_start"] : row["window_end"]]
        result[station] = df

    return result


def resample_10min(df: pd.DataFrame) -> pd.DataFrame:
    """Force df onto a strict, regularly-spaced 10-minute time index.

    Raw bins are already nominally 10-minute, but real timestamps drift
    slightly and gaps leave holes in the index — resampling onto a fixed
    grid (mean within each bin) is what makes downstream lag/rolling
    operations (ccf, rolling %missing) well-defined.
    """
    return df.resample("10min").mean()


def fill_gaps(df: pd.DataFrame, max_gap: str = "6h") -> pd.DataFrame:
    """Linearly interpolate NaN runs no longer than `max_gap`; leave longer runs as NaN.

    `max_gap` defaults to 6 hours as a judgment call, not a universal
    constant: pressure and light both vary smoothly enough over a few hours
    that a short linear bridge is a reasonable stand-in for the true signal,
    but a multi-day outage shouldn't be invented data — those longer gaps
    are left as NaN and relied on to be pairwise-excluded downstream (e.g.
    by `DataFrame.corr()`, or by `dropna()` immediately before an operation
    that requires a complete index, such as `asfreq`).
    """
    max_gap_td = pd.Timedelta(max_gap)
    max_gap_bins = int(max_gap_td / pd.Timedelta("10min"))

    filled = df.copy()
    for column in filled.columns:
        filled[column] = filled[column].interpolate(
            method="linear", limit=max_gap_bins, limit_area="inside"
        )
    return filled
