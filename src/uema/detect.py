"""Shared output schema and Step 3 statistical detectors (Z-score family).

Every anomaly-detection method in this project (Steps 3-6) emits results in
the same schema (`SCHEMA_COLUMNS`) so Step 7 can align and compare all
methods' flagged sets at matching station/sensor/timestamp combinations
without per-method special-casing. This module owns that schema plus the
first two methods: standard Z-score and Modified (MAD-based) Z-score, both
computed as centered rolling statistics rather than over each station's
whole history — pressure and especially luminous_intensity are strongly
non-stationary (diurnal cycle dominates their mean/variance), so a plain
whole-history Z-score would mostly just rediscover "is it currently
daytime" rather than surface local point deviations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SCHEMA_COLUMNS = ["station", "sensor", "timestamp", "method", "flagged", "score"]

# 6h centered window at 10-minute resolution — matches the local rolling-mean
# point-anomaly criterion Step 2 wrote for pressure, reused unchanged for
# all three sensors so the same detector logic applies uniformly.
WINDOW_BINS = 36
MIN_PERIODS = WINDOW_BINS // 2

# A local window's std/MAD can collapse to (near) zero — a flat, rain-free
# precipitation window, or a still, clear-sky lux window — which would
# otherwise blow up the z-score for any small subsequent wiggle. Flooring
# the local scale at a fraction of the station/sensor's whole-history scale
# keeps the score meaningful (a deviation still has to be large relative to
# how this sensor normally varies) without reintroducing the whole-history
# non-stationarity problem the rolling window exists to avoid.
GLOBAL_FLOOR_FRAC = 0.1

Z_THRESHOLD = 3.0
MODIFIED_Z_THRESHOLD = 3.5  # Iglewicz & Hoaglin (1993) convention for modified z


def rolling_zscore(
    series: pd.Series,
    window: int = WINDOW_BINS,
    min_periods: int = MIN_PERIODS,
    floor_frac: float = GLOBAL_FLOOR_FRAC,
) -> pd.Series:
    """Centered rolling standard Z-score, floored at `floor_frac` of the
    series' whole-history standard deviation.

    Assumes `series` is already on a regular time grid (e.g.
    `uema.silver.consolidate_stations` + `resample_10min`) — a rolling
    window over irregular timestamps would not represent a fixed span of
    time.
    """
    roll = series.rolling(window, center=True, min_periods=min_periods)
    mean = roll.mean()
    std = roll.std()

    global_std = series.std(skipna=True)
    floor = floor_frac * global_std if global_std and not np.isnan(global_std) else 0.0
    scale = std.clip(lower=floor)

    z = (series - mean) / scale
    return z.where(scale > 0)


def rolling_modified_zscore(
    series: pd.Series,
    window: int = WINDOW_BINS,
    min_periods: int = MIN_PERIODS,
    floor_frac: float = GLOBAL_FLOOR_FRAC,
) -> pd.Series:
    """Centered rolling Modified (MAD-based) Z-score, same floor logic as
    `rolling_zscore` but against the whole-history MAD.

    Two-pass rolling-median approach: a centered rolling median gives the
    local center, then a second centered rolling median of the absolute
    residuals from that center gives the local MAD — both vectorized, no
    per-window Python callback.
    """
    roll = series.rolling(window, center=True, min_periods=min_periods)
    med = roll.median()
    resid = (series - med).abs()
    mad = resid.rolling(window, center=True, min_periods=min_periods).median()

    clean = series.dropna()
    global_med = clean.median()
    global_mad = (clean - global_med).abs().median()
    # For a zero-inflated sensor (precipitation: most 10-minute bins are
    # exactly 0.0mm) the *global* MAD is itself exactly 0 — more than half
    # of all readings sit at the median — so flooring against it would
    # floor at 0 and leave almost every score NaN (mad.clip(lower=0) does
    # nothing). Fall back to the whole-history standard deviation as the
    # floor anchor in that case; it stays informative even when MAD can't.
    floor_anchor = global_mad if global_mad else clean.std(skipna=True)
    floor = floor_frac * floor_anchor if floor_anchor and not np.isnan(floor_anchor) else 0.0
    scale = mad.clip(lower=floor)

    z = 0.6745 * (series - med) / scale
    return z.where(scale > 0)


def hour_of_day_zscore(series: pd.Series) -> pd.Series:
    """Z-score against the station's own historical distribution for that
    hour-of-day, not a rolling local window.

    This is the context-conditioned variant Step 2's contextual-anomaly
    criterion for `luminous_intensity` calls for specifically: a reading
    that is unremarkable in the full 24h-pooled distribution can still be
    wrong for its specific hour (e.g. moderate lux at 3am). Only meaningful
    for a sensor whose contextual criterion is itself a plain per-hour
    comparison — Step 2's pressure/precipitation contextual criteria are
    cross-variable (lagged pressure-lux coupling, daytime-conditioned
    precipitation-lux coupling) and are not captured by this univariate
    method.
    """
    hours = series.index.hour
    grouped = series.groupby(hours)
    mean_by_hour = grouped.transform("mean")
    std_by_hour = grouped.transform("std")

    z = (series - mean_by_hour) / std_by_hour
    return z.where(std_by_hour > 0)


def build_detection_table(
    score: pd.Series,
    station: str,
    sensor: str,
    method: str,
    threshold: float,
) -> pd.DataFrame:
    """Assemble one method's scores into the shared output schema.

    Rows are emitted only where `score` is non-NaN — a timestamp the method
    couldn't score (e.g. inside a long unfilled gap, or before enough
    history accumulates for the rolling window) is omitted rather than
    recorded as an unflagged 0, so downstream agreement analysis (Step 7)
    can distinguish "this method looked and found nothing" from "this
    method had nothing to look at here."
    """
    score = score.dropna()
    return pd.DataFrame(
        {
            "station": station,
            "sensor": sensor,
            "timestamp": score.index,
            "method": method,
            "flagged": score.abs() > threshold,
            "score": score.to_numpy(),
        }
    )[SCHEMA_COLUMNS]
