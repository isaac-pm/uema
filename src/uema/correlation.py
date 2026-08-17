"""Pairwise, lagged, and time-conditioned correlation between sensor variables.

Operates on one station's consolidated DataFrame (see `uema.silver`) with
columns `pressure`, `precipitation`, `luminous_intensity`. Precipitation is
zero-inflated (most 10-minute bins are 0.0, per `data/stations/raw/README.md`),
which distorts Pearson correlation for any pair involving it — Spearman
(rank-based, insensitive to the zero spike's shape) is the primary statistic
for those pairs. Pearson is still computed and reported alongside for
comparison, just not treated as the primary read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRECIPITATION = "precipitation"


def primary_method(var1: str, var2: str) -> str:
    """Which correlation method to treat as primary for this variable pair.

    Spearman for any pair involving precipitation (zero-inflation distorts
    Pearson); Pearson otherwise.
    """
    return "spearman" if PRECIPITATION in (var1, var2) else "pearson"


def pairwise_correlation(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pearson and Spearman correlation matrices over all columns of df.

    pandas' `.corr()` excludes NaNs pairwise per pair of columns, so this
    works directly on a gap-containing (but not necessarily gap-filled)
    DataFrame.
    """
    return {
        "pearson": df.corr(method="pearson"),
        "spearman": df.corr(method="spearman"),
    }


def ccf(df: pd.DataFrame, var1: str, var2: str, max_lag: int) -> pd.Series:
    """Cross-correlation of var1 and var2 across lags -max_lag..+max_lag.

    Computed manually (lagged Pearson correlation via pandas, not
    statsmodels.tsa.stattools.ccf) so the result is directly comparable
    across statsmodels versions, whose `ccf` signature/normalization has
    changed between releases. At `lag`, this is
    `corr(var1[t], var2[t + lag])` — a positive lag means var2's later
    values line up with var1's current ones (var2 lags var1 in time); rows
    are pairwise NaN-dropped per lag before correlating, since `fill_gaps`
    intentionally leaves long gaps as NaN. Index is the integer lag in
    10-minute sampling-interval units.
    """
    x = df[var1]
    y = df[var2]
    lags = range(-max_lag, max_lag + 1)
    values = []
    for lag in lags:
        shifted_y = y.shift(-lag)
        pair = pd.concat([x, shifted_y], axis=1).dropna()
        if len(pair) < 2:
            values.append(np.nan)
        else:
            values.append(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
    return pd.Series(values, index=pd.Index(lags, name="lag"), name=f"ccf({var1}, {var2})")


def _hour_bucket(index: pd.DatetimeIndex) -> pd.Series:
    """Day (06:00-17:59 local) vs night (18:00-05:59 local) bucket per timestamp."""
    return pd.Series(np.where((index.hour >= 6) & (index.hour < 18), "day", "night"), index=index)


def season_bucket(index: pd.DatetimeIndex) -> pd.Series:
    """Costa Rica's two-season convention: dry (Dec-Apr) vs wet (May-Nov).

    Not the four-season meteorological convention — Costa Rica's climate is
    tropical and doesn't have one; this is the locally meaningful split.
    Public (not `_`-prefixed): also used by `uema.audit` to check whether a
    station's decision window has enough of each season to trust a
    dry/wet-conditioned correlation split.
    """
    is_dry = index.month.isin([12, 1, 2, 3, 4])
    return pd.Series(np.where(is_dry, "dry", "wet"), index=index)


_BUCKET_FUNCS = {"hour": _hour_bucket, "season": season_bucket}


def conditioned_correlation(df: pd.DataFrame, group_by: str = "hour") -> dict[str, dict[str, pd.DataFrame]]:
    """Pearson/Spearman correlation matrices computed separately per time bucket.

    `group_by="hour"` splits into day/night buckets; `group_by="season"`
    splits into Costa Rica's dry/wet buckets. Reveals sensor pairs whose
    relationship shifts conditionally (e.g. light-pressure coupling that
    only holds during the day) rather than assuming one global correlation
    applies everywhere.
    """
    if group_by not in _BUCKET_FUNCS:
        raise ValueError(f"group_by must be one of {list(_BUCKET_FUNCS)}, got {group_by!r}")

    bucket = _BUCKET_FUNCS[group_by](df.index)
    result: dict[str, dict[str, pd.DataFrame]] = {}
    for label, group_mask in bucket.groupby(bucket).groups.items():
        subset = df.loc[group_mask]
        result[label] = pairwise_correlation(subset)
    return result
