"""Step 4 density/isolation detectors: Local Outlier Factor and Isolation Forest.

The density/isolation corner of the five-method comparison. Both methods
here write to the shared schema established in `uema.detect`
(`SCHEMA_COLUMNS` / `build_detection_table`) so Step 7 can align them
against the Step 3 statistical detectors and the Steps 5-6 reconstruction
detectors at matching station/sensor/timestamp combinations.

Two design decisions carry over from earlier steps rather than being made
here:

*Per-sensor, not joint.* Step 1 found global lag-0 correlation between the
three sensors to be weak at every in-scope station (|r| < 0.15), so the
feature strategy it recommended — and that the plan then routes into this
step — is independent per-sensor treatment. LOF and Isolation Forest are
therefore fit separately per station x sensor, and `sensor` in the output
schema keeps its plain meaning (no joint-feature identifier is needed).
The two conditional cross-variable couplings Step 1 did find are gated on
completeness or time-of-day and are left to methods that can condition on
them explicitly.

*Window features, not bare values.* Neither method is sequence-aware: fed
one scalar reading at a time, both would reduce to a univariate
distributional outlier test, largely duplicating Step 3 and wasting the
one thing they offer over it — the ability to judge a point in a
multi-dimensional feature space. Each reading is therefore represented by
a 4-dimensional vector combining the raw value with three summaries of the
1-hour window centered on it (mean, standard deviation, slope), so that a
reading can be flagged for sitting in an unusual *local dynamic context*
(e.g. an ordinary pressure value occurring mid-way through an unusually
steep local trend) and not only for having an unusual value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

# 1 hour at 10-minute resolution. Deliberately shorter than Step 3's 6-hour
# scoring window: these features exist to describe a reading's immediate
# dynamic neighbourhood, not to re-derive the same local baseline the
# Z-score family already computes over a longer span.
FEATURE_WINDOW_BINS = 6
FEATURE_COLUMNS = ["value", "window_mean", "window_std", "window_slope"]

# Breunig et al. (2000) treat MinPts as the one parameter that matters and
# recommend a lower bound around 10-20 to keep the local density estimate
# from being dominated by a single close neighbour; 20 is that upper end
# and scikit-learn's default, taken as-is rather than tuned, since there
# are no labels here to tune against.
LOF_N_NEIGHBORS = 20

# Liu et al. (2008) show path-length convergence well before 100 trees;
# 200 buys stability of the averaged score at negligible cost. max_samples
# is the 256 subsample the same paper recommends and scikit-learn defaults
# to — the method is designed around small subsamples, not full data.
IFOREST_N_ESTIMATORS = 200
IFOREST_MAX_SAMPLES = 256
RANDOM_STATE = 0

# Both thresholds are each method's own published convention, matching
# scikit-learn's `contamination="auto"` offsets, rather than a
# contamination rate assumed for this network. Assuming a contamination
# rate would mean fixing the flagged rate by hand in advance and then
# reporting it as a finding.
LOF_THRESHOLD = 1.5
IFOREST_THRESHOLD = 0.5


def window_features(series: pd.Series, window: int = FEATURE_WINDOW_BINS) -> pd.DataFrame:
    """Represent each reading as [value, window mean, window std, window slope].

    The window is centered on the reading and every feature requires it to
    be complete: a reading whose surrounding window is clipped by the start
    or end of the record, or punctured by a gap the <=6h fill left as NaN,
    yields NaN features and is dropped before fitting rather than being
    scored off a partial window. `slope` is an ordinary least-squares fit
    over the window, expressed in sensor units per hour so it stays
    readable independent of the bin size.

    Assumes `series` is on a strict 10-minute grid (`uema.silver`), since
    both the window length and the slope's time unit are counted in bins.
    """
    roll = series.rolling(window, center=True, min_periods=window)
    mean = roll.mean()
    std = roll.std()

    # OLS slope over a fixed, equally-spaced window reduces to a fixed
    # linear filter, sum(u_i * x_i) / sum(u_i^2) with u the bin offsets
    # from the window's center — one convolution instead of a per-window
    # Python callback. Convolution zero-pads the edges instead of
    # propagating NaN, so incomplete windows are masked afterwards using
    # `mean`, which does propagate them.
    offsets = np.arange(window) - (window - 1) / 2
    weights = offsets / (offsets**2).sum()
    bins_per_hour = 6
    slope = pd.Series(
        np.convolve(series.to_numpy(dtype=float), weights[::-1], mode="same") * bins_per_hour,
        index=series.index,
    ).where(mean.notna())

    return pd.DataFrame(
        {
            "value": series,
            "window_mean": mean,
            "window_std": std,
            "window_slope": slope,
        }
    )[FEATURE_COLUMNS]


def _scaled_matrix(features: pd.DataFrame) -> np.ndarray:
    """Drop incomplete rows and put the four features on a common scale.

    Both methods are distance- or split-based over the joint feature space,
    so a feature's raw units would otherwise set its influence: lux values
    run to five figures while their slope does not. `RobustScaler` centers
    on the median and scales by the IQR, so the extreme readings the
    detectors are meant to isolate do not inflate the scale they are
    measured against — the same reasoning behind Step 3's Modified Z-score.
    It also degrades safely on `precipitation`, whose IQR is exactly zero
    (most bins are 0.0mm): scikit-learn leaves a zero scale at 1.0 rather
    than dividing by it.
    """
    return RobustScaler().fit_transform(features.to_numpy(dtype=float))


def lof_scores(features: pd.DataFrame, n_neighbors: int = LOF_N_NEIGHBORS) -> pd.Series:
    """Local Outlier Factor per reading, indexed like `features`.

    The returned score is the LOF factor itself: the reading's local
    density relative to that of its `n_neighbors` nearest neighbours, so
    ~1.0 means "as dense as its neighbourhood" and larger means
    progressively more isolated than the points around it. Rows with an
    incomplete feature window are omitted rather than scored.

    LOF is used in its unsupervised (`novelty=False`) form: each station x
    sensor is scored against its own readings, with no train/test split,
    which is what makes "locally sparse" mean locally sparse *for that
    station* rather than relative to some pooled network-wide density.
    """
    complete = features.dropna()
    if complete.empty:
        return pd.Series(dtype=float, index=features.index[:0])

    estimator = LocalOutlierFactor(n_neighbors=n_neighbors, n_jobs=-1)
    estimator.fit(_scaled_matrix(complete))
    return pd.Series(-estimator.negative_outlier_factor_, index=complete.index)


def isolation_forest_scores(
    features: pd.DataFrame,
    n_estimators: int = IFOREST_N_ESTIMATORS,
    max_samples: int = IFOREST_MAX_SAMPLES,
    random_state: int = RANDOM_STATE,
) -> pd.Series:
    """Isolation Forest anomaly score per reading, indexed like `features`.

    The returned score is Liu et al. (2008)'s s(x, n) = 2^(-E[h(x)]/c(n)),
    bounded in (0, 1): a reading that random axis-aligned splits isolate in
    far fewer splits than average scores near 1, one needing average or
    more splits scores at or below 0.5. Rows with an incomplete feature
    window are omitted rather than scored.

    `random_state` is fixed because the method is stochastic in both its
    subsampling and its split selection — without it, re-running the
    notebook would produce a slightly different flagged set and Step 7's
    agreement numbers would not be reproducible.
    """
    complete = features.dropna()
    if complete.empty:
        return pd.Series(dtype=float, index=features.index[:0])

    estimator = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1,
    )
    matrix = _scaled_matrix(complete)
    estimator.fit(matrix)
    # score_samples returns -s(x, n); negate to restore the published
    # orientation, where higher means more anomalous.
    return pd.Series(-estimator.score_samples(matrix), index=complete.index)
