"""Cross-method agreement primitives, shared by the Step 4 controls and Step 7.

Step 7's central analysis compares which timestamps each detection method
flagged. That comparison is not as simple as intersecting the `flagged`
columns, for two reasons this module exists to handle.

*Different methods score different rows.* Each method emits a row only
where it could actually produce a score (`uema.detect.build_detection_table`),
and those row sets differ — Step 3's 6-hour rolling statistics and Step 4's
1-hour feature windows are clipped by gaps and record edges at different
places. Any agreement number therefore has to state which rows it was
computed over; `align_scores` restricts to rows *every* method scored, so a
method is never counted as "missing a detection" at a timestamp it was
never in a position to score.

*Not every method applies to every sensor.* A method can be absent from a
(sensor, timestamp) because it could not score that reading — a gap, a
record edge — or because it does not apply to that sensor at all, as with
`z_score_context_hour`, which Step 3 defines for `luminous_intensity`
alone. The first is incidental and listwise dropping is the right response;
the second is structural, and dropping rows for it deletes every row of the
sensors that method never covered. `align_by_sensor` keeps the two apart,
and `align_scores` refuses a roster that mixes them rather than silently
returning a one-channel answer that still looks network-wide.

*Different methods flag wildly different shares of the data.* Each method's
threshold is its own published convention, and those conventions do not
imply comparable alarm rates: on this network they range from well under
1% to nearly 30% of readings. Jaccard similarity between a 20%-flagging
method and a 0.3%-flagging one is bounded near zero no matter how well the
two agree about *which* readings are most anomalous, so a matrix built on
native thresholds largely ranks threshold looseness. `matched_budget_flags`
provides the complementary view: give every method the same alarm budget
and compare the readings each considers most anomalous. Neither view is
the "true" one — the native view says what each method actually decides,
the matched view says whether they are looking at the same phenomena — so
Step 7 should report both.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

INDEX_COLUMNS = ["station", "sensor", "timestamp"]
DEFAULT_BUDGET = 0.01


def applicable_methods(detections: pd.DataFrame) -> pd.DataFrame:
    """Boolean sensor x method table: did this method score this sensor at all.

    Distinguishes a method that is *out of scope* for a sensor from one that
    merely failed to score particular timestamps — see `align_by_sensor` for
    why conflating the two silently destroys an analysis.
    """
    counts = detections.groupby(["sensor", "method"]).size().unstack(fill_value=0)
    return counts.gt(0)


def align_scores(detections: pd.DataFrame, require_uniform: bool = True) -> pd.DataFrame:
    """Wide table of |score| per method, over rows every method scored.

    Takes the long-form concatenation of any number of detection tables in
    the shared schema and returns one column per `method`, indexed by
    station/sensor/timestamp. Scores are compared by magnitude: Step 3's
    Z-scores are signed (direction of deviation) while the Step 4 density
    scores are strictly positive, so |score| is the only orientation on
    which all methods mean the same thing — "further from normal".

    Listwise, not pairwise: a timestamp missing from any one method is
    dropped for all of them, so every pair in a resulting matrix is
    computed over the identical row set and the matrix is internally
    comparable. This costs rows (Step 3 alone scores more than the
    intersection) but avoids each cell of an agreement matrix silently
    describing a different subset of the network.

    That listwise rule is only correct when every method in `detections` is
    *applicable* to every sensor in it. `z_score_context_hour` is defined for
    `luminous_intensity` alone, so pooling it with the rest makes listwise
    dropping delete every `pressure` and `precipitation` row — collapsing the
    whole analysis to one channel while every number computed from it still
    looks network-wide. `require_uniform` refuses that case rather than
    returning the wrong answer quietly; pass `align_by_sensor` the same frame
    instead. Set it to False only for a deliberate pooled call whose method
    roster is already known to be uniform.
    """
    if require_uniform:
        applicable = applicable_methods(detections)
        missing = [
            (sensor, method)
            for sensor, row in applicable.iterrows()
            for method, present in row.items()
            if not present
        ]
        if missing:
            pairs = ", ".join(f"{method} on {sensor}" for sensor, method in missing)
            raise ValueError(
                "align_scores is listwise and would drop every row of a sensor whose "
                f"roster is incomplete; no scores for: {pairs}. Use align_by_sensor() "
                "to align each sensor over its own applicable methods, or pass "
                "require_uniform=False if a pooled listwise view is genuinely intended."
            )

    wide = detections.assign(abs_score=detections["score"].abs()).pivot_table(
        index=INDEX_COLUMNS, columns="method", values="abs_score", aggfunc="first"
    )
    return wide.dropna()


def align_by_sensor(detections: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-sensor wide |score| tables, each over that sensor's own methods.

    Returns `{sensor: wide frame}`. Within a sensor the rule is unchanged
    from `align_scores` — listwise over the methods that scored that sensor —
    but a method absent from a sensor entirely is dropped as a *column* there
    rather than deleting the sensor's *rows*.

    That distinction is the whole point. A method can be missing from a
    (sensor, timestamp) for two unrelated reasons: it could not score that
    particular reading (a gap, a record edge — incidental, and listwise
    dropping is the right response), or it does not apply to that sensor at
    all (structural, and dropping rows for it is a category error). Treating
    the second as the first is what collapses a six-method network-wide
    comparison into a one-channel one.

    Each frame keeps the full (station, sensor, timestamp) index so
    `matched_budget_flags` can still group by station and sensor unchanged.
    """
    applicable = applicable_methods(detections)
    aligned = {}
    for sensor, group in detections.groupby("sensor", sort=True):
        methods = applicable.columns[applicable.loc[sensor]].tolist()
        aligned[sensor] = align_scores(
            group[group["method"].isin(methods)], require_uniform=False
        )
    return aligned


def native_flags(
    detections: pd.DataFrame,
    index: pd.MultiIndex | None = None,
    methods: list[str] | None = None,
) -> pd.DataFrame:
    """Wide boolean table of each method's own threshold decision.

    `index` restricts to a row set computed elsewhere (normally
    `align_scores(...).index`, or one frame's index from `align_by_sensor`)
    so the native and matched-budget views describe the same rows and can be
    read side by side. `methods` restricts the columns, which matters when
    `index` covers a single sensor: without it the frame would carry an
    all-NaN column for every method that sensor never had.
    """
    flags = detections.pivot_table(
        index=INDEX_COLUMNS, columns="method", values="flagged", aggfunc="first"
    )
    if methods is not None:
        flags = flags[list(methods)]
    if index is not None:
        flags = flags.loc[index]
    return flags.astype(bool)


def matched_budget_flags(
    wide: pd.DataFrame,
    budget: float = DEFAULT_BUDGET,
    by: tuple[str, ...] = ("station", "sensor"),
) -> pd.DataFrame:
    """Flag each method's top `budget` share of |score|, within each group.

    Ranking within station x sensor rather than pooling the whole network
    keeps a station with an intrinsically wider score distribution from
    consuming the entire budget, and matches how every detector here was
    fit — per station and sensor.
    """
    ranks = wide.groupby(level=list(by)).rank(pct=True, ascending=False)
    return ranks <= budget


def jaccard_matrix(flags: pd.DataFrame) -> pd.DataFrame:
    """Symmetric pairwise Jaccard similarity between methods' flagged sets.

    |A and B| / |A or B|, i.e. of every reading at least one method
    flagged, the share both flagged. Chosen over Cohen's kappa as the
    headline statistic because these flagged sets are extremely small
    relative to the data: with base rates well under 1%, the overwhelming
    majority of rows are agreed negatives, which kappa's chance-correction
    term is known to handle erratically (the "kappa paradox" — very high
    raw agreement can coexist with near-zero kappa). Jaccard ignores the
    agreed negatives entirely, which is the quantity of interest here.
    """
    methods = list(flags.columns)
    matrix = pd.DataFrame(np.eye(len(methods)), index=methods, columns=methods)
    for i, left in enumerate(methods):
        for right in methods[i + 1:]:
            union = (flags[left] | flags[right]).sum()
            value = (flags[left] & flags[right]).sum() / union if union else np.nan
            matrix.loc[left, right] = matrix.loc[right, left] = value
    return matrix


def jaccard_by_sensor(flags: pd.DataFrame | Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Long-form pairwise Jaccard computed separately per sensor.

    Pooling sensors hides the dominant structure in this network: the
    zero-inflated `precipitation` channel behaves very differently from
    `pressure` and `luminous_intensity`, so a pooled matrix mostly reports
    whichever channel contributes the most rows.

    Accepts either one wide flags frame covering every sensor, or the
    `{sensor: frame}` mapping that follows from `align_by_sensor`. The
    mapping form is what allows sensors to carry different method rosters —
    a contextual variant defined for one channel only appears in that
    channel's pairs and nowhere else.
    """
    if isinstance(flags, Mapping):
        groups = list(flags.items())
    else:
        groups = list(flags.groupby(level="sensor"))

    rows = []
    for sensor, group in groups:
        matrix = jaccard_matrix(group)
        methods = list(matrix.columns)
        for i, left in enumerate(methods):
            for right in methods[i + 1:]:
                rows.append(
                    {
                        "sensor": sensor,
                        "method_a": left,
                        "method_b": right,
                        "jaccard": matrix.loc[left, right],
                    }
                )
    return pd.DataFrame(rows)
