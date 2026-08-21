"""Step 7 cross-method analysis: what the aligned flagged sets are made of.

`uema.agree` answers *how much* two methods overlap. This module answers the
questions that follow, and that an overlap statistic cannot settle on its own.

*How much of a flagged set is one method's alone.* With eight method entries
on one channel, the interesting quantity is not only pairwise Jaccard but the
share of a method's flags that no other method reaches at the same alarm
budget (`exclusive_share`), which is what makes a method a candidate for
manual inspection rather than a redundant copy of its neighbors.

*What shape the flagged readings have in time.* Step 2's criteria separate
point anomalies (one bin, neighbors at baseline) from collective ones (a
multi-hour pattern with no abnormal single step). Those are statements about
the temporal footprint of a detection, so they are testable directly:
`flag_runs` groups each method's flags into maximal runs of consecutive
10-minute bins and `run_length_summary` reduces them to the share of flags
that are isolated and the share that sit in runs of an hour or more. At a
matched alarm budget every method flags the same number of readings, so any
difference in run structure is a difference in what the methods are looking
for rather than in how much they flag.

*Whether a contextual flag is actually contextual.* `percentile_context`
places every reading on two axes at once: its rank in its station's full
24-hour-pooled distribution for that sensor, and its rank among readings from
the same hour of day at the same station. Step 2's `luminous_intensity`
contextual criterion is exactly a claim about that pair — unremarkable on the
first axis, extreme on the second — so it can be checked against real flags
instead of inferred from a low agreement number.

*One vote per family, not one per method.* Step 6 established that the three
reconstruction entries move together (Jaccard 0.32-0.80) far more closely
than any cross-family pair (never above 0.229), so counting them as three
independent votes lets the largest family outvote the others by construction.
`family_vote_flags` gives each family a single flagged set at the same budget
as every other, by averaging its members' within-cell score ranks before
thresholding. Averaging ranks rather than taking a union or an intersection
keeps the alarm budget identical across families regardless of how many
members a family has.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BIN = pd.Timedelta("10min")
LONG_RUN_BINS = 6  # one hour at 10-minute resolution
EXTREME_TAIL = 0.05  # two-sided: below the 5th or above the 95th percentile

# The Step 7 roster: every method entry Steps 3-6 produced, grouped by the
# anomaly-type corner it represents. `z_score_context_hour` is deliberately
# not a member of the point family — it is the one contextual entry in the
# study, is defined for `luminous_intensity` alone, and is reported on its own
# rather than averaged into a family vote (see `family_vote_flags`).
FAMILIES = {
    "statistical": ["z_score", "modified_z_score"],
    "density": ["lof", "isolation_forest"],
    "reconstruction": ["autoencoder", "lstm_ae", "gru_ae"],
}
CONTEXTUAL_METHOD = "z_score_context_hour"
CONTEXTUAL_SENSOR = "luminous_intensity"

# Step 4b's window-length variants. Controls, never roster members: they exist
# to bound how much of a cross-family difference is attributable to method
# identity at all, and adding them to the matrix would double-count two
# methods (see Step 4b's closing note).
CONTROL_METHODS = ("lof_6h", "isolation_forest_6h")

STEP_FILES = (
    "step3_zscore_family.parquet",
    "step4_density_isolation.parquet",
    "step5_autoencoder.parquet",
    "step6_lstm_gru.parquet",
)


def load_detections(detect_dir, files=STEP_FILES) -> pd.DataFrame:
    """Concatenate the Steps 3-6 outputs into one long table.

    Timestamps are normalized to microsecond resolution first: Step 6 wrote
    nanosecond-resolution timestamps and the earlier steps microsecond, and a
    pivot over mixed resolutions would align nothing.
    """
    frames = []
    for name in files:
        frame = pd.read_parquet(detect_dir / name)
        frame["timestamp"] = frame["timestamp"].astype("datetime64[us]")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def flag_runs(flags: pd.Series) -> pd.DataFrame:
    """Maximal runs of consecutive flagged 10-minute bins, per station.

    `flags` is one method's boolean column, indexed by station/sensor/
    timestamp. A run ends where the next flagged bin is not the next bin in
    time, so a flag isolated in time and a flag adjacent to an unscored gap
    both terminate a run — the aligned row set is not a complete grid, and
    treating a gap as continuity would merge episodes on either side of an
    outage.
    """
    runs = []
    for station, group in flags.groupby(level="station"):
        stamps = group.index.get_level_values("timestamp")
        flagged = np.flatnonzero(group.to_numpy())
        if flagged.size == 0:
            continue
        adjacent = np.diff(stamps[flagged]) == BIN
        breaks = np.flatnonzero(~adjacent)
        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks, [flagged.size - 1]])
        runs.append(
            pd.DataFrame(
                {
                    "station": station,
                    "start": stamps[flagged[starts]],
                    "end": stamps[flagged[ends]],
                    "length": ends - starts + 1,
                }
            )
        )
    return pd.concat(runs, ignore_index=True) if runs else pd.DataFrame(
        columns=["station", "start", "end", "length"]
    )


def run_length_summary(flags: pd.DataFrame, sensor: str) -> pd.DataFrame:
    """Temporal footprint of each method's flagged set on one sensor.

    `share_isolated` and `share_in_long_runs` are shares of *flags*, not of
    runs: the question Step 2 poses is what a typical detection looks like,
    and a method producing a handful of very long runs alongside many single
    bins should not be described by whichever count happens to be larger.
    """
    rows = []
    for method in flags.columns:
        runs = flag_runs(flags[method])
        total = int(runs["length"].sum())
        if total == 0:
            continue
        rows.append(
            {
                "sensor": sensor,
                "method": method,
                "n_flags": total,
                "n_runs": len(runs),
                "median_run_bins": float(runs["length"].median()),
                "mean_run_bins": float(runs["length"].mean()),
                "max_run_bins": int(runs["length"].max()),
                "share_isolated": float(runs.loc[runs["length"] == 1, "length"].sum() / total),
                "share_in_long_runs": float(
                    runs.loc[runs["length"] >= LONG_RUN_BINS, "length"].sum() / total
                ),
            }
        )
    return pd.DataFrame(rows)


def exclusive_share(flags: pd.DataFrame) -> pd.Series:
    """Share of each method's flags that no other method in the frame flags."""
    shares = {}
    for method in flags.columns:
        own = flags[method]
        others = flags.drop(columns=[method]).any(axis=1)
        shares[method] = float((own & ~others).sum() / own.sum()) if own.any() else np.nan
    return pd.Series(shares, name="share_exclusive")


def consensus_counts(flags: pd.DataFrame) -> pd.Series:
    """How many rows were flagged by exactly k methods, for each k."""
    counts = flags.sum(axis=1).value_counts().sort_index()
    counts.index.name = "n_methods_flagging"
    return counts.rename("n_rows")


def family_vote_flags(
    wide: pd.DataFrame,
    budget: float = 0.01,
    families: dict[str, list[str]] | None = None,
    by: tuple[str, ...] = ("station", "sensor"),
) -> pd.DataFrame:
    """One flagged set per family, at the same alarm budget as every other.

    Each member's |score| is converted to a within-cell percentile rank, the
    ranks are averaged across the family's members, and the family flags its
    top `budget` share of the resulting consensus rank. This is what keeps
    the reconstruction family — three entries that Step 6 showed move
    together — from carrying three times the weight of a two-member family in
    any tally of how many families a reading is flagged by.
    """
    families = families or FAMILIES
    ranks = wide.groupby(level=list(by)).rank(pct=True, ascending=False)
    consensus = pd.DataFrame(
        {
            family: ranks[[m for m in members if m in ranks.columns]].mean(axis=1)
            for family, members in families.items()
        }
    )
    # Low percentile rank == high |score|, so the family's most anomalous
    # readings are the smallest consensus ranks.
    within = consensus.groupby(level=list(by)).rank(pct=True, ascending=True)
    return within <= budget


def percentile_context(
    flags: pd.DataFrame, station_frames: dict[str, pd.DataFrame], sensor: str
) -> pd.DataFrame:
    """Each aligned reading's rank in its pooled and its hour-of-day distribution.

    Two columns, both per station and computed over the aligned rows only:
    `global_pct` ranks a reading against every hour of the station's record,
    `hour_pct` against the same hour of day alone. Step 2's contextual
    criterion for `luminous_intensity` is the conjunction "central on the
    first, extreme on the second", which `contextual_profile` then counts.
    """
    parts = []
    for station, group in flags.groupby(level="station"):
        stamps = group.index.get_level_values("timestamp")
        values = station_frames[station][sensor].reindex(stamps)
        parts.append(
            pd.DataFrame(
                {
                    "value": values.to_numpy(),
                    "global_pct": values.rank(pct=True).to_numpy(),
                    "hour_pct": values.groupby(stamps.hour).rank(pct=True).to_numpy(),
                    "hour": stamps.hour,
                },
                index=group.index,
            )
        )
    return pd.concat(parts).reindex(flags.index)


def _is_extreme(pct: pd.Series, tail: float = EXTREME_TAIL) -> pd.Series:
    return (pct < tail) | (pct > 1 - tail)


def contextual_profile(
    flags: pd.DataFrame, context: pd.DataFrame, tail: float = EXTREME_TAIL
) -> pd.DataFrame:
    """Where each method's flags fall on the pooled/hour-conditioned axes.

    `share_contextual_only` is the quantity Step 2's criterion names: flags
    that are unremarkable in the station's full 24-hour distribution but
    extreme for their own hour of day. The same share computed over every
    aligned row is the chance level a method would hit by flagging at random,
    and is returned as the `baseline` row so no reading of the table has to
    assume one.
    """
    global_extreme = _is_extreme(context["global_pct"], tail)
    hour_extreme = _is_extreme(context["hour_pct"], tail)
    night = (context["hour"] >= 18) | (context["hour"] < 6)

    def profile(mask):
        selected = mask.to_numpy()
        return {
            "n_flags": int(selected.sum()),
            "share_global_extreme": float(global_extreme[selected].mean()),
            "share_hour_extreme": float(hour_extreme[selected].mean()),
            "share_contextual_only": float((~global_extreme & hour_extreme)[selected].mean()),
            "share_neither": float((~global_extreme & ~hour_extreme)[selected].mean()),
            "median_global_pct": float(context["global_pct"][selected].median()),
            "share_at_night": float(night[selected].mean()),
        }

    rows = {method: profile(flags[method]) for method in flags.columns}
    rows["baseline (all aligned rows)"] = profile(pd.Series(True, index=flags.index))
    return pd.DataFrame(rows).T.rename_axis("method")


def episodes(mask: pd.Series, min_bins: int = 1, top: int | None = None) -> pd.DataFrame:
    """Contiguous runs of `mask`, longest first — the case-study candidates."""
    runs = flag_runs(mask)
    runs = runs[runs["length"] >= min_bins].sort_values("length", ascending=False)
    return runs.head(top) if top else runs


def mean_offdiagonal(matrix: pd.DataFrame) -> float:
    """Mean of a symmetric agreement matrix's upper triangle."""
    values = matrix.to_numpy()[np.triu_indices(len(matrix), 1)]
    return float(np.nanmean(values))
