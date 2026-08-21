"""Alignment behaviour of `uema.agree`.

These exist for one specific failure mode. `align_scores` is listwise, which
is correct when every method scores every sensor and catastrophic when one
does not: `z_score_context_hour` is defined for `luminous_intensity` alone,
so pooling it with the rest silently deleted every `pressure` and
`precipitation` row and left a one-channel result that still read as
network-wide. Nothing errored. The tests below pin the distinction between a
method being *out of scope* for a sensor and a method merely *failing to
score* a timestamp, and pin the backward-compatibility claim that protects
the Step 4b numbers already published in `reports/detections/README.md`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uema.agree import (
    align_by_sensor,
    align_scores,
    applicable_methods,
    jaccard_by_sensor,
    jaccard_matrix,
)

SENSORS = ("pressure", "precipitation", "luminous_intensity")


def detections(methods_by_sensor: dict[str, tuple[str, ...]], n: int = 10) -> pd.DataFrame:
    """Long-form detection table in the shared schema, one station."""
    stamps = pd.date_range("2025-01-01", periods=n, freq="10min")
    rows = []
    for sensor, methods in methods_by_sensor.items():
        for method in methods:
            rows.append(
                pd.DataFrame(
                    {
                        "station": "s0",
                        "sensor": sensor,
                        "timestamp": stamps,
                        "method": method,
                        "flagged": np.arange(n) < 2,
                        "score": np.arange(n, dtype=float),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


UNIFORM = {sensor: ("z_score", "lof") for sensor in SENSORS}
LUX_ONLY_EXTRA = {
    "pressure": ("z_score", "lof"),
    "precipitation": ("z_score", "lof"),
    "luminous_intensity": ("z_score", "lof", "z_score_context_hour"),
}


def test_applicable_methods_marks_the_sensor_specific_method():
    applicable = applicable_methods(detections(LUX_ONLY_EXTRA))
    assert applicable.loc["luminous_intensity", "z_score_context_hour"]
    assert not applicable.loc["pressure", "z_score_context_hour"]
    assert not applicable.loc["precipitation", "z_score_context_hour"]


def test_align_by_sensor_keeps_every_sensor_despite_a_lux_only_method():
    """The regression test: a lux-only method must not delete other sensors."""
    aligned = align_by_sensor(detections(LUX_ONLY_EXTRA))

    assert set(aligned) == set(SENSORS)
    assert all(len(frame) == 10 for frame in aligned.values())
    # ...and it appears only in the channel it applies to.
    assert "z_score_context_hour" in aligned["luminous_intensity"].columns
    assert "z_score_context_hour" not in aligned["pressure"].columns


def test_align_scores_refuses_a_non_uniform_roster():
    with pytest.raises(ValueError, match="z_score_context_hour on pressure"):
        align_scores(detections(LUX_ONLY_EXTRA))


def test_align_scores_accepts_a_uniform_roster():
    assert len(align_scores(detections(UNIFORM))) == 30


def test_require_uniform_false_reproduces_the_old_collapsing_behaviour():
    """The opt-out is still available, and still collapses -- deliberately."""
    wide = align_scores(detections(LUX_ONLY_EXTRA), require_uniform=False)
    assert set(wide.index.get_level_values("sensor")) == {"luminous_intensity"}


def test_align_by_sensor_matches_align_scores_when_every_method_is_uniform():
    """Backward compatibility: this is what protects Step 4b's published numbers.

    Each row belongs to exactly one sensor and `dropna` is row-wise, so
    per-sensor listwise alignment should select the identical row set as the
    pooled call whenever the method roster does not vary by sensor.
    """
    frame = detections(UNIFORM)
    pooled = align_scores(frame)
    per_sensor = pd.concat(align_by_sensor(frame).values())

    assert len(per_sensor) == len(pooled)
    assert set(per_sensor.index) == set(pooled.index)


def test_align_by_sensor_still_drops_rows_a_method_could_not_score():
    """Structural absence drops a column; incidental absence still drops a row."""
    frame = detections(UNIFORM)
    unscored = (frame["sensor"] == "pressure") & (frame["method"] == "lof")
    frame = frame[~(unscored & (frame["timestamp"] == frame["timestamp"].min()))]

    aligned = align_by_sensor(frame)
    assert len(aligned["pressure"]) == 9
    assert len(aligned["precipitation"]) == 10


def test_jaccard_by_sensor_accepts_the_mapping_form():
    aligned = align_by_sensor(detections(LUX_ONLY_EXTRA))
    flags = {sensor: frame < 2 for sensor, frame in aligned.items()}
    pairs = jaccard_by_sensor(flags)

    assert set(pairs["sensor"]) == set(SENSORS)
    lux = pairs[pairs["sensor"] == "luminous_intensity"]
    others = pairs[pairs["sensor"] == "pressure"]
    assert len(lux) == 3  # three methods -> three pairs
    assert len(others) == 1  # two methods -> one pair


def test_jaccard_matrix_endpoints():
    index = pd.MultiIndex.from_product(
        [["s0"], ["pressure"], pd.date_range("2025-01-01", periods=4, freq="10min")],
        names=["station", "sensor", "timestamp"],
    )
    flags = pd.DataFrame(
        {
            "a": [True, True, False, False],
            "b": [True, True, False, False],
            "c": [False, False, True, True],
            "empty": [False, False, False, False],
        },
        index=index,
    )
    matrix = jaccard_matrix(flags)

    assert matrix.loc["a", "b"] == 1.0  # identical flagged sets
    assert matrix.loc["a", "c"] == 0.0  # disjoint

    # Empty union is undefined rather than perfect agreement: two methods that
    # flagged nothing have not agreed about anything.
    empty = jaccard_matrix(flags[["empty"]].assign(empty2=False))
    assert np.isnan(empty.loc["empty", "empty2"])
