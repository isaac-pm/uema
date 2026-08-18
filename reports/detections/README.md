# Detections — shared schema for Steps 3-6

This directory holds the output of every anomaly-detection method in this
project (Steps 3-6), all written to the same schema so Step 7 can align and
compare them directly:

```
station, sensor, timestamp, method, flagged, score
```

`score` is a numeric anomaly score, always oriented so higher magnitude
means more anomalous, though the underlying scale differs by method.
`flagged` is that method's own threshold decision. A row exists only where
the method could actually produce a score at that timestamp — a gap the
6-hour fill (`uema.silver.fill_gaps`) left as `NaN`, or a timestamp too
close to a rolling window's edge, produces no row rather than a fabricated
"not flagged." This is defined in `uema.detect.SCHEMA_COLUMNS` /
`build_detection_table`.

Each step writes its own file(s) and does not touch another step's output,
so re-running any one notebook only regenerates that notebook's files.
Per-timestamp tables are bulky and fully reproducible, so they're
gitignored (`*.parquet` in this directory); each step's small
cross-station summary CSV stays tracked since this README quotes exact
numbers from it.

## Step 3 — Z-score / Modified Z-score (point-anomaly family)

Produced by `notebooks/03_zscore_detector.ipynb`, using
`src/uema/detect.py`. Outputs: `step3_zscore_family.parquet` (4,466,190
rows — every scored station x sensor x method x timestamp combination) and
`step3_summary.csv` (flagged counts/rates per station x sensor x method,
tracked).

**Scope:** all 12 Step 0 in-scope stations (GO + CONDITIONAL-GO), all 3
sensors, same joint-span/short-gap-filled data as Steps 1-2
(`uema.silver.consolidate_stations` + `fill_gaps`, <=6h linear bridge).

**Method.** Both Z-score and Modified (MAD-based) Z-score are computed as
a **6-hour centered rolling window** (36 bins at 10-minute resolution)
rather than against each station's whole history — `pressure` and
especially `luminous_intensity` are strongly non-stationary (lux swings
from ~0 at night to tens of thousands of lux at midday), so a whole-history
Z-score would mostly measure "is it daytime," not a genuine point
deviation. The window matches Step 2's own pressure point-anomaly
criterion ("deviation from the station's trailing 6h rolling mean"),
applied uniformly to all three sensors here.

A local window's scale (std or MAD) can collapse to near zero — a calm,
rain-free precipitation window, or a still, clear-sky lux stretch — which
would blow up the z-score for any small subsequent wiggle. Both methods
floor the local scale at 10% of the station/sensor's whole-history scale
(`GLOBAL_FLOOR_FRAC` in `uema.detect`) before dividing. `precipitation`'s
zero-inflation (over half of all 10-minute bins are exactly 0.0mm at most
stations) pushes this further: the *global* MAD is itself exactly 0 for a
zero-inflated series, so `rolling_modified_zscore` falls back to the
whole-history standard deviation as the floor anchor specifically when the
global MAD is degenerate — found and fixed during this notebook's
development (an earlier version silently scored almost no precipitation
rows because the floor itself floored at 0).

Thresholds — no labels exist to tune against, so both are the standard
literature convention rather than fit to this data: `|z| > 3` for standard
Z-score, `|modified z| > 3.5` for Modified Z-score (Iglewicz & Hoya, 1993).

**Context-conditioned variant — `luminous_intensity` only.** Step 2's
contextual-anomaly criterion for `luminous_intensity` is a plain per-hour
comparison (a value unremarkable in the full distribution but wrong for
its specific hour) — computed as `z_score_context_hour`: each reading's
Z-score against that station's own historical distribution *for that
hour-of-day*, not a rolling local window. Step 2's `pressure` and
`precipitation` contextual criteria are cross-variable (lagged
pressure-lux coupling; daytime-conditioned precipitation-lux coupling),
which a univariate per-hour Z-score can't capture — no context-conditioned
variant is computed for those two sensors.

**Results.** Standard Z-score comes back consistently far more
conservative than Modified Z-score at every station: mean flagged rate
across the 12 stations is ~0.08%-0.32% for `z_score` (pressure lowest,
precipitation highest) versus ~1.7%-3.3% for `modified_z_score` — expected,
not a bug, since `mean`/`std` are themselves pulled by the same extreme
points the detector is trying to catch, diluting the score, while
`median`/MAD stay robust to them. `z_score_context_hour` flags ~0.4%-1.4%
of `luminous_intensity` readings per station, roughly midway between the
other two. Full per-station numbers in `step3_summary.csv`.

Sanity-check plots in the notebook (largest `|score|` `z_score` flags per
sensor) confirm the detector is catching genuinely irregular episodes, not
a systematic artifact — e.g. a single-bin pressure dropout from ~1017 hPa
to ~900 hPa at `sede-central_finca-2`, correctly isolated as a point
anomaly. This is a sanity check only, not validation — there is no ground
truth in this project to validate against (see Step 2's closing note).

**Role in the overall study:** this is the "point-anomaly corner" of the
five-method comparison, and the first of the five to write to this shared
schema. Steps 4-6 must conform to `uema.detect.SCHEMA_COLUMNS` exactly so
Step 7 can concatenate and compare all methods' flagged sets at matching
station/sensor/timestamp combinations.
