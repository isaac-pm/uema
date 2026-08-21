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
criterion ("deviation from the station's surrounding 6h rolling mean"),
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
Z-score, `|modified z| > 3.5` for Modified Z-score (Iglewicz & Hoaglin, 1993).

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

## Step 4 — Local Outlier Factor / Isolation Forest (density/isolation family)

Produced by `notebooks/04_density_isolation_detector.ipynb`, using
`src/uema/density.py`. Outputs: `step4_density_isolation.parquet`
(3,859,374 rows — 1,929,687 scored timestamps per method), plus three
tracked CSVs: `step4_summary.csv` (flagged counts/rates per station x
sensor x method), `step4_threshold_sensitivity.csv`, and
`step4_tie_degeneracy.csv`.

**Scope:** identical to Step 3 — all 12 in-scope stations, all 3 sensors,
the same `consolidate_stations` + `fill_gaps` (<=6h) data. Using the same
prepared input as the Z-score family is what makes Step 7's per-timestamp
comparison attributable to the methods rather than to preprocessing.

**Feature strategy.** Step 1 found weak global lag-0 correlation between
the three sensors at every station (`|r| < 0.15`), so both methods are fit
separately per station x sensor and `sensor` keeps its plain meaning in the
schema — the plan's "combined identifier for joint features" case does not
arise. Each reading is represented by a 4-dimensional vector (`value`,
`window_mean`, `window_std`, `window_slope`) over the **1-hour centered
window** around it, rather than by its bare value: fed one scalar at a
time, both methods would collapse into a univariate distributional test
and largely duplicate Step 3. Keeping the raw value alongside the window
summaries lets a reading be flagged either for its own level or for the
local dynamic context it sits in. The 1-hour span is deliberately shorter
than Step 3's 6-hour scoring window, so the two steps describe different
time scales rather than re-deriving the same local baseline. Every feature
requires a complete window, so edge and gap-punctured readings emit no row
(same convention as Step 3). Features are scaled per station x sensor with
`RobustScaler` (median/IQR), which also degrades safely on
`precipitation`, whose IQR is exactly 0.

**Parameters** — each taken as its published/default value rather than
tuned, since there are no labels to tune against. LOF: `n_neighbors=20`
(Breunig et al., 2000, whose recommended MinPts lower bound is ~10-20).
Isolation Forest: `n_estimators=200`, `max_samples=256` (the subsample size
Liu et al., 2008 recommend), with `random_state` fixed so Step 7's
agreement numbers are reproducible across re-runs of a stochastic method.

**Thresholds** — each method's published convention, matching
scikit-learn's `contamination="auto"` offsets rather than a contamination
rate assumed for this network: LOF factor `> 1.5`, Isolation Forest
`s(x, n) > 0.5`. `step4_threshold_sensitivity.csv` records what the
flagged rate would be at other cutoffs so this can be revisited from the
stored scores without recomputing.

**Results.** LOF flags 0.5%-1.9% of readings by sensor (`precipitation`
lowest at 0.53% mean, `luminous_intensity` highest at 1.89%; per-station
means across sensors span 1.07%-2.42%) — the same order of magnitude as
Step 3's Modified Z-score. Isolation Forest at its published cutoff flags
far more: 7.5% (`precipitation`), 15.0% (`pressure`), 29.1%
(`luminous_intensity`), or 12.75%-19.89% per station. This is a property
of the cutoff, not a fault in the fit: the median IF score in this data is
~0.36-0.45 and the 99th-percentile cutoff is only ~0.65-0.77, so `s > 0.5`
falls inside the bulk of the distribution rather than in its tail. Liu et
al. note that when scores cluster near 0.5 the data shows no strongly
distinct anomalies and 0.5 does not separate them — which is what the
score distribution here indicates. **Step 7 should not compare raw flagged
sets across these two methods without accounting for that base-rate gap**;
comparing at matched alarm budgets (equal flagged share per method) from
the stored scores is the more informative comparison, with the base-rate
difference itself reported as a finding.

**Tie degeneracy — a LOF artifact on the zero-inflated channels.** LOF is
defined through local reachability density, which is undefined when a
point has `n_neighbors` exact duplicates (reachability distance 0 =>
infinite density). This is routine here, not hypothetical: on average
96.3% of `precipitation` feature windows are exact duplicates of another,
with a single all-zero vector accounting for 89.8% of windows on its own
(up to 97.3% at one station); `luminous_intensity` carries a smaller tie
mass (39.5% duplicated) from flat overnight stretches; `pressure` has
essentially none (0.01%). scikit-learn resolves the division with a small
epsilon, which correctly leaves points *inside* a tied cluster scored as
inliers (~1.0) but gives points sitting just outside one enormous,
epsilon-determined scores (up to ~3e8). The flagged decision for those
points remains meaningful — they genuinely are isolated relative to an
extraordinarily dense neighbourhood — but the score *magnitude* in that
regime is an implementation artifact, not a measure of severity. It
affects ~0.3% of `precipitation` and ~0.24% of `luminous_intensity` rows
(`step4_tie_degeneracy.csv`, `lof_degenerate_share`). Step 7 should treat
this as one of the method-specific artifacts it is explicitly meant to
characterize; the notebook's sanity plots exclude this regime so the
plotted examples reflect what the score actually means.

**Role in the overall study:** the "density/isolation corner" — the only
methods in the comparison that judge a reading by its position in a
multi-dimensional feature space, and the first to bring short-term
temporal structure in (via window features) without modeling sequence
directly.

## Step 4b — controls: window length and precipitation zero provenance

Produced by `notebooks/04b_step4_controls.ipynb`, using `src/uema/agree.py`
(new, and reused by Step 7) and `uema.silver.suspect_zero_precipitation`.
These are **controls on the Step 4 result, not detection methods** — the
`lof_6h` / `isolation_forest_6h` entries must not enter Step 7's method
roster as additional detectors.

Motivation: an interim comparison of the four detectors from Steps 3-4
found their flagged sets barely overlap (pairwise Jaccard 0.06-0.23 at
matched 1% alarm budgets, near zero on `pressure` and
`luminous_intensity`). Two alternative explanations had to be measured
before that could be read as evidence about method *families*.

Outputs: `step4b_window_control.parquet` (gitignored) plus tracked
`step4b_jaccard_native.csv`, `step4b_jaccard_matched.csv`,
`step4b_jaccard_matched_by_sensor.csv`, `step4b_window_effect.csv`,
`step4b_precip_provenance.csv`, `step4b_precip_flag_provenance.csv`.

**Agreement is reported two ways, and both are needed.** At native
thresholds, methods flag between 0.21% and 16.8% of readings, so Jaccard
between a loose and a tight method is bounded near zero regardless of how
well they agree about *which* readings are most anomalous — a native-only
matrix largely ranks threshold looseness. `uema.agree.matched_budget_flags`
gives every method the same 1% alarm budget per station x sensor as the
complementary view. The two disagree about which pairs are most similar
(native: `isolation_forest`-`modified_z_score` at 0.092; matched:
`modified_z_score`-`z_score` at 0.229), so Step 7 must report both.
Jaccard is preferred over Cohen's kappa as the headline statistic because
base rates this low put kappa in its well-known paradox regime.

### Control 1 — window length is a first-order confound

Re-running LOF and Isolation Forest on **6-hour** feature windows (36 bins,
matching Step 3's scoring window) and changing nothing else gives the
decisive number in `step4b_window_effect.csv` — how much each method
agrees with *itself* under only that change:

- `lof` at 1h vs `lof` at 6h: **Jaccard 0.053**
- `isolation_forest` at 1h vs 6h: **Jaccard 0.242**

For LOF that self-agreement (0.053) is *lower* than its agreement with
methods from an entirely different family (0.065 with `z_score`, 0.070
with `modified_z_score`). For Isolation Forest, self-agreement under a
window change (0.242) is barely above the agreement between the two
distinct statistical methods (`z_score`-`modified_z_score`, 0.229). Put
plainly: **which readings a density method flags depends at least as much
on an arbitrary window-length choice as on which method it is.**

Matching the window did *not* improve agreement with Step 3 — all four
density-vs-statistical pairs got *worse* at 6h (changes of -0.028 to
-0.068). So the Step 3 / Step 4 disagreement is not a simple time-scale
mismatch that aligning windows repairs; window length instead perturbs
each method idiosyncratically.

**Consequence for the study's central claim:** disagreement between the
corners cannot be attributed to method family while an unexamined
hyperparameter moves the flagged set by as much or more. Step 7 must
either hold time scale fixed across methods or treat it as an explicit
factor, and Steps 5-6 cannot choose the autoencoder's window and the
LSTM/GRU sequence length independently — the plan's stated goal for Step 6
(isolating the effect of *recurrence* by comparing against Step 5) is
confounded unless those two are deliberately matched.

### Control 2 — the zero-fill hypothesis is refuted

`uema.silver.suspect_zero_precipitation` marks precipitation bins that are
exactly 0.0 while *both* other sensors are missing — i.e. the station was
probably not reporting, so the zero is the export-time fill rather than an
observed dry ten minutes. These are numerous: 19.8% of all zero bins
network-wide, ranging 1.9% (`sede-central_sabanilla`) to 44.5%
(`sede-central_losic-norte-2`). That they are a fill artifact rather than
dry weather is confirmed by a clean signature — across all 12 stations,
precipitation is **never** nonzero while both other sensors are missing
(0 bins). Real dry-weather periods would occasionally contain rain.

The hypothesis under test was that precipitation's comparatively high
cross-method agreement rests on these fabricated bins. **It does not.**
Every method places far fewer precipitation flags on suspect zeros than
chance would (`step4b_precip_flag_provenance.csv`): 2.0% (`lof`), 2.1%
(`isolation_forest`), 3.2% (`lof_6h`), 5.5% (`isolation_forest_6h`), 0.04%
(`z_score`), 0.0% (`modified_z_score`) — against a 20.0% chance baseline.
No detector is drawn to the fill artifact; all avoid it, which is what one
would expect since a filled zero sits in the densest, most ordinary-looking
region of the data.

The suspect zeros still matter as a data-quality fact — they make up a
fifth of the zero mass that LOF's tie degeneracy arises from — but
precipitation's agreement is about the rain, not about the fill. Whether
that agreement is *informative* (methods converging on real events) or
*trivial* (any method finds a nonzero bin in a channel that is 96% zeros)
is not settled by this control and belongs to Step 7's case studies.

### Where the methods diverge most

Per sensor at matched budget (`step4b_jaccard_matched_by_sensor.csv`),
`luminous_intensity` is where agreement collapses almost completely —
`lof`-`modified_z_score` 0.005, `lof`-`z_score` 0.006, and no pair above
0.084. `pressure` is modest (0.012-0.379) and `precipitation` highest
(0.051-0.416). The channel with the richest diurnal structure is the one
on which these methods share almost nothing.

---

## Step 5 — non-recurrent Autoencoder (reconstruction family)

`notebooks/05_autoencoder_detector.ipynb`, logic in `src/uema/reconstruct.py`.
Third of the five method families, same 12 stations / 3 sensors / same
prepared data as Steps 3-4, writing the same schema. Method name
`autoencoder`.

A feed-forward autoencoder is fit per station x sensor (36 cells) on
**flattened raw 36-bin (6h) windows**, stride 1, labelled at the window
centre. Architecture `36 -> 17 -> 8 -> 17 -> 36` (1,574 parameters), ReLU,
linear output, hidden width at the geometric mean of input and latent size.
Earliest 80% of each cell's span by time trains, the rest is held out;
RobustScaler is fit on training rows only. Adam, MSE, batch 256, lr 1e-3,
early stopping on held-out loss with the best state restored. Score is
per-window MSE; flagged above the **99th percentile of the held-out split's
error**.

Outputs: `step5_autoencoder.parquet` (gitignored) plus tracked
`step5_summary.csv`, `step5_threshold_sensitivity.csv`,
`step5_bottleneck_ablation.csv`, `step5_seed_stability.csv`,
`step5_window_control.csv`, and three `step5_top_flagged_*.png`.

### The threshold is not the same kind of object as Steps 3-4's

`|z| > 3`, `LOF > 1.5` and `s(x,n) > 0.5` are published conventions: the
data decides what share of readings clears them, which is why Step 4's
"Isolation Forest flags 7.5-29.1%" was a finding at all. Reconstruction
error has no published cutoff, so the threshold here is a quantile, which
**fixes the alarm rate by construction**. Flagged rates land at 0.27%-3.99%
per cell (`step5_summary.csv`) and 0.92%-1.21% per sensor — near 1% because
1% was chosen, not because 1% was discovered. Step 7's native-threshold view
must say so rather than reading this rate as a property of the method.

Threshold sensitivity (`step5_threshold_sensitivity.csv`) at the 90th/95th/
99th percentiles gives overall rates of 10.31%/5.14%/1.21%
(`luminous_intensity`), 8.68%/4.41%/1.06% (`precipitation`), 7.40%/3.73%/
0.92% (`pressure`). These exceed `1 - pct` slightly and consistently, which
says the held-out (later) portion of each record reconstructs marginally
better than the whole, not worse — so the temporal split is not smuggling in
a drift artifact.

### Robustness: three passes

All three quantify displacement of the flagged set with `uema.agree`, since
with no labels the only measurable sensitivity is how far the decision moves.

**Seed stability** (`step5_seed_stability.csv`, 5 seeds, 10 pairs): mean
pairwise Jaccard **0.776** (`precipitation`), **0.601** (`luminous_intensity`),
**0.583** (`pressure`); minimum pair 0.769/0.511/0.547. Flagged-rate
standard deviation across seeds is tiny (0.0001-0.0006), so reseeding moves
*which* readings are flagged, not how many. This is the number that bounds
what Step 5 can claim: roughly 40% of the flagged set on pressure and lux is
attributable to initialization rather than to the method. It is still far
above every cross-method agreement in the project, so the method has a
stable core — but any Step 7 claim about the reconstruction family must be
read against it.

**Bottleneck ablation** (`step5_bottleneck_ablation.csv`, latent 2/4/8/16 vs
the primary 8): `precipitation` is nearly indifferent (0.66-0.73),
`pressure` moderate (latent 4: 0.40, latent 16: 0.57, latent 2: 0.17), and
`luminous_intensity` highly sensitive (latent 4: 0.28, latent 16: 0.32,
latent 2: **0.042**). Latent 2 is a genuinely different detector on the two
structured channels; 4/8/16 are variations on one. This mirrors the
zero-inflation pattern seen throughout the project — the channel that is 96%
zeros is the one whose flagged set no configuration choice can move much.

**Window control** (`step5_window_control.csv`, 36 bins vs 6 bins, holding
compression at 4.5:1 rather than latent size, since a latent of 8 over 6
inputs would be over-complete): self-agreement **0.128** overall — 0.210
(`precipitation`), 0.172 (`pressure`), 0.053 (`luminous_intensity`).
Directly comparable to Step 4b's LOF 0.053 and Isolation Forest 0.242: the
autoencoder sits between them, well below its own seed stability. **Window
length still moves this method's flagged set roughly four to five times as
much as reseeding does**, which is the third method family in a row to show
that window length is a first-order factor. On `luminous_intensity` the
window control (0.053) lands exactly on LOF's — the two least window-stable
results in the project are both on the channel every step has found the most
divergent.

### An unseeded initialization found and fixed (second run superseded)

Found while preparing Step 6, and the reason the numbers in this section
changed once more. `train_reconstructor` called `torch.manual_seed` *after*
the caller had already constructed the model, so `seed` governed the batch
order and **not the weight initialization** — the initial weights came from
wherever the ambient RNG happened to be, which depends on how many models
were built earlier in the same process. The docstring claimed the seed
covered both.

The symptom is narrow enough to miss entirely: every fit after the first in a
process reproduces exactly, and only the first one differs, so a whole-notebook
re-run reproduced itself perfectly while no individual cell could be
reproduced on its own. Re-fitting one `precipitation` cell in isolation ended
training at 145, 154, 176 and 205 epochs on four attempts, and stopping one
epoch apart moved that cell's flagged set by ~20% (Jaccard 0.80).

`train_reconstructor` now re-initializes parameters after seeding, so a fit
depends on `seed` alone. Step 5 was re-run in full; **what moved and what did
not is the useful part**:

- **Unchanged, to the digit:** the cross-method placement below
  (0.148/0.067/0.042/0.041), the bottleneck ablation (latent 2 at 0.042 on
  lux, 0.171 on pressure, 0.655 on precipitation), the threshold sensitivity
  table, per-cell flagged rates (0.27%-3.99%), and the epoch distribution
  (median 97, range 24-293, no cell capped).
- **Moved slightly:** seed stability, 0.780/0.606/0.602 -> 0.776/0.601/0.583.
- **Moved materially:** the window control, 0.187 -> **0.128** overall, driven
  almost entirely by `luminous_intensity` (0.163 -> 0.053).

That split is itself informative. Aggregate properties of the score
distribution are insensitive to initialization; *which specific readings* land
in a small flagged set is not, and the shorter 6-bin control window — fewer
inputs, a 2-dimensional latent — is where that sensitivity is largest. It also
raises the same caution Step 6 inherits: a Jaccard difference of a few
hundredths between two configurations of a neural detector is not necessarily
a difference between the configurations.

### A training defect found and fixed (first run superseded)

The first full run capped training at 100 epochs, and early stopping almost
never fired: `train_reconstructor` reset its patience counter on *any*
improvement in held-out loss, including noise-level ones, so on a plateau
the counter kept resetting and runs ended by exhausting the budget rather
than by converging. Patience is now counted from the last improvement
clearing `MIN_DELTA` (1e-4 relative) and `MAX_EPOCHS` raised to 500 as a
genuine safety bound. **All 36 cells now end by early stopping** (median 97
epochs, range 24-293; `ended_early` in `step5_summary.csv`).

Converging the models improved reconstruction substantially — held-out loss
fell 4-35% on the three cells spot-checked (34.9% and 27.6% on two
pressure cells, but only 4.0% on a lux one) — **but changed almost nothing about
the flagged sets**. Seed stability moved 0.601->0.606, 0.784->0.780,
0.603->0.602; the window control 0.186->0.187; flagged rates by under 0.002
at nearly every cell. The hypothesis that moderate seed stability was an
artifact of unconverged models is therefore **refuted**: it is a property of
the method. That the *ranking* of reconstruction error is this insensitive
to a large improvement in absolute reconstruction quality is itself worth
carrying into Step 6, where the same training machinery is reused.

One inference recorded here because it was wrong: the defect was diagnosed
from the 8 smallest cells, 6 of which hit the cap, and generalized to the
whole run. With the fix in place the median cell converges at 97 epochs, so
the old 100-epoch cap was about right for the typical cell and truncated
only the slower half.

### Cross-method placement (preliminary, for Step 7 to redo properly)

At matched 1% budgets over the five non-contextual methods (1,911,399
listwise-aligned rows), the autoencoder's agreement with the others is
0.148 (`isolation_forest`), 0.067 (`lof`), 0.042 (`z_score`), 0.041
(`modified_z_score`) — its strongest pairing is with Isolation Forest, and
`autoencoder`-`isolation_forest` at 0.148 is the second-highest pair in the
whole matrix after `z_score`-`modified_z_score` at 0.229. Per sensor the
autoencoder agrees most on `precipitation` (0.044-0.203) and least on
`luminous_intensity` (0.006-0.098), the same channel ordering every earlier
step found.

### Fixed: the contextual variant no longer breaks listwise alignment

`z_score_context_hour` is computed **only** for `luminous_intensity` (Step 3
restricted it there deliberately), so it scores 0 rows on `pressure` and
`precipitation`. `uema.agree.align_scores` is listwise, so pooling it with
the rest **silently collapsed the analysis to `luminous_intensity` alone** —
1,911,399 aligned rows down to 590,352, all lux, while every number computed
from it still looked network-wide. Nothing errored. Step 4b had avoided it
with a manual filter in the notebook; Step 5's cross-method check did not,
and got one-channel numbers.

The cause was that `align_scores` conflated two unrelated kinds of absence: a
method that *could not score* a given timestamp (a gap, a record edge —
incidental, and listwise row-dropping is correct), and a method that *does
not apply to a sensor at all* (structural, where dropping rows is a category
error). `uema.agree` now separates them:

- `applicable_methods` reports which methods scored which sensors.
- `align_by_sensor` aligns each sensor over its own applicable methods,
  listwise *within* the sensor — dropping an inapplicable method as a
  **column** rather than deleting the sensor's **rows**.
- `align_scores` raises on a non-uniform roster instead of returning a
  quietly wrong answer (`require_uniform=False` opts out deliberately).

This is backward compatible, and that is asserted rather than argued:
`tests/test_agree.py` pins that per-sensor alignment selects the identical
row set as the pooled call when no method is sensor-specific. Every Step 4b
number recomputes exactly through the new path — LOF self-agreement 0.053,
Isolation Forest 0.242, and no lux pair above 0.084.

Aligned rows per sensor: `luminous_intensity` 590,352 over **6** methods,
`precipitation` 726,687 and `pressure` 592,356 over **5**. The pooled matrix
uses the five methods common to all sensors.

### The contextual variant, compared for the first time

Because it was excluded from every previous agreement analysis, Step 2's
cleanest contextual criterion had never been tested against another method.
At matched 1% budgets on `luminous_intensity`, `z_score_context_hour` agrees
with `modified_z_score` 0.026, `isolation_forest` 0.016, `lof` 0.012,
`z_score` 0.012, and `autoencoder` 0.007 — the lowest row in the lux matrix,
and its strongest pairing is with its own statistical family.

Step 2 predicted exactly this shape: it argued a point-only method comparing
against the full 24h-pooled marginal distribution should *largely miss* what
an hour-conditioned method catches. The observed near-disjointness is
consistent with that prediction. It is not yet evidence for it — agreement is
low between essentially every pair on this channel (no pair above 0.098), so
a low number here is not distinctive on its own. Step 7's case studies have
to establish that the readings the contextual variant flags are the
contextual ones, which is a claim about *content* that no agreement statistic
can settle.

### Incidental finding: physically impossible pressure values

The sanity-check plot for the top-scoring `pressure` window
(`step5_top_flagged_pressure.png`) shows `recinto-santa-cruz` swinging from
roughly +1000 hPa to **-1078 hPa** and back. Absolute pressure cannot be
negative, so this is sensor or transport corruption, not weather.

Counting readings outside a deliberately generous 500-1100 hPa surface range
across all 12 in-scope stations: **322 of 581,571 pressure readings (0.055%)
are physically impossible**, and they are overwhelmingly concentrated at one
station — `recinto-santa-cruz`, 310 readings (0.54% of its record, range
-1078 to +1371 hPa). `sede-central_finca-2` and `sede-central_losic-norte-2`
have 6 each; the remaining nine stations have none. This is consistent with
`recinto-santa-cruz` having the highest pressure flagged rate in the network
(2.33% vs a 0.92% sensor mean).

Worth stating plainly: **Step 0's coverage audit does not test value
plausibility.** It gates on missingness, gap length and season coverage, so a
station can pass as CONDITIONAL-GO — as this one did — while carrying
hundreds of impossible values. That is not an error in the audit's own terms,
but it means "in scope" has never meant "values are physically valid," and
nothing downstream has checked. Two consequences for Step 7: these readings
are almost certainly in every method's flagged set, so a chunk of whatever
cross-method agreement exists on `pressure` may be agreement about corruption
rather than about weather; and they are obvious candidates for the
disagreement case studies, since a method that *misses* a -1078 hPa reading
has a real problem. Whether to add a plausibility gate upstream is a decision
for Step 7's writeup, not something this step changed.

---

## Step 6 — LSTM/GRU sequence autoencoders (reconstruction family, recurrent)

`notebooks/06_lstm_gru_detector.ipynb`, logic in `src/uema/reconstruct.py`
(the same module Step 5 uses). Fifth and final method family, same 12
stations / 3 sensors / same prepared data as Steps 3-5, writing the same
schema. Method names `lstm_ae` and `gru_ae` — two separate entries, never one
combined entry.

Encoder RNN over the 36-bin (6h) window, final hidden state projected to an
8-dimensional latent; the latent repeated across all 36 steps, a decoder RNN
over that repetition, and a per-step projection back to one value. One
recurrent layer per side, hidden width 17, no dropout. `nn.LSTM` (4,123
parameters) against `nn.GRU` (3,171) is the *only* difference between the two
variants, and the flattened-window dense model of Step 5 (1,574) is the only
other difference against that step: same data, same window, same split, same
scaler, same optimizer, same stopping rule, same threshold rule, all shared
through `uema.reconstruct`.

504 fits: 5 seeds and one window control per architecture over all 36 cells,
plus a latent-size sweep over a 12-cell subset (see below). Run on CUDA,
which is 3.3x faster than CPU for the LSTM and 10.1x for the GRU — the
opposite of Step 5, whose dense model is *slower* on GPU. Fits were verified
bit-identical across repeated runs on both devices first.

Outputs: `step6_lstm_gru.parquet` (gitignored) plus tracked
`step6_summary.csv`, `step6_threshold_sensitivity.csv`,
`step6_seed_stability.csv`, `step6_window_control.csv`,
`step6_bottleneck_ablation.csv`, `step6_lstm_vs_gru.csv`,
`step6_recurrence_comparison.csv`, `step6_displacement_hierarchy.csv`, and
six `step6_top_flagged_*.png`.

All 72 primary cells ended by early stopping (median 72 epochs LSTM / 74.5
GRU, max 354 / 422) — none hit the 500-epoch cap. Flagged rates are 0.18%-3.25%
per cell and 1.01%-1.25% per sensor, which is the 99th-percentile threshold
doing what it does by construction, not a property of the method; the caveat
recorded for Step 5 now applies to two more entries.

### LSTM vs GRU: not distinguishable in this data

The plan required GRU be run as a genuinely separate variant rather than
assumed equivalent to LSTM. It was, and the answer is that **on this network
the two are not distinguishable**.

The test is pooled rather than a comparison of two averages. All ten primary
fits — five seeds of each architecture — are aligned together, all 45 pairs
scored, and the pairs partitioned into *within LSTM*, *within GRU*, and
*cross-architecture*. Mean, and (min-max) across pairs:

| sensor | within `lstm_ae` | within `gru_ae` | cross-architecture |
|---|---|---|---|
| `precipitation` | 0.799 (0.775-0.817) | 0.782 (0.749-0.812) | 0.771 (0.748-0.795) |
| `pressure` | 0.757 (0.688-0.800) | 0.697 (0.545-0.813) | 0.687 (0.549-0.807) |
| `luminous_intensity` | 0.544 (0.504-0.601) | 0.582 (0.526-0.667) | 0.542 (0.447-0.667) |

The cross-architecture range overlaps both within-architecture ranges on
every sensor, and its mean sits inside them. Swapping `nn.LSTM` for `nn.GRU`
therefore moves the flagged set no more than re-running the same architecture
under a different seed.

**A weaker test would have reported the opposite, and this is worth
recording.** Comparing group *means* alone, cross-architecture agreement is
below the lower of the two within-architecture means on all three sensors —
by 0.033 (lux), 0.011 (precipitation) and 0.010 (pressure). Read as a
threshold rule that says "the architectures differ." But Step 5 established
that a few hundredths of Jaccard is inside what initialization alone
produces, so margins of that size are not evidence, which is exactly what the
overlapping ranges show. The mean-comparison version of this table was
computed first and is superseded by the pooled one.

### What moves a flagged set, ranked

Every control in Steps 4b, 5 and 6 measures one quantity — how far a flagged
set moves when one thing changes — so they go on one scale
(`step6_displacement_hierarchy.csv`, mean Jaccard, higher means *less*
displacement):

| change | lux | precip | pressure | overall |
|---|---|---|---|---|
| reseed the same architecture | 0.563 | 0.790 | 0.727 | **0.693** |
| swap LSTM for GRU | 0.542 | 0.771 | 0.687 | **0.667** |
| add recurrence (dense vs recurrent) | 0.447 | 0.650 | 0.327 | **0.475** |
| change the window (36 vs 6 bins) | 0.144 | 0.213 | 0.255 | **0.204** |
| change method family (`autoencoder` vs the four Steps 3-4 methods) | 0.006-0.098 | 0.044-0.203 | 0.021-0.137 | **0.041-0.148** |

This is the summary Step 7 needs, and it is the clearest statement the
project has produced of which differences are large enough to interpret:

1. **Recurrence is real but modest.** Dense-vs-recurrent agreement (0.475)
   sits clearly below both reseeding and the architecture swap, so adding
   recurrence does move the flagged set beyond initialization noise — unlike
   the LSTM/GRU choice. It is strongest on `pressure` (0.327), the channel
   whose Step 2 collective criterion is a sustained multi-hour trend, which
   is the one place the taxonomy predicts recurrence should matter most.
2. **Window length still beats recurrence.** A pure 36-vs-6-bin change
   (0.204) displaces a flagged set more than swapping a dense model for a
   recurrent one does. Fourth and fifth method families in a row to confirm
   Step 4b's finding, and this time the two things being compared are inside
   the same family and share every other setting.
3. **Method family still beats everything.** Cross-family agreement never
   reaches 0.23 anywhere in the project, well below every within-family
   number above.

### Recurrence, measured directly

`step6_recurrence_comparison.csv`, Step 5 against Step 6 at matched window,
split, scaler, optimizer and threshold rule — the comparison both steps were
designed for. Matched-budget Jaccard (native view differs by at most 0.05,
since all three already share a 99th-percentile budget):

| pair | lux | precip | pressure |
|---|---|---|---|
| `autoencoder`-`gru_ae` | 0.515 | 0.651 | 0.335 |
| `autoencoder`-`lstm_ae` | 0.379 | 0.648 | 0.319 |
| `gru_ae`-`lstm_ae` | 0.502 | 0.802 | 0.724 |

The reconstruction family is far more internally consistent than any
cross-family pair in the project — the *lowest* number here (0.319) is above
the *highest* cross-family pair (0.229, `z_score`-`modified_z_score`). Step 7
should expect the three reconstruction entries to behave as a bloc, and
should not treat `autoencoder`, `lstm_ae` and `gru_ae` as three independent
votes when counting agreement.

### Robustness

**Seed stability** (`step6_seed_stability.csv`, 5 seeds, 10 pairs each): mean
pairwise Jaccard 0.799/0.757/0.544 (`lstm_ae`: precipitation, pressure, lux)
and 0.782/0.697/0.582 (`gru_ae`). Comparable to Step 5's 0.776/0.583/0.601,
with one difference worth noting: **the recurrent models are markedly more
seed-stable on `pressure`** (0.757 and 0.697, against the dense model's
0.583). On the channel whose anomalies are expected to be temporal, giving
the model access to temporal order makes what it finds more reproducible, not
just different.

**Window control** (`step6_window_control.csv`, 36 vs 6 bins, compression
held at 4.5:1): self-agreement **0.205** (`lstm_ae`) and **0.206** (`gru_ae`)
overall. Both sit above Step 5's dense 0.128 and just below Step 4b's
Isolation Forest 0.242 — recurrent models are somewhat *less* window-sensitive
than the dense autoencoder, though still far more window-sensitive than
seed-sensitive.

**Bottleneck ablation** (`step6_bottleneck_ablation.csv`, latent 2/4/16 vs
the primary 8). **Scope limit: 4 of 12 stations** (12 cells), chosen by
spanning the range of training-set sizes; Step 5 ran the equivalent sweep
across the full network on identical data and windows, so this pass asks
whether that conclusion carries rather than establishing it. It does, with
the same shape: `precipitation` nearly indifferent (0.687-0.822), the
structured channels more sensitive, and latent 2 the outlier — 0.132 on lux
for `lstm_ae`, against Step 5's 0.042. Severe compression is a different
detector; 4, 8 and 16 are variations on one.

**Threshold sensitivity** (`step6_threshold_sensitivity.csv`) at the
90th/95th/99th percentiles gives overall rates of 9.9%/4.8%/1.0%
(`lstm_ae`, lux) through 8.0%/4.2%/1.1% (`gru_ae`, pressure) — slightly above
`1 - pct` throughout, the same direction Step 5 found, meaning the held-out
tail of each record reconstructs marginally better than the record as a
whole.
