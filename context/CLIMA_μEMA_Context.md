# CLIMA-µEMA — Context: Prior Prototype

**Status of this document:** background for a *new, deeper* project built on the same
µEMA network data. CLIMA-µEMA was a fast, low-rigor prototype (single author,
short timeline, "does this basic idea even work" scope). Nothing here should be
treated as a settled architecture — it is a record of what was tried, what the
result actually showed, and where the prototype's own evidence says its
weaknesses were.

---

## 1. What problem it addressed

UCR (Universidad de Costa Rica) operates 10 decentralized automatic micro
weather stations (µEMA) across different microclimates. The prototype asked:
can unsupervised reconstruction-error anomaly detection on this station
telemetry line up with real emergency alerts issued by CNE (Costa Rica's
national emergency commission), well enough to be a useful local early-warning
signal?

Framing choice worth flagging: this was **not** weather forecasting. It never
predicted future sensor values — it only measured how well a model trained on
"normal" data could reconstruct a given 24h window, on the premise that
anomalous/pre-emergency conditions reconstruct worse.

## 2. Data

- 10 stations, 3 raw sensor streams each: pressure (hPa), precipitation (mm),
  luminous intensity (lux). 10-minute native resolution after resampling.
- Station operational histories are wildly uneven — from ~1.2 years of data
  down to a few weeks (see Figure 1 timeline in the paper). This heterogeneity
  turned out to be the dominant factor in the results, more than any modeling
  choice.
- Emergency alert labels came from CNE's historical alert archive, digitized
  via LLM-based OCR extraction from PDF alert bulletins (not a clean
  structured feed — this is a real weak point: label quality is only as good
  as OCR + LLM extraction + regional text matching against station location).
- One station (`recinto-guapiles`) was excluded outright for insufficient/faulty
  data and never entered any experiment.

## 3. Pipeline (Medallion architecture)

**Raw layer:** per-station, per-sensor CSVs downloaded from UCR's µEMA portal;
alerts extracted from PDF bulletins via docling (OCR) + Gemini (structured
extraction) into a single `alerts_data.csv`.

**Silver layer:** per station —
- consolidate the 3 sensor streams on a shared time index
- apply station-specific sensor cutoffs (e.g. one station's luminous sensor
  was dead until 2025-05-20; data before that is simply dropped for that
  channel)
- resample to strict 10-min frequency
- trim to the overlapping valid range across sensors, interpolate pressure
  gaps linearly, zero-fill precipitation/lux gaps
- add cyclical time features (hour, day-of-year — sin/cos)
- match each timestep against CNE alerts by matching alert region text
  against each station's assigned region list (`STATION_REGIONS`), using
  `merge_asof` with a tight (10 min) tolerance

**Gold layer:**
- **Anomaly mask**: alert-active timesteps dilated **asymmetrically**:
  48h *before* alert issue (developing conditions / forecast lead time) and
  120h *after* (event duration + sensor recovery). This buffer choice was a
  judgment call, not derived from data — worth re-examining, not assuming.
- Feature scaling fit **only on non-anomalous training-portion data**
  (StandardScaler for pressure, MinMaxScaler for precipitation/lux) to avoid
  leaking anomaly statistics into normalization.
- 144-step (24h) sliding windows, stride 6.
- Temporal 80/20 split per station (earliest 80% train, latest 20% test) —
  **not random** — anomalous windows are reserved entirely for test.
- A separate calibration set of "boundary-normal" windows (normal windows near
  but not inside an anomaly) is built for threshold calibration, to avoid
  calibrating thresholds against a distribution the model never sees anomalies
  near.
- A global variant concatenates all stations (after per-station temporal
  splitting, so no station leaks entirely into train or test) for a
  cross-station model.

## 4. Model

Symmetric LSTM-Autoencoder:
- Input: 144 timesteps × 7 features (3 continuous: pressure, precipitation,
  lux; 4 cyclical: hour/day-of-year sin-cos).
- Encoder: 3-layer LSTM funnel 128 → 64 → *d*, keeping only the final hidden
  state as the bottleneck (*d*-dimensional).
- Decoder: mirrors it, *d* → 64 → 128, bottleneck repeated 144× before a
  linear projection back to the 3 continuous features only (cyclical features
  are excluded from reconstruction — they're context, not signal to detect
  anomalies in).
- Dropout (p=0.2) on the first two layers of encoder/decoder; bottleneck and
  final pre-projection layers deliberately left uncorrupted.
- Loss: feature-weighted MSE over continuous features only (precipitation
  weighted highest, 2.0, since CNE alerts are predominantly rainfall-driven;
  pressure 1.5; lux 1.0).
- Optional augmentation regime: Gaussian noise (simulating calibration drift)
  + temporal masking (zeroing short spans to force context-based
  reconstruction).

## 5. Experimental design

- 4 pipeline configs: {Local, Global} × {Baseline, Augmented}
- 4 bottleneck sizes: *d* ∈ {8, 16, 32, 64} (compression ratios 126:1 to 16:1)
- 5 fixed seeds
- → 80 total training runs, all metrics reported as mean ± std across seeds
- Thresholds calibrated **independently of test labels**: local models sweep
  85th/90th/95th percentile of validation reconstruction error; global model
  uses 95th percentile of the boundary-normal calibration set.
- Metrics: standard pointwise Precision/Recall/F1/FPR, plus **Point-Adjustment
  (PA)** variants — a ground-truth anomalous block counts as "detected" if
  ≥10% of its windows are flagged (~17h of the dilated 48h/120h block). PA
  metrics are the headline numbers in the paper; raw pointwise numbers are
  much weaker (e.g. pointwise F1 around 0.3–0.6 even at "viable" stations) —
  **this gap is worth keeping in view**, since PA-F1 credits a full block from
  a fairly loose detection threshold and is a generous metric, not a strict one.

## 6. What the results actually showed

**Four stations came out "viable"** (recinto-esparza, recinto-santa-cruz,
sede-atlantico_turrialba, sede-sur_golfito) — at *d*=16, PA-F1 ranged
0.938–0.980, and 3 of 4 hit PA-Recall = 1.000 across all seeds. These four
share ≥4,440 training windows and continuously-functioning sensors — i.e.
enough clean history, not something particular about the model.

**Five stations failed, for distinct, diagnosable reasons — this is the
prototype's most useful finding, more than the "good" numbers:**
- `finca-2`: high PA-F1 (0.936) but FPR ≈0.757 — the luminous sensor was dead
  until 2025-05-20, so "good" score is not trustworthy signal here; it's a
  sensor-fault artifact.
- `liberia`: PA-Recall ≈0.086 consistently across all bottleneck sizes —
  event-label misalignment (alerts not lining up with this station's actual
  local conditions/region matching), not a modeling failure.
- `finca-3`: fails locally (PA-Recall ≤0.055) at most dims, one seed spikes to
  1.000 at *d*=16 (bimodal, mean 0.241±0.424) — borderline detectability.
  Recovered by the **global** model (PA-F1 = 0.926±0.001, PA-Recall = 1.000
  across all *d*) — the one case where cross-station pooling clearly helped.
- `caribe_limon`, `finca-1`: all-zero output across every seed/dim — under
  ~1,650 training windows, not enough data to stabilize a calibration
  distribution at all.

**Ablation-level takeaways:**
- Results stable across *d* ∈ {8, 16, 32}; at *d*=64, the most data-scarce
  viable station (`sede-atlantico_turrialba`) shows bimodal, seed-sensitive
  behavior (σ=0.387) — larger bottlenecks amplify initialization sensitivity
  when data is scarce.
- Denoising augmentation gave **no consistent benefit** (|ΔPA-F1| ≤0.006 at
  *d*=16) and destabilized some seeds at atypical bottleneck sizes. Worth not
  carrying this forward as an assumed-good technique without re-testing.
- Global pooling helped only where local data was fundamentally insufficient
  (finca-3); it did not improve already-viable local stations.

## 7. Honest limitations of the prototype (carry these into the new project)

- **Single author, fast prototype, "no much thought" per your framing** —
  architectural choices (buffer hours, feature weights, bottleneck grid, PA
  10% threshold) were reasonable defaults, not tuned or justified from data.
- **Label quality is a real bottleneck**, not a footnote: alerts came through
  OCR + LLM extraction from PDFs, then fuzzy region-text matching to assign
  alerts to stations. The `liberia` and `finca-3` failures are as plausibly
  label problems as model problems, and the paper itself concludes this.
- **PA-F1 is a lenient metric.** A 10%-of-block detection criterion over a
  large dilated window (up to 168h combined) is not equivalent to "the model
  reliably flags emergencies with useful lead time." Anyone using this as a
  headline claim later should re-derive pointwise/lead-time numbers too.
- **Sample size per station is small** (5 seeds, 4 dims) relative to the
  amount of station-to-station heterogeneity found — the "four operational
  states" framing (viable / sensor-limited / event-limited / data-limited) is
  a useful categorization but was reached from 9 stations, not a large-N result.
- **No comparison against simpler baselines** (e.g. plain statistical
  thresholding, non-recurrent autoencoder, or even a persistence model) — so
  "LSTM-AE reconstruction error works" is really "LSTM-AE reconstruction error
  is *plausible*," not benchmarked against alternatives.
- Data ownership: the µEMA datasets are exclusive property of UCR (LOSIC-UCR
  hardware); this is an academic proof-of-concept only, explicitly **not** an
  official emergency warning system, and outputs must not be treated as
  substitutes for CNE alerts.

## 8. Repo state at the point this prototype stopped

Full Medallion pipeline (raw → silver → gold) implemented and working;
`run_experiments.py` runs the 80-run ablation with multi-GPU seed
distribution; results written as per-seed and aggregated (mean/var) CSVs.
Reference implementation: `github.com/isaac-pm/clima-uema` (AGPLv3 for code;
data excluded from that license, UCR-owned).

## 9. What "going deeper" means for the new project

The new project is a systematic anomaly-detection methodology study on the
µEMA data, not a continuation/patch of the LSTM-AE prototype. Concretely:

- **Compare methods across three families** rather than assuming deep
  learning is the right tool from the start:
  - *Statistical*: e.g. Z-score / modified Z-score (MAD-based), rolling
    statistical thresholds.
  - *Unsupervised ML*: Local Outlier Factor, Isolation Forest, and similar
    density/isolation-based methods.
  - *Deep learning*: Autoencoder (non-recurrent, as a baseline against the
    LSTM version), LSTM/GRU-based sequence models.
- **Explicitly separate anomaly types** rather than treating "anomaly" as one
  bucket, since each method family targets a different type by construction:
  - *Point anomalies* — a single reading far from the norm (Z-score, IForest,
    LOF territory).
  - *Contextual anomalies* — normal in general but abnormal given the local
    context (e.g. time of day/season) — needs the cyclical/time context, not
    just raw values.
  - *Collective/temporal anomalies* — no single point is abnormal, but the
    sequence/pattern is (LSTM/GRU-AE territory; this is what the prototype
    implicitly assumed was the only relevant type).
- **Variable correlation analysis** across pressure, precipitation, and
  luminous intensity — both as its own exploratory step (do sensors move
  together in ways that matter for detecting joint/collective anomalies?) and
  as a possible feature/diagnostic for the multivariate methods above.
- The prototype's findings on data volume and label quality
  (Section 6–7) remain relevant **as things to check for when interpreting
  any method's results** — a "good" score from a data-starved or
  sensor-faulty station means the same thing regardless of which detection
  method produced it — but they are not themselves the object of study this
  time. The object of study is: which method(s) suit which anomaly type(s) in
  this data, and why.
