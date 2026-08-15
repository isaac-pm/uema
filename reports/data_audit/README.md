# Step 0 — Data Audit

Per-station, per-sensor coverage audit of the raw µEMA CSVs, run before any
downstream analysis. Every downstream method's results are meaningless without
first knowing which stations have enough clean data to evaluate on, so this is
done first rather than discovered later after running a full method
comparison. Produced by `notebooks/00_data_audit.ipynb`
(code in `src/uema/audit.py`). Regenerate by re-running that notebook —
`coverage_table.csv`, `station_recommendation.csv`, `go_station_date_ranges.csv`,
`timeline.png`, and `timeline_grouped.png` in this directory are its outputs,
not hand-edited.

## Method

For each of the 10 stations' 3 raw sensor CSVs (`pressure`, `precipitation`,
`luminous_intensity`):

- time range (`start`, `end`), and `span_days` between them
- `expected_rows` — number of 10-minute bins the span should contain
- `actual_rows`, `missing_rows`, `pct_missing`
- `n_gaps` — count of breaks longer than one sampling interval
- `max_gap` — the single largest gap

`timeline.png` plots this as filled/gap segments per station × sensor, so
coverage and gaps are visible directly rather than inferred from a table.
`timeline_grouped.png` is the same per-sensor segments, but split into three
blocks by `windowed_decision` (see below): GO, WINDOWED-GO, and NO-GO, each
sorted by overall data availability (`uema.audit.station_availability` —
actual/expected rows summed across a station's three sensors), highest
first, with a dashed divider between blocks. WINDOWED-GO stations additionally
get a dashed bracket outlining their best common window across all three
sensor rows. Font and color palette are the project-wide fixed style
(`src/uema/style.py`) — see that module for the palette hexes and font
fallback chain.

### Windowed layer

`decision` (below) is computed over each sensor's *entire* recorded span, so
it can't distinguish "uniformly mediocre for its whole history" from "has a
long, clean sub-period inside a longer patchy record." `uema.audit.find_best_common_window`
adds a second, independent layer: for a station, it computes a trailing
30-day rolling %missing per sensor over the intersection of all sensors'
recorded spans, marks bins where that stays under the same 25%-missing
threshold used above, intersects across the three sensors, and returns the
longest resulting contiguous stretch (if any clears the same 90-day minimum
span). This yields `best_window_start`, `best_window_end`, `best_window_days`,
and a derived `windowed_decision`: GO stations keep `"GO"` (their whole span
already applies); everything else becomes `"WINDOWED-GO"` if a valid window
exists, else stays `"NO-GO"`. This runs for every station, including GO
ones — for GO stations the recovered window should closely track their
existing full span, which is a useful sanity check that the windowed method
agrees with the whole-span method where the whole-span method already says
"clean."

`windowed_decision` is independent of `MANUAL_OVERRIDES` (below): overriding
a station's whole-span `decision` doesn't suppress or force its windowed
result, so a manually-overridden station's window (if any) still shows up
in the table rather than being silently masked.

`go_station_date_ranges.csv` gives, for each `decision == "GO"` station, the
single date range where all three sensors are simultaneously present (max of
each sensor's own start, min of each sensor's own end) — not any individual
sensor's own, wider span. Any analysis using a GO station's full sensor set
at once needs to stay within this range.

## Go/no-go thresholds

Deliberately about raw coverage only — not modeling-readiness (window counts,
label alignment), which is out of scope for this phase. A sensor is:

- **absent** — 0 rows, or ≥90% of expected bins missing
- **short-history** — span under 90 days
- **degraded** — >25% of expected bins missing
- **ok** — otherwise

A station is:

- **NO-GO** — its pressure sensor (the one every station is expected to
  have) is absent or short-history
- **CONDITIONAL-GO** — usable, but one or more sensors are degraded /
  short-history / absent and must be scoped around explicitly (see rationale
  per station below) — not silently dropped later in a pipeline
- **GO** — all three sensors clean

These are then subject to `uema.audit.MANUAL_OVERRIDES` — an explicit,
editable dict of `station -> (decision, reason)` for analyst judgment calls
that override the numeric-threshold decision (e.g. a station that clears
the thresholds on paper but is known/observed to be unreliable). Currently:

| Station              | Override | Reason                                                  |
| --------------------- | -------- | -------------------------------------------------------- |
| `sede-caribe_limon`   | NO-GO    | flagged by manual review despite clearing numeric thresholds |

Edit `MANUAL_OVERRIDES` in `src/uema/audit.py` to add/remove/change these as
the assessment changes; every entry there is reflected automatically in
`station_recommendation.csv` and both timeline figures.

## Result

| Station                  | Decision           | Windowed decision | Best window            | Rationale                                                                                                                                          |
| ------------------------ | ------------------ | ------------------ | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| recinto-esparza          | **GO**             | GO                  | (full span, 300 days)   | all sensors within thresholds                                                                                                                      |
| sede-atlantico_turrialba | **GO**             | GO                  | (full span, 499 days)   | all sensors within thresholds                                                                                                                      |
| sede-central_finca-1     | **GO**             | GO                  | (full span, 112 days)   | all sensors within thresholds                                                                                                                      |
| sede-guanacaste_liberia  | **GO**             | GO                  | (full span, 266 days)   | all sensors within thresholds                                                                                                                      |
| sede-sur_golfito         | **GO**             | GO                  | (full span, 234 days)   | all sensors within thresholds                                                                                                                      |
| recinto-santa-cruz       | CONDITIONAL-GO      | **WINDOWED-GO**     | 2025-11-30 to 2026-03-01 (91 days) | pressure & luminous_intensity degraded (~47% missing, gaps up to 42 days) over the whole span, but has a 91-day clean common window at the end     |
| sede-central_finca-2     | CONDITIONAL-GO      | **NO-GO**           | none found              | pressure & luminous_intensity degraded (~39% missing) — see the raw README for the documented luminous_intensity outage before 2025-05-20; no 90-day common window clears the threshold |
| sede-central_finca-3     | CONDITIONAL-GO      | **NO-GO**           | none found              | pressure & luminous_intensity degraded (~51% missing, gaps up to 106 days); no 90-day common window clears the threshold                          |
| recinto-guapiles         | **NO-GO**           | NO-GO               | none found              | pressure sensor span is 0.12 days (18 rows total) — under the 90-day minimum, effectively no usable pressure history                               |
| sede-caribe_limon        | **NO-GO**           | NO-GO               | none found              | manual override — clears whole-span numeric thresholds but flagged by manual review (see `MANUAL_OVERRIDES` above); frequent short pressure gaps also mean no 90-day windowed stretch clears the threshold either, reinforcing the override |

Whole-span: 5 GO, 3 CONDITIONAL-GO, 2 NO-GO (one of the NO-GOs is a manual
override, not a threshold failure — see above). Windowed: 5 GO, 1
WINDOWED-GO, 4 NO-GO — one former CONDITIONAL-GO station (`recinto-santa-cruz`)
has a usable 91-day window, but the other two CONDITIONAL-GO stations and
both original NO-GOs don't clear the windowed bar either.

## GO station date ranges

`go_station_date_ranges.csv` — the range where all three sensors are
simultaneously present, for each `decision == "GO"` station:

| Station                  | Start               | End                 | Span (days) |
| ------------------------ | ------------------- | ------------------- | ----------- |
| recinto-esparza          | 2024-11-21 15:00:00 | 2026-03-01 00:00:00 | 464.4       |
| sede-atlantico_turrialba | 2024-10-03 11:40:00 | 2026-03-01 00:00:00 | 513.5       |
| sede-central_finca-1     | 2025-11-08 17:40:00 | 2026-03-01 00:00:00 | 112.3       |
| sede-guanacaste_liberia  | 2024-12-12 16:10:00 | 2026-02-10 00:00:00 | 424.3       |
| sede-sur_golfito         | 2024-12-12 16:10:00 | 2026-02-18 10:10:00 | 432.8       |

## Notable findings (beyond the per-station verdict)

- **Pressure and luminous_intensity track together, station by station.**
  At every station except `sede-caribe_limon` and `recinto-guapiles`, the two
  sensors' start time, end time, gap count, and %missing are identical or
  near-identical. This isn't documented in `data/stations/raw/README.md` and
  suggests a shared cause (e.g. power/connectivity outages affecting the
  whole station board, not one sensor) rather than two independent sensor
  histories. Worth confirming with whoever runs the hardware before relying
  on "only the lux sensor was affected" framing for any station other than
  the one case the README does call out (`sede-central_finca-2` lux,
  pre-2025-05-20).
