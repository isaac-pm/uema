# Step 0 — Data Audit

Per-station, per-sensor coverage audit of the raw µEMA CSVs, run before any
downstream analysis. Every downstream method's results are meaningless without
first knowing which stations have enough clean data to evaluate on, so this is
done first rather than discovered later after running a full method
comparison. Produced by `notebooks/00_data_audit.ipynb`
(code in `src/uema/audit.py`). Regenerate by re-running that notebook —
`coverage_table.csv`, `station_recommendation.csv`, and `timeline.png` in this
directory are its outputs, not hand-edited.

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
Font and color palette are the project-wide fixed style (`src/uema/style.py`)
— see that module for the palette hexes and font fallback chain.

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

## Result

| Station                  | Decision           | Rationale                                                                                                                                          |
| ------------------------ | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| recinto-esparza          | **GO**             | all sensors within thresholds                                                                                                                      |
| sede-atlantico_turrialba | **GO**             | all sensors within thresholds                                                                                                                      |
| sede-caribe_limon        | **GO**             | all sensors within thresholds                                                                                                                      |
| sede-central_finca-1     | **GO**             | all sensors within thresholds                                                                                                                      |
| sede-guanacaste_liberia  | **GO**             | all sensors within thresholds                                                                                                                      |
| sede-sur_golfito         | **GO**             | all sensors within thresholds                                                                                                                      |
| recinto-santa-cruz       | **CONDITIONAL-GO** | pressure & luminous_intensity degraded (~47% missing, gaps up to 42 days) — precipitation only if using this station without special gap handling  |
| sede-central_finca-2     | **CONDITIONAL-GO** | pressure & luminous_intensity degraded (~39% missing) — see the raw README for the documented luminous_intensity outage before 2025-05-20          |
| sede-central_finca-3     | **CONDITIONAL-GO** | pressure & luminous_intensity degraded (~51% missing, gaps up to 106 days) — precipitation only if using this station without special gap handling |
| recinto-guapiles         | **NO-GO**          | pressure sensor span is 0.12 days (18 rows total) — under the 90-day minimum, effectively no usable pressure history                               |

6 GO, 3 CONDITIONAL-GO, 1 NO-GO.

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
