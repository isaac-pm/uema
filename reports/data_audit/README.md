# Step 0 — Data Audit

Per-station, per-sensor coverage audit of the raw µEMA CSVs, run before any
downstream analysis. Every downstream method's results are meaningless without
first knowing which stations have enough clean data to evaluate on, so this is
done first rather than discovered later after running a full method
comparison. Produced by `notebooks/00_data_audit.ipynb`
(code in `src/uema/audit.py`). Regenerate by re-running that notebook —
`coverage_table.csv`, `station_recommendation.csv`, `timeline.png`, and
`timeline_grouped.png` in this directory are its outputs, not hand-edited.

## Method

For each of the 13 stations' 3 raw sensor CSVs (`pressure`, `precipitation`,
`luminous_intensity`), `coverage_table.csv` reports each sensor's own,
independent recorded history: time range (`start`, `end`) and `span_days`
between them, `expected_rows`/`actual_rows`/`missing_rows`/`pct_missing`,
`n_gaps` (breaks longer than one sampling interval), and `max_gap`. This is
informational context — "how much total history does this sensor have" — not
what the go/no-go decision is scored on (see below).

`timeline.png` plots this as filled/gap segments per station × sensor, so
coverage and gaps are visible directly rather than inferred from a table.
`timeline_grouped.png` is the same per-sensor segments, but split into three
blocks by `decision` (see below): GO, CONDITIONAL-GO, and NO-GO, each sorted
by overall data availability (`uema.audit.station_availability` —
actual/expected rows summed across a station's three sensors), highest
first, with a dashed divider between blocks. Font and color palette are the
project-wide fixed style (`src/uema/style.py`) — see that module for the
palette hexes and font fallback chain.

### Scoring on the joint window, not each sensor's own span

An earlier version of this audit tiered each sensor against its own
individual recorded span. That's the wrong axis for a *joint* go/no-go call:
a sensor with a long, mostly-clean individual history but only a short
overlap with its sibling sensors would pass a tier it doesn't deserve for
joint analysis, since the extra history outside the overlap can't be used
for anything needing all three sensors together. `station_recommendation.csv`
scores each sensor's completeness against the station's **joint window** —
`joint_span_start`/`joint_span_end`, the intersection of all three sensors'
recorded spans (max of each sensor's own start, min of each sensor's own
end) — instead. A sensor tier is:

- **absent** — 0 rows in the joint window, or ≥90% of its expected bins
  missing
- **short-history** — the joint window itself is under 90 days (same for
  every sensor at a station, since it's a shared window)
- **degraded** — either >25% of expected bins missing within the joint
  window, **or** the single longest gap inside it exceeds 7 days — even a
  low aggregate %missing can hide one unbridgeable multi-day outage
  (`silver.py` only linearly bridges gaps ≤6h), so gap length is checked
  independently of the aggregate
- **ok** — otherwise

A station is:

- **NO-GO** — its sensors never overlap at all, or its pressure sensor (the
  one every station is expected to have) is absent or short-history within
  the joint window
- **CONDITIONAL-GO** — usable, but one or more sensors are degraded /
  short-history / absent within the joint window, or the joint window has
  thin coverage of one Costa Rica season (see below) — scoped around
  explicitly (see rationale per station below), not silently dropped later
  in a pipeline
- **GO** — all three sensors clean within the joint window, and both
  seasons are adequately represented

A GO station is further checked for **season coverage**: if the joint
window has fewer than 30 days of either Costa Rica dry-season (Dec–Apr) or
wet-season (May–Nov) days (`uema.correlation.season_bucket`), it's
downgraded to CONDITIONAL-GO. Step 1 (`uema.correlation`) conditions some
correlations on dry/wet season, and a window sitting almost entirely in one
season would silently bias that split without this check — `dry_days`/
`wet_days` in `station_recommendation.csv` report the count for every
station regardless of decision.

`decision` is the **only** go/no-go signal — there is deliberately no
second "windowed" layer searching for a per-station optimal clean
sub-window (an earlier version of this audit had one). That's closer to
breakpoint/homogeneity analysis than standard coverage gating, and it let a
station that fails on its own numbers get quietly re-admitted the moment a
new data pull happened to contain a rescuing stretch. Standard
meteorological practice is simpler and is what this audit does instead:
accept a station's actual joint window as-is, interpolate only short gaps
downstream (`uema.silver.fill_gaps`, ≤6h), leave longer gaps as missing
(handled by pairwise exclusion in `uema.correlation`), and gate on the
completeness thresholds above over that one window.

These are then subject to `uema.audit.MANUAL_OVERRIDES` — an explicit,
editable dict of `station -> (decision, reason)` for analyst judgment calls
that override the numeric-threshold decision (e.g. a station that clears
the thresholds on paper but is known/observed to be unreliable). Currently
empty — every station's `decision` below is the unmodified numeric-threshold
result.

Edit `MANUAL_OVERRIDES` in `src/uema/audit.py` to add entries as the
assessment changes; every entry there is reflected automatically in
`station_recommendation.csv` and both timeline figures.

## Result

| Station                      | Decision            | Joint span                            | Rationale |
| ----------------------------- | -------------------- | -------------------------------------- | --------- |
| recinto-esparza                | **GO**               | 2024-11-21 → 2026-08-16 (632.4 days)  | all sensors within coverage thresholds |
| sede-atlantico_turrialba       | **GO**               | 2024-10-03 → 2026-08-16 (681.5 days)  | all sensors within coverage thresholds |
| sede-central_finca-1           | **GO**               | 2025-11-08 → 2026-08-16 (280.3 days)  | all sensors within coverage thresholds |
| recinto-santa-cruz             | CONDITIONAL-GO       | 2024-12-12 → 2026-08-16 (611.3 days)  | pressure & luminous_intensity degraded within the joint window |
| sede-central_finca-2           | CONDITIONAL-GO       | 2025-04-30 → 2026-07-21 (447.3 days)  | pressure & luminous_intensity degraded — see raw README's documented lux outage before 2025-05-20 |
| sede-central_finca-3           | CONDITIONAL-GO       | 2024-12-12 → 2026-08-16 (611.3 days)  | pressure, precipitation & luminous_intensity all degraded within the joint window |
| sede-central_losic-norte-1     | CONDITIONAL-GO*      | 2026-04-07 → 2026-08-16 (130.2 days)  | all sensors clean, but only 24 dry-season days in the joint window (<30d) |
| sede-central_losic-norte-2     | CONDITIONAL-GO       | 2025-10-12 → 2026-04-04 (174.3 days)  | pressure, precipitation & luminous_intensity all degraded within the joint window |
| sede-central_sabanilla         | CONDITIONAL-GO*      | 2026-04-29 → 2026-08-16 (108.6 days)  | all sensors clean, but only 2 dry-season days in the joint window (<30d) |
| sede-guanacaste_liberia        | CONDITIONAL-GO       | 2024-12-12 → 2026-08-16 (611.3 days)  | pressure, precipitation & luminous_intensity all degraded within the joint window |
| sede-sur_golfito                | CONDITIONAL-GO       | 2024-12-12 → 2026-08-16 (611.3 days)  | pressure & luminous_intensity degraded within the joint window |
| sede-caribe_limon               | CONDITIONAL-GO       | 2025-11-19 → 2026-08-16 (269.5 days)  | pressure & luminous_intensity degraded within the joint window |
| recinto-guapiles                | **NO-GO**            | 2025-11-19 → 2025-11-19 (0.1 days)    | joint sensor overlap is 0.12 days — effectively no usable joint history |

\* CONDITIONAL-GO purely on the season-coverage check, not on any sensor
tier — all three sensors are individually `ok`.

3 GO, 9 CONDITIONAL-GO, 1 NO-GO. Every GO/CONDITIONAL-GO station (12 of 13)
is used over its full joint span in Step 1 — accept the blocky missingness
the CONDITIONAL-GO rationale describes, don't clip to a shorter clean
sub-window.

## Notable findings (beyond the per-station verdict)

- **Pressure and luminous_intensity track together, station by station.**
  At most stations the two sensors' gap count and %missing within the joint
  window are identical or near-identical (exceptions: `recinto-guapiles`
  and `sede-central_losic-norte-2`, where precipitation is also degraded).
  This isn't documented in `data/stations/raw/README.md`
  and suggests a shared cause (e.g. power/connectivity outages affecting
  the whole station board, not one sensor) rather than two independent
  sensor histories.
- **Joint span length says nothing about how clean it is.** Several
  stations (`sede-central_finca-3`, `sede-guanacaste_liberia`,
  `sede-sur_golfito`) have a joint span over 600 days but 30-45% missing
  within it — a lot of history, most of it gappy. Step 1's post-fill
  missingness table (`reports/variable_correlation/post_fill_missingness.csv`)
  makes this concrete per station: it ranges from ~1% (`sede-central_sabanilla`)
  to ~51% (`sede-central_losic-norte-2`) even after short-gap interpolation,
  and that spread directly affects how much any given station's correlation
  results can be trusted (see `reports/variable_correlation/README.md`).
