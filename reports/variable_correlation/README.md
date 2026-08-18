# Step 1 — Variable Correlation

How `pressure`, `precipitation`, and `luminous_intensity` relate to each
other at the stations the Step 0 audit rates in scope for analysis: every
**GO** or **CONDITIONAL-GO** station, each used over its full joint
sensor-overlap span (the intersection of all three sensors' recorded
spans) — no per-station rescue-window trimming. The roster is derived at
run time from `reports/data_audit/station_recommendation.csv`
(`uema.silver.analysis_stations`) — see `station_scope.csv` in this
directory for the exact stations and date ranges used on the run that
produced these results. As of this run: 12 of 13 stations (3 GO, 9
CONDITIONAL-GO); `recinto-guapiles` is NO-GO and excluded. Produced by
`notebooks/01_variable_correlation.ipynb` (data prep
in `src/uema/silver.py`, correlation logic in `src/uema/correlation.py`).
Regenerate by re-running that notebook — every file in this directory is
its output, not hand-edited.

Step 0's go/no-go methodology was simplified since an earlier version of
this report: there is no longer a "windowed" layer that searched each
station for an optimal rolling clean sub-window and clipped CONDITIONAL-GO
stations to it. Standard meteorological practice is to accept a station's
actual joint window as-is — interpolate short gaps, leave longer gaps as
missing, gate on straightforward completeness thresholds — rather than
hunt for a bespoke best window per station. The direct consequence: 9 of
the 12 in-scope stations here now run over their full, sometimes gappy,
joint span (post-fill missingness up to ~51% at the worst station — see
`post_fill_missingness.csv`) rather than a clipped clean window. That
tradeoff is visible in the Result section below, not hidden.

## Method

Each in-scope station's three raw sensor CSVs are consolidated
(`uema.silver.consolidate_stations`) onto a shared 10-minute grid, trimmed to
its full joint sensor-overlap span, and short gaps (<= 6h, a judgment call,
not a universal constant) are linearly bridged; longer gaps are left as
`NaN` and pairwise-excluded by the correlation calls rather than filled.

Per station: Pearson and Spearman correlation for all three sensor pairs
(`*_pearson.csv`, `*_spearman.csv`, `*_correlation_heatmap.png`); cross-
correlation across a +-6h lag window per pair (`*_ccf_<var1>_<var2>.csv`,
`*_ccf.png`, peak lags summarized in `ccf_peak_summary.csv`); and correlation
matrices conditioned on time-of-day (day 06:00-17:59 / night 18:00-05:59,
local) and Costa Rica's two-season convention (dry Dec-Apr / wet May-Nov)
(`*_<hour|season>_<bucket>_pearson.csv`, `*_<hour|season>_<bucket>_spearman.csv`,
summarized in `conditioned_correlation_summary.csv` and
`conditioned_correlation_bars.png`).

Spearman is treated as the primary statistic for any pair involving
`precipitation` (zero-inflated — most 10-minute bins are exactly 0.0mm, which
distorts Pearson); Pearson is still reported for every pair for comparison.
`primary_correlation_summary.csv` pulls out just the primary-method value per
station × pair. `post_fill_missingness.csv` reports each station's %missing
after short-gap interpolation — the single most useful number for judging
how much to trust any given station's correlation results, now that stations
aren't pre-filtered to a clean window.

Region labels (`REGION` in the notebook, from the station-slug prefix) are
shown for context only, not used as a grouping key — matrices are computed
per station and not pooled into one "average station." Several in-scope
stations share a region prefix (`recinto-esparza`/`recinto-santa-cruz`;
`sede-central_finca-1`/`finca-2`/`finca-3`/`losic-norte-1`/`losic-norte-2`/
`sabanilla`), so region is a loose contextual label, not evidence of
independence.

## Result

Global (lag-0) correlation is small everywhere: the primary statistic never
exceeds `|r| = 0.14` across all 12 stations x 3 pairs
(`primary_correlation_summary.csv`; the single largest value is
`pressure`-`luminous_intensity` at `sede-central_finca-3`, `r = 0.14`,
Pearson). Still far from what would justify treating the three sensors as
one coupled multivariate system by default.

One conditional exception is real, but its strength now tracks data
completeness closely enough to need an explicit caveat:

- **`pressure`-`luminous_intensity` lagged coupling is clear only at the
  cleanest stations.** Sorting `ccf_peak_summary.csv` by each station's
  post-fill %missing shows a sharp split. The five stations under 5%
  missing all show a clear peak (`r = -0.24` to `-0.43`, clustering
  3.3-4.5h lag): `sede-central_sabanilla` (-0.32, 1.2% missing),
  `sede-atlantico_turrialba` (-0.24, 2.1%), `recinto-esparza` (-0.34,
  3.1%), `sede-central_losic-norte-1` (-0.35, 3.6%), `sede-central_finca-1`
  (-0.43, 4.2%). Every station above ~11% missing instead mostly shows a
  weak, inconsistent, or sign-flipped peak: `sede-sur_golfito` (-0.12,
  11.3%), `sede-central_finca-2` (-0.08, 26.9%), `sede-guanacaste_liberia`
  (+0.06, 28.3%), `recinto-santa-cruz` (-0.06, 33.0%), `sede-central_finca-3`
  (+0.21, 36.4% — opposite sign), `sede-central_losic-norte-2` (-0.11,
  51.0%) — with one exception, `sede-caribe_limon` (-0.28, 20.7% missing),
  whose peak is comparable in strength to the cleanest stations, so
  missingness alone doesn't fully explain when this coupling survives. The
  overall split is still the direct, visible cost of dropping the
  windowed-rescue mechanism: real signal at clean stations gets washed out
  (or produces artifacts, as at `finca-3`) once missingness climbs past
  roughly 10-15%, just not uniformly so.
- **`precipitation`-`luminous_intensity` daytime coupling holds at all 12
  stations regardless of completeness.** Daytime `r` ranges `-0.10`
  (`sede-guanacaste_liberia`) to `-0.26` (`sede-sur_golfito`) vs. `|r| <= 0.08`
  at night at every station (`conditioned_correlation_summary.csv`) —
  consistent with rain clouds suppressing daytime light readings. Unlike
  the lagged pressure-lux coupling, this pattern is robust even at the
  messiest stations, suggesting it's a coarser effect than a precise
  multi-hour lag estimate.

**Feature strategy decision** (full reasoning and numbers in the notebook's
final cell): treat the three sensors as independent per-sensor series by
default for Step 3, but carry a lagged/joint feature for
`pressure`-`luminous_intensity` weighted by each station's data completeness
(strong evidence under 5% missing, unreliable above ~10-15%) rather than as
a uniform network-wide constant, and account for the daytime-conditioned
`precipitation`-`luminous_intensity` coupling in any time-of-day-sensitive
method — that one holds regardless of completeness.
