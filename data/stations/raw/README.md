# Station Raw Data — Origin, Assumptions & Processing

## Data Origin

Raw meteorological data comes from the **UCR-µEMA network** (Universidad de Costa
Rica, *Micro Estaciones Meteorológicas Automáticas*), 13 low-cost weather
stations deployed across Costa Rica. Each station reports three sensor
channels:

| Feature              | Unit | Notes                                         |
| -------------------- | ---- | --------------------------------------------- |
| Atmospheric pressure | hPa  | Calibration-corrected per station (see below) |
| Precipitation        | mm   | Accumulated per 10-minute bin                 |
| Luminous intensity   | lux  | Raw sensor reading                            |

Data is collected via MQTT into an InfluxDB instance and queried through a
Grafana proxy (`relampagos.ucr.ac.cr`) using Flux queries. It was pulled with a
one-off downloader tool (not included in this project) that issued queries in
30-day chunks per station/feature to avoid timing out the Grafana API, and
saved the results as **one CSV per station per feature**.

This project stores the downloaded CSVs verbatim, flattened into a single
`data/stations/raw/` folder (no per-feature subfolders), one file per
station/feature combination.

## Stations

13 stations, identified by a `filename` slug (also used as the CSV filename
component) and an index prefix `00`–`12`. Indices `10`–`12` came online after
the original ten and were added by a later pull:

- 00 `sede-central_finca-1`
- 01 `recinto-esparza`
- 02 `sede-sur_golfito`
- 03 `recinto-guapiles`
- 04 `sede-guanacaste_liberia`
- 05 `sede-caribe_limon`
- 06 `sede-atlantico_turrialba`
- 07 `sede-central_finca-2`
- 08 `sede-central_finca-3`
- 09 `recinto-santa-cruz`
- 10 `sede-central_sabanilla`
- 11 `sede-central_losic-norte-1`
- 12 `sede-central_losic-norte-2`

13 stations x 3 channels = 39 CSVs in `data/stations/raw/`.

CSV naming convention: `{index}_{feature}_{station}.csv`, e.g.
`00_pressure_sede-central_finca-1.csv`.

## Processing Applied at Download Time

The following transformations are **already baked into the raw CSVs** — they
were applied server-side in the Flux query before the data was ever written to
disk, so they are not "raw sensor values" in the strictest sense:

1. **10-minute aggregation window.** All series are aggregated with
   `aggregateWindow(every: 10m, ...)`:
   - Precipitation: **summed** per 10-min window (accumulation).
   - Pressure & luminous intensity: **averaged (mean)** per 10-min window.
   - Timestamps in the CSV mark the _end_ of each 10-minute bin.

2. **Precipitation unit conversion.** Raw tipping-bucket counts are multiplied
   by `0.2794` to convert to millimeters (`value * 0.2794`). This constant is the mm-per-tip calibration factor for the bucket hardware used.

3. **Precipitation empty-bin fill.** Windows with no readings are created
   explicitly (`createEmpty: true`) and filled with `0.0` — i.e., "no data
   reported" is assumed to mean "no rain," not treated as missing. This is a
   deliberate assumption: it is reasonable for precipitation but should **not**
   be replicated for other sensor types.

4. **Pressure calibration offset.** A **station-specific constant** (in hPa)
   is added to every raw pressure reading before it is saved, to correct for
   sensor/altitude calibration bias. This offset is _baked into the CSV
   values_ — the numbers in `pressure` CSVs are already corrected, not
   sensor-raw. Offsets used (hPa):

   | Station                  | Offset (hPa) |
   | ------------------------ | ------------ |
   | sede-central_finca-1     | +136.3       |
   | recinto-esparza          | +23.8        |
   | sede-sur_golfito         | +3.1         |
   | recinto-guapiles         | +32.6        |
   | sede-guanacaste_liberia  | +15.3        |
   | sede-caribe_limon        | +7.4         |
   | sede-atlantico_turrialba | +75.1        |
   | sede-central_finca-2     | +138.3       |
   | sede-central_finca-3     | +138.3       |
   | recinto-santa-cruz       | +5.9         |

   The offsets above were recorded for the original ten stations only. The
   offsets applied to `sede-central_sabanilla`, `sede-central_losic-norte-1`
   and `sede-central_losic-norte-2` were not captured when those stations were
   pulled, so they are unknown here — the values in their pressure CSVs are
   still calibration-corrected, but by an amount this file does not record.

5. **Timezone.** Timestamps are converted from UTC (native InfluxDB storage)
   to **Costa Rica local time (UTC−06:00, fixed offset, no DST)** before
   writing. The `time` column format is `YYYY-MM-DD HH:MM:SS`, naive
   (no timezone suffix), but is implicitly CR local time.

6. **Luminous intensity and precipitation had no map/offset transform**
   beyond what's listed above (lux is stored as the raw sensor mean; no
   calibration offset applied).

## CSV Schema

Each raw CSV has exactly two columns:

time,value\_\<unit\>

- `pressure/*.csv` → `time,value_hPa`
- `precipitation/*.csv` → `time,value_mm`
- `luminous_intensity/*.csv` → `time,value_lux`

## Known Data Quality Caveats (inherited, not fixed at download time)

- **Sensor hardware differences.** Some stations use a `BME` pressure sensor, others use `LPS`. 
  The calibration offsets above were derived per-hardware/per-station and are not necessarily 
  interchangeable if a sensor is swapped.
- **`sede-central_finca-2` luminous intensity** is known to be unreliable/
  absent before **2025-05-20 18:20:00** — the sensor was not operational until
  that date. Data before that timestamp should be discarded or flagged, not
  treated as valid zero readings.
- **`recinto-guapiles`** has known data quality issues across all sensors and
  was excluded entirely from downstream modeling in the source project.
- Gaps (missing 10-min bins) can still occur for pressure and luminous
  intensity despite the `fill(value: 0.0)` logic — that fill only applies to
  precipitation. Pressure/lux gaps are true missing data and must be handled
  downstream (interpolation, masking, etc.), not assumed to be zero.
- No outlier filtering, deduplication, or unit sanity-checking was performed
  at download time — CSVs are the direct Flux query output.
- **`sede-sur_golfito` and `sede-guanacaste_liberia`** carry duplicate `time`
  values: a ~30-day block of precipitation (2026-02-19 → 2026-03-21, 4320
  rows, mostly but not entirely matching values on the ~23 bins where actual
  rain was recorded) plus a single stray duplicated timestamp each in
  pressure and luminous_intensity — most likely two overlapping export/append
  runs over the same period, not two genuine readings. `uema.io.load_raw_series`
  drops these, keeping the first occurrence, rather than silently letting them
  break any downstream column-alignment (`pd.concat`) that needs a unique
  index. Worth checking the export/append pipeline for the root cause.

## Time Range

Data was downloaded incrementally over time; per-station coverage windows are
not uniform (some stations came online later, some had sensor outages). Always
check each CSV's actual min/max timestamp rather than assuming a shared range
across stations.
