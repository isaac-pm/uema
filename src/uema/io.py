"""Discovery and loading of raw µEMA station CSVs.

See data/stations/raw/README.md for the full data-origin and caveats
documentation this module assumes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "stations" / "raw"

FEATURES = ("pressure", "precipitation", "luminous_intensity")

VALUE_COLUMNS = {
    "pressure": "value_hPa",
    "precipitation": "value_mm",
    "luminous_intensity": "value_lux",
}

SAMPLING_INTERVAL = pd.Timedelta(minutes=10)

_FILENAME_RE = re.compile(
    r"^(?P<index>\d{2})_(?P<feature>pressure|precipitation|luminous_intensity)_(?P<station>.+)\.csv$"
)


@dataclass(frozen=True)
class RawFile:
    index: str
    feature: str
    station: str
    path: Path


def discover_raw_files(raw_dir: Path = RAW_DIR) -> list[RawFile]:
    """Find and parse all raw station/sensor CSVs in raw_dir."""
    files = []
    for path in sorted(raw_dir.glob("*.csv")):
        match = _FILENAME_RE.match(path.name)
        if match is None:
            continue
        files.append(
            RawFile(
                index=match["index"],
                feature=match["feature"],
                station=match["station"],
                path=path,
            )
        )
    return files


def list_stations(raw_dir: Path = RAW_DIR) -> dict[str, str]:
    """Map station index ('00'..'09') to station slug, in index order."""
    stations = {f.index: f.station for f in discover_raw_files(raw_dir)}
    return dict(sorted(stations.items()))


def load_raw_series(raw_file: RawFile) -> pd.Series:
    """Load a single raw CSV as a time-indexed Series, sorted by time.

    A handful of raw files carry duplicate `time` values — observed as a
    duplicated ~30-day block in `sede-sur_golfito`/`sede-guanacaste_liberia`
    precipitation, plus a stray single-timestamp duplicate in their
    pressure/luminous_intensity — most likely two overlapping export/append
    runs over the same period rather than two genuine readings. Downstream
    code (`uema.silver.consolidate_stations`) needs a unique DatetimeIndex to
    column-align sensors, so duplicates are dropped here, keeping the first
    occurrence. This is a documented, deterministic choice, not a fix to the
    raw CSVs — flagged as a caveat in data/stations/raw/README.md.
    """
    df = pd.read_csv(raw_file.path, parse_dates=["time"])
    value_col = VALUE_COLUMNS[raw_file.feature]
    series = df.set_index("time")[value_col].sort_index()
    series = series[~series.index.duplicated(keep="first")]
    series.name = value_col
    return series
