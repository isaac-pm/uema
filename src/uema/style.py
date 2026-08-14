"""Shared plotting style (font + color palette) for every plot in this project.

Fixed brand palette, six colors. Chosen for identity, not derived from a
generic categorical-design system: `INDIGO`/`BURNT_SIENNA`/`VIOLET` separate
cleanly for colorblind viewers (CVD deltaE 21-30, normal-vision deltaE 27-32
on all pairs); `INDIGO` and `AZURE` sit outside a "loud" categorical
lightness/chroma band and `BURNT_SIENNA`/`MATCHA`/`AZURE` fall under 3:1
contrast on white — so charts using this palette must always carry a visible
legend/direct labels rather than relying on color alone (already the default
for every plot in this project).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
FONT_DIR = REPO_ROOT / "assets" / "fonts"

PALETTE = {
    "matcha": "#8FB78F",
    "amethyst": "#8B8DC1",
    "indigo": "#28536B",
    "azure": "#B4CCCF",
    "burnt_sienna": "#E97451",
    "violet": "#6A3D9A",
}

# Fixed draw order for categorical series across all project plots — assign
# colors by this order, never cycle/reassign when a series count changes.
PALETTE_ORDER = ["indigo", "burnt_sienna", "violet", "matcha", "amethyst", "azure"]

_FONT_FALLBACK_CHAIN = [
    ("Times New Roman", FONT_DIR / "Times_New_Roman.ttf"),
    ("Liberation Serif", FONT_DIR / "Liberation_Serif.ttf"),
]

DPI = 300


def categorical_colors(n: int | None = None) -> list[str]:
    """Hex colors for n categorical series, in the fixed palette order."""
    names = PALETTE_ORDER if n is None else PALETTE_ORDER[:n]
    return [PALETTE[name] for name in names]


def apply_style() -> None:
    """Register the project font and palette as matplotlib defaults.

    Call once per notebook/script before plotting. Font falls back
    Times New Roman -> Liberation Serif -> generic serif, depending on
    what's available under assets/fonts/.
    """
    serif_names = []
    for name, path in _FONT_FALLBACK_CHAIN:
        if path.exists():
            fm.fontManager.addfont(str(path))
            serif_names.append(name)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = serif_names + ["Liberation Serif", "serif"]
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=categorical_colors())
    plt.rcParams["figure.dpi"] = DPI
    plt.rcParams["savefig.dpi"] = DPI
