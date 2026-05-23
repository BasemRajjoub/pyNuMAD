"""Colour palette + chord-group classification for blade visualisations.

A single source of truth for "what colour is HP_TE_REINF" so that every
plot in this package — interactive HTML, PNG snapshots, continuity
reports — uses the same convention. If you want a different palette,
override ``GROUP_COLORS`` after import (it is a plain dict).

The CHORD_GROUP_TAGS order is important: ``classify_set_name`` walks
it in order and returns the first matching tag, so longer / more
specific labels must come before their substrings.  ``HP_TE_REINF``
must be checked before ``HP_TE`` (which doesn't actually exist as a
tag here but illustrates the principle).
"""
from __future__ import annotations


# Order matters: longer / more specific tags first. The classifier
# below relies on this to disambiguate substrings (HP_TE_REINF vs
# HP_TE_PANEL etc.).
CHORD_GROUP_TAGS: tuple[str, ...] = (
    "HP_TE_REINF", "LP_TE_REINF",
    "HP_TE_PANEL", "LP_TE_PANEL",
    "HP_LE_PANEL", "LP_LE_PANEL",
    "HP_TE_FLAT",  "LP_TE_FLAT",
    "HP_SPAR",     "LP_SPAR",
    "HP_LE",       "LP_LE",
    "SW",
)

# 13-colour qualitative palette (tab10 + a few extras).  Distinct
# enough to tell adjacent zones apart in a 3D view.
GROUP_COLORS: dict[str, str] = {
    "HP_SPAR":      "#d62728",  # red
    "LP_SPAR":      "#1f77b4",  # blue
    "HP_TE_REINF":  "#e377c2",  # pink
    "LP_TE_REINF":  "#9467bd",  # purple
    "HP_LE":        "#2ca02c",  # green
    "LP_LE":        "#17becf",  # cyan
    "HP_TE_PANEL":  "#bcbd22",  # olive
    "LP_TE_PANEL":  "#7f7f7f",  # grey
    "HP_LE_PANEL":  "#ff7f0e",  # orange
    "LP_LE_PANEL":  "#aec7e8",  # light blue
    "HP_TE_FLAT":   "#8c564b",  # brown
    "LP_TE_FLAT":   "#ffbb78",  # light orange
    "SW":           "#7c3f00",  # dark brown
}

# Bondline colours.  Vivid so they stand out against the shell.
ADHESIVE_COLORS: dict[str, str] = {
    "TE_BOND": "#ff0000",
    "LE_BOND": "#0000ff",
}
DEFAULT_ADHESIVE_COLOR = "#ff00ff"  # magenta — for unrecognised tags

# Legend grouping so the HTML legend can collapse 13 entries into a
# few logical buckets.
LEGEND_GROUPS: dict[str, str] = {
    "HP_SPAR": "Spar caps",
    "LP_SPAR": "Spar caps",
    "HP_TE_REINF": "TE reinforcements",
    "LP_TE_REINF": "TE reinforcements",
    "HP_LE": "LE chord strips",
    "LP_LE": "LE chord strips",
    "HP_TE_PANEL": "Trailing-edge panels",
    "LP_TE_PANEL": "Trailing-edge panels",
    "HP_LE_PANEL": "Leading-edge panels",
    "LP_LE_PANEL": "Leading-edge panels",
    "HP_TE_FLAT": "TE flats",
    "LP_TE_FLAT": "TE flats",
    "SW": "Shear webs",
}


def classify_set_name(set_name: str) -> str | None:
    """Map an element-set name (``01_30_HP_TE_REINF``) to its chord group.

    Returns the matching tag from ``CHORD_GROUP_TAGS`` (e.g.
    ``"HP_TE_REINF"``) or ``None`` if no tag matches.  Adhesive sets
    (``TE_BOND``, ``LE_BOND``) are intentionally not classified here —
    they are handled separately by ``classify_adhesive_set_name``.
    """
    for tag in CHORD_GROUP_TAGS:
        if tag in set_name:
            return tag
    return None


def classify_adhesive_set_name(set_name: str) -> str | None:
    """Map an adhesive set name to ``"TE_BOND"`` or ``"LE_BOND"``."""
    if set_name.startswith("TE_BOND"):
        return "TE_BOND"
    if set_name.startswith("LE_BOND"):
        return "LE_BOND"
    return None


def get_color(tag: str, default: str = "#cccccc") -> str:
    """Look up the palette entry for a chord-group or adhesive tag."""
    if tag in GROUP_COLORS:
        return GROUP_COLORS[tag]
    if tag in ADHESIVE_COLORS:
        return ADHESIVE_COLORS[tag]
    return default
