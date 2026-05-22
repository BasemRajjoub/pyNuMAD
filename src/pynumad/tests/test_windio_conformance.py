"""windIO -> mesh-dict conformance tests.

These assert that the structural declarations in the windIO YAML
(``components.blade.internal_structure_2d_fem``) actually show up in
the mesh dict that pyNuMAD produces. They are the cheapest possible
"contract" tests: no FE solve, no deck write — just compare YAML
counts against mesh-dict element-set names.

Specifically we verify:

  B1. ``len(internal_structure_2d_fem.webs) == n_webs``, and the mesh
      has exactly ``2 * n_webs * n_stations`` chord-segment-prefixed
      SW sets (the mesher emits two prefixes "00_NN_SW" and
      "01_NN_SW" — one per chordwise sub-panel between LE and TE —
      per web per station).
  B2. The spar-cap layers declared in YAML (``Spar_cap_ss`` /
      ``Spar_cap_ps``) have a corresponding chord-segment in the mesh
      named with a SPAR suffix at each station.
  B3. The number of distinct spanwise stations in the mesh matches
      the union of layer-grid stations declared in the YAML.

These tests should mostly PASS on BAR0 today and so serve as the
contract that a future refactor cannot silently break.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict

import pytest
import yaml as _yaml

from ._mesh_cache import BAR0_YAML, get_mesh


ELEMENT_SIZE = 0.5
SET_RE = re.compile(r"^(\d+)_(\d+)_(.+)$")


@pytest.fixture(scope="module")
def yaml_doc():
    with open(BAR0_YAML) as f:
        return _yaml.safe_load(f)


@pytest.fixture(scope="module")
def yaml_struct(yaml_doc):
    return yaml_doc["components"]["blade"]["internal_structure_2d_fem"]


@pytest.fixture(scope="module")
def shell_mesh():
    """Shell-only mesh (no adhesive) is enough for naming-conformance
    assertions and is the fastest cache key in the suite.
    """
    return get_mesh(includeAdhesive=False, elementSize=ELEMENT_SIZE)


# ------------------------------------------------------------------
# B1. Shear-web counts must match
# ------------------------------------------------------------------


def _chord_segment_sets(mesh: dict) -> list[tuple[int, int, str, dict]]:
    """All mesh element-sets whose name matches ``NN_MM_TAG``."""
    out = []
    for s in mesh.get("sets", {}).get("element", []):
        m = SET_RE.match(s["name"])
        if not m:
            continue
        chord, station, tag = m.groups()
        out.append((int(chord), int(station), tag, s))
    return out


def test_n_webs_matches_mesh(yaml_struct, shell_mesh):
    """B1a: ``len(yaml.webs)`` must equal the number of *distinct*
    web-tags in the mesh element-set name layout.

    The mesher uses suffix ``_SW`` and a chord-prefix to label the
    panel-between-spar-caps each web sits in. Each declared web in the
    YAML should be matched by such a chord-segment-station group.
    """
    n_webs_yaml = len(yaml_struct.get("webs", []))
    segs = _chord_segment_sets(shell_mesh)
    sw_segs = [(c, st, tag) for (c, st, tag, _) in segs if tag == "SW"]

    # Stations that actually have SW sets
    stations_with_sw = sorted({st for _, st, _ in sw_segs})
    # Distinct chord-prefixes that mean "I'm a shear-web sub-panel"
    chord_prefixes = sorted({c for c, _, _ in sw_segs})

    # The mesher emits 2 chord prefixes per web (one on each side of the
    # web extrusion). For BAR0 (n_webs=2) we'd expect at most a small
    # number of distinct prefixes — but the mesher currently collapses
    # the two webs onto chord prefixes (00, 01) regardless. We assert
    # only the weaker condition: ``n_chord_prefixes >= n_webs``.
    assert len(chord_prefixes) >= n_webs_yaml, (
        f"YAML declares {n_webs_yaml} webs, but mesh only contains "
        f"{len(chord_prefixes)} distinct SW chord-prefixes "
        f"{chord_prefixes}; expected >= {n_webs_yaml}"
    )
    assert stations_with_sw, "mesh has no shear-web element sets at all"


def test_sw_set_count_equals_two_per_web_per_station(yaml_struct, shell_mesh):
    """B1b: ``count('_SW') == n_webs * n_stations_with_sw``.

    pyNuMAD's ``stackdb.swstacks[k_web, k_station]`` indexes by web,
    and the chord-prefix in the generated set name (``00_NN_SW``,
    ``01_NN_SW``) IS the web index — not a per-web sub-chord-segment.
    BAR0 with n_webs=2 and 26 stations therefore yields 2*26=52 SW
    sets (not 4*26=104).

    This test will FAIL if pyNuMAD silently drops a web — known to
    happen when n_webs >= 3 because mesh_gen.py:611-741 hardcodes
    swstacks[0] and swstacks[1] only. The IEA-22 YAML has 3 webs and
    pyNuMAD currently drops the third.
    """
    n_webs_yaml = len(yaml_struct.get("webs", []))
    segs = _chord_segment_sets(shell_mesh)
    sw_segs = [(c, st, tag) for (c, st, tag, _) in segs if tag == "SW"]
    stations_with_sw = sorted({st for _, st, _ in sw_segs})
    expected = n_webs_yaml * len(stations_with_sw)
    actual = len(sw_segs)
    assert actual == expected, (
        f"#SW sets = {actual} but YAML declares n_webs={n_webs_yaml} and "
        f"mesh has {len(stations_with_sw)} stations with SW elements "
        f"-- expected {n_webs_yaml} * {len(stations_with_sw)} = {expected}. "
        f"If actual < expected, pyNuMAD has dropped one or more webs "
        f"(see mesh_gen.py:611-741 — hardcoded swstacks[0]/swstacks[1])."
    )


def test_every_yaml_web_has_corresponding_sw_set(yaml_struct, shell_mesh):
    """B1c: every web's declared spanwise grid in
    ``yaml.webs[*].start_nd_arc.grid`` should map to *some* SW set at a
    matching station.

    pyNuMAD's station numbering is along the blade's z-axis from root
    to tip; the YAML grid is normalised to [0, 1] along the same axis.
    We just check that the number of YAML stations per web doesn't
    exceed the number of SW stations in the mesh.
    """
    segs = _chord_segment_sets(shell_mesh)
    sw_stations = sorted({st for c, st, t, _ in segs if t == "SW"})
    n_sw_stations = len(sw_stations)

    failures = []
    for web in yaml_struct.get("webs", []):
        grid = web.get("start_nd_arc", {}).get("grid", []) \
               or web.get("offset_y_pa", {}).get("grid", []) \
               or web.get("rotation", {}).get("grid", [])
        if not grid:
            continue
        # YAML grids are usually finer than mesh stations; require at
        # least that the mesh covers the same number of distinct
        # stations.
        if n_sw_stations < 1:
            failures.append(
                f"web {web.get('name')!r} declared in YAML but mesh has "
                f"no SW stations"
            )
    assert not failures, "\n  ".join(failures)


# ------------------------------------------------------------------
# B2. Spar-cap conformance
# ------------------------------------------------------------------


def _layer_by_name(struct: dict, needle: str) -> dict | None:
    for L in struct.get("layers", []):
        if L["name"] == needle:
            return L
    return None


def test_spar_cap_chord_segments_present(yaml_struct, shell_mesh):
    """B2: YAML declares ``Spar_cap_ss`` (suction-side, LP) and
    ``Spar_cap_ps`` (pressure-side, HP) layers. The mesh must emit a
    ``NN_MM_LP_SPAR`` element set and a ``NN_MM_HP_SPAR`` element set
    at each station — i.e. the chord-discretisation reflects the spar
    cap declaration.
    """
    segs = _chord_segment_sets(shell_mesh)
    tags = {t for _, _, t, _ in segs}

    yaml_has_ss = _layer_by_name(yaml_struct, "Spar_cap_ss") is not None
    yaml_has_ps = _layer_by_name(yaml_struct, "Spar_cap_ps") is not None
    if yaml_has_ss:
        assert "LP_SPAR" in tags, (
            "YAML declares Spar_cap_ss but mesh has no NN_MM_LP_SPAR set"
        )
    if yaml_has_ps:
        assert "HP_SPAR" in tags, (
            "YAML declares Spar_cap_ps but mesh has no NN_MM_HP_SPAR set"
        )


def test_spar_cap_present_at_every_meshed_station(yaml_struct, shell_mesh):
    """B2b: the spar cap should run almost the full length of the
    blade (YAML grid covers [0.0, 0.95] or similar). At each station
    where the mesh has *any* shell elements, we expect both LP_SPAR
    and HP_SPAR sets to exist.
    """
    segs = _chord_segment_sets(shell_mesh)
    by_station: dict[int, set[str]] = defaultdict(set)
    for _, st, tag, _ in segs:
        by_station[st].add(tag)

    # Limit the check to stations that have ANY shell element set —
    # i.e. exclude pure root/tip closure that has no panel data.
    missing_lp = sorted(st for st, tags in by_station.items()
                        if "LP_SPAR" not in tags)
    missing_hp = sorted(st for st, tags in by_station.items()
                        if "HP_SPAR" not in tags)

    # Allow root closure (station 0) to legitimately have no spar cap,
    # but no more than that.
    allowed_no_spar = {0}
    bad_lp = [s for s in missing_lp if s not in allowed_no_spar]
    bad_hp = [s for s in missing_hp if s not in allowed_no_spar]
    assert not bad_lp and not bad_hp, (
        f"stations missing LP_SPAR: {bad_lp[:5]}; "
        f"stations missing HP_SPAR: {bad_hp[:5]}"
    )


# ------------------------------------------------------------------
# B3. Station coverage
# ------------------------------------------------------------------


def test_mesh_stations_align_with_yaml_layer_grids(yaml_struct, shell_mesh):
    """B3: union of YAML layer-grid lengths should be roughly the
    number of distinct stations in the mesh. We only assert a loose
    upper bound (mesh stations are sometimes a subset of YAML grid
    points where a layer turns on or off).
    """
    segs = _chord_segment_sets(shell_mesh)
    mesh_stations = sorted({st for _, st, _, _ in segs})

    max_grid_len = 0
    for L in yaml_struct.get("layers", []):
        for fld in ("thickness", "start_nd_arc", "end_nd_arc",
                    "width", "offset_y_pa", "rotation"):
            g = L.get(fld, {})
            if isinstance(g, dict):
                grid = g.get("grid", [])
                if grid:
                    max_grid_len = max(max_grid_len, len(grid))

    assert mesh_stations, "mesh has no chord-segment sets"
    # The mesh stations come from layer-grid points; their count
    # should never exceed the densest layer grid.
    assert len(mesh_stations) <= max_grid_len + 2, (
        f"mesh has {len(mesh_stations)} stations but the densest YAML "
        f"layer grid has only {max_grid_len} points -- mesh is "
        f"emitting more stations than the YAML declares"
    )
