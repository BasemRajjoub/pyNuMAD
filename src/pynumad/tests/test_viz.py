"""Smoke + invariant tests for the ``pynumad.viz`` package.

These are deliberately fast tests that exercise the public surface
on the BAR0 fixture and assert the output files exist + look sane.
We don't do pixel-level comparison — matplotlib backends differ
between versions — but we do verify counts, basic content, and that
no exception escapes any writer.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from pynumad.tests._mesh_cache import get_mesh
from pynumad.viz import (
    ADHESIVE_COLORS,
    CHORD_GROUP_TAGS,
    GROUP_COLORS,
    classify_adhesive_set_name,
    classify_set_name,
    get_color,
    write_adhesive_snapshots,
    write_blade_html,
    write_component_snapshots,
    write_overview_snapshots,
)
from pynumad.viz._mesh_utils import (
    collect_adhesive_bond_indices,
    collect_chord_group_indices,
    quads_to_polygons,
    quads_to_triangles,
    solid_outer_face_triangles,
)


@pytest.fixture(scope="module")
def mesh_with_adhesive():
    return get_mesh(includeAdhesive=True, elementSize=0.8)


@pytest.fixture(scope="module")
def mesh_shell_only():
    return get_mesh(includeAdhesive=False, elementSize=0.8)


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------


class TestPalette:
    def test_every_chord_group_has_a_color(self):
        for tag in CHORD_GROUP_TAGS:
            assert tag in GROUP_COLORS, f"missing colour for {tag}"
            assert re.fullmatch(r"#[0-9a-fA-F]{6}", GROUP_COLORS[tag]), (
                f"{tag}: colour must be a #rrggbb hex string"
            )

    def test_adhesive_colors_distinct_from_shell(self):
        shell_palette = set(GROUP_COLORS.values())
        for tag, c in ADHESIVE_COLORS.items():
            assert c not in shell_palette, (
                f"{tag} colour {c} collides with a shell zone"
            )

    def test_classify_specific_before_generic(self):
        # HP_TE_REINF and HP_TE_PANEL both contain "HP_TE", so the
        # classifier must walk specific-first.
        assert classify_set_name("01_30_HP_TE_REINF") == "HP_TE_REINF"
        assert classify_set_name("02_30_HP_TE_PANEL") == "HP_TE_PANEL"
        assert classify_set_name("00_30_HP_TE_FLAT") == "HP_TE_FLAT"
        # HP_LE classifier shouldn't match HP_LE_PANEL by mistake.
        assert classify_set_name("04_05_HP_LE_PANEL") == "HP_LE_PANEL"
        assert classify_set_name("05_05_HP_LE") == "HP_LE"

    def test_classify_returns_none_for_unknown(self):
        assert classify_set_name("nonsense") is None
        assert classify_set_name("") is None

    def test_classify_adhesive_set_name(self):
        assert classify_adhesive_set_name("TE_BOND") == "TE_BOND"
        assert classify_adhesive_set_name("TE_BOND__LP_TGT") == "TE_BOND"
        assert classify_adhesive_set_name("LE_BOND__ALL_TGT") == "LE_BOND"
        assert classify_adhesive_set_name("HP_SPAR") is None

    def test_get_color_fallback(self):
        assert get_color("HP_SPAR") == GROUP_COLORS["HP_SPAR"]
        assert get_color("TE_BOND") == ADHESIVE_COLORS["TE_BOND"]
        assert get_color("nonsense", default="#abcdef") == "#abcdef"


# ---------------------------------------------------------------------------
# _mesh_utils helpers
# ---------------------------------------------------------------------------


class TestMeshUtils:
    def test_quads_to_polygons_filters_tris(self):
        nodes = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0], [2,0,0]], float)
        # One quad + one tri (last slot -1)
        elements = np.array([[0,1,2,3], [0,1,4,-1]], int)
        polys = quads_to_polygons(nodes, elements)
        assert polys.shape == (1, 4, 3), \
            "only the quad should survive, the tri must be dropped"
        np.testing.assert_array_equal(polys[0, 0], nodes[0])

    def test_quads_to_triangles_doubles_quads(self):
        conn = np.array([[0,1,2,3], [4,5,6,7]], int)
        tris = quads_to_triangles(conn)
        assert tris.shape == (4, 3), "each quad => 2 tris"

    def test_solid_outer_face_dedup_drops_interior_face(self):
        # Two adjacent hex bricks sharing one face (the (1,2,5,6) face on
        # brick 1 == (0,3,4,7) face on brick 2). 6+6 = 12 faces; the
        # shared face appears twice and is dropped, leaving 10 faces.
        # 8 quad faces => 16 tris; we expect 10 faces => 20 tris.
        a = np.array([[0,1,2,3,4,5,6,7]])
        # second brick translated +x by one unit (shares right face with a)
        b = np.array([[1,8,9,2,5,10,11,6]])
        elements = np.vstack([a, b])
        tris = solid_outer_face_triangles(elements)
        # 10 quad faces × 2 tris each = 20 tris
        assert tris.shape == (20, 3), tris.shape

    def test_solid_outer_face_empty_input(self):
        assert solid_outer_face_triangles(np.empty((0, 8), int)).shape == (0, 3)

    def test_collect_chord_group_indices_uses_zero_indexed_labels(self, mesh_shell_only):
        """Regression: labels in mesh['sets']['element'] are 0-indexed.

        An earlier collect_groups implementation did ``int(label) - 1``
        and returned element indices off-by-one, which made adjacent-
        region elements appear as if they belonged to the wrong group.
        """
        groups = collect_chord_group_indices(mesh_shell_only)
        elements = np.asarray(mesh_shell_only["elements"], int)
        for tag, ids in groups.items():
            for el_id in ids:
                assert 0 <= el_id < elements.shape[0], (
                    f"{tag} has out-of-range element id {el_id}"
                )

    def test_collect_adhesive_uses_bond_sets_when_available(self, mesh_with_adhesive):
        bonds = collect_adhesive_bond_indices(mesh_with_adhesive)
        # IEA-22 / BAR0 fixtures emit TE + LE bondlines.
        assert "TE_BOND" in bonds, sorted(bonds)
        n_adh = len(mesh_with_adhesive.get("adhesiveEls", []))
        for tag, ids in bonds.items():
            for i in ids:
                assert 0 <= i < n_adh, (
                    f"{tag} has out-of-range adhesive id {i}"
                )


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


class TestSnapshots:
    def test_overview_snapshots_produce_three_views_plus_zooms(
        self, mesh_shell_only, tmp_path
    ):
        out = write_overview_snapshots(
            mesh_shell_only, tmp_path, include_adhesive=False,
            zoom_band_size=30.0,
        )
        # 3 overviews + at least 1 zoom
        assert len(out) >= 4
        names = {p.name for p in out}
        for canonical in ("overview_iso.png", "overview_side_TE.png",
                           "overview_side_LE.png"):
            assert canonical in names, f"missing {canonical}"
        assert any(n.startswith("zoom_z") for n in names), (
            "no zoom-band snapshot produced"
        )
        for p in out:
            assert p.exists() and p.stat().st_size > 1024, (
                f"{p.name} too small to be a real PNG"
            )

    def test_component_snapshots_one_per_present_group(
        self, mesh_shell_only, tmp_path
    ):
        out = write_component_snapshots(mesh_shell_only, tmp_path)
        groups = collect_chord_group_indices(mesh_shell_only)
        assert len(out) == len(groups), (
            "should produce exactly one PNG per chord group present in mesh"
        )
        names = {p.stem for p in out}
        for tag in groups:
            assert tag in names, f"missing component snapshot for {tag}"
        for p in out:
            assert p.exists() and p.stat().st_size > 1024

    def test_adhesive_snapshots_skip_when_no_adhesive(
        self, mesh_shell_only, tmp_path
    ):
        out = write_adhesive_snapshots(mesh_shell_only, tmp_path)
        assert out == [], (
            "should return empty list when mesh has no adhesive volumes"
        )

    def test_adhesive_snapshots_produce_overview_and_zooms(
        self, mesh_with_adhesive, tmp_path
    ):
        out = write_adhesive_snapshots(
            mesh_with_adhesive, tmp_path, zoom_band_size=30.0,
        )
        assert len(out) >= 2  # overview + at least 1 zoom
        names = {p.name for p in out}
        assert "adhesive_overview.png" in names
        assert any(n.startswith("adhesive_zoom_z") for n in names)


# ---------------------------------------------------------------------------
# Interactive HTML
# ---------------------------------------------------------------------------


class TestInteractive:
    def test_html_writer_creates_self_contained_file(
        self, mesh_shell_only, tmp_path
    ):
        out = write_blade_html(
            mesh_shell_only, tmp_path / "blade.html",
            include_adhesive=False, title="test blade",
        )
        assert out.exists()
        html = out.read_text(encoding="utf-8")
        # plotly.js must be embedded inline
        assert "<script" in html
        assert "Plotly.newPlot" in html or "Plotly.react" in html
        # external CDNs must have been scrubbed
        assert "unpkg.com" not in html
        assert "cdn.plot.ly" not in html
        # side panel must have come through
        assert "Trace inventory" in html
        # all chord groups present in mesh should appear in trace names
        groups = collect_chord_group_indices(mesh_shell_only)
        for tag in groups:
            assert tag in html, f"trace for {tag} missing from HTML"

    def test_html_includes_adhesive_when_present(
        self, mesh_with_adhesive, tmp_path
    ):
        out = write_blade_html(
            mesh_with_adhesive, tmp_path / "blade.html",
            include_adhesive=True,
        )
        html = out.read_text(encoding="utf-8")
        assert "TE_BOND" in html or "LE_BOND" in html, (
            "adhesive traces missing from HTML"
        )

    def test_html_omits_adhesive_when_requested(
        self, mesh_with_adhesive, tmp_path
    ):
        out = write_blade_html(
            mesh_with_adhesive, tmp_path / "blade.html",
            include_adhesive=False,
        )
        html = out.read_text(encoding="utf-8")
        # adhesive trace names contain "adhesive ("; their absence
        # confirms the flag is honoured even when mesh has bondlines.
        assert "adhesive (" not in html


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_help_runs():
    """``python -m pynumad.viz --help`` must succeed."""
    from pynumad.viz.cli import _build_arg_parser
    p = _build_arg_parser()
    # SystemExit(0) from argparse on --help
    with pytest.raises(SystemExit) as exc:
        p.parse_args(["--help"])
    assert exc.value.code == 0
