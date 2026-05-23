"""Unit tests for ``write_ansys_solid_general`` — the ANSYS APDL writer
for full 3D solid blade meshes.

These tests parse the emitted ``.mac`` deck as plain text and assert
structural properties (element-type definition, node/element counts,
material blocks, CSKP coordinate systems, ESYS assignment per section,
no NaN/inf literals from the failure-criteria sanitiser). They do NOT
invoke ANSYS — that lives in ``test_solid_solver_run.py`` (Phase 5,
``@pytest.mark.integration``).

Fixture: bundled BAR0 blade at elementSize=0.5 (coarse, ~3 s build).
"""
from __future__ import annotations

import os
import re
import tempfile
import unittest

from pynumad.analysis.ansys.write import write_ansys_solid_general
from pynumad.tests._mesh_cache import get_blade, get_solid_mesh_cached


def _build_deck(element_size: float = 0.5) -> tuple[object, dict, str]:
    """Build the BAR0 solid mesh + emit the ANSYS deck. Returns
    (blade, mesh, deck_text). The deck file is written to a temp
    file that is deleted before return."""
    blade = get_blade()
    mesh = get_solid_mesh_cached(elementSize=element_size)
    with tempfile.NamedTemporaryFile(suffix=".mac", delete=False) as f:
        path = f.name
    try:
        write_ansys_solid_general(path, blade, mesh)
        with open(path) as f:
            return blade, mesh, f.read()
    finally:
        os.unlink(path)


class TestDeckEmission(unittest.TestCase):
    """The deck must be written, parseable, and contain the expected
    APDL constructs for a SOLID185 composite blade."""

    @classmethod
    def setUpClass(cls):
        cls.blade, cls.mesh, cls.deck = _build_deck(element_size=0.5)
        cls.lines = cls.deck.splitlines()

    def test_deck_written_non_empty(self):
        self.assertGreater(len(self.deck), 1000,
                           "deck is < 1 KB — likely truncated")

    def test_solid185_element_type_defined(self):
        # Case-insensitive match: APDL accepts mixed case.
        self.assertRegex(self.deck, r"(?im)^\s*et,\s*11,\s*solid185\b",
                         "ET,11,SOLID185 missing from deck")

    def test_enhanced_strain_keyopt_set(self):
        # KEYOPT(2)=2 is required for skewed root elements; the
        # adhesive helper uses the same setting.
        self.assertRegex(self.deck, r"(?im)^\s*keyopt,\s*11,\s*2,\s*2\b",
                         "KEYOPT(2)=2 (enhanced strain) missing")

    def test_homogeneous_solid_keyopt_set(self):
        self.assertRegex(self.deck, r"(?im)^\s*keyopt,\s*11,\s*3,\s*0\b",
                         "KEYOPT(3)=0 (homogeneous structural solid) missing")

    def test_node_count_matches_mesh(self):
        # N,<id>,x,y,z lines — count must equal mesh nodes.
        n_lines = sum(1 for L in self.lines if re.match(r"^N,\s*\d+,", L))
        self.assertEqual(n_lines, self.mesh["nodes"].shape[0],
                         f"deck has {n_lines} N lines vs mesh "
                         f"{self.mesh['nodes'].shape[0]} nodes")

    def test_element_count_matches_mesh(self):
        # EN,<id>,n1..n8 lines — count must equal mesh elements
        # (both hex and degenerate-brick wedge variants).
        en_lines = sum(1 for L in self.lines if L.startswith("EN,"))
        self.assertEqual(en_lines, self.mesh["elements"].shape[0],
                         f"deck has {en_lines} EN lines vs mesh "
                         f"{self.mesh['elements'].shape[0]} elements")

    def test_wedges_emitted_as_degenerate_brick(self):
        # Wedges: positions 4==3 AND positions 8==7 in the EN command
        # (1-indexed in the APDL tokens). Count must equal wedge count.
        wedge_count = int((self.mesh["elements"][:, 6] == -1).sum())
        if wedge_count == 0:
            self.skipTest("no wedge elements in fixture mesh")
        # Parse EN lines: EN,eid,n1,n2,n3,n4,n5,n6,n7,n8
        deg_count = 0
        for L in self.lines:
            if not L.startswith("EN,"):
                continue
            parts = L.strip().rstrip(",").split(",")
            if len(parts) != 10:
                continue
            n3, n4 = parts[4], parts[5]   # 4th and 5th tokens are nodes 3,4
            n7, n8 = parts[8], parts[9]   # 9th and 10th tokens are nodes 7,8
            if n3 == n4 and n7 == n8:
                deg_count += 1
        self.assertEqual(deg_count, wedge_count,
                         f"deck has {deg_count} degenerate-brick wedges "
                         f"vs mesh {wedge_count}")


class TestMaterialEmission(unittest.TestCase):
    """Every material in the blade definition must produce MP,EX and
    MP,DENS commands. Failure-criteria sanitiser must not leak NaN/inf."""

    @classmethod
    def setUpClass(cls):
        cls.blade, cls.mesh, cls.deck = _build_deck(element_size=0.5)

    def test_mp_ex_per_material(self):
        n_mats = len(self.blade.definition.materials)
        for matid in range(1, n_mats + 1):
            with self.subTest(matid=matid):
                pattern = rf"(?im)^\s*mp,ex,\s*{matid}\b"
                self.assertRegex(self.deck, pattern,
                                 f"MP,EX,{matid} missing from deck")

    def test_mp_dens_per_material(self):
        n_mats = len(self.blade.definition.materials)
        for matid in range(1, n_mats + 1):
            with self.subTest(matid=matid):
                pattern = rf"(?im)^\s*mp,dens,\s*{matid}\b"
                self.assertRegex(self.deck, pattern,
                                 f"MP,DENS,{matid} missing from deck")

    def test_no_nan_or_inf_literals_in_deck(self):
        # The _apdl_finite sanitiser must convert NaN/inf to defaults
        # before formatting. Literal 'nan' / 'inf' tokens in the deck
        # would crash ANSYS R2023 with a parse error.
        for token in ("nan", "inf", "-inf"):
            # Match as a standalone numeric token (comma- or space-delimited).
            pattern = rf"(?i)[,\s]{re.escape(token)}[,\s\n]"
            self.assertNotRegex(self.deck, pattern,
                                f"literal {token!r} leaked into deck — "
                                "_apdl_finite sanitiser failed")


class TestSectionEmission(unittest.TestCase):
    """Each section must produce a CSKP coordinate system and a paired
    EMODIF,ALL,ESYS assignment. Catches off-by-one in csys_id math."""

    @classmethod
    def setUpClass(cls):
        cls.blade, cls.mesh, cls.deck = _build_deck(element_size=0.5)

    def test_cskp_count_matches_section_count(self):
        n_cskp = sum(1 for L in self.deck.splitlines() if L.startswith("CSKP,"))
        n_sections = len(self.mesh["sections"])
        self.assertEqual(n_cskp, n_sections,
                         f"deck has {n_cskp} CSKP vs {n_sections} sections")

    def test_emodif_esys_count_matches_section_count(self):
        # One EMODIF,ALL,ESYS per section (not per element).
        n_esys = sum(1 for L in self.deck.splitlines()
                     if L.upper().startswith("EMODIF,ALL,ESYS"))
        n_sections = len(self.mesh["sections"])
        self.assertEqual(n_esys, n_sections,
                         f"deck has {n_esys} EMODIF,ALL,ESYS vs "
                         f"{n_sections} sections")

    def test_emodif_mat_count_matches_section_count(self):
        n_mat = sum(1 for L in self.deck.splitlines()
                    if L.upper().startswith("EMODIF,ALL,MAT"))
        n_sections = len(self.mesh["sections"])
        self.assertEqual(n_mat, n_sections,
                         f"deck has {n_mat} EMODIF,ALL,MAT vs "
                         f"{n_sections} sections")

    def test_csys_ids_start_at_100_and_are_unique(self):
        ids = []
        for L in self.deck.splitlines():
            if L.startswith("CSKP,"):
                # CSKP,<csys_id>,<kcs>,<kp1>,<kp2>,<kp3>
                csys_id = int(L.split(",")[1])
                ids.append(csys_id)
        self.assertEqual(min(ids), 100,
                         "CSYS IDs should start at 100 (plan convention)")
        self.assertEqual(len(ids), len(set(ids)),
                         "CSYS IDs collide — section indexing broken")


class TestBoundaryConditions(unittest.TestCase):
    """Clamped-root BCs must be present so the deck is solver-ready."""

    @classmethod
    def setUpClass(cls):
        cls.blade, cls.mesh, cls.deck = _build_deck(element_size=0.5)

    def test_root_node_selection_present(self):
        self.assertRegex(self.deck, r"(?im)^\s*nsel,s,loc,z,0\b",
                         "root node selection (NSEL,S,LOC,Z,0) missing")

    def test_clamped_constraint_present(self):
        self.assertRegex(self.deck, r"(?im)^\s*d,all,all\b",
                         "clamped constraint (D,ALL,ALL) missing — "
                         "deck cannot be solved without BCs")

    def test_save_command_present(self):
        # SAVE writes the .db that downstream postprocessing reads.
        self.assertRegex(self.deck, r"(?im)^\s*save\b",
                         "SAVE command missing — database not persisted")


if __name__ == "__main__":
    unittest.main()
