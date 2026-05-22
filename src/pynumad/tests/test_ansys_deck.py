"""End-to-end regression tests for the ANSYS deck writer.

These pin two specific bugs we hit during Phase 0.7c of the IEA-22
mesh-convergence work:

  1. ``includeAdhesive=True`` raised ``NameError: 'adhesive_sec_id'``
     in ``_write_ansys_adhesive`` — every adhesive run crashed.
  2. ``EMODIF`` for SECN assignments silently failed on a subset of
     elements — the mesh dict said HP_SPAR_r20 contained elements
     [887..890] but the actual ANSYS run had those elements stuck on
     the previous station's SECN (133 instead of 164). ESEL,S,SEC,,164
     returned 0 elements.

The fixture is BAR0 (the bundled minimal blade), not IEA-22, so the
tests are fast (~10 s end-to-end per case) but exercise the same
deck-writer code paths as the production blade.
"""
from __future__ import annotations

import os
import re
import unittest

import numpy as np

from pynumad.analysis.ansys.write import write_ansys_shell_model
from pynumad.mesh_gen.mesh_gen import get_shell_mesh
from pynumad.objects.blade import Blade

test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")


def _build_deck(includeAdhesive: bool, esize: float = 0.5) -> tuple[Blade, dict, str]:
    """Build mesh + write the ANSYS deck on BAR0. Returns (blade, mesh, deck_text)."""
    yamlfile = os.path.join(test_data_dir, "blades", "blade.yaml")
    blade = Blade(yamlfile)
    mesh = get_shell_mesh(blade, includeAdhesive=int(includeAdhesive),
                          elementSize=esize)
    deck_path = write_ansys_shell_model(
        blade, mesh,
        {"elementType": "181", "blade_name": "BAR0_TEST"},
    )
    with open(deck_path, "r") as f:
        return blade, mesh, f.read()


class TestAdhesiveEmission(unittest.TestCase):
    """``includeAdhesive=True`` used to crash with NameError. Regression
    test that it (a) doesn't raise and (b) actually writes adhesive
    elements into the deck.
    """

    def test_adhesive_does_not_raise(self):
        # Just calling _build_deck is the regression test — pre-patch
        # this raised NameError inside _write_ansys_adhesive.
        _, mesh, deck = _build_deck(includeAdhesive=True)
        self.assertIsNotNone(deck)
        self.assertGreater(len(deck), 1000)

    def test_adhesive_section_present_in_deck(self):
        _, mesh, deck = _build_deck(includeAdhesive=True)
        self.assertIn(
            "Adhesive nodes", deck,
            "deck missing adhesive-node block — adhesive emission silently skipped",
        )
        self.assertIn(
            "Adhesive elements", deck,
            "deck missing adhesive-element block",
        )

    def test_adhesive_elements_use_solid185(self):
        _, _, deck = _build_deck(includeAdhesive=True)
        # The adhesive element type should be defined and used.
        self.assertRegex(
            deck,
            r"et,\d+,solid185",
            msg="adhesive element type (SOLID185) not defined in deck",
        )

    def test_no_undefined_adhesive_sec_id(self):
        """Guards against the original bug: a stray ``adhesive_sec_id``
        in a format-string argument. If pyNuMAD regresses, the bare
        identifier would re-appear in the deck or raise on write.
        """
        _, _, deck = _build_deck(includeAdhesive=True)
        # The variable name should NEVER appear as a token in the deck
        # (it's a Python local that should never get format-stringed in).
        self.assertNotIn("adhesive_sec_id", deck)


class TestSecNumberConsistency(unittest.TestCase):
    """For each element in pyNuMAD's mesh-dict element sets, the
    emodif command in the generated .mac must assign the same SECN
    that the section TYPE comment claims. Pre-patch the off-by-one
    + silent-failure combo caused 156/4811 elements to be on the
    wrong layup.
    """

    @classmethod
    def setUpClass(cls):
        cls.blade, cls.mesh, cls.deck = _build_deck(includeAdhesive=False)

    def _parse_section_ids(self) -> dict[str, int]:
        """sectype N defined for each named section comment."""
        out: dict[str, int] = {}
        pat = re.compile(
            r"!\s*(\d+_\d+_[A-Z_]+)\s*\n\s*sectype,(\d+),shell",
            re.IGNORECASE,
        )
        for m in pat.finditer(self.deck):
            out[m.group(1)] = int(m.group(2))
        return out

    def _parse_emodif_secnum(self) -> dict[int, int]:
        """element_id -> final secnum from the emodif batch."""
        out: dict[int, int] = {}
        for m in re.finditer(
            r"emodif,\s*(\d+),secnum,\s*(\d+)", self.deck
        ):
            out[int(m.group(1))] = int(m.group(2))
        return out

    def test_every_section_tag_has_sectype(self):
        """Skip aggregator sets like ``allOuterShellEls`` — those are
        meta-collections, not real chord-segment sections. The
        chord-segment-station naming convention is ``NN_MM_TAG``.
        """
        sec_ids = self._parse_section_ids()
        el_sets = self.mesh.get("sets", {}).get("element", [])
        chord_seg = re.compile(r"^\d+_\d+_")
        for s in el_sets:
            tag = s["name"]
            if not s.get("labels"):
                continue
            if not chord_seg.match(tag):
                continue  # meta-aggregator set
            self.assertIn(
                tag, sec_ids,
                f"section '{tag}' has elements in the mesh but no sectype "
                f"definition in the deck",
            )

    def test_every_section_has_at_least_one_emodif(self):
        """Each section that owns mesh elements should produce at
        least one ``emodif,*,secnum,SID`` line — otherwise the
        elements end up on a stale SECN and ANSYS computes stress
        with the wrong layup.
        """
        sec_ids = self._parse_section_ids()
        emodif = self._parse_emodif_secnum()
        secnum_to_count = {}
        for sid in emodif.values():
            secnum_to_count[sid] = secnum_to_count.get(sid, 0) + 1
        el_sets = self.mesh.get("sets", {}).get("element", [])
        missing = []
        for s in el_sets:
            tag = s["name"]
            if not s.get("labels"):
                continue
            sid = sec_ids.get(tag)
            if sid is None:
                continue  # caught by the previous test
            if secnum_to_count.get(sid, 0) == 0:
                missing.append((tag, sid))
        self.assertEqual(
            missing, [],
            msg=f"{len(missing)} sections own mesh elements but have NO "
                f"emodif lines assigning them — those elements will use "
                f"a stale SECN: {missing[:5]} ...",
        )

    def test_emodif_count_matches_mesh_dict(self):
        """The mesh dict says section S contains N elements. The deck
        should EMODIF exactly N elements to SECNUM=sid_of_S (the bug
        we hit had a +1 offset between mesh-dict labels and the
        actual element-IDs that got emodif'd, but the *count* should
        match exactly).
        """
        sec_ids = self._parse_section_ids()
        emodif = self._parse_emodif_secnum()
        secnum_to_count: dict[int, int] = {}
        for sid in emodif.values():
            secnum_to_count[sid] = secnum_to_count.get(sid, 0) + 1
        el_sets = self.mesh.get("sets", {}).get("element", [])
        mismatches = []
        for s in el_sets:
            tag = s["name"]
            n_expected = len(s.get("labels", []))
            if n_expected == 0:
                continue
            sid = sec_ids.get(tag)
            if sid is None:
                continue
            n_actual = secnum_to_count.get(sid, 0)
            if n_actual != n_expected:
                mismatches.append(
                    (tag, sid, n_expected, n_actual)
                )
        self.assertEqual(
            mismatches, [],
            msg=f"emodif element count mismatches: {mismatches[:5]} "
                f"(... {len(mismatches)} total)",
        )

    def test_emodif_element_ids_in_valid_range(self):
        """EMODIF should only ever target element IDs that the deck
        will actually create. If pyNuMAD has off-by-one math the
        EMODIF target may be N+1 where only N elements exist — the
        EMODIF is then silently skipped because no element with that
        ID exists in the selection.
        """
        elements = np.asarray(self.mesh["elements"], dtype=int)
        n_elem = elements.shape[0]
        emodif = self._parse_emodif_secnum()
        bad = [eid for eid in emodif if eid < 1 or eid > n_elem]
        self.assertEqual(
            bad, [],
            msg=f"{len(bad)} emodif lines target element IDs outside "
                f"[1, {n_elem}]: {bad[:10]}",
        )


class TestDeckSyntacticCorrectness(unittest.TestCase):
    """Cheap syntactic sanity on the generated deck — every command
    line should be a valid MAPDL command or a comment.
    """

    @classmethod
    def setUpClass(cls):
        cls.blade, cls.mesh, cls.deck = _build_deck(includeAdhesive=True)

    def test_deck_has_required_blocks(self):
        for required in (
            "/prep7",          # preprocessor entry
            "et,",             # at least one element type definition
            "mp,",             # at least one material property
            "sectype,",        # at least one shell section
            "n,",              # at least one node
            "e,",              # at least one element
        ):
            self.assertIn(
                required, self.deck,
                f"deck missing required block / command: {required!r}",
            )

    def test_no_python_repr_leaks(self):
        """Pre-patch the deck contained the literal token
        ``adhesive_sec_id`` because the format string left an
        undefined Python var in the output. Generally no Python
        identifier or undefined-variable repr should leak into APDL.
        """
        for leaked in ("<built-in", "NameError", "adhesive_sec_id"):
            self.assertNotIn(
                leaked, self.deck,
                f"deck contains suspicious leaked token: {leaked!r}",
            )

    def test_no_non_numeric_in_tbdata(self):
        """``tb,fcli`` failure-criteria limit tables take only numeric
        arguments. pyNuMAD currently writes ``tbdata,17,nan,None,-inf,0.0``
        for materials whose failure-stress fields aren't fully populated
        in the YAML — ANSYS errors on those tokens. We work around it by
        post-processing the deck (see ``run_static_v2.py`` regex sub),
        but the writer should be fixed at source.

        This test is xfail-style: if it PASSES, pyNuMAD has been fixed
        upstream and our regex workaround can be removed.
        """
        bad_tokens = ("nan", "None", "inf", "-inf")
        offending: list[str] = []
        for line in self.deck.splitlines():
            stripped = line.strip()
            if not stripped.lower().startswith(("tbdata,", "mp,", "secdata,")):
                continue
            for tok in bad_tokens:
                # Match as a comma-bounded token so we don't false-positive
                # on identifiers like "infinity_..." or "Nonempty_...".
                if (f",{tok}," in stripped + ","
                        or f",{tok}\n" in (line + "\n")):
                    offending.append(line.rstrip())
                    break
        self.assertEqual(
            offending, [],
            msg=f"{len(offending)} numeric APDL commands contain non-numeric "
                f"tokens (nan/None/inf): {offending[:3]} ...",
        )


if __name__ == "__main__":
    unittest.main()
