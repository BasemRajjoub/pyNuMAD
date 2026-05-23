"""Unit tests for pynumad.analysis.convergence.

These tests don't touch ANSYS — they verify that the APDL emitters
produce the expected string patterns, the parser handles the CSV
schema, and the drift analyser correctly identifies the smallest
element size that satisfies a per-category criterion.
"""
from __future__ import annotations

import re
from pathlib import Path
from textwrap import dedent

import pytest

from pynumad.analysis.convergence import (
    ConvergenceResult,
    ConvergenceSpec,
    PatchSpec,
    SectionSpec,
    TipDeflectionSpec,
    analyse,
    emit_csv_close,
    emit_csv_open,
    emit_patch,
    emit_post1,
    emit_section,
    emit_tip_deflection,
    filter_element_sets,
    iea22_default_spec,
    parse_results,
    print_drift_table,
)


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


class TestSpecs:
    def test_iea22_default_has_15_patches(self):
        s = iea22_default_spec()
        assert len(s.patches) == 15
        # Naming sanity: every patch has the substring that matches it
        for p in s.patches:
            assert p.element_set_substr in p.name

    def test_iea22_default_has_5_sections(self):
        s = iea22_default_spec()
        assert len(s.sections) == 5
        assert set(int(sec.z_m) for sec in s.sections) == {20, 35, 50, 70, 90}

    def test_iea22_default_has_tip(self):
        s = iea22_default_spec()
        assert s.tip_deflection is not None
        assert s.tip_deflection.z_band_m[1] > s.tip_deflection.z_band_m[0]


# ---------------------------------------------------------------------------
# Filter helper
# ---------------------------------------------------------------------------


class TestFilterElementSets:
    def test_substring_match_is_orderpreserving(self):
        names = ["03_05_HP_SPAR", "01_05_HP_TE_REINF", "03_06_HP_SPAR",
                  "10_06_LP_TE_REINF"]
        assert filter_element_sets(names, "HP_SPAR") == \
            ["03_05_HP_SPAR", "03_06_HP_SPAR"]
        assert filter_element_sets(names, "TE_REINF") == \
            ["01_05_HP_TE_REINF", "10_06_LP_TE_REINF"]

    def test_substring_no_match_returns_empty(self):
        assert filter_element_sets(["a", "b"], "ZZZ") == []

    def test_substring_is_case_sensitive(self):
        # Documenting the contract; ANSYS components are case-preserving.
        assert filter_element_sets(["HP_SPAR"], "hp_spar") == []


# ---------------------------------------------------------------------------
# APDL emitters
# ---------------------------------------------------------------------------


class TestApdlEmit:
    def test_emit_csv_open_writes_header(self):
        s = emit_csv_open("qoi.csv")
        assert "*CFOPEN,qoi.csv" in s
        assert "category,name,key,value" in s

    def test_emit_tip_deflection_writes_one_row(self):
        spec = TipDeflectionSpec(name="tip", z_band_m=(135.0, 140.0))
        s = emit_tip_deflection(spec)
        assert "NSEL,S,LOC,Z,135.000000,140.000000" in s
        assert "NSORT,U,SUM" in s
        # Single *VWRITE for the umax row
        assert s.count("*VWRITE") == 1
        assert "'tip,'" in s and "umax_m" in s

    def test_emit_patch_includes_all_matching_components(self):
        spec = PatchSpec(
            name="HP_SPAR_r20",
            element_set_substr="HP_SPAR",
            z_band_m=(18.0, 22.0),
            layers=(3,),
            surfaces=("TOP", "BOT"),
        )
        matching = ["03_05_HP_SPAR", "03_06_HP_SPAR", "03_07_HP_SPAR"]
        s = emit_patch(spec, matching)
        for c in matching:
            assert f"CMSEL,A,{c}" in s
        # z-band restriction
        assert "ESEL,R,CENT,Z,18.000000,22.000000" in s
        # Both surfaces emitted
        assert "LAYER,3" in s
        assert "SHELL,TOP" in s
        assert "SHELL,BOT" in s
        # Area-weighted formula: σ·V summed then divided by V summed
        assert "ETABLE,vol_,VOLU" in s
        assert "ETABLE,svm_,S,EQV" in s
        assert "ETABLE,svw_,*,svm_,vol_" in s
        assert "sum_sw_ / sum_v_" in s

    def test_emit_patch_with_no_matching_components_emits_warning_only(self):
        spec = PatchSpec(name="empty", element_set_substr="ZZ", z_band_m=(0, 10))
        s = emit_patch(spec, [])
        assert "WARNING" in s
        assert "CMSEL" not in s

    def test_emit_section_writes_six_resultant_rows(self):
        spec = SectionSpec(name="section_r20", z_m=20.0)
        s = emit_section(spec)
        assert "ESEL,S,CENT,Z,20.000000,1.0e10" in s
        assert "WPOFFS,0,0,20.000000" in s
        # 6 *VWRITE rows: Fx, Fy, Fz, Mx, My, Mz
        for key in ("Fx_N", "Fy_N", "Fz_N", "Mx_Nm", "My_Nm", "Mz_Nm"):
            assert f"'{key}'" in s
        assert s.count("*VWRITE") == 6

    def test_emit_post1_full_round_trip(self):
        spec = iea22_default_spec()
        # Manufacture set names matching every patch's substring.
        set_names = []
        for patch in spec.patches:
            for k in range(3):
                set_names.append(f"00_{k:02d}_{patch.element_set_substr}")
        s = emit_post1(spec, set_names)
        # Sanity: opens + closes CSV exactly once
        assert s.count("*CFOPEN") == 1
        assert s.count("*CFCLOSE") == 1
        # Block headers for every patch + section + tip
        for patch in spec.patches:
            assert f"patch block: {patch.name}" in s
        for sec in spec.sections:
            assert f"section cut block: {sec.name}" in s


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestParser:
    def test_parses_canonical_schema(self, tmp_path):
        csv = tmp_path / "qoi.csv"
        csv.write_text(dedent("""\
            category,name,key,value
            tip,tip,umax_m,    1.234e+01
            patch,HP_SPAR_r20,L3_BOT_svm_Pa,   2.5e+08
            patch,HP_SPAR_r20,L3_BOT_volu_m3,  1.2e-02
            patch,HP_SPAR_r20,L3_BOT_n_elem,   42.
            section,section_r20,Fx_N,          5e5
            section,section_r20,My_Nm,         6.93e7
            """))
        r = parse_results(csv)
        assert r.tip["tip"]["umax_m"] == pytest.approx(12.34)
        assert r.patch_svm_pa("HP_SPAR_r20", layer=3, surface="BOT") == \
            pytest.approx(2.5e8)
        assert r.patches["HP_SPAR_r20"]["L3_BOT_n_elem"] == pytest.approx(42.0)
        assert r.section_my_nm("section_r20") == pytest.approx(6.93e7)
        assert r.sections["section_r20"]["Fx_N"] == pytest.approx(5e5)

    def test_parse_empty_file(self, tmp_path):
        csv = tmp_path / "empty.csv"
        csv.write_text("")
        r = parse_results(csv)
        assert r.tip == {} and r.patches == {} and r.sections == {}

    def test_parse_rejects_wrong_header(self, tmp_path):
        csv = tmp_path / "bad.csv"
        csv.write_text("foo,bar,baz,qux\nrow1,row1,row1,1.0\n")
        with pytest.raises(ValueError, match="expected the 4-column"):
            parse_results(csv)

    def test_parse_skips_garbage_rows(self, tmp_path):
        csv = tmp_path / "noisy.csv"
        csv.write_text(dedent("""\
            category,name,key,value
            tip,tip,umax_m,1.0
            something_unparseable
            tip,tip,bad,not_a_float
            patch,p,k,2.0
            """))
        r = parse_results(csv)
        assert r.tip["tip"]["umax_m"] == 1.0
        assert r.patches["p"]["k"] == 2.0


# ---------------------------------------------------------------------------
# Drift analyser
# ---------------------------------------------------------------------------


def _result_with_patch_svm(value: float, patch_name: str = "HP_SPAR_r20") -> ConvergenceResult:
    r = ConvergenceResult()
    r.patches[patch_name] = {"L3_BOT_svm_Pa": value}
    return r


def _result_with_tip(value: float) -> ConvergenceResult:
    r = ConvergenceResult()
    r.tip["tip"] = {"umax_m": value}
    return r


class TestAnalyse:
    def test_drifts_decrease_with_refinement_passes(self):
        # Linear approach to a limit: drifts go 10%, 5%, 2%, 1%
        # Tip drift criterion is 1% → smallest passing h is 0.1.
        results = {
            0.8: _result_with_tip(10.0),
            0.4: _result_with_tip(11.0),     # +10% drift vs 0.8
            0.2: _result_with_tip(11.55),    # +5%  vs 0.4
            0.1: _result_with_tip(11.78),    # +2%  vs 0.2
        }
        rows = analyse(results, tip_drift_pct=2.5)
        assert len(rows) == 1
        row = rows[0]
        # Only one drift < 2.5%: between 0.2 and 0.1 (2% drift).
        assert row.passes_at_h == 0.1

    def test_no_passing_h_when_diverging(self):
        results = {
            0.8: _result_with_patch_svm(100.0),
            0.4: _result_with_patch_svm(120.0),  # +20%
            0.2: _result_with_patch_svm(150.0),  # +25%
            0.1: _result_with_patch_svm(200.0),  # +33%
        }
        rows = analyse(results, patch_drift_pct=5.0)
        # Should have at least the σ_vM row
        svm_row = next(r for r in rows
                       if r.category == "patch" and r.key.endswith("svm_Pa"))
        assert svm_row.passes_at_h is None

    def test_ignores_non_svm_patch_keys(self):
        # Patch dict contains volume + count too; analyser should only
        # produce a drift row for *_svm_Pa.
        r1 = ConvergenceResult()
        r1.patches["P"] = {"L3_BOT_svm_Pa": 100.0,
                            "L3_BOT_volu_m3": 1.0,
                            "L3_BOT_n_elem": 10.0}
        r2 = ConvergenceResult()
        r2.patches["P"] = {"L3_BOT_svm_Pa": 102.0,
                            "L3_BOT_volu_m3": 1.0,
                            "L3_BOT_n_elem": 20.0}
        rows = analyse({0.2: r2, 0.1: r1})
        patch_rows = [r for r in rows if r.category == "patch"]
        assert len(patch_rows) == 1
        assert patch_rows[0].key == "L3_BOT_svm_Pa"

    def test_print_drift_table_runs(self, capsys):
        # Smoke test — table printer shouldn't crash on a mixed row set.
        results = {
            0.2: _result_with_tip(10.0),
            0.1: _result_with_tip(10.05),
        }
        rows = analyse(results)
        print_drift_table(rows)
        out = capsys.readouterr().out
        assert "drift_finest_%" in out
        assert "tip" in out
