"""Unit tests for pynumad.analysis.convergence.forces_src."""
from __future__ import annotations

import numpy as np
import pytest

from pynumad.analysis.convergence.forces_src import (
    read_forces_src,
    section_resultants,
    section_resultants_at,
)


def test_read_forces_src_handles_ansys_1_indexed_ids(tmp_path):
    """ANSYS labels are 1-indexed; we should write to row label-1."""
    path = tmp_path / "forces.src"
    path.write_text(
        "finish\n/prep7\n"
        "f,1,FX,1000.0\n"
        "f,2,FY,-500.0\n"
        "f,3,FZ,250.0\n"
    )
    F = read_forces_src(path, n_nodes=5)
    assert F.shape == (5, 3)
    assert F[0, 0] == 1000.0
    assert F[1, 1] == -500.0
    assert F[2, 2] == 250.0
    np.testing.assert_array_equal(F[3:], np.zeros((2, 3)))


def test_read_forces_src_handles_e_notation_and_signs(tmp_path):
    path = tmp_path / "f.src"
    path.write_text(
        "f,1,FX, 1.23e+05\n"
        "f,1,FY,-4.5E-2\n"
        "f,1,FZ, .5\n"
    )
    F = read_forces_src(path, n_nodes=2)
    assert F[0, 0] == pytest.approx(1.23e5)
    assert F[0, 1] == pytest.approx(-0.045)
    assert F[0, 2] == pytest.approx(0.5)


def test_read_forces_src_ignores_unknown_lines(tmp_path):
    path = tmp_path / "noisy.src"
    path.write_text(
        "! a comment\n"
        "finish\n"
        "/prep7\n"
        "f,1,FX,100.0\n"
        "esel,s,elem,,1,5\n"
        "f,2,FY,50.0\n"
    )
    F = read_forces_src(path, n_nodes=3)
    assert F[0, 0] == 100.0
    assert F[1, 1] == 50.0


def test_section_resultants_returns_zero_outboard_of_tip():
    # One node, one applied load — but request a section beyond the
    # node. Nothing outboard → all zeros.
    nodes = np.array([[0, 0, 50.0]])
    F = np.array([[100.0, 0, 0]])
    r = section_resultants(nodes, F, z_section=100.0)
    assert all(v == 0 for v in r.values())


def test_section_resultants_pure_axial_load():
    # Single nodal load applied 10 m outboard of the section centroid:
    # the section sees Fx = applied Fx, all moments zero except My.
    nodes = np.array([[0, 0, 100.0]])
    F = np.array([[100.0, 0, 0]])      # 100 N in +x at z=100
    r = section_resultants(nodes, F, z_section=90.0)
    assert r["Fx_N"] == pytest.approx(100.0)
    assert r["Fy_N"] == pytest.approx(0.0)
    assert r["Fz_N"] == pytest.approx(0.0)
    # Moment about z=90: arm = (0, 0, 10); F = (100, 0, 0); arm x F = (0, 1000, 0)
    assert r["Mx_Nm"] == pytest.approx(0.0)
    assert r["My_Nm"] == pytest.approx(1000.0)
    assert r["Mz_Nm"] == pytest.approx(0.0)


def test_section_resultants_summation_of_two_loads():
    # Two loads, each contributes a moment; check superposition.
    nodes = np.array([
        [0, 0, 50.0],
        [0, 0, 100.0],
    ])
    F = np.array([
        [200.0, 0, 0],
        [300.0, 0, 0],
    ])
    r = section_resultants(nodes, F, z_section=0.0)
    assert r["Fx_N"] == pytest.approx(500.0)
    # arm x F: (0,0,50)x(200,0,0)=(0,10000,0); (0,0,100)x(300,0,0)=(0,30000,0)
    # sum My = 40000
    assert r["My_Nm"] == pytest.approx(40000.0)


def test_section_resultants_excludes_inboard_nodes():
    # Two loads — one outboard of the section, one inboard. Only the
    # outboard one contributes.
    nodes = np.array([
        [0, 0, 50.0],
        [0, 0, 100.0],
    ])
    F = np.array([
        [-999.0, 0, 0],   # inboard, ignored
        [200.0, 0, 0],    # outboard, counted
    ])
    r = section_resultants(nodes, F, z_section=75.0)
    assert r["Fx_N"] == pytest.approx(200.0)
    assert r["My_Nm"] == pytest.approx(200.0 * 25.0)


def test_section_resultants_at_dispatches_correctly(tmp_path):
    nodes = np.array([
        [0, 0, 50.0],
        [0, 0, 100.0],
    ])
    path = tmp_path / "f.src"
    path.write_text(
        "f,1,FX,200.0\n"
        "f,2,FX,300.0\n"
    )
    results = section_resultants_at(nodes, path, [0.0, 75.0])
    assert results[0.0]["Fx_N"] == pytest.approx(500.0)
    assert results[75.0]["Fx_N"] == pytest.approx(300.0)
