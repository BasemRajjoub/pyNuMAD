"""Tests for native SHELL281 (quadratic) deck generation in the ANSYS writer.

pyNuMAD's ANSYS shell writer historically emitted 4-node elements for
SHELL281 and relied on ``EMID,ADD`` to retrofit mid-side nodes. That
silently skipped some boundary elements (IEA-22 root TE_FLAT, shear-web
tip), leaving SHELL281 elements with dropped mid-side nodes and aborting
the solve. The writer now generates the mid-side nodes in Python
(``_quadratic_midside_nodes``) and emits 8-node elements directly, which
guarantees coverage.

Unit tests cover the mid-side node builder; integration tests parse a
real BAR0 SHELL281 deck and assert every element is a valid 8-node
element with all mid-side nodes defined. A regression test confirms the
SHELL181 path is unchanged (still 4-node).
"""
from __future__ import annotations

import os
import re

import numpy as np
import pytest

from pynumad.analysis.ansys.write import (
    _quadratic_midside_nodes,
    write_ansys_shell_model,
)

from ._mesh_cache import get_blade, get_mesh, BAR0_YAML


# ----------------------------------------------------------------------
# Unit: mid-side node builder
# ----------------------------------------------------------------------
def test_midside_nodes_shared_across_common_edge():
    """Two quads sharing edge (1,2) must share a single mid-side node on
    that edge -> 7 unique mid-side nodes for 2 quads (4+4 edges minus the
    1 shared)."""
    # quad A: corners 0,1,2,3 ; quad B: corners 1,4,5,2 (shares edge 1-2)
    nodes = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],   # 0..3
        [2, 0, 0], [2, 1, 0],                          # 4,5
    ], dtype=float)
    elements = np.array([[0, 1, 2, 3], [1, 4, 5, 2]])
    edge_mid, mid_nodes = _quadratic_midside_nodes(nodes, elements, first_id=100)
    assert len(mid_nodes) == 7
    # shared edge (1,2) -> exactly one node, referenced by both quads
    assert (1, 2) in edge_mid
    # ids are contiguous from first_id
    ids = [m[0] for m in mid_nodes]
    assert ids == list(range(100, 107))


def test_midside_node_sits_at_edge_midpoint():
    nodes = np.array([[0, 0, 0], [2, 0, 0], [2, 4, 0], [0, 4, 0]], dtype=float)
    elements = np.array([[0, 1, 2, 3]])
    edge_mid, mid_nodes = _quadratic_midside_nodes(nodes, elements, first_id=1)
    by_id = {m[0]: np.array(m[1:]) for m in mid_nodes}
    # edge (0,1): midpoint (1,0,0)
    assert np.allclose(by_id[edge_mid[(0, 1)]], [1, 0, 0])
    # edge (1,2): midpoint (2,2,0)
    assert np.allclose(by_id[edge_mid[(1, 2)]], [2, 2, 0])


def test_midside_skips_degenerate_quads():
    """A collapsed quad (fewer than 4 unique corners) is not written as an
    element and must not contribute mid-side nodes."""
    nodes = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    elements = np.array([[0, 1, 2, 3], [0, 1, 1, 2]])  # 2nd is degenerate
    _, mid_nodes = _quadratic_midside_nodes(nodes, elements, first_id=1)
    assert len(mid_nodes) == 4  # only the valid quad's 4 edges


# ----------------------------------------------------------------------
# Integration: parse a real BAR0 deck
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def bar0_mesh():
    return get_mesh(includeAdhesive=False, elementSize=0.5)


def _write_deck(blade, mesh, element_type, workdir):
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        mac = write_ansys_shell_model(
            blade, mesh, {"elementType": element_type, "blade_name": "BAR0"}
        )
        return open(mac).read()
    finally:
        os.chdir(cwd)


def test_shell281_deck_all_elements_are_8node(bar0_mesh, tmp_path):
    blade = get_blade(BAR0_YAML)
    txt = _write_deck(blade, bar0_mesh, "281", tmp_path)
    e8 = re.findall(r"^e, \d+,\d+,\d+,\d+,\d+,\d+,\d+,\d+", txt, re.M)
    e4 = re.findall(r"^e, \d+, \d+, \d+, \d+  !", txt, re.M)
    n_quads = int(sum(np.unique(bar0_mesh["elements"][i, :4]).size == 4
                      for i in range(bar0_mesh["elements"].shape[0])))
    assert len(e8) == n_quads, f"expected {n_quads} 8-node elements, got {len(e8)}"
    assert len(e4) == 0, f"{len(e4)} elements still emitted as 4-node under SHELL281"


def test_shell281_deck_every_referenced_midside_node_is_defined(bar0_mesh, tmp_path):
    """No element may reference a mid-side node that isn't defined with an
    ``n,`` card — that is exactly the 'dropped mid-side node' failure the
    EMID path produced."""
    blade = get_blade(BAR0_YAML)
    txt = _write_deck(blade, bar0_mesh, "281", tmp_path)
    defined = set(int(m) for m in re.findall(r"^n, (\d+),", txt, re.M))
    referenced = set()
    for line in re.findall(r"^e, ([\d,]+)", txt, re.M):
        ids = [int(x) for x in line.split(",") if x.strip()]
        if len(ids) >= 8:
            referenced.update(ids[:8])
    missing = referenced - defined
    assert not missing, f"{len(missing)} mid-side/corner node ids referenced but undefined"


def test_shell181_deck_still_4node(bar0_mesh, tmp_path):
    """Regression: the SHELL181 path is unchanged — elements stay 4-node and
    no mid-side node section is emitted."""
    blade = get_blade(BAR0_YAML)
    txt = _write_deck(blade, bar0_mesh, "181", tmp_path)
    e4 = re.findall(r"^e, \d+, \d+, \d+, \d+  !", txt, re.M)
    e8 = re.findall(r"^e, \d+,\d+,\d+,\d+,\d+,\d+,\d+,\d+", txt, re.M)
    assert len(e8) == 0
    assert len(e4) > 0
    assert "MIDSIDE NODES" not in txt
