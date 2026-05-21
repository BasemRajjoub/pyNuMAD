"""Property-based tests for ShellRegion's structured-mesh algorithm.

These tests systematically scan the `edgeEls` parameter space using
``hypothesis`` to ensure the unit-level invariants hold across topologies
much broader than the hand-picked cases in
``test_shell_region_repro.py``. The two invariants checked here:

  1. ``edges_matched``  =>  mesh is fully clean (no jflips, no slivers,
                            no coincident pairs, no collapsed triangles)
  2. ``edges_mismatched``  =>  mesh has no sign-flipped Jacobians at the
                               unit-square level; collapsed triangles
                               are allowed (snap-and-merge is supposed
                               to handle the geometric impossibility)

After the Phase-4 fix lands, callers will no longer be able to pass
mismatched edgeEls into ShellRegion (the helper in mesh_gen will force
opposite-edge equality), but ShellRegion still has to behave sanely if
someone passes mismatched edges directly. These tests guarantee that.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from pynumad.mesh_gen.shell_region import ShellRegion
from pynumad.testing.mesh_quality import (
    analyse_mesh,
    assert_mesh_clean,
    assert_no_jacobian_flips,
)


def _quad3_unit_square_keypoints() -> np.ndarray:
    r3 = 1.0 / 3.0
    nat = np.array(
        [
            [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0],
            [-r3, -1.0], [r3, -1.0], [1.0, -r3], [1.0, r3],
            [r3, 1.0], [-r3, 1.0], [-1.0, r3], [-1.0, -r3],
            [-r3, -r3], [r3, -r3], [r3, r3], [-r3, r3],
        ]
    )
    kp = np.zeros((16, 3))
    kp[:, :2] = 0.5 * (nat + 1.0)
    return kp


_KP = _quad3_unit_square_keypoints()


def _build_quad3(edge_els):
    region = ShellRegion(
        regType="quad3",
        keyPoints=_KP,
        numEdgeEls=list(edge_els),
        meshMethod="structured",
        name=f"prop_test_{edge_els}",
    )
    return region.createShellMesh()


# Strategy: 4 integers in [2, 10]. Keeps the search space tractable and
# matches the realistic range of per-region edge counts on IEA-22-class
# blades at element sizes 0.10-0.80 m.
_edge_count = st.integers(min_value=2, max_value=10)
_edge_els_4 = st.lists(_edge_count, min_size=4, max_size=4)


@settings(max_examples=80, deadline=2000)
@given(edge_els=_edge_els_4)
def test_no_sign_flipped_jacobian_for_any_edge_combo(edge_els):
    """Across ALL edgeEls combinations in [2,10]^4, the unit-square mesh
    must never produce a sign-flipped Jacobian. Triangles from collapse
    are allowed; slivers are not asserted against here (they can happen
    with very mismatched edges)."""
    mesh = _build_quad3(edge_els)
    assert_no_jacobian_flips(mesh, msg_prefix=f"edgeEls={edge_els}")


@settings(max_examples=40, deadline=2000)
@given(
    a=_edge_count,
    b=_edge_count,
)
def test_matched_edges_always_produce_clean_mesh(a, b):
    """For any matched (a, b, a, b) pattern in [2,10]^2, the mesh must be
    fully clean: no triangles, no slivers, no jflips, no coincident pairs."""
    edge_els = [a, b, a, b]
    mesh = _build_quad3(edge_els)
    report = assert_mesh_clean(mesh, msg_prefix=f"edgeEls={edge_els}")
    assert report.n_valid_tris == 0, f"matched edges should not collapse: {report.summary()}"
    assert report.n_valid_quads == a * b


@settings(max_examples=80, deadline=2000)
@given(edge_els=_edge_els_4)
def test_element_count_matches_xnodes_ynodes(edge_els):
    """The total element count must equal max(ee0,ee2) * max(ee1,ee3) — the
    algorithm builds an xNodes-by-yNodes grid based on the larger of each
    opposite-edge pair."""
    mesh = _build_quad3(edge_els)
    expected = max(edge_els[0], edge_els[2]) * max(edge_els[1], edge_els[3])
    actual = mesh["elements"].shape[0]
    assert actual == expected, (
        f"edgeEls={edge_els}: expected {expected} elements "
        f"(max(ee0,ee2)*max(ee1,ee3)), got {actual}"
    )


@pytest.mark.parametrize(
    "edge_els",
    [
        # extreme mismatch — exercises the node-pulling under more stress
        [10, 5, 2, 5],
        [2, 5, 10, 5],
        [5, 10, 5, 2],
        [5, 2, 5, 10],
    ],
)
def test_extreme_mismatch_still_no_sign_flip(edge_els):
    """Even with 5:1 opposite-edge ratios, no sign-flipped Jacobian should
    appear at the unit-square level."""
    mesh = _build_quad3(edge_els)
    assert_no_jacobian_flips(mesh, msg_prefix=f"extreme {edge_els}")
