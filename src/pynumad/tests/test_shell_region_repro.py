"""Focused unit reproducer for the mesh-quality wall at the root transition.

Background
----------
When opposite edges of a structured `ShellRegion` have unequal element counts
(e.g. ``edgeEls = [5, 5, 4, 5]``), the node-pulling branches in
``createShellMesh`` snap boundary-row nodes onto a coarser segment via a
nearest-neighbour rule and then collapse coincident quads to triangles.

At the **ShellRegion unit level on flat keypoints** this produces:
  - higher aspect ratios than the matched-edge case
  - some collapsed triangles
  - **no Jacobian flips** in natural space

The Jacobian flips that crash ANSYS only appear when this distorted natural-
space topology is mapped through the quad3 Lagrange basis onto a strongly
non-uniform 3D blade-shell surface (the root-to-airfoil transition). That
end-to-end behaviour is exercised by ``test_iea22_mesh_quality.py``.

These unit tests therefore serve two distinct purposes:

  1. **Control tests** (``edgeEls=[5,5,5,5]`` etc.) — must always be clean.
     Failure here means the bug has spread beyond the node-pulling branches.

  2. **Characterization tests** — pin the current asymmetric-edge behaviour
     (collapsed-triangle count, max aspect ratio) as golden values. Any fix
     that changes the topology at this level must explicitly update the
     golden values, forcing the author to confirm the change is intentional.

See ``docs/dev/mesh_bug_trace.md`` for the full algorithm walkthrough.
"""

import numpy as np
import pytest

from pynumad.mesh_gen.shell_region import ShellRegion
from pynumad.testing.mesh_quality import (
    analyse_mesh,
    assert_mesh_clean,
    assert_no_jacobian_flips,
)


def _quad3_keypoints_unit_square() -> np.ndarray:
    """16 keypoints mapping natural-space [-1,1]^2 to the unit square in z=0."""
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


def _build_structured(edge_els, keypoints):
    region = ShellRegion(
        regType="quad3",
        keyPoints=keypoints,
        numEdgeEls=edge_els,
        meshMethod="structured",
    )
    return region.createShellMesh()


# =========================================================================
# CONTROL: matched edges must always produce clean meshes
# =========================================================================


@pytest.mark.parametrize("n", [3, 4, 5, 6, 8, 10])
def test_matched_edges_clean(n):
    """When all four edges have the same count, no node-pulling fires.
    Resulting mesh must have zero Jacobian flips, no coincident nodes,
    no collapsed triangles."""
    mesh = _build_structured([n, n, n, n], _quad3_keypoints_unit_square())
    report = assert_mesh_clean(mesh, msg_prefix=f"edgeEls=[{n},{n},{n},{n}]")
    assert report.n_valid_tris == 0, f"matched edges should produce no triangles: {report.summary()}"
    assert report.n_valid_quads == n * n


# =========================================================================
# CHARACTERIZATION: pin current asymmetric-edge behaviour
# =========================================================================


def test_edgeEls_5_5_4_5_unit_level_topology():
    """The canonical paper2_fem failure pattern, isolated to ShellRegion.

    At this level (unit-square keypoints, no 3D blade mapping) the bug
    manifests as elevated aspect ratio + one collapsed triangle, NOT as a
    Jacobian flip. The flip only emerges downstream in
    ``test_iea22_mesh_quality.py``. We pin the unit-level topology so any
    algorithm change is visible here too.
    """
    mesh = _build_structured([5, 5, 4, 5], _quad3_keypoints_unit_square())
    report = analyse_mesh(mesh)
    # No jflips at this level (the bug only manifests with curved 3D mapping)
    assert report.n_jacobian_flips == 0, report.summary()
    # Golden values for the current (buggy) algorithm. A fix at
    # mesh_gen.py:322-340 will prevent ShellRegion from EVER receiving this
    # edgeEls combination, so this test will be untouched. A fix at
    # shell_region.py itself would change the topology here — and that's
    # exactly what we want this test to catch.
    assert report.n_valid_tris == 1, f"expected 1 collapsed tri, got {report.summary()}"
    assert report.n_valid_quads == 24
    assert report.max_aspect_ratio == pytest.approx(1.25, rel=1e-3)


def test_edgeEls_5_5_3_5_more_aggressive_mismatch():
    """ee[2] = 3 vs ee[0] = 5: larger mismatch, more triangles created."""
    mesh = _build_structured([5, 5, 3, 5], _quad3_keypoints_unit_square())
    report = analyse_mesh(mesh)
    assert report.n_jacobian_flips == 0, report.summary()
    assert report.n_valid_tris == 2
    assert report.max_aspect_ratio == pytest.approx(5.0 / 3.0, rel=1e-3)


@pytest.mark.parametrize(
    "edge_els, expected_tris",
    [
        ([4, 5, 5, 5], 1),  # bottom shorter: ee[0] < ee[2]
        ([5, 5, 4, 5], 1),  # top shorter:    ee[2] < ee[0]  -- canonical
        ([5, 4, 5, 5], 1),  # right shorter:  ee[1] < ee[3]
        ([5, 5, 5, 4], 1),  # left shorter:   ee[3] < ee[1]
    ],
)
def test_asymmetric_branches_all_clean_jflip_free(edge_els, expected_tris):
    """All four node-pulling branches fire correctly at the unit level:
    one collapsed triangle, no Jacobian flips. If a future ShellRegion
    refactor breaks any branch, this test catches it."""
    mesh = _build_structured(edge_els, _quad3_keypoints_unit_square())
    report = analyse_mesh(mesh)
    assert report.n_jacobian_flips == 0, f"{edge_els}: {report.summary()}"
    assert report.n_valid_tris == expected_tris, f"{edge_els}: {report.summary()}"


# =========================================================================
# FORENSIC: report-only test, prints metrics for the canonical case
# =========================================================================


def test_diagnostic_report_for_canonical_case(capsys):
    """Run the canonical failing case and print metrics. Pure forensic data
    for future debuggers comparing pre- and post-fix output. Always passes."""
    mesh = _build_structured([5, 5, 4, 5], _quad3_keypoints_unit_square())
    report = analyse_mesh(mesh)
    print(f"\nedgeEls=[5,5,4,5] unit-level: {report.summary()}")
    print(f"  jflip indices: {report.jacobian_flip_indices}")
    print(f"  coincident node pairs: {report.coincident_node_pairs}")
    assert report.n_elements > 0
