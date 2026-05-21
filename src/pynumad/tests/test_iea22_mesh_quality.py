"""Integration reproducer for the IEA-22 root-transition mesh-quality wall.

Background
----------
Below ~0.45 m element size on the IEA-22 reference blade, ``shell_mesh_general``
produces Jacobian-flipped quads at the root transition (z ~ 5-10 m). The flip
count grows monotonically with refinement (paper2_fem characterisation):

    esize=0.80m → 19 jflips     esize=0.30m → 109 jflips
    esize=0.45m → 108 jflips    esize=0.20m → 131 jflips
    esize=0.10m → 226 jflips

ANSYS aborts below 0.40 m. The bug is in the node-pulling code triggered by
``edgeEls[0] != edgeEls[2]`` (opposite chord edges have unequal element counts),
which produces warped quads that flip when mapped through the quad3 Lagrange
basis to the curved root-transition surface.

These tests pin the failure and define the Phase-4 acceptance criterion:
zero Jacobian flips at every refinement level from 0.80 m down to 0.10 m.

The full IEA-22 YAML is large (the IEA-22-280-RWT submodule); we locate it
relative to the package root so the test runs from anywhere in the repo.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


def _find_iea22_yaml() -> Path | None:
    """Locate the IEA-22 windIO YAML by walking up from this file. Returns
    None if not present (the IEA-22 submodule is optional)."""
    here = Path(__file__).resolve()
    candidates = [
        # pyNuMAD lives under /bigwork/.../pce/pyNuMAD; IEA-22 is a sibling
        here.parents[5] / "IEA-22-280-RWT" / "windIO" / "IEA-22-280-RWT.yaml",
        here.parents[4] / "IEA-22-280-RWT" / "windIO" / "IEA-22-280-RWT.yaml",
        here.parents[6] / "IEA-22-280-RWT" / "windIO" / "IEA-22-280-RWT.yaml",
    ]
    for p in candidates:
        if p.is_file():
            return p
    env = os.environ.get("PYNUMAD_IEA22_YAML")
    if env and Path(env).is_file():
        return Path(env)
    return None


IEA22_YAML = _find_iea22_yaml()
_skip_no_iea22 = pytest.mark.skipif(
    IEA22_YAML is None,
    reason="IEA-22-280-RWT YAML not found; set PYNUMAD_IEA22_YAML to enable",
)


# Root-transition geometric band where the bug lives (from paper2_fem
# characterisation, see continue.md). All assertions on jflip count are
# global-mesh, not region-restricted, because the root band is where they
# cluster but a robust fix should eliminate them everywhere.
ROOT_Z_LOW, ROOT_Z_HIGH = 5.0, 10.0


@pytest.fixture(scope="module")
def iea22_blade():
    """Load the IEA-22 blade once per test module — YAML parsing is slow."""
    if IEA22_YAML is None:
        pytest.skip("IEA-22 YAML not available")
    import pynumad as pynu

    blade = pynu.Blade()
    blade.read_yaml(str(IEA22_YAML))
    return blade


def _build_iea22_mesh(blade, element_size: float) -> dict:
    from pynumad.mesh_gen.mesh_gen import shell_mesh_general

    return shell_mesh_general(
        blade,
        forSolid=False,
        includeAdhesive=False,
        elementSize=element_size,
    )


# =========================================================================
# Phase-4 acceptance: zero Jacobian flips at every refinement level
# =========================================================================


@_skip_no_iea22
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="Phase-4 fix not yet applied: opposite-edge mismatch creates "
    "sliver/jflip quads at root transition; expected to fail today",
)
@pytest.mark.parametrize("element_size", [0.80, 0.60, 0.45])
def test_iea22_no_jflips_coarse(iea22_blade, element_size):
    """Coarse element sizes (≥ 0.45 m) currently work in ANSYS but already
    have a non-trivial jflip count. A fix must drive these to zero.

    The handoff notes 19 / 44 / 108 flips at 0.80 / 0.60 / 0.45 m today.
    """
    from pynumad.testing.mesh_quality import analyse_mesh, assert_no_jacobian_pathology

    mesh = _build_iea22_mesh(iea22_blade, element_size)
    report = analyse_mesh(mesh)
    print(
        f"\nIEA-22 @ {element_size} m: nodes={report.n_nodes}, elems={report.n_elements}, "
        f"jflips={report.n_jacobian_flips}, low_jac={report.n_low_jacobian}, "
        f"severe_AR={report.n_severe_aspect_ratio}, "
        f"min_jac_ratio={report.min_jacobian_ratio:.2e}, "
        f"max_aspect={report.max_aspect_ratio:.2f}"
    )
    assert_no_jacobian_pathology(mesh, msg_prefix=f"IEA-22 @ {element_size} m")


@_skip_no_iea22
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="Phase-4 fix not yet applied: ANSYS aborts on these sizes today",
)
@pytest.mark.parametrize("element_size", [0.40, 0.35, 0.30, 0.20, 0.10])
def test_iea22_no_jflips_fine(iea22_blade, element_size):
    """Fine element sizes (< 0.40 m) currently cause ANSYS to abort. These
    must pass for paper2_fem to be unblocked. Acceptance criterion (c).

    The handoff notes 107 / 98 / 109 / 131 / 226 flips at 0.40 / 0.35 /
    0.30 / 0.20 / 0.10 m today.
    """
    from pynumad.testing.mesh_quality import analyse_mesh, assert_no_jacobian_pathology

    mesh = _build_iea22_mesh(iea22_blade, element_size)
    report = analyse_mesh(mesh)
    print(
        f"\nIEA-22 @ {element_size} m: nodes={report.n_nodes}, elems={report.n_elements}, "
        f"jflips={report.n_jacobian_flips}, low_jac={report.n_low_jacobian}, "
        f"severe_AR={report.n_severe_aspect_ratio}, "
        f"min_jac_ratio={report.min_jacobian_ratio:.2e}, "
        f"max_aspect={report.max_aspect_ratio:.2f}"
    )
    assert_no_jacobian_pathology(mesh, msg_prefix=f"IEA-22 @ {element_size} m")


# =========================================================================
# Forensic: produce the convergence table the handoff doc references
# =========================================================================


@_skip_no_iea22
@pytest.mark.integration
@pytest.mark.slow
def test_iea22_jflip_convergence_table(iea22_blade, capsys):
    """Forensic: rebuild the paper2_fem convergence table inside the test
    suite. This test is INFORMATIONAL — it never fails. Use it to track
    how a candidate fix affects the jflip count at each refinement level.

    Run with::

        pytest src/pynumad/tests/test_iea22_mesh_quality.py::test_iea22_jflip_convergence_table -s
    """
    from pynumad.testing.mesh_quality import analyse_mesh, select_elements_in_z_band

    sizes = [0.80, 0.60, 0.45, 0.30, 0.20, 0.10]
    print(f"\n{'esize':>6s} {'n_elem':>8s} {'jflips':>7s} {'root_jflips':>12s} {'max_AR':>7s}")
    print("-" * 50)
    for s in sizes:
        try:
            mesh = _build_iea22_mesh(iea22_blade, s)
        except Exception as e:
            print(f"{s:>6.2f}  build failed: {e}")
            continue
        report = analyse_mesh(mesh)
        root_mask = select_elements_in_z_band(mesh, ROOT_Z_LOW, ROOT_Z_HIGH)
        root_jflip_count = sum(
            1 for ei in report.jacobian_flip_indices if root_mask[ei]
        )
        print(
            f"{s:>6.2f} {report.n_elements:>8d} {report.n_jacobian_flips:>7d} "
            f"{root_jflip_count:>12d} {report.max_aspect_ratio:>7.2f}"
        )
    assert True  # always passes


# =========================================================================
# Phase-4 acceptance: monotonic eigenfreq convergence (placeholder)
# =========================================================================
#
# Acceptance criterion (b) — first 10 eigenfrequencies converge monotonically
# through 0.10 m — requires an ANSYS solve and lives at the cluster level.
# It is covered by paper2_fem's sbatch_mesh_convergence.sh, not here. This
# file deliberately stays a pyNuMAD-only test (no ANSYS dependency).
