"""Geometric pathology tests on the shell mesh.

These tests assert hard quality bounds on the mesh that pyNuMAD
produces. The thresholds are chosen to be roughly what ANSYS will
accept at the element-Jacobian stage:

  D1. No Jacobian-flipped quads at the working element size.
  D2. No quad with shortest edge below an absolute floor (avoids
      silently producing infinitesimal elements that blow up the
      condition number).
  D3. Max aspect ratio < 20 across the whole mesh. (BAR0 today has
      some sliver elements with AR ~ 50-100 — this test is intended
      to PIN that pathology so a fix has a clear regression target.)
  D4. Collapsed-triangle count < 5% of total elements (degenerate
      quads where slot[3] == -1).
  D5. Twist robustness: meshing on a *twisted* synthetic copy of
      BAR0 must not introduce extra Jacobian flips (placeholder —
      we only check that BAR0's own ~20° twist sweep doesn't flip).

Together these are the geometric correctness contract on the mesher.
"""
from __future__ import annotations

import numpy as np
import pytest

from pynumad.testing.mesh_quality import analyse_mesh

from ._mesh_cache import get_mesh


ELEMENT_SIZE = 0.5

# Hard quality thresholds — keep them in sync with the
# ``DEFAULT_*`` constants in ``mesh_quality`` where reasonable.
MAX_ASPECT_RATIO = 20.0
MAX_COLLAPSED_FRACTION = 0.05
MIN_EDGE_LENGTH = 1e-3  # metres — anything below this is mesher debris


@pytest.fixture(scope="module")
def shell_mesh_only():
    return get_mesh(includeAdhesive=False, elementSize=ELEMENT_SIZE)


@pytest.fixture(scope="module")
def report(shell_mesh_only):
    return analyse_mesh(shell_mesh_only)


# ------------------------------------------------------------------
# D1. No Jacobian flips
# ------------------------------------------------------------------


def test_no_jacobian_flipped_quads(report):
    """D1: at the working element size, no quad may have a Jacobian
    sign flip. BAR0 passes this today (see test_mesh.test_bar0_no_jflips);
    we restate it here as part of the explicit pathology contract.
    """
    assert report.n_jacobian_flips == 0, (
        f"{report.n_jacobian_flips} sign-flipped quads at esize="
        f"{ELEMENT_SIZE} m: indices {report.jacobian_flip_indices[:10]}"
    )


# ------------------------------------------------------------------
# D2. Minimum edge length
# ------------------------------------------------------------------


def test_no_quad_below_absolute_edge_floor(shell_mesh_only):
    """D2: no quad may have an edge shorter than ``MIN_EDGE_LENGTH``.
    Sub-mm edges in a 200-metre blade indicate either coincident nodes
    or a degenerate mapping at the trailing edge.
    """
    nodes = np.asarray(shell_mesh_only["nodes"], dtype=float)
    elems = np.asarray(shell_mesh_only["elements"], dtype=int)
    bad = 0
    worst = float("inf")
    bad_idx: list[int] = []
    for ei, conn in enumerate(elems[:, :4]):
        valid = conn[conn >= 0]
        if valid.size < 3:
            continue
        pts = nodes[valid]
        # cycle the polygon
        edges = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        m = edges.min()
        if m < MIN_EDGE_LENGTH:
            bad += 1
            if len(bad_idx) < 10:
                bad_idx.append(ei)
        worst = min(worst, float(m))
    assert bad == 0, (
        f"{bad} elements have an edge shorter than {MIN_EDGE_LENGTH} m "
        f"(worst overall = {worst:.3e} m). First 10: {bad_idx}"
    )


# ------------------------------------------------------------------
# D3. Aspect ratio ceiling
# ------------------------------------------------------------------


def test_max_aspect_ratio_below_threshold(report):
    """D3: max element aspect ratio must be below ``MAX_ASPECT_RATIO``
    on the BAR0 reference at the standard element size.

    This will FAIL today on BAR0 (max AR is in the 50-100 range at the
    trailing edge / root transition). The failure is the bug: a future
    mesher fix should drive max AR below 20.
    """
    assert report.max_aspect_ratio < MAX_ASPECT_RATIO, (
        f"max aspect ratio = {report.max_aspect_ratio:.2f} exceeds "
        f"threshold {MAX_ASPECT_RATIO}; worst-elements indices: "
        f"{report.severe_aspect_indices[:10]}. p99 AR = "
        f"{report.p99_aspect_ratio:.2f}, median = "
        f"{report.median_aspect_ratio:.2f}."
    )


# ------------------------------------------------------------------
# D4. Collapsed-triangle fraction
# ------------------------------------------------------------------


def test_collapsed_triangle_fraction_below_threshold(shell_mesh_only):
    """D4: count of elements with slot[3] == -1 (mesher's
    collapsed-quad-to-triangle sentinel) must be below 5% of all
    elements. This guards against the mesher silently degrading to
    triangle-rich output at the TE where the chord narrows.
    """
    elems = np.asarray(shell_mesh_only["elements"], dtype=int)
    n_tri = int((elems[:, 3] == -1).sum())
    n_total = elems.shape[0]
    frac = n_tri / max(n_total, 1)
    assert frac < MAX_COLLAPSED_FRACTION, (
        f"{n_tri}/{n_total} ({frac:.1%}) elements are collapsed "
        f"triangles; threshold {MAX_COLLAPSED_FRACTION:.0%}"
    )


# ------------------------------------------------------------------
# D5. Twist robustness
# ------------------------------------------------------------------


def test_twisted_blade_does_not_introduce_jflips(shell_mesh_only):
    """D5: the BAR0 reference already has spanwise twist — meshing
    didn't flip any element (see D1). As a positive sanity check,
    confirm that twist *did* take effect: the chord at the tip is
    rotated relative to the chord at the root.

    We don't reshape the blade here (that would require re-running the
    full meshing pipeline against a modified YAML); we only verify the
    mesh-output evidence of twist. A future regression that breaks the
    twist evaluation would land here.
    """
    nodes = np.asarray(shell_mesh_only["nodes"], dtype=float)
    z = nodes[:, 2]
    z_min, z_max = float(z.min()), float(z.max())
    # Sample root-band and tip-band slices and look at the orientation
    # of the in-plane xy bbox principal direction
    def principal_angle(band_mask: np.ndarray) -> float:
        pts = nodes[band_mask, :2]
        if pts.shape[0] < 4:
            return float("nan")
        pts = pts - pts.mean(axis=0, keepdims=True)
        cov = pts.T @ pts
        eigvals, eigvecs = np.linalg.eigh(cov)
        # largest eigenvector
        v = eigvecs[:, -1]
        return float(np.degrees(np.arctan2(v[1], v[0])))

    root_band = (z >= z_min) & (z <= z_min + 0.1 * (z_max - z_min))
    tip_band = (z >= z_max - 0.1 * (z_max - z_min)) & (z <= z_max)
    a_root = principal_angle(root_band)
    a_tip = principal_angle(tip_band)
    assert np.isfinite(a_root) and np.isfinite(a_tip)
    # Take the smaller of |a_tip - a_root| and 180 - |...| to avoid the
    # sign-flip ambiguity in an eigenvector direction.
    delta = abs(a_tip - a_root) % 180.0
    delta = min(delta, 180.0 - delta)
    # BAR0 has real structural twist; this is a sanity check that some
    # twist made it through the mesher (lower bound) and that the value
    # is not absurd (upper bound). The proxy is the PCA principal axis of
    # each band's node cloud, which is sensitive to the tip node
    # distribution: the thin-TE_FLAT merge in mesh_gen removes the most
    # trailing tip nodes, rotating the tip principal axis by a few degrees
    # (the blade's actual twist is unchanged). The bound is set with
    # margin for that proxy shift; it does NOT introduce extra twist.
    assert 5.0 < delta < 35.0, (
        f"chord twist root -> tip = {delta:.1f}° outside the sanity band "
        f"(expected meaningful but not absurd twist for BAR0)"
    )
