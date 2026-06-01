"""Tests for the thin-TE_FLAT -> TE_REINF merge in ``mesh_gen``.

The merge folds a thin trailing-edge "flat" strip into the adjacent
TE_REINF region when the flat's chord is thinner than
``elementSize / _MERGE_TE_FLAT_AR``. Those lone 1-element-wide strips
otherwise become high-aspect-ratio slivers that pass the static AR<20
check but warp past ANSYS's element-formulation limit under NLGEOM at
large tip deflection (documented in docs/dev/TODO_shell_mesh_pipeline.md
#1).

Two layers of tests:

* **Unit** — the per-station merge *decision* (`_station_region_plan`)
  and the arc-length-resampled merged patch builder
  (`_shell_kp_merged`), on synthetic spline grids. Fast, deterministic,
  no full mesh build.
* **Integration** — the merged BAR0 shell mesh: TE region aspect ratio
  is bounded (no slivers), and the absorbing TE_REINF stays a single
  node-connected component (trailing-edge closure preserved).
"""
from __future__ import annotations

import numpy as np
import pytest

from pynumad.mesh_gen.mesh_gen import (
    _MERGE_TE_FLAT_AR,
    _compute_edge_nels,
    _shell_kp,
    _shell_kp_merged,
    _station_region_plan,
)
from pynumad.mesh_gen.shell_region import ShellRegion
from pynumad.testing import mesh_continuity as mc

from ._mesh_cache import get_mesh


# ----------------------------------------------------------------------
# Synthetic spline grid: monotone chord coordinate with controllable
# HP-flat (cols 0..3) and LP-flat (cols 33..36) widths.
# ----------------------------------------------------------------------
def _synth_spline(hp_flat_chord, lp_flat_chord, body_chord=1.0,
                  ncols=37, nrows=4, dz=0.5):
    xs = np.zeros(ncols)
    for c in range(1, 4):          # HP flat: cols 0->3
        xs[c] = xs[c - 1] + hp_flat_chord / 3.0
    for c in range(4, 34):         # main body: cols 3->33
        xs[c] = xs[c - 1] + body_chord / 30.0
    for c in range(34, 37):        # LP flat: cols 33->36
        xs[c] = xs[c - 1] + lp_flat_chord / 3.0
    X = np.tile(xs, (nrows, 1))
    Y = np.zeros((nrows, ncols))
    Z = np.tile((np.arange(nrows) * dz)[:, None], (1, ncols))
    return X, Y, Z


def _stack_indices(plan):
    return [stack_j for stack_j, _ in plan]


# ----------------------------------------------------------------------
# Unit: merge decision
# ----------------------------------------------------------------------
ELEMENT_SIZE = 0.20
THRESH = ELEMENT_SIZE / _MERGE_TE_FLAT_AR  # 0.05 m at es=0.20


def test_station_plan_merges_both_thin_flats():
    """Both flats below threshold -> both folded into TE_REINF.

    The flat stacks (0 = HP_TE_FLAT, 11 = LP_TE_FLAT) drop out and the
    reinf entries (1, 10) carry a ``("merge", ...)`` spec spanning the
    widened chord with the TE-most column preserved.
    """
    X, Y, Z = _synth_spline(0.5 * THRESH, 0.5 * THRESH)
    plan = _station_region_plan(X, Y, Z, 0, ELEMENT_SIZE)
    idx = _stack_indices(plan)
    assert 0 not in idx and 11 not in idx          # flats gone
    assert idx == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 10 regions, ordered
    specs = dict(plan)
    assert specs[1] == ("merge", 0, 6)             # HP reinf spans flat+reinf
    assert specs[10] == ("merge", 30, 36)          # LP reinf spans reinf+flat


def test_station_plan_keeps_both_wide_flats():
    """Both flats above threshold -> all 12 regions, no merge."""
    X, Y, Z = _synth_spline(4.0 * THRESH, 4.0 * THRESH)
    plan = _station_region_plan(X, Y, Z, 0, ELEMENT_SIZE)
    idx = _stack_indices(plan)
    assert idx == list(range(12))                  # all 12 present
    assert all(spec[0] == "cols" for _, spec in plan)


def test_station_plan_is_per_side():
    """Thin HP flat + wide LP flat -> HP merges, LP does not.

    This mirrors BAR0's asymmetric root TE, where the HP-side flat is a
    sliver but the LP-side flat is not. The merge decision is taken
    independently per side, by that side's own chord.
    """
    X, Y, Z = _synth_spline(0.5 * THRESH, 4.0 * THRESH)
    plan = _station_region_plan(X, Y, Z, 0, ELEMENT_SIZE)
    idx = _stack_indices(plan)
    assert 0 not in idx          # HP flat merged away
    assert 11 in idx             # LP flat retained
    specs = dict(plan)
    assert specs[1] == ("merge", 0, 6)
    assert specs[11] == ("cols", (33, 34, 35, 36))


def test_station_plan_merge_disabled_keeps_flats():
    """enable_merge=False (the solid-seed path) keeps all 12 regions even
    when both flats are thin. The merge is a shell-only NLGEOM fix; the
    solid pipeline has its own post-extrusion untangler."""
    X, Y, Z = _synth_spline(0.5 * THRESH, 0.5 * THRESH)
    plan = _station_region_plan(X, Y, Z, 0, ELEMENT_SIZE, enable_merge=False)
    assert _stack_indices(plan) == list(range(12))


# ----------------------------------------------------------------------
# Unit: merged patch geometry
# ----------------------------------------------------------------------
def test_shell_kp_merged_endpoints_lie_on_spline_columns():
    """The merged patch's four corners must land exactly on the original
    spline columns c_lo / c_hi at rows stPt and stPt+3 — that is what
    keeps the trailing-edge closure edge (and the outboard edge shared
    with TE_PANEL) unchanged, preserving continuity."""
    X, Y, Z = _synth_spline(0.02, 0.02)
    kp = _shell_kp_merged(X, Y, Z, 0, 0, 6)

    def col(row, c):
        return np.array([X[row, c], Y[row, c], Z[row, c]])

    assert np.allclose(kp[0], col(0, 0))   # G(0,0): row stPt, chord frac 0 -> c_lo
    assert np.allclose(kp[1], col(0, 6))   # G(0,3): row stPt, chord frac 1 -> c_hi
    assert np.allclose(kp[2], col(3, 6))   # G(3,3): row stPt+3, c_hi
    assert np.allclose(kp[3], col(3, 0))   # G(3,0): row stPt+3, c_lo


def _mesh_max_ar(kp, elementSize):
    nEl = _compute_edge_nels(kp, elementSize)
    assert nEl is not None
    reg = ShellRegion("quad3", kp, nEl, elType="quad", meshMethod="structured")
    md = reg.createShellMesh()
    nodes = np.asarray(md["nodes"], dtype=float)
    elems = np.asarray(md["elements"], dtype=int)
    worst = 0.0
    for conn in elems[:, :4]:
        v = conn[conn >= 0]
        if v.size < 3:
            continue
        p = nodes[v]
        e = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
        if e.min() > 0:
            worst = max(worst, float(e.max() / e.min()))
    return worst


def test_shell_kp_merged_resample_beats_naive_every_other_column():
    """Arc-length resampling is load-bearing: naively taking every-other
    spline column (0,2,4,6) bunches the cubic controls in the thin flat
    and produces high-AR slivers. The resampled merged patch must mesh to
    a substantially lower max aspect ratio on the same geometry.

    Regression guard: do not "simplify" _shell_kp_merged back to column
    selection.
    """
    # Thin HP flat (cols 0..3) + wider body (cols 3..6): the pathological
    # case the merge targets.
    X, Y, Z = _synth_spline(0.02, 0.02, body_chord=1.0)
    es = 0.05
    naive = _mesh_max_ar(_shell_kp(X, Y, Z, 0, (0, 2, 4, 6)), es)
    resampled = _mesh_max_ar(_shell_kp_merged(X, Y, Z, 0, 0, 6), es)
    assert resampled < naive, (
        f"resampled max AR {resampled:.2f} should beat naive {naive:.2f}"
    )
    assert resampled < 6.0, f"resampled merged patch still slivered: AR={resampled:.2f}"


# ----------------------------------------------------------------------
# Integration: merged BAR0 shell mesh
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def bar0_mesh():
    return get_mesh(includeAdhesive=False, elementSize=0.5)


def _region_element_indices(mesh, suffix):
    import re
    out = []
    for s in mesh["sets"]["element"]:
        name = s["name"]
        if name.startswith("all") or not re.match(r"\d+_\d+_", name):
            continue
        if name.endswith(suffix):
            out.extend(int(x) for x in s["labels"])
    return out


def _max_ar_of(mesh, eids):
    nodes = np.asarray(mesh["nodes"], dtype=float)
    elems = np.asarray(mesh["elements"], dtype=int)
    worst = 0.0
    for ei in eids:
        v = elems[ei, :4]
        v = v[v >= 0]
        if v.size < 3:
            continue
        p = nodes[v]
        e = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
        if e.min() > 0:
            worst = max(worst, float(e.max() / e.min()))
    return worst


def test_bar0_te_region_aspect_ratio_bounded(bar0_mesh):
    """No TE_FLAT or TE_REINF element on BAR0 may be a high-AR sliver.
    Bound is the global D3 ceiling (AR<20) so the test works under both
    the merge-only path and the conforming-mesh path (which handles AR via
    its own AR-guard segmentation and may give a slightly higher but still
    safe TE-region AR vs the merge-only value)."""
    te_eids = (
        _region_element_indices(bar0_mesh, "TE_FLAT")
        + _region_element_indices(bar0_mesh, "TE_REINF")
    )
    assert te_eids, "no TE region elements found"
    worst = _max_ar_of(bar0_mesh, te_eids)
    assert worst < 20.0, f"TE region still has an AR={worst:.1f} sliver"


def test_bar0_te_reinf_single_component(bar0_mesh):
    """The TE_REINF groups that absorb the flats must each remain a single
    node-connected component — i.e. the merge did not open the trailing
    edge or fragment the reinforcement."""
    sizes = mc.all_group_components(bar0_mesh, min_shared_nodes=1)
    for tag in ("HP_TE_REINF", "LP_TE_REINF"):
        cc = sizes.get(tag, [])
        assert cc, f"{tag} missing"
        assert len(cc) == 1, f"{tag} fragmented into {len(cc)} components: {cc[:5]}"
