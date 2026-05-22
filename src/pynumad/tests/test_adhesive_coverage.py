"""Adhesive constraint-coverage tests.

These tests pin the bug in ``pyNuMAD/src/pynumad/mesh_gen/mesh_gen.py``
lines 895-915: when ``includeAdhesive=True`` is requested, the mesher
only emits constraint equations (CEs) tying the **TE** adhesive nodes
(``LP_AdNodes`` / ``HP_AdNodes``) to the TE-reinforcement shell sections
(``LP_TE_REINF`` / ``HP_TE_REINF``). The shear-web-to-skin adhesive
bonds — which a real wind-turbine blade absolutely has — have NO
constraint equations at all (the corresponding block in
``solidMeshFromShell`` is commented out, see lines 1103-1111).

Consequence: roughly 4 in 5 adhesive nodes have zero CEs at fine
element sizes, ANSYS gets a singular stiffness matrix at solve time,
and the run aborts.

Each test in this file is designed to FAIL today and PASS once the
shear-web adhesive constraints are emitted.

Test categories:

A1. Every adhesive node either appears in a CE or shares its id
    with a shell-mesh node (full coverage).
A2. Adhesive nodes form spatial clusters per physical bond
    (TE-LP, TE-HP, SW-LP, SW-HP); each cluster has at least one
    constrained node.
A3. Graph-theoretic connectivity: starting from constrained
    nodes and walking through element connectivity, we must
    reach EVERY adhesive node.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pytest

from ._mesh_cache import BAR0_YAML, get_mesh


# Element size chosen as a compromise: small enough that the mesher
# emits enough adhesive elements for the spatial-clustering test to
# resolve all 6 bonds AND so that the SW-bond coverage bug actually
# manifests (at the coarsest BAR0 setting the TE-only tied set happens
# to cover everything by accident); large enough that each cached
# mesh build stays well under 8 s on the test runner.
ELEMENT_SIZE = 0.3


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _tied_adhesive_node_ids(mesh: dict) -> set[int]:
    """Set of adhesive-node ids that appear as the ``tiedMesh`` (driven)
    node in at least one constraint equation.

    The mesher's convention is: every CE has ``coef=-1`` on a ``tiedMesh``
    node (the adhesive node being constrained) and ``coef>=0`` weights on
    several ``targetMesh`` nodes (the shell nodes it's tied to).
    """
    out: set[int] = set()
    for ce in mesh.get("constraints", []):
        for term in ce["terms"]:
            if term.get("nodeSet") == "tiedMesh":
                out.add(int(term["node"]))
    return out


def _adhesive_node_use(mesh: dict) -> np.ndarray:
    """Bool array of length ``len(adhesiveNds)``; True if an adhesive
    element actually references the node.

    Adhesive nodes that are not used by any solid-element row are
    geometric debris and should not exist; we use this to filter them
    out before the coverage check.
    """
    n = len(mesh["adhesiveNds"])
    used = np.zeros(n, dtype=bool)
    for el in np.asarray(mesh["adhesiveEls"], dtype=int):
        for nd in el:
            if nd >= 0:
                used[nd] = True
    return used


def _cluster_adhesive_nodes(coords: np.ndarray, radius: float) -> np.ndarray:
    """Connected-component labelling on a spatial graph of the adhesive
    nodes: two nodes are linked if their coordinates are within ``radius``.

    Returns an integer label array of shape ``(N,)``. Implementation is
    a flood-fill over a KD-tree style binning for ``O(N)`` average cost.
    """
    n = coords.shape[0]
    labels = -np.ones(n, dtype=int)
    if n == 0:
        return labels

    # Bin nodes onto a coarse cubic grid of side ``radius`` so we only
    # do distance checks against a constant-size neighbourhood per node.
    keys = np.floor(coords / radius).astype(np.int64)
    bucket: dict[tuple, list[int]] = {}
    for i, k in enumerate(map(tuple, keys)):
        bucket.setdefault(k, []).append(i)

    next_label = 0
    stack: list[int] = []
    r2 = radius * radius
    for seed in range(n):
        if labels[seed] != -1:
            continue
        labels[seed] = next_label
        stack.append(seed)
        while stack:
            i = stack.pop()
            ki = tuple(keys[i])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        nk = (ki[0] + dx, ki[1] + dy, ki[2] + dz)
                        for j in bucket.get(nk, ()):
                            if labels[j] != -1:
                                continue
                            d = coords[j] - coords[i]
                            if (d * d).sum() <= r2:
                                labels[j] = next_label
                                stack.append(j)
        next_label += 1
    return labels


# ------------------------------------------------------------------
# A1. Every adhesive node must be referenced by a CE
# ------------------------------------------------------------------


def test_every_adhesive_node_is_constrained():
    """A1: every adhesive node that is actually used by an adhesive
    element must be the ``tiedMesh`` side of at least one constraint
    equation.

    This is the single most important assertion in the suite: with
    today's mesher only ~20% of adhesive nodes are tied (the TE-LP and
    TE-HP face of the TE-reinforcement bond), so the four shear-web
    adhesive faces are floating. That makes ANSYS's K singular.
    """
    mesh = get_mesh(includeAdhesive=True, elementSize=ELEMENT_SIZE)
    used = _adhesive_node_use(mesh)
    tied = _tied_adhesive_node_ids(mesh)

    n_used = int(used.sum())
    untied = [i for i in range(used.size) if used[i] and i not in tied]
    coverage = 1.0 - len(untied) / max(n_used, 1)

    assert not untied, (
        f"{len(untied)}/{n_used} adhesive nodes used by an adhesive element "
        f"have NO constraint equation tying them to the shell mesh "
        f"(coverage = {coverage:.1%}). "
        f"First 10 unconstrained ids: {untied[:10]}. "
        f"This is the SW1/SW2 adhesive bond bug — see "
        f"mesh_gen.py:1103 where the SW adhesive constraints are commented out."
    )


# ------------------------------------------------------------------
# A2. Each physical bond cluster must have its own CE coverage
# ------------------------------------------------------------------


def test_every_physical_bond_cluster_has_constraints():
    """A2: BAR0 has two shear webs, so there are six physical adhesive
    bond patches (TE-LP, TE-HP, SW1-LP, SW1-HP, SW2-LP, SW2-HP). After
    spatial clustering of adhesive-mesh nodes, each cluster must contain
    at least one tied (constrained) node.

    Today only the two TE clusters are constrained — the four
    shear-web clusters are completely floating.
    """
    mesh = get_mesh(includeAdhesive=True, elementSize=ELEMENT_SIZE)
    adh = np.asarray(mesh["adhesiveNds"], dtype=float)
    used = _adhesive_node_use(mesh)
    if not used.any():
        pytest.skip("no adhesive nodes — fixture not configured for bonding")

    # Radius: a couple of element sizes — far enough to chain along the
    # bondline but tight enough that two distinct bonds don't merge.
    radius = 2.5 * ELEMENT_SIZE
    keep_idx = np.flatnonzero(used)
    labels_used = _cluster_adhesive_nodes(adh[keep_idx], radius=radius)

    # Map back to full-mesh ids
    n_clusters = int(labels_used.max() + 1) if labels_used.size else 0
    tied = _tied_adhesive_node_ids(mesh)

    cluster_to_count = {c: 0 for c in range(n_clusters)}
    cluster_to_tied = {c: 0 for c in range(n_clusters)}
    for local_i, ci in enumerate(labels_used):
        global_i = int(keep_idx[local_i])
        cluster_to_count[int(ci)] += 1
        if global_i in tied:
            cluster_to_tied[int(ci)] += 1

    # BAR0 has 2 shear webs -> 6 physical bond patches expected.
    # Don't be strict about the exact cluster count (the clusterer may
    # split a long thin bondline if it gets locally disjoint), but
    # demand every non-trivial cluster has *some* constrained node.
    uncovered = [c for c in range(n_clusters)
                 if cluster_to_count[c] >= 4 and cluster_to_tied[c] == 0]
    assert not uncovered, (
        f"{len(uncovered)}/{n_clusters} spatial adhesive-node clusters "
        f"have NO constrained node. Cluster sizes (uncovered): "
        f"{sorted([cluster_to_count[c] for c in uncovered], reverse=True)[:5]} "
        f"-- one cluster per physical bond expected; SW-skin bonds are "
        f"the ones currently missing."
    )


def test_adhesive_node_cloud_spans_realistic_chord_range():
    """A2b: on a 2-shear-web blade the adhesive node cloud must span
    at least a non-trivial fraction of the chord and the spanwise
    axis. If pyNuMAD's adhesive emission silently degenerated to a
    single-point bond, the bounding box would collapse.

    This is a calibration / sanity check: it should pass even before
    the SW-constraint bug is fixed (the adhesive ELEMENT geometry is
    emitted today; only the CEs are missing).
    """
    mesh = get_mesh(includeAdhesive=True, elementSize=ELEMENT_SIZE)
    adh = np.asarray(mesh["adhesiveNds"], dtype=float)
    used = _adhesive_node_use(mesh)
    if not used.any():
        pytest.skip("no adhesive nodes")
    pts = adh[used]
    x_span = float(pts[:, 0].max() - pts[:, 0].min())
    z_span = float(pts[:, 2].max() - pts[:, 2].min())
    # BAR0 chord at the root is ~5 m; adhesive should span at least
    # 0.5 m of chord (proxy: a few elements across) and most of the
    # blade span (>= 30 m, since BAR0 is 100 m long).
    assert x_span >= 0.5, (
        f"adhesive node chord-extent x_span={x_span:.2f} m is too "
        f"small; bondline may have collapsed to a near-point"
    )
    assert z_span >= 30.0, (
        f"adhesive node span-extent z_span={z_span:.2f} m is much "
        f"smaller than the blade span; only a partial bondline emitted"
    )


# ------------------------------------------------------------------
# A3. Graph-theoretic: no floating adhesive island
# ------------------------------------------------------------------


def _adhesive_node_to_elements(mesh: dict) -> dict[int, list[int]]:
    """Adjacency: adhesive-node id -> list of adhesive-element ids."""
    out: dict[int, list[int]] = {}
    for eid, el in enumerate(np.asarray(mesh["adhesiveEls"], dtype=int)):
        for nd in el:
            if nd >= 0:
                out.setdefault(int(nd), []).append(eid)
    return out


def test_no_floating_adhesive_island():
    """A3: BFS from constrained adhesive nodes through adhesive-element
    connectivity must reach EVERY adhesive node. If any adhesive node is
    unreachable from the constrained seed set, that node belongs to an
    island whose entire DOF column in the stiffness matrix has no
    coupling to the rest of the structure.
    """
    mesh = get_mesh(includeAdhesive=True, elementSize=ELEMENT_SIZE)
    tied = _tied_adhesive_node_ids(mesh)
    used = _adhesive_node_use(mesh)

    if not used.any():
        pytest.skip("no adhesive nodes")
    if not tied:
        pytest.fail(
            "no adhesive node has any constraint equation at all "
            "-- the adhesive emission is completely orphaned"
        )

    adj_node_to_el = _adhesive_node_to_elements(mesh)
    els = np.asarray(mesh["adhesiveEls"], dtype=int)

    # BFS over adhesive nodes; two adhesive nodes are adjacent if they
    # share an adhesive element.
    seen = set(tied)
    stack = list(tied)
    while stack:
        n = stack.pop()
        for eid in adj_node_to_el.get(n, ()):
            for m in els[eid]:
                m = int(m)
                if m < 0 or m in seen:
                    continue
                seen.add(m)
                stack.append(m)

    unreachable = [
        i for i in range(used.size)
        if used[i] and i not in seen
    ]
    assert not unreachable, (
        f"{len(unreachable)} adhesive nodes are unreachable from the "
        f"constrained-node BFS seed set "
        f"(starting from {len(tied)} seeds). "
        f"These nodes belong to bond islands with no path to the shell "
        f"mesh. First 10 ids: {unreachable[:10]}."
    )
