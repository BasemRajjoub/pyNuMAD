"""Connectivity diagnostics for pyNuMAD shell meshes.

Each material zone in a windIO blade (HP_SPAR, HP_TE_REINF, etc.) is
emitted as a collection of per-spanwise-station element sets. This
module exposes two questions:

  * **node_connected_components(group, mesh)** — how many disconnected
    islands does this material zone form when you walk through
    *shared-node* neighbour-of-neighbour edges?  A clean, plate-like
    mesh would have 1 component.

  * **edge_connected_components(group, mesh)** — same walk but requiring
    two shared nodes (an edge) instead of one.  pyNuMAD's shell meshes
    are intrinsically T-junction-based — adjacent chordwise regions
    share only corner nodes — so edge-cc is almost always >> 1.

The numbers therefore characterise the **structure** of the mesh
rather than diagnosing a localised bug.  They are useful as a
regression baseline:

  - HP/LP symmetry: ``node_cc(HP_TE_REINF) ≈ node_cc(LP_TE_REINF)``
    is a sanity check; if HP fragments much more than LP, the
    keypoint-clamp asymmetry between
    ``keypoints.py:200-210`` (HP) and ``keypoints.py:246-256`` (LP)
    has changed.
  - Spar / spar-cap continuity: ``node_cc(HP_SPAR) == 1`` and
    ``node_cc(LP_SPAR) == 1`` — spar caps should be one piece.
  - Shear webs: ``node_cc(SW) == N_webs`` (one component per web).

ANSYS handles the T-junctions correctly via the constraint equations
that pyNuMAD emits, so the high edge-cc values are not a structural
defect — they are how pyNuMAD's chord/span subdivision works.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


CHORD_GROUP_TAGS = (
    # order matters: longer/more-specific labels first so substring
    # matching catches HP_TE_REINF before HP_TE.
    "HP_TE_REINF", "LP_TE_REINF",
    "HP_TE_PANEL", "LP_TE_PANEL",
    "HP_LE_PANEL", "LP_LE_PANEL",
    "HP_TE_FLAT",  "LP_TE_FLAT",
    "HP_SPAR",     "LP_SPAR",
    "HP_LE",       "LP_LE",
    "SW",
)


def classify_set_name(set_name: str) -> str | None:
    """Map an element-set name (``01_30_HP_TE_REINF``) to its chord group."""
    for tag in CHORD_GROUP_TAGS:
        if tag in set_name:
            return tag
    return None


def collect_groups(mesh: dict) -> dict[str, list[int]]:
    """Return ``{group_tag: [0-indexed element ids ...]}``.

    Each ``elements`` row in the mesh dict is 1-indexed in the
    element-set labels (Abaqus / ANSYS convention), so we convert to
    0-indexed ints for downstream BFS.
    """
    el_sets = mesh.get("sets", {}).get("element", []) or []
    groups: dict[str, list[int]] = defaultdict(list)
    for s in el_sets:
        tag = classify_set_name(s.get("name", ""))
        if tag is None:
            continue
        for lbl in s.get("labels", []):
            groups[tag].append(int(lbl) - 1)
    return groups


def _bfs_components(
    elem_ids: np.ndarray,
    connectivity: np.ndarray,
    min_shared_nodes: int,
) -> list[int]:
    """Return component sizes (descending) after BFS over ``elem_ids``.

    Two elements are considered adjacent when they share at least
    ``min_shared_nodes`` of their corner-node ids (``connectivity`` is
    the (Nx4) corner-node table, ``-1`` for unused slots).
    """
    # node -> [local element idx]
    node_to_els: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(connectivity):
        for n in c:
            if n >= 0:
                node_to_els[int(n)].append(i)

    visited = np.zeros(len(elem_ids), bool)
    sizes: list[int] = []
    for seed in range(len(elem_ids)):
        if visited[seed]:
            continue
        stack = [seed]
        visited[seed] = True
        size = 0
        while stack:
            cur = stack.pop()
            size += 1
            cur_nodes = {int(n) for n in connectivity[cur] if n >= 0}
            candidates: dict[int, int] = defaultdict(int)
            for n in cur_nodes:
                for j in node_to_els[n]:
                    if not visited[j]:
                        candidates[j] += 1
            for j, n_shared in candidates.items():
                if n_shared >= min_shared_nodes and not visited[j]:
                    visited[j] = True
                    stack.append(j)
        sizes.append(size)
    sizes.sort(reverse=True)
    return sizes


def component_sizes(
    mesh: dict,
    group_tag: str,
    *,
    min_shared_nodes: int = 1,
) -> list[int]:
    """Return descending list of component sizes for one chord group.

    Parameters
    ----------
    mesh
        The dict returned by ``shell_mesh_general``.
    group_tag
        One of ``CHORD_GROUP_TAGS`` (e.g. ``"HP_TE_REINF"``).
    min_shared_nodes
        ``1`` for node-connectivity (default; loose), ``2`` for
        edge-connectivity (strict, sees T-junctions as cuts).
    """
    groups = collect_groups(mesh)
    ids = sorted(set(groups.get(group_tag, [])))
    if not ids:
        return []
    elements = np.asarray(mesh["elements"], int)
    elem_ids = np.asarray(ids, int)
    elem_ids = elem_ids[(elem_ids >= 0) & (elem_ids < len(elements))]
    conn = elements[elem_ids][:, :4]
    return _bfs_components(elem_ids, conn, min_shared_nodes)


def all_group_components(
    mesh: dict,
    *,
    min_shared_nodes: int = 1,
) -> dict[str, list[int]]:
    """Run ``component_sizes`` for every group present in the mesh."""
    out: dict[str, list[int]] = {}
    for tag in CHORD_GROUP_TAGS:
        sizes = component_sizes(mesh, tag, min_shared_nodes=min_shared_nodes)
        if sizes:
            out[tag] = sizes
    return out


def assert_group_continuity(
    mesh: dict,
    *,
    expected: dict[str, int],
    min_shared_nodes: int = 1,
) -> None:
    """Assert each named group has the expected number of components.

    ``expected`` maps group tag -> expected n_components. Tags absent
    from ``expected`` are not checked. Raises ``AssertionError`` with a
    descriptive message on first mismatch.

    Use this in regression tests so any future pyNuMAD change that
    silently fragments a previously-continuous spar (for example) gets
    flagged immediately.
    """
    sizes = all_group_components(mesh, min_shared_nodes=min_shared_nodes)
    for tag, n_expected in expected.items():
        got = sizes.get(tag, [])
        if len(got) != n_expected:
            raise AssertionError(
                f"{tag}: expected {n_expected} components "
                f"(min_shared_nodes={min_shared_nodes}), got "
                f"{len(got)} (sizes={got[:10]}{'...' if len(got)>10 else ''})"
            )


def stray_elements(mesh: dict, group_tag: str) -> list[int]:
    """Return 0-indexed element ids that form 1-element singleton islands.

    A singleton means the element doesn't share any node with another
    element of the same chord group. Usually a labeling bug — the
    canonical example is one shear-web element ending up in the wrong
    station set at the very root or tip.
    """
    groups = collect_groups(mesh)
    ids = sorted(set(groups.get(group_tag, [])))
    if not ids:
        return []
    elements = np.asarray(mesh["elements"], int)
    elem_ids = np.asarray(ids, int)
    elem_ids = elem_ids[(elem_ids >= 0) & (elem_ids < len(elements))]
    conn = elements[elem_ids][:, :4]
    node_to_els: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(conn):
        for n in c:
            if n >= 0:
                node_to_els[int(n)].append(i)
    strays: list[int] = []
    for i, c in enumerate(conn):
        cur_nodes = {int(n) for n in c if n >= 0}
        neighbours = set()
        for n in cur_nodes:
            for j in node_to_els[n]:
                if j != i:
                    neighbours.add(j)
        if not neighbours:
            strays.append(int(elem_ids[i]))
    return strays
