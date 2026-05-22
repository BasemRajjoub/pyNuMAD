"""Shell-solid connectivity tests.

The pyNuMAD mesh, when ``includeAdhesive=True``, is a *hybrid* mesh:
the outer skin + shear webs are shell quads (``elements`` array), the
adhesive bonds are 8-node hex solids (``adhesiveEls`` array), and the
two are coupled through constraint equations (``constraints``).

For the assembled FE model to be solvable, the union of
(shell elements + adhesive elements + CE-mediated couplings) must
form a single connected graph. If even one adhesive node is floating
— no CE and no shared node id with a shell element — its three
displacement DOFs are unconstrained and the stiffness matrix is
singular.

Tests here:

  E1. For every adhesive element, at least one of its 8 nodes is
      either (a) the ``tiedMesh`` side of a CE, OR (b) is a shell
      mesh node (shared id).
  E2. The shell-adhesive coupled connectivity graph has a single
      connected component (or, if not, every component contains at
      least one root-clamp node so it is grounded).

Both of these tests are designed to FAIL today and PASS once the
shear-web adhesive CEs are emitted.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from ._mesh_cache import get_mesh


# Use a finer element size than other test files so the SW-bond
# coverage bug actually surfaces (at 0.5 m on BAR0 the TE-only CE
# emission happens to cover every adhesive node by accident).
ELEMENT_SIZE = 0.3


def _tied_adhesive_node_ids(mesh: dict) -> set[int]:
    out: set[int] = set()
    for ce in mesh.get("constraints", []):
        for t in ce["terms"]:
            if t.get("nodeSet") == "tiedMesh":
                out.add(int(t["node"]))
    return out


# ------------------------------------------------------------------
# E1. Every adhesive element has a path to ground
# ------------------------------------------------------------------


def test_every_adhesive_element_has_a_constrained_node():
    """E1: for every adhesive solid element, at least one of its 8
    corner nodes must be tied to the shell mesh (via a CE) OR must be
    a shell-mesh node id.

    Today, only adhesive elements that touch the TE-reinforcement
    bondline pass this test; the shear-web adhesive elements (which
    are the majority) are floating.

    Note on the "shared node id" case: pyNuMAD's adhesive mesh uses
    its OWN id space (``adhesiveNds``) that is disjoint from the shell
    mesh's ``nodes``. So in practice condition (b) never fires in the
    current code path — meaning this test reduces to "every adhesive
    element must have at least one tied node".
    """
    mesh = get_mesh(includeAdhesive=True, elementSize=ELEMENT_SIZE)
    adh_els = np.asarray(mesh["adhesiveEls"], dtype=int)
    if adh_els.size == 0:
        pytest.skip("no adhesive elements")

    tied = _tied_adhesive_node_ids(mesh)

    bad_elements = []
    for ei, el in enumerate(adh_els):
        if not any(int(n) in tied for n in el if n >= 0):
            bad_elements.append(ei)
    assert not bad_elements, (
        f"{len(bad_elements)}/{adh_els.shape[0]} adhesive elements have "
        f"NO tied node (no CE on any corner); they are floating. "
        f"First 10 element ids: {bad_elements[:10]}."
    )


# ------------------------------------------------------------------
# E2. Connectivity-component count
# ------------------------------------------------------------------


def test_adhesive_connectivity_single_component():
    """E2: build the undirected graph G where vertices are adhesive
    nodes and edges are "two adhesive nodes co-occur in an adhesive
    element OR are linked via a CE that shares a target shell node".

    G should have a single connected component on each shear-web /
    TE bondline, and every component must contain at least one tied
    node. We report the number of components that have ZERO tied
    nodes; this should be zero.
    """
    mesh = get_mesh(includeAdhesive=True, elementSize=ELEMENT_SIZE)
    adh_els = np.asarray(mesh["adhesiveEls"], dtype=int)
    n_adh = len(mesh["adhesiveNds"])
    if n_adh == 0 or adh_els.size == 0:
        pytest.skip("no adhesive mesh")

    tied = _tied_adhesive_node_ids(mesh)

    # adjacency on adhesive nodes via element co-membership
    adj: dict[int, set[int]] = defaultdict(set)
    for el in adh_els:
        nodes = [int(n) for n in el if n >= 0]
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                adj[a].add(b)
                adj[b].add(a)

    # Union-find / iterative BFS to label components
    label = -np.ones(n_adh, dtype=int)
    cur = 0
    for s in range(n_adh):
        if label[s] != -1 or s not in adj:
            continue
        stack = [s]
        label[s] = cur
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if label[v] == -1:
                    label[v] = cur
                    stack.append(v)
        cur += 1
    n_components = cur

    # Components with zero tied nodes are the bug
    comp_has_tie: dict[int, bool] = {c: False for c in range(n_components)}
    for nd in tied:
        if 0 <= nd < n_adh and label[nd] >= 0:
            comp_has_tie[int(label[nd])] = True

    untied_components = [c for c, t in comp_has_tie.items() if not t]
    comp_sizes = np.bincount(label[label >= 0])
    untied_sizes = sorted((int(comp_sizes[c]) for c in untied_components),
                          reverse=True)

    assert not untied_components, (
        f"{len(untied_components)}/{n_components} adhesive connected "
        f"components contain ZERO tied nodes; they are floating "
        f"islands. Component sizes (untied, sorted): "
        f"{untied_sizes[:10]}. Total untied nodes: {sum(untied_sizes)} "
        f"out of {n_adh} adhesive nodes."
    )


def test_adhesive_shell_coupling_one_component_with_root():
    """E2b: the FULL coupled graph (shell-quad connectivity + adhesive
    connectivity + CE-mediated edges) must reach the root clamp from
    every non-isolated node.

    We build a bipartite graph: shell nodes and adhesive nodes are
    *both* vertices, with a special "ADHESIVE+n" prefix on adhesive
    ids to keep namespaces disjoint. Edges:

      - any two nodes that co-occur in a shell quad
      - any two nodes that co-occur in an adhesive solid
      - any (adhesive_node, shell_node) pair where the shell node
        appears as a CE target for that adhesive node
    """
    mesh = get_mesh(includeAdhesive=True, elementSize=ELEMENT_SIZE)
    shell_els = np.asarray(mesh["elements"], dtype=int)
    adh_els = np.asarray(mesh["adhesiveEls"], dtype=int)
    n_shell = len(mesh["nodes"])
    n_adh = len(mesh["adhesiveNds"])

    if n_adh == 0:
        pytest.skip("no adhesive mesh")

    # Vertex ids: shell node i -> i; adhesive node j -> n_shell + j
    OFFSET = n_shell
    adj: dict[int, set[int]] = defaultdict(set)

    def link(a: int, b: int):
        if a != b:
            adj[a].add(b)
            adj[b].add(a)

    for el in shell_els:
        nodes = [int(n) for n in el if n >= 0]
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                link(a, b)

    for el in adh_els:
        nodes = [OFFSET + int(n) for n in el if n >= 0]
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                link(a, b)

    for ce in mesh.get("constraints", []):
        tied = [t for t in ce["terms"] if t.get("nodeSet") == "tiedMesh"]
        targets = [t for t in ce["terms"] if t.get("nodeSet") != "tiedMesh"]
        for ti in tied:
            ti_v = OFFSET + int(ti["node"])
            for tg in targets:
                tg_v = int(tg["node"])
                link(ti_v, tg_v)

    # Find a root-clamp seed: take the shell node-set "RootNodes" if it
    # exists, otherwise any shell node.
    root_seeds: list[int] = []
    for ns in mesh.get("sets", {}).get("node", []):
        if "Root" in ns.get("name", ""):
            root_seeds.extend(int(n) for n in ns.get("labels", []))
            break
    if not root_seeds:
        root_seeds = [0]

    # BFS from root seeds across the unified graph
    seen = set(root_seeds)
    stack = list(root_seeds)
    while stack:
        u = stack.pop()
        for v in adj.get(u, ()):
            if v not in seen:
                seen.add(v)
                stack.append(v)

    # Adhesive nodes that should have been reached
    unreached = [j for j in range(n_adh) if (OFFSET + j) not in seen]
    # ignore truly-unused adhesive nodes (no element)
    used = np.zeros(n_adh, dtype=bool)
    for el in adh_els:
        for n in el:
            if n >= 0:
                used[int(n)] = True
    unreached = [j for j in unreached if used[j]]

    assert not unreached, (
        f"{len(unreached)}/{used.sum()} used adhesive nodes are not "
        f"reachable from the root clamp through "
        f"(shell ⊕ adhesive ⊕ CE) graph traversal. "
        f"First 10 ids: {unreached[:10]}. "
        f"These DOFs are unconstrained at solve time -> singular K."
    )
