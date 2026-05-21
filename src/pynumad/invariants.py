"""Checked predicates over pyNuMAD data structures.

A small library of inexpensive runtime checks used by both:

- ``assert`` statements on hot paths inside pyNuMAD modules (cheap dev
  invariants; can be disabled with ``python -O``).
- explicit ``raise ValueError(...)`` calls at module boundaries (input
  validation; survives ``-O``).

The convention in this file: every predicate has a `check_<name>` variant
that returns ``(ok: bool, message: str)`` and a `require_<name>` variant
that raises with ``message`` if the check fails. Use ``check_*`` when you
want to log a warning without aborting; use ``require_*`` when the
condition is load-bearing.

These checks are intentionally cheap. Anything that needs a full mesh-quality
sweep belongs in :mod:`pynumad.testing.mesh_quality` instead.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    # edgeEls
    "check_edge_els",
    "require_edge_els",
    "opposite_edges_match",
    # quad corners
    "check_quad_corners",
    "quad_signed_area_in_plane",
    "quad_is_positive_jacobian",
    "require_quad_positive_jacobian",
    # mesh dict
    "check_mesh_dict",
    "no_unreferenced_node_ids",
    "no_negative_node_ids_except_sentinel",
    "quad_elements_have_distinct_first_three_nodes",
]

# ---------------------------------------------------------------------------
# edgeEls / region geometry invariants
# ---------------------------------------------------------------------------


def check_edge_els(edge_els: Sequence[int]) -> tuple[bool, str]:
    """A `ShellRegion`'s ``edgeEls`` must have 4 strictly-positive ints."""
    if edge_els is None:
        return False, "edge_els is None"
    if len(edge_els) != 4:
        return False, f"edge_els must have length 4, got {len(edge_els)}"
    for i, n in enumerate(edge_els):
        if not isinstance(n, (int, np.integer)):
            return False, f"edge_els[{i}] is not an integer ({type(n).__name__})"
        if int(n) <= 0:
            return False, f"edge_els[{i}] = {int(n)} <= 0 (must be positive)"
    return True, ""


def require_edge_els(edge_els: Sequence[int]) -> None:
    ok, msg = check_edge_els(edge_els)
    if not ok:
        raise ValueError(f"invalid edge_els: {msg}")


def opposite_edges_match(edge_els: Sequence[int]) -> tuple[bool, str]:
    """Soft check: a structured ``quad`` region produces clean meshes only
    when opposite edges have equal element counts. Returns ``(True, "")``
    when matched; otherwise ``(False, "...")`` for logging or warning.
    """
    if len(edge_els) != 4:
        return False, "edge_els has wrong length"
    if edge_els[0] != edge_els[2] or edge_els[1] != edge_els[3]:
        return (
            False,
            f"edge_els={list(edge_els)} has unequal opposite edges "
            f"(ee[0]={edge_els[0]} vs ee[2]={edge_els[2]}, "
            f"ee[1]={edge_els[1]} vs ee[3]={edge_els[3]})",
        )
    return True, ""


# ---------------------------------------------------------------------------
# Quad / element invariants
# ---------------------------------------------------------------------------


_EPS = 1e-12


def check_quad_corners(corners: np.ndarray) -> tuple[bool, str]:
    """A quad's corners must be a (4, 3) float array."""
    arr = np.asarray(corners)
    if arr.shape != (4, 3):
        return False, f"quad corners shape {arr.shape} != (4, 3)"
    if not np.isfinite(arr).all():
        return False, "quad corners contain NaN or inf"
    return True, ""


def quad_signed_area_in_plane(corners: np.ndarray) -> float:
    """Return the signed in-plane area of a quad, using the plane spanned by
    edge 0->1 and a direction orthogonal within the (0,1,2) triangle plane.
    Negative means the quad is wound the wrong way (or sign-flipped).
    """
    p0, p1, p2, p3 = corners
    n = np.cross(p1 - p0, p2 - p0)
    nn = np.linalg.norm(n)
    if nn < _EPS:
        return 0.0
    n_hat = n / nn
    u = p1 - p0
    un = np.linalg.norm(u)
    if un < _EPS:
        return 0.0
    u = u / un
    v = np.cross(n_hat, u)
    pts = np.array([[np.dot(c - p0, u), np.dot(c - p0, v)] for c in corners])
    # Shoelace over the 4 vertices (signed)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(
        x[0] * (y[1] - y[3])
        + x[1] * (y[2] - y[0])
        + x[2] * (y[3] - y[1])
        + x[3] * (y[0] - y[2])
    )


def quad_is_positive_jacobian(corners: np.ndarray, tol: float = 0.0) -> tuple[bool, str]:
    """The two diagonal-triangle areas of the quad must have the same sign
    and both magnitudes must exceed ``tol``."""
    ok, msg = check_quad_corners(corners)
    if not ok:
        return False, msg
    p0, p1, p2, p3 = corners
    n = np.cross(p1 - p0, p2 - p0)
    nn = np.linalg.norm(n)
    if nn < _EPS:
        return False, "first triangle (0,1,2) is degenerate"
    n_hat = n / nn
    u = p1 - p0
    un = np.linalg.norm(u)
    if un < _EPS:
        return False, "edge 0->1 has zero length"
    u = u / un
    v = np.cross(n_hat, u)
    pts = np.array([[np.dot(c - p0, u), np.dot(c - p0, v)] for c in corners])
    a1 = 0.5 * (
        (pts[1, 0] - pts[0, 0]) * (pts[2, 1] - pts[0, 1])
        - (pts[2, 0] - pts[0, 0]) * (pts[1, 1] - pts[0, 1])
    )
    a2 = 0.5 * (
        (pts[2, 0] - pts[0, 0]) * (pts[3, 1] - pts[0, 1])
        - (pts[3, 0] - pts[0, 0]) * (pts[2, 1] - pts[0, 1])
    )
    if a1 * a2 < -tol * tol:
        return False, f"diagonal-triangle areas have opposite signs: a1={a1:.3e}, a2={a2:.3e}"
    if abs(a1) <= tol or abs(a2) <= tol:
        return False, f"diagonal-triangle area near zero: |a1|={abs(a1):.3e}, |a2|={abs(a2):.3e}"
    return True, ""


def require_quad_positive_jacobian(corners: np.ndarray, tol: float = 0.0) -> None:
    ok, msg = quad_is_positive_jacobian(corners, tol=tol)
    if not ok:
        raise ValueError(f"quad has non-positive Jacobian: {msg}")


# ---------------------------------------------------------------------------
# Mesh-level invariants
# ---------------------------------------------------------------------------


def check_mesh_dict(mesh: dict) -> tuple[bool, str]:
    """A pyNuMAD mesh dict has ``nodes`` (N,3) and ``elements`` (M,K) with
    K >= 4."""
    if not isinstance(mesh, dict):
        return False, f"mesh is {type(mesh).__name__}, expected dict"
    for key in ("nodes", "elements"):
        if key not in mesh:
            return False, f"mesh missing key {key!r}"
    nodes = np.asarray(mesh["nodes"])
    elements = np.asarray(mesh["elements"])
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        return False, f"nodes shape {nodes.shape} not (N, 3)"
    if elements.ndim != 2 or elements.shape[1] < 4:
        return False, f"elements shape {elements.shape} not (M, K>=4)"
    return True, ""


def no_unreferenced_node_ids(mesh: dict) -> tuple[bool, str]:
    """Every element node-id slot (excluding -1 sentinels) must point to a
    real node row. Catches off-by-one errors during mesh assembly.
    """
    nodes = np.asarray(mesh["nodes"])
    elements = np.asarray(mesh["elements"])
    n_nodes = nodes.shape[0]
    flat = elements.flatten()
    bad = flat[(flat >= 0) & (flat >= n_nodes)]
    if bad.size:
        return (
            False,
            f"{bad.size} element slot(s) reference out-of-range node ids "
            f"(max id in elements = {int(flat.max())}, n_nodes = {n_nodes})",
        )
    return True, ""


def no_negative_node_ids_except_sentinel(mesh: dict) -> tuple[bool, str]:
    """Element node ids may be -1 (collapsed-quad sentinel) but no other
    negatives are allowed."""
    elements = np.asarray(mesh["elements"])
    flat = elements.flatten()
    bad = flat[(flat < 0) & (flat != -1)]
    if bad.size:
        return False, f"{bad.size} element slot(s) have negative id != -1"
    return True, ""


def quad_elements_have_distinct_first_three_nodes(mesh: dict) -> tuple[bool, str]:
    """A valid quad or collapsed-tri element must have at least 3 distinct
    node ids in its first three slots. Catches the degenerate
    line/point elements that can appear if ``mergeDuplicateNodes`` over-merges.
    """
    elements = np.asarray(mesh["elements"])
    bad: list[int] = []
    for ei in range(elements.shape[0]):
        first_three = [int(x) for x in elements[ei, :3] if x >= 0]
        if len(set(first_three)) < 3 and len(first_three) >= 3:
            bad.append(ei)
            if len(bad) >= 10:
                break
    if bad:
        return False, f"{len(bad)}+ elements have degenerate first-three nodes (e.g. {bad[:5]})"
    return True, ""
