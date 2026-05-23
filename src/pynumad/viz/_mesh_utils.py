"""Mesh-geometry helpers shared by the snapshot + interactive backends.

These take the (nodes, elements) arrays from
``mesh_gen.shell_mesh_general`` and produce render-ready polygons or
triangle lists. Pure numpy, no matplotlib / plotly imports — so the
viz package's heavy graphics deps stay optional.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def quads_to_polygons(
    nodes: np.ndarray,
    elements: np.ndarray,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    """Extract the 4-corner polygons of the requested quad elements.

    Tris (``-1`` in the 4th slot) are skipped — for the high-level
    plots we want clean rectangles only.

    Returns an ``(n_quads, 4, 3)`` array suitable for matplotlib's
    ``Poly3DCollection`` or for triangulating into plotly mesh i/j/k.
    """
    if indices is None:
        indices = np.arange(elements.shape[0])
    else:
        indices = np.asarray(indices, int)
        indices = indices[(indices >= 0) & (indices < elements.shape[0])]
    if indices.size == 0:
        return np.empty((0, 4, 3), dtype=float)
    conn = elements[indices][:, :4]
    quad_mask = (conn >= 0).all(axis=1)
    quads = conn[quad_mask]
    if quads.size == 0:
        return np.empty((0, 4, 3), dtype=float)
    return nodes[quads]  # (n_quads, 4, 3)


def quads_to_triangles(connectivity: np.ndarray) -> np.ndarray:
    """Split each (Nx>=4) quad-connectivity row into 2 triangle index rows.

    Tris pass through as a single row. Returns ``(M, 3)`` int array.
    """
    if connectivity.size == 0:
        return np.empty((0, 3), dtype=int)
    out: list[list[int]] = []
    for row in connectivity:
        valid = [int(x) for x in row if x >= 0]
        if len(valid) >= 4:
            out.append([valid[0], valid[1], valid[2]])
            out.append([valid[0], valid[2], valid[3]])
        elif len(valid) == 3:
            out.append([valid[0], valid[1], valid[2]])
    if not out:
        return np.empty((0, 3), dtype=int)
    return np.asarray(out, dtype=int)


# ---------------------------------------------------------------------------
# 3-D solid (SOLID185) outer-face extraction for adhesive bondlines
# ---------------------------------------------------------------------------

# Per-element face definitions for the two element types we see in
# pyNuMAD's adhesive output: 8-node bricks and 6-node wedges. Each
# entry is a list of vertex-index lists (length 3 or 4).
_BRICK_FACES = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)
_WEDGE_FACES = (
    (0, 1, 2),
    (3, 4, 5),
    (0, 1, 4, 3),
    (1, 2, 5, 4),
    (2, 0, 3, 5),
)


def solid_outer_face_triangles(elements: np.ndarray) -> np.ndarray:
    """Return triangle indices for the OUTER boundary faces of a brick mesh.

    A face is "outer" if it appears in exactly one element. Internal
    faces (shared between two adjacent elements) are dropped — without
    that filter you get z-fighting and a "checkerboard" artefact at
    bondline interiors.

    Parameters
    ----------
    elements : (N, M) int array
        Per-element node-ids. ``M`` must be 6, 8, or more (extra slots
        are ignored). ``-1`` entries are treated as missing.

    Returns
    -------
    tris : (T, 3) int array
        Triangle index rows. Empty array if no input.
    """
    if elements.size == 0:
        return np.empty((0, 3), dtype=int)

    faces: list[tuple[int, ...]] = []
    for el in elements:
        real = [int(n) for n in el if n >= 0]
        if len(real) == 8:
            for fv in _BRICK_FACES:
                faces.append(tuple(real[i] for i in fv))
        elif len(real) == 6:
            for fv in _WEDGE_FACES:
                faces.append(tuple(real[i] for i in fv))
        # other lengths are ignored — pyNuMAD doesn't emit them today

    counts: dict[frozenset, int] = {}
    for f in faces:
        key = frozenset(f)
        counts[key] = counts.get(key, 0) + 1

    out: list[list[int]] = []
    for f in faces:
        if counts[frozenset(f)] != 1:
            continue
        if len(f) == 4:
            out.append([f[0], f[1], f[2]])
            out.append([f[0], f[2], f[3]])
        else:  # tri
            out.append([f[0], f[1], f[2]])

    if not out:
        return np.empty((0, 3), dtype=int)
    return np.asarray(out, dtype=int)


# ---------------------------------------------------------------------------
# Element-set grouping
# ---------------------------------------------------------------------------


def collect_chord_group_indices(mesh: dict) -> dict[str, list[int]]:
    """``{group_tag: [0-indexed element ids ...]}`` for every chord group.

    Labels in pyNuMAD's mesh dict are 0-indexed (the ANSYS deck writer
    adds ``+1`` when emitting EMODIF — see
    ``analysis/ansys/write.py:1272``). Older code that did
    ``int(label) - 1`` was reading the wrong element and produced
    misleading "fragmented" snapshots; do not reintroduce that.
    """
    from pynumad.viz._palette import classify_set_name

    el_sets = mesh.get("sets", {}).get("element", []) or []
    out: dict[str, list[int]] = defaultdict(list)
    for s in el_sets:
        tag = classify_set_name(s.get("name", ""))
        if tag is None:
            continue
        for lbl in s.get("labels", []):
            out[tag].append(int(lbl))
    return dict(out)


def collect_adhesive_bond_indices(mesh: dict) -> dict[str, list[int]]:
    """``{bond_tag: [0-indexed adhesive-element ids ...]}``.

    Prefers ``mesh["adhesiveBondSets"]`` (full bondline element list,
    populated by ``_emit_adhesive_volume``).  Falls back to the
    tied-face subsets in ``sets.element`` named ``TE_BOND_*`` /
    ``LE_BOND_*`` when ``adhesiveBondSets`` is absent.
    """
    from pynumad.viz._palette import classify_adhesive_set_name

    bond_sets = mesh.get("adhesiveBondSets", []) or []
    out: dict[str, list[int]] = defaultdict(list)
    for bs in bond_sets:
        tag = bs.get("name", "ADHESIVE")
        for lbl in bs.get("labels", []):
            out[tag].append(int(lbl))
    if out:
        return dict(out)

    el_sets = mesh.get("sets", {}).get("element", []) or []
    for s in el_sets:
        tag = classify_adhesive_set_name(s.get("name", ""))
        if tag is None:
            continue
        for lbl in s.get("labels", []):
            out[tag].append(int(lbl))
    return dict(out)
