"""Detect and tie shell-mesh T-junctions (hanging nodes).

pyNuMAD's structured shell mesher derives each chord region's spanwise
element count independently, so adjacent regions (and the merged
TE_FLAT/TE_REINF stations) can disagree on subdivision. Where they do, a
node of the finer region lands on the *interior* of a coarser region's
element edge — a "hanging node" / T-junction. The deck only emits
``nummrg,all`` (coincident-node merge), which cannot tie a hanging node
because it is not coincident with anything: it sits mid-edge. The result
is a displacement incompatibility at every T-junction (a local gap/overlap
under load), which concentrates stress and can seed local instabilities
under NLGEOM.

``find_hanging_nodes`` locates them; ``constraint_records`` turns each into
a linear multipoint constraint ``u_H = (1-t)*u_A + t*u_B`` tying the
hanging node H to the two corner nodes A,B of the edge it splits. All node
ids are 0-indexed (mesh-dict convention); callers add +1 for ANSYS.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def find_hanging_nodes(mesh, perp_tol_frac=0.01, t_margin=0.02):
    """Return a list of ``(H, A, B, t)`` hanging-node records (0-indexed).

    H sits on the interior of the edge (A, B) at parametric position
    ``t`` in (0, 1): ``coord(H) ~= (1-t)*coord(A) + t*coord(B)``.

    Parameters
    ----------
    mesh : dict
        pyNuMAD mesh dict with ``nodes`` (N,3) and ``elements`` (M,>=4).
    perp_tol_frac : float
        A node counts as "on the edge" if its perpendicular distance to
        the edge is below ``perp_tol_frac`` times the edge length.
    t_margin : float
        Exclude the edge endpoints: only ``t_margin < t < 1-t_margin``.

    Returns
    -------
    list[tuple[int, int, int, float]]
        One record per unique hanging node (its first/closest host edge).
    """
    nodes = np.asarray(mesh["nodes"], dtype=float)
    elems = np.asarray(mesh["elements"], dtype=int)
    tree = cKDTree(nodes)
    seen: dict[int, tuple[int, int, int, float]] = {}
    best_perp: dict[int, float] = {}
    for ei in range(elems.shape[0]):
        conn = elems[ei, :4]
        conn = conn[conn >= 0]
        if conn.size < 4:
            continue
        cset = set(int(x) for x in conn)
        for k in range(4):
            a = int(conn[k]); b = int(conn[(k + 1) % 4])
            pa = nodes[a]; pb = nodes[b]
            ab = pb - pa
            L = float(np.linalg.norm(ab))
            if L < 1e-9:
                continue
            mid = 0.5 * (pa + pb)
            for j in tree.query_ball_point(mid, 0.5 * L):
                if j in cset:
                    continue
                t = float(np.dot(nodes[j] - pa, ab) / (L * L))
                if not (t_margin < t < 1.0 - t_margin):
                    continue
                perp = float(np.linalg.norm(nodes[j] - (pa + t * ab)))
                if perp < perp_tol_frac * L:
                    if j not in best_perp or perp < best_perp[j]:
                        best_perp[j] = perp
                        seen[j] = (j, a, b, t)
    return list(seen.values())


def constraint_records(hanging, dofs=("UX", "UY", "UZ")):
    """Turn hanging-node records into constraint dicts the ANSYS writer
    understands: ``u_H - (1-t)*u_A - t*u_B = 0`` per DOF.

    Returns a list of ``{"dof": str, "terms": [(node0idx, coef), ...],
    "rhs": 0.0}`` records (nodes 0-indexed)."""
    out = []
    for (H, A, B, t) in hanging:
        for dof in dofs:
            out.append({
                "dof": dof,
                "terms": [(H, 1.0), (A, -(1.0 - t)), (B, -t)],
                "rhs": 0.0,
            })
    return out


def emit_apdl_ce(hanging, dofs=("UX", "UY", "UZ")):
    """Emit an APDL constraint-equation block (1-indexed) tying each
    hanging node to its host edge. One ``CE,NEW`` per (node, DOF); the
    three triples (H, A, B) fit a single CE command.

    NOTE: this ties only the listed (translational) DOFs. Shell elements
    also carry rotational DOFs (ROTX/Y/Z); leaving them free at a hanging
    node creates a local bending hinge that can stall NLGEOM convergence.
    For shell meshes prefer ``emit_apdl_ceintf`` (ANSYS CEINTF), which ties
    all DOFs via the host element's shape functions.
    """
    lines = ["! ===== T-junction constraint equations (hanging-node ties) ====="]
    for (H, A, B, t) in hanging:
        for dof in dofs:
            # CE,NEXT (not NEW — "NEW" is not a valid NEQN keyword; ANSYS
            # then reads the equation number as 0 and silently ignores the
            # command). NEXT assigns the next available equation number.
            lines.append(
                "CE,NEXT,0, %d,%s,%g, %d,%s,%g, %d,%s,%g"
                % (H + 1, dof, 1.0, A + 1, dof, -(1.0 - t), B + 1, dof, -t)
            )
    lines.append("ALLSEL,ALL")
    return "\n".join(lines) + "\n"


def emit_apdl_ceintf(hanging, toler=0.05):
    """Emit an APDL block that ties hanging nodes to the elements they sit
    on using ANSYS ``CEINTF`` (constraint-equation interface generation).

    Unlike ``emit_apdl_ce`` (manual, translation-only, edge-linear), CEINTF
    has ANSYS locate each selected node inside an element and generate the
    constraint equations from that element's **shape functions across all
    active DOFs** — including the rotational shell DOFs. This restores both
    translational and rotational continuity at the T-junction, removing the
    free-hinge that stalls NLGEOM, and uses ANSYS's own (less
    over-constraining) CE formulation.

    The hanging nodes are selected into a node component and CEINTF is run
    with ALL OTHER nodes also selected (CEINTF ties the *selected* nodes to
    the *selected* elements). ``toler`` is the normalized element-coordinate
    tolerance for deciding a node lies on an element.

    Parameters
    ----------
    hanging : list[tuple[int, int, int, float]]
        Output of :func:`find_hanging_nodes`; only the hanging node id (H,
        0-indexed) of each record is used.
    toler : float
        CEINTF tolerance (normalized element coords). 0.05 (5 %) matches the
        scale of floating-point imprecision in 3-D shell node placement:
        hanging nodes lie exactly on coarse element edges by construction, but
        round-off can push them ~0.001–0.01 outside in parametric space. 1e-4
        left ~40 % of nodes unbound on the IEA-22 mesh (confirmed in testing).

    Returns
    -------
    str
        APDL block. Assumes the full mesh (nodes + elements) is already
        defined when this runs.
    """
    ids = sorted({int(H) + 1 for (H, _A, _B, _t) in hanging})
    lines = [
        "! ===== T-junction ties via CEINTF (shape-function, all DOFs) =====",
        "! CEINTF ties each SELECTED node to whichever SELECTED element it",
        "! lies on (other than its own elements). Select the hanging nodes,",
        "! keep ALL elements selected so the coarse-side host is available.",
        "ALLSEL,ALL",
        "ESEL,ALL",
        "NSEL,NONE",
    ]
    # Select the hanging nodes. Explicit per-node NSEL,A keeps it robust to
    # non-contiguous ids (the alternative range form risks pulling in
    # untargeted nodes).
    for nid in ids:
        lines.append("NSEL,A,NODE,,%d" % nid)
    lines.append("CEINTF,%g" % toler)
    lines.append("ALLSEL,ALL")
    return "\n".join(lines) + "\n"
