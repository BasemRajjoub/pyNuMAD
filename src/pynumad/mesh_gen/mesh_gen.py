import numpy as np
import warnings
from os import getcwd
from os.path import join
import subprocess
import copy as cp
from scipy import interpolate

import pynumad
from pynumad._logging import get_logger
from pynumad import invariants as _inv
from pynumad.utils.interpolation import interpolator_wrap

##from pynumad.mesh_gen.shellClasses import shellRegion, elementSet, NuMesh3D, spatialGridList2D, spatialGridList3D
from pynumad.mesh_gen.boundary2d import *
from pynumad.mesh_gen.surface import Surface
from pynumad.mesh_gen.mesh2d import *
from pynumad.mesh_gen.mesh3d import Mesh3D
from pynumad.mesh_gen.shell_region import ShellRegion
from pynumad.mesh_gen.mesh_tools import *
from pynumad.mesh_gen.element_utils import *
#from pynumad.analysis.ansys.write import writeAnsysShellModel

_log = get_logger(__name__)


# Patches whose shortest edge is below this fraction of the requested
# elementSize are flagged as "degenerate" and skipped. Threshold chosen so
# that legitimate small-chord regions still mesh (a tapered TE flat with
# chord ~ 0.5 * elementSize meshes fine) but the tip/root cylinder-end
# degeneracies (chord ~ 0.001 m on a 0.45 m elementSize, ratio ~ 0.002) get
# caught.
_DEGENERATE_EDGE_RATIO = 0.05


def _emit_adhesive_volume(
    blade,
    shellData,
    splineXi,
    splineYi,
    splineZi,
    frstXS,
    elementSize,
    bond_name,
    corner_cols,
    target_shell_set_substrs,
):
    """Build one swept adhesive volume + tie constraints for a single bondline.

    The helper reproduces the swept-quad cross-section construction that
    used to live inline for the TE bondline, but parameterised so the
    same code can emit the LE bond (and, later, the shear-web-to-skin
    bonds). It

      * builds a 2D quad cross-section at ``frstXS`` from four spline
        columns ``corner_cols = (c0, c1, c2, c3)``,
      * sweeps that cross-section spanwise (3 spline-rows per layer) to
        ``stPt = splineXi.shape[0]``,
      * appends the produced nodes/elements to
        ``shellData['adhesiveNds']`` / ``shellData['adhesiveEls']``
        (creating the keys if absent),
      * appends an element-set named ``bond_name`` to
        ``shellData['adhesiveElSet']`` (also created if absent — for the
        first bondline the value is the dict itself for backward
        compatibility),
      * generates LP/HP/residual tie constraints against the shell sets
        identified by ``target_shell_set_substrs`` and appends them to
        ``shellData['constraints']``.

    Parameters
    ----------
    blade : Blade
        Blade object (unused at the moment but kept in the signature so
        future phases can read e.g. ``blade.keypoints.web_indices``).
    shellData : dict
        Mesh accumulator. Modified in-place.
    splineXi, splineYi, splineZi : 2D ndarrays
        Spline grids built by :func:`shell_mesh_general`.
    frstXS : int
        First spanwise spline-row where the cross-section is non-degenerate.
    elementSize : float
        Target element edge length.
    bond_name : str
        Logical bondline tag (``"TE_BOND"``, ``"LE_BOND"``, ...). Used
        as the element-set name for *this* bondline so each bondline can
        be identified separately by downstream code / tests.
    corner_cols : tuple[int, int, int, int]
        Spline-keypoint column indices for the 4 cross-section corners,
        in the order pyNuMAD's existing TE code uses (shellKp slots
        0, 1, 2, 3). For TE: ``(4, 6, 30, 32)``. For LE: ``(17, 15, 21, 19)``.
        Convention: slots 0/3 are on the *outer* edge (closer to the
        TE-tip / LE-tip), slots 1/2 are on the *inner* edge (closer to
        the spar caps). Slots 0/1 lie on the HP-side skin, 2/3 on the
        LP-side skin. The +y outward face is the LP face (edge 2-3),
        the -y outward face is the HP face (edge 0-1).
    target_shell_set_substrs : list[str | tuple[str, ...]]
        Per-side suffix substrings identifying the shell element-sets
        this adhesive ties to. Matched against
        ``shellData['sets']['element'][i]['name']`` with an
        ``endswith('_' + substr)`` test so e.g. ``'HP_LE'`` matches
        ``05_NN_HP_LE`` but NOT ``04_NN_HP_LE_PANEL``. Each side may
        be a single string OR a tuple of strings: the union of all
        matching sets is used as the tie target. The first entry must
        be the LP-side, the second the HP-side (matching the LP / HP
        outward face convention above). For TE:
        ``['LP_TE_REINF', 'HP_TE_REINF']``. For LE:
        ``[('LP_LE', 'LP_LE_PANEL'), ('HP_LE', 'HP_LE_PANEL')]`` — the
        ``_PANEL`` sets are included so adhesive nodes that extend past
        the LE chord-segment (e.g. into the tip-taper region where the
        bare LE stack has zero thickness) still find a target.
    """
    if len(corner_cols) != 4:
        raise ValueError(
            f"_emit_adhesive_volume({bond_name}): corner_cols must have 4 entries, got {corner_cols}"
        )
    if len(target_shell_set_substrs) != 2:
        raise ValueError(
            f"_emit_adhesive_volume({bond_name}): target_shell_set_substrs "
            f"must have 2 entries (LP, HP), got {target_shell_set_substrs}"
        )
    c0, c1, c2, c3 = corner_cols
    lp_substr, hp_substr = target_shell_set_substrs

    # ------------------------------------------------------------------
    # Determine per-edge element counts from the first cross-section.
    # ------------------------------------------------------------------
    stPt = frstXS
    v1x = splineXi[stPt, c1] - splineXi[stPt, c0]
    v1y = splineYi[stPt, c1] - splineYi[stPt, c0]
    v1z = splineZi[stPt, c1] - splineZi[stPt, c0]
    mag1 = np.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
    v2x = splineXi[stPt, c2] - splineXi[stPt, c3]
    v2y = splineYi[stPt, c2] - splineYi[stPt, c3]
    v2z = splineZi[stPt, c2] - splineZi[stPt, c3]
    mag2 = np.sqrt(v2x * v2x + v2y * v2y + v2z * v2z)
    v3x = splineXi[stPt, c1] - splineXi[stPt, c2]
    v3y = splineYi[stPt, c1] - splineYi[stPt, c2]
    v3z = splineZi[stPt, c1] - splineZi[stPt, c2]
    mag3 = np.sqrt(v3x * v3x + v3y * v3y + v3z * v3z)
    v4x = splineXi[stPt, c0] - splineXi[stPt, c3]
    v4y = splineYi[stPt, c0] - splineYi[stPt, c3]
    v4z = splineZi[stPt, c0] - splineZi[stPt, c3]
    mag4 = np.sqrt(v4x * v4x + v4y * v4y + v4z * v4z)
    nE1 = np.ceil(mag1 / elementSize).astype(int)
    nE2 = np.ceil(mag3 / elementSize).astype(int)
    nE3 = np.ceil(mag2 / elementSize).astype(int)
    nE4 = np.ceil(mag4 / elementSize).astype(int)
    nEl = np.array([nE1, nE2, nE3, nE4])

    # ------------------------------------------------------------------
    # Build cross-section at frstXS and sweep along span.
    # ------------------------------------------------------------------
    sweepElements = []
    guideNds = []
    adhesMesh = None
    while stPt < splineXi.shape[0]:
        shellKp = np.zeros((9, 3))
        shellKp[0, :] = np.array(
            [splineXi[stPt, c0], splineYi[stPt, c0], splineZi[stPt, c0]]
        )
        shellKp[1, :] = np.array(
            [splineXi[stPt, c1], splineYi[stPt, c1], splineZi[stPt, c1]]
        )
        shellKp[2, :] = np.array(
            [splineXi[stPt, c2], splineYi[stPt, c2], splineZi[stPt, c2]]
        )
        shellKp[3, :] = np.array(
            [splineXi[stPt, c3], splineYi[stPt, c3], splineZi[stPt, c3]]
        )
        # Midside nodes — use the +1 spline column between c0 and c1 for
        # the HP-skin midpoint, and the +1 between c2 and c3 for the LP
        # one. The +1 column is the next intermediate sample produced by
        # XSCurvePts so it lies on the actual skin curve, not on a
        # straight chord between c0 and c1. Same convention as the
        # original TE code (which used spl5 for the HP midpoint between
        # spl4-spl6 and spl31 for the LP midpoint between spl30-spl32).
        c0_mid = c0 + 1
        c3_mid = c3 - 1
        shellKp[4, :] = np.array(
            [splineXi[stPt, c0_mid], splineYi[stPt, c0_mid], splineZi[stPt, c0_mid]]
        )
        shellKp[5, :] = 0.5 * shellKp[1, :] + 0.5 * shellKp[2, :]
        shellKp[6, :] = np.array(
            [splineXi[stPt, c3_mid], splineYi[stPt, c3_mid], splineZi[stPt, c3_mid]]
        )
        shellKp[7, :] = 0.5 * shellKp[0, :] + 0.5 * shellKp[3, :]
        shellKp[8, :] = 0.5 * shellKp[4, :] + 0.5 * shellKp[6, :]
        sReg = ShellRegion("quad2", shellKp, nEl, elType="quad", meshMethod="free")
        regMesh = sReg.createShellMesh()

        if stPt == frstXS:
            adhesMesh = Mesh3D(regMesh["nodes"], regMesh["elements"])
        else:
            guideNds.append(regMesh["nodes"])
            layerSwEl = np.ceil(
                (splineZi[stPt, c0] - splineZi[(stPt - 3), c0]) / elementSize
            ).astype(int)
            sweepElements.append(layerSwEl)
        stPt = stPt + 3

    adMeshData = adhesMesh.createSweptMesh(
        "toDestNodes", sweepElements, destNodes=guideNds, interpMethod="smooth"
    )

    # ------------------------------------------------------------------
    # Merge this bondline's nodes/elements into shellData. The first
    # call just stores the arrays; subsequent calls concatenate and
    # shift the new element connectivity by the existing node count.
    #
    # For backward compatibility ``shellData['adhesiveElSet']`` stays a
    # single dict naming the combined "adhesiveElements" element-set
    # (the Abaqus writer accesses it as a dict). Per-bondline subsets
    # live in ``shellData['adhesiveBondSets']`` — a list of
    # ``{name, labels}`` dicts so each physical bondline can be
    # identified separately by downstream code / tests.
    # ------------------------------------------------------------------
    new_nodes = np.asarray(adMeshData["nodes"], dtype=float)
    new_elements = np.asarray(adMeshData["elements"], dtype=int)
    has_existing = (
        "adhesiveNds" in shellData
        and shellData["adhesiveNds"] is not None
        and len(shellData["adhesiveNds"]) > 0
    )
    if has_existing:
        existing_nodes = np.asarray(shellData["adhesiveNds"], dtype=float)
        existing_elements = np.asarray(shellData["adhesiveEls"], dtype=int)
        nd_offset = existing_nodes.shape[0]
        el_offset = existing_elements.shape[0]
        shifted_elements = np.where(new_elements >= 0, new_elements + nd_offset, -1)
        merged_nodes = np.vstack([existing_nodes, new_nodes])
        merged_elements = np.vstack([existing_elements, shifted_elements])
        shellData["adhesiveNds"] = merged_nodes
        shellData["adhesiveEls"] = merged_elements
        new_el_labels = list(range(el_offset, el_offset + new_elements.shape[0]))
        combined_n_el = merged_elements.shape[0]
        shellData["adhesiveElSet"] = {
            "name": "adhesiveElements",
            "labels": list(range(0, combined_n_el)),
        }
        shellData.setdefault("adhesiveBondSets", [])
        shellData["adhesiveBondSets"].append(
            {"name": bond_name, "labels": list(new_el_labels)}
        )
        adMeshData_for_ties = {
            "nodes": merged_nodes,
            "elements": merged_elements,
            "sets": {
                "element": [
                    {"name": "adhesiveElements",
                     "labels": list(range(0, combined_n_el))},
                    {"name": bond_name, "labels": list(new_el_labels)},
                ],
                "node": [],
            },
        }
    else:
        shellData["adhesiveNds"] = new_nodes
        shellData["adhesiveEls"] = new_elements
        adEls = new_elements.shape[0]
        labList = list(range(0, adEls))
        adhesSet = {"name": "adhesiveElements", "labels": labList}
        shellData["adhesiveElSet"] = adhesSet
        shellData["adhesiveBondSets"] = [
            {"name": bond_name, "labels": list(labList)}
        ]
        adMeshData_for_ties = {
            "nodes": new_nodes,
            "elements": new_elements,
            "sets": {
                "element": [adhesSet, {"name": bond_name, "labels": list(labList)}],
                "node": [],
            },
        }

    # ------------------------------------------------------------------
    # Tie constraints. Same logic as the original TE-only code but
    # parameterised by the target shell-set substrings and the bond
    # element set (so surface filtering on the LP/HP normals operates
    # on *this* bondline's elements, not on every adhesive element
    # accumulated so far). Use an endswith match so 'HP_LE' matches
    # only ``NN_NN_HP_LE`` and not ``NN_NN_HP_LE_PANEL``.
    # ------------------------------------------------------------------
    # Normalise per-side substring(s) to a tuple so callers can pass a
    # single string or a tuple covering several shell-set families.
    lp_substrs = (lp_substr,) if isinstance(lp_substr, str) else tuple(lp_substr)
    hp_substrs = (hp_substr,) if isinstance(hp_substr, str) else tuple(hp_substr)
    lp_suffixes = tuple("_" + s for s in lp_substrs)
    hp_suffixes = tuple("_" + s for s in hp_substrs)
    lpEls = []
    hpEls = []
    for es in shellData["sets"]["element"]:
        name = es["name"]
        if any(name.endswith(sfx) for sfx in lp_suffixes):
            lpEls.extend(es["labels"])
        elif any(name.endswith(sfx) for sfx in hp_suffixes):
            hpEls.extend(es["labels"])
    lp_set_name = f"{bond_name}__LP_TGT"
    hp_set_name = f"{bond_name}__HP_TGT"
    all_set_name = f"{bond_name}__ALL_TGT"
    shellData = add_element_set(shellData, {"name": lp_set_name, "labels": lpEls})
    shellData = add_element_set(shellData, {"name": hp_set_name, "labels": hpEls})

    # Classify surface nodes by their outward face normal direction.
    # Same 60-degree-cone convention as the original TE code: edge 2-3
    # is the LP face (outward +y), edge 0-1 the HP face (outward -y).
    lp_nodes_setname = f"{bond_name}__LP_AdNodes"
    hp_nodes_setname = f"{bond_name}__HP_AdNodes"
    nDir = np.array([0.0, 1.0, 0.0])
    adMeshData_for_ties = get_surface_nodes(
        adMeshData_for_ties, bond_name, lp_nodes_setname, nDir, normTol=60.0
    )
    nDir = np.array([0.0, -1.0, 0.0])
    adMeshData_for_ties = get_surface_nodes(
        adMeshData_for_ties, bond_name, hp_nodes_setname, nDir, normTol=60.0
    )

    # Estimate the LP-HP gap for the fallback maxDist.
    adhBondGap = float(mag3)
    fallbackDist = max(1.5 * adhBondGap, 4.0 * float(elementSize))

    constraints = tie_2_meshes_constraints(
        adMeshData_for_ties, lp_nodes_setname,
        shellData, lp_set_name,
        0.5 * elementSize,
    )
    hpConst = tie_2_meshes_constraints(
        adMeshData_for_ties, hp_nodes_setname,
        shellData, hp_set_name,
        0.5 * elementSize,
    )
    constraints.extend(hpConst)

    # Combined LP+HP target set for the residual fallback.
    shellData = add_element_set(
        shellData,
        {"name": all_set_name, "labels": list(set(lpEls) | set(hpEls))},
    )

    already_tied = set()
    for ce in constraints:
        for term in ce["terms"]:
            if term.get("nodeSet") == "tiedMesh":
                already_tied.add(int(term["node"]))

    # Residual fallback: tie any adhesive node of *this* bondline that
    # wasn't picked up by the LP/HP surface filter to the nearest face
    # in the combined target set.
    bond_elements = np.asarray(adMeshData_for_ties["elements"], dtype=int)
    bond_label_set = set()
    for s in adMeshData_for_ties["sets"]["element"]:
        if s["name"] == bond_name:
            bond_label_set = set(s["labels"])
            break
    used_nds = set()
    for eid, el in enumerate(bond_elements):
        if eid not in bond_label_set:
            continue
        for nd in el:
            if nd >= 0:
                used_nds.add(int(nd))
    residual_labels = sorted(used_nds - already_tied)
    if residual_labels:
        residual_set_name = f"{bond_name}__AdNodes_residual"
        residual_set = {"name": residual_set_name, "labels": residual_labels}
        adMeshData_for_ties["sets"]["node"].append(residual_set)
        resConst = tie_2_meshes_constraints(
            adMeshData_for_ties, residual_set_name,
            shellData, all_set_name,
            fallbackDist,
        )
        constraints.extend(resConst)

    if "constraints" in shellData and shellData["constraints"] is not None:
        shellData["constraints"].extend(constraints)
    else:
        shellData["constraints"] = constraints

    return shellData


# When a TE_FLAT chord (the blunt trailing-edge strip) is thinner than
# ``elementSize / _MERGE_TE_FLAT_AR`` it produces a lone 1-element-wide
# spanwise strip whose aspect ratio (span/chord) exceeds this value.
# Those slivers pass the static AR<20 check but warp past ANSYS's
# element-formulation limit under NLGEOM at large tip deflection. When
# the threshold is crossed the flat is folded into the adjacent TE_REINF
# region (see _station_region_plan), widening that region's chord so the
# sliver disappears. See docs/dev/TODO_shell_mesh_pipeline.md #1.
_MERGE_TE_FLAT_AR = 4.0


def _shell_kp(splineXi, splineYi, splineZi, stPt, cols):
    """Build the 16-point quad3 control array for one (non-merged) shell
    patch from four consecutive chord columns.

    Parameters
    ----------
    stPt : int
        Spanwise start row; the patch spans rows ``stPt .. stPt+3``.
    cols : tuple[int, int, int, int]
        The four chord control columns ``(stSp, stSp+1, stSp+2, stSp+3)``.

    Returns
    -------
    (16, 3) ndarray
        quad3 keypoints in the canonical order: 4 corners, then the 12
        edge/interior points (identical to what the station loop has
        always emitted).
    """
    c0, c1, c2, c3 = cols

    def P(dr, c):
        return [splineXi[stPt + dr, c], splineYi[stPt + dr, c], splineZi[stPt + dr, c]]

    return np.array([
        P(0, c0), P(0, c3), P(3, c3), P(3, c0),
        P(0, c1), P(0, c2), P(1, c3), P(2, c3),
        P(3, c2), P(3, c1), P(2, c0), P(1, c0),
        P(1, c1), P(1, c2), P(2, c2), P(2, c1),
    ])


def _shell_kp_merged(splineXi, splineYi, splineZi, stPt, c_lo, c_hi):
    """Build the 16-point quad3 control array for a *merged* TE_FLAT+TE_REINF
    patch spanning chord columns ``c_lo .. c_hi`` (7 columns).

    The merged region's chord controls are resampled at four **arc-length**
    fractions (0, 1/3, 2/3, 1) of the chord polyline through columns
    ``c_lo .. c_hi`` for each spanwise row. This is essential: picking
    every-other spline column directly (e.g. 0,2,4,6) gives control points
    bunched in the thin flat and sparse in the wide reinf, which the cubic
    map turns into high-AR slivers. Arc-length resampling produces an
    evenly-proportioned patch.

    The chord endpoints (fraction 0 -> column ``c_lo``; fraction 1 ->
    column ``c_hi``) land exactly on the original spline columns, so the
    TE-most closure edge and the outboard edge shared with TE_PANEL are
    unchanged — mesh continuity is preserved. Only interior controls move.
    """
    rows = (stPt, stPt + 1, stPt + 2, stPt + 3)
    fracs = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
    cols = np.arange(c_lo, c_hi + 1)
    grid = np.empty((4, 4, 3))  # [row_idx, chord_frac_idx, xyz]
    for ri, r in enumerate(rows):
        pts = np.stack(
            [splineXi[r, cols], splineYi[r, cols], splineZi[r, cols]], axis=1
        )
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = s[-1]
        for fi, f in enumerate(fracs):
            tgt = f * total
            grid[ri, fi] = [np.interp(tgt, s, pts[:, k]) for k in range(3)]

    def G(ri, fi):
        return grid[ri, fi]

    return np.array([
        G(0, 0), G(0, 3), G(3, 3), G(3, 0),
        G(0, 1), G(0, 2), G(1, 3), G(2, 3),
        G(3, 2), G(3, 1), G(2, 0), G(1, 0),
        G(1, 1), G(1, 2), G(2, 2), G(2, 1),
    ])


def _station_region_plan(splineXi, splineYi, splineZi, stPt, elementSize,
                         enable_merge=True):
    """Return the per-station region list as ``(stack_index, spec)`` tuples.

    ``spec`` is either ``("cols", (c0, c1, c2, c3))`` for a normal region
    (four consecutive chord columns) or ``("merge", c_lo, c_hi)`` for a
    TE_REINF region that has absorbed an adjacent thin TE_FLAT.

    Normally this is the 12 chord regions, each consuming three spline
    columns. When a TE_FLAT chord is thinner than
    ``elementSize / _MERGE_TE_FLAT_AR`` the flat is folded into the
    adjacent TE_REINF: the flat's own region is dropped and the reinf
    region spans both. The TE-most chord column (0 for HP, 36 for LP) is
    preserved, so the trailing-edge closure edge is unchanged.

    ``enable_merge`` is False on the solid-seed path (``forSolid``): the
    thin-flat merge is a shell-mesh fix for NLGEOM, and the solid pipeline
    has its own post-extrusion untangler that the merged TE seed would
    otherwise disrupt (it regresses solid-element Jacobians). Solid seeds
    therefore keep the original 12-region layout.
    """
    def chord_len(ca, cb):
        pa = np.array([splineXi[stPt, ca], splineYi[stPt, ca], splineZi[stPt, ca]])
        pb = np.array([splineXi[stPt, cb], splineYi[stPt, cb], splineZi[stPt, cb]])
        return float(np.linalg.norm(pb - pa))

    thresh = elementSize / _MERGE_TE_FLAT_AR
    hp_merge = enable_merge and 0.0 < chord_len(0, 3) < thresh
    lp_merge = enable_merge and 0.0 < chord_len(33, 36) < thresh

    plan = []
    if hp_merge:
        plan.append((1, ("merge", 0, 6)))            # HP_TE_REINF absorbs HP_TE_FLAT
    else:
        plan.append((0, ("cols", (0, 1, 2, 3))))     # HP_TE_FLAT
        plan.append((1, ("cols", (3, 4, 5, 6))))     # HP_TE_REINF
    for j in range(2, 10):                           # middle regions unchanged
        b = 3 * j
        plan.append((j, ("cols", (b, b + 1, b + 2, b + 3))))
    if lp_merge:
        plan.append((10, ("merge", 30, 36)))         # LP_TE_REINF absorbs LP_TE_FLAT
    else:
        plan.append((10, ("cols", (30, 31, 32, 33))))  # LP_TE_REINF
        plan.append((11, ("cols", (33, 34, 35, 36))))  # LP_TE_FLAT
    return plan


def _compute_edge_nels(shellKp, elementSize, region_name="", force_match_opposite=True):
    """Compute the four per-edge element counts of a shell patch from its
    corner keypoints and the target element size.

    The shell mesher derives the per-edge element count from chord length
    divided by ``elementSize``, rounded up. When the patch is trapezoidal
    (opposite edges of different length), the two opposite chord edges
    naturally end up with different counts — which fires the buggy
    node-pulling code in ``ShellRegion.createShellMesh``.

    Fix: when ``force_match_opposite`` is True (the default), opposite
    edges are clamped to ``max(nEl_a, nEl_b)`` so the structured-mesh
    branch never has to do node-pulling. The cost is mild over-refinement
    in strongly trapezoidal patches (chord-direction count is governed by
    the longer of the two chord edges); the benefit is no Jacobian flips,
    no slivers, no warped quads. Neighbour-cell connectivity is preserved
    because both neighbours' ``ceil(|same_edge|/elementSize)`` evaluations
    agree (same vector, same divisor); the ``max()`` of equal numbers
    against equal numbers stays equal across the boundary.

    Set ``force_match_opposite=False`` to recover the original buggy
    behaviour (useful for regression testing the fix's impact).

    Parameters
    ----------
    shellKp : (>=4, 3) ndarray
        Shell keypoints; only rows 0..3 (corners) are read here.
    elementSize : float
        Target element size in metres.
    region_name : str
        Optional label for the JSONL ``context.region_name`` field.
    force_match_opposite : bool, default True
        When True, force opposite edges to share the larger element count.

    Returns
    -------
    nEl : (4,) ndarray of int
        Element counts on the four edges in walk order (0->1, 1->2, 2->3, 3->0).
    """
    assert elementSize > 0, f"_compute_edge_nels: elementSize must be positive, got {elementSize}"
    edge_lens = np.array(
        [
            np.linalg.norm(shellKp[1, :] - shellKp[0, :]),
            np.linalg.norm(shellKp[2, :] - shellKp[1, :]),
            np.linalg.norm(shellKp[3, :] - shellKp[2, :]),
            np.linalg.norm(shellKp[0, :] - shellKp[3, :]),
        ]
    )
    if not np.all(np.isfinite(edge_lens)):
        _log.error(
            "non-finite edge lengths in shell patch",
            extra={"stage": "edge_nels", "region_name": region_name, "edge_lens": edge_lens},
        )
        raise ValueError(
            f"shell patch {region_name!r} has non-finite edge lengths: {edge_lens}"
        )
    # Detect degenerate patches (chord-zero at tip/root) before clamping.
    # A patch whose shortest edge is < 5% of elementSize cannot be meshed
    # without producing sliver/triangle elements that ANSYS rejects. Return
    # None so the caller can skip this patch entirely.
    edge_min = float(edge_lens.min())
    if edge_min < _DEGENERATE_EDGE_RATIO * elementSize:
        _log.warning(
            "degenerate shell patch detected (shortest edge < threshold) — skipping",
            extra={
                "stage": "edge_nels",
                "region_name": region_name,
                "edge_lens": edge_lens.tolist(),
                "min_edge": edge_min,
                "threshold": _DEGENERATE_EDGE_RATIO * elementSize,
                "ratio": edge_min / elementSize,
            },
        )
        return None

    # Detect duplicate keypoints in the full 16-point shellKp grid (if
    # available). At cylindrical root/tip regions, XSCurvePts[i] can contain
    # repeated geometry-curve indices after np.round() of close linspace
    # values; that produces interior keypoints coincident with boundary
    # keypoints, and the quad3 Lagrange basis interpolant becomes degenerate
    # — yielding bow-tied elements deep inside the patch even though the
    # four corners look fine.
    if shellKp.shape[0] >= 16:
        # Spatial-hash tolerance: 1% of the minimum edge length (already
        # validated as >= 5% of elementSize above)
        tol = max(1e-9, 0.01 * edge_min)
        rounded = np.round(shellKp[:16] / tol).astype(np.int64)
        seen: dict[tuple, int] = {}
        for i, key in enumerate(map(tuple, rounded)):
            if key in seen:
                _log.warning(
                    "shellKp has duplicate keypoints (likely XSCurvePts collision "
                    "from np.round in linspace) — skipping patch to avoid degenerate "
                    "Lagrange interpolant",
                    extra={
                        "stage": "edge_nels",
                        "region_name": region_name,
                        "duplicate_indices": [seen[key], i],
                        "duplicate_coord": shellKp[i].tolist(),
                        "tol": tol,
                    },
                )
                return None
            seen[key] = i

    # Final check: are the four CORNER keypoints in a consistently-wound
    # (non-bow-tied) configuration? At narrow airfoil-transition regions
    # the chord-direction sampling can produce slight inconsistencies
    # between adjacent z-stations that don't trip the duplicate-keypoint
    # check (corners differ by ~1e-3 m but in opposite directions). We
    # detect this by running the same Jacobian-flip test we apply to the
    # final elements: project the four corners onto their dominant plane
    # and check that the two diagonal triangles have the same signed area.
    from pynumad.testing.mesh_quality import quad_has_jacobian_flip
    if quad_has_jacobian_flip(shellKp[:4]):
        _log.warning(
            "twist-induced bow-tie in patch corners — skipping",
            extra={
                "stage": "edge_nels",
                "region_name": region_name,
                "corner_coords": shellKp[:4].tolist(),
                "edge_lens": edge_lens.tolist(),
            },
        )
        return None

    nEl_raw = np.ceil(edge_lens / elementSize).astype(int)
    # Guard against zero counts (would produce a degenerate region)
    if np.any(nEl_raw <= 0):
        _log.warning(
            "edge_nels: zero count produced, clamping to 1",
            extra={
                "stage": "edge_nels",
                "region_name": region_name,
                "edge_lens": edge_lens.tolist(),
                "nEl_before_clamp": nEl_raw.tolist(),
            },
        )
        nEl_raw = np.maximum(nEl_raw, 1)

    mismatch_chord = int(nEl_raw[0] - nEl_raw[2])
    mismatch_span = int(nEl_raw[1] - nEl_raw[3])
    has_mismatch = mismatch_chord != 0 or mismatch_span != 0

    if force_match_opposite and has_mismatch:
        chord_nel = int(max(nEl_raw[0], nEl_raw[2]))
        span_nel = int(max(nEl_raw[1], nEl_raw[3]))
        nEl = np.array([chord_nel, span_nel, chord_nel, span_nel], dtype=int)
        _log.info(
            "opposite-edge mismatch clamped to max (force_match_opposite)",
            extra={
                "stage": "edge_nels",
                "region_name": region_name,
                "edge_lens": edge_lens.tolist(),
                "nEl_raw": nEl_raw.tolist(),
                "nEl_clamped": nEl.tolist(),
                "mismatch_chord": mismatch_chord,
                "mismatch_span": mismatch_span,
            },
        )
    elif has_mismatch:
        # Unfixed path — log at WARNING so the JSONL sidecar exposes the
        # bug trigger immediately.
        nEl = nEl_raw
        _log.warning(
            "opposite-edge mismatch (triggers node-pulling in ShellRegion)",
            extra={
                "stage": "edge_nels",
                "region_name": region_name,
                "edge_lens": edge_lens.tolist(),
                "nEl": nEl.tolist(),
                "mismatch_chord": mismatch_chord,
                "mismatch_span": mismatch_span,
            },
        )
    else:
        nEl = nEl_raw
        _log.debug(
            "edge_nels matched",
            extra={"stage": "edge_nels", "region_name": region_name, "nEl": nEl.tolist()},
        )
    return nEl


def shell_mesh_general(blade, forSolid, includeAdhesive, elementSize):
    """
    This method generates a finite element shell mesh for the blade, based on what is
    stored in blade.geometry.coordinates, blade.keypoints.key_points, 
    and blade.geometry.profiles.  Output is given as a python dictionary.

    Parameters
    -----------
    blade: Blade
    forSolid: bool
    includeAdhesive: bool
    elementSize: float

    Returns
    -------
    meshData:
    Nodes and elements for outer shell and shear webs:
    nodes:
        - [x, y, z]
        - [x, y, z]
        ...
    elements:
        - [n1,n2,n3,n4]
        - [n1,n2,n3,n4]
        ...
    Set list and section list for the outer shell and shear webs.
    These are companion lists with the same length and order,
    so meshData['sets']['element'][i] corresponds to meshData['sections'][i]
    sets:
        element:
            - name: set1Name
              labels: [e1, e2, e3 ...]
            - name: set2Name
              labels: [e1, e2, e3 ...]
            ...
    sections:
        - type: 'shell'
          layup:
             - [materialid,thickness,angle] ## layer 1
             - [materialid,thickness,angle] ## layer 2
             ...
          elementSet: set1Name
        - type: 'shell'
          layup:
             - [materialid,thickness,angle] ## layer 1
             - [materialid,thickness,angle] ## layer 2
             ...
          elementSet: set2Name
    Nodes, elements and set for adhesive elements
    adhesiveNds:
        - [x, y, z]
        ...
    adhesiveEls:
        - [n1,n2,n3,n4,n5,n6,n7,n8]
        ...
    adhesiveElSet:
        name: 'adhesiveElements'
        labels: [0,1,2,3....(number of adhesive elements)]
    """
    _stacks_attr = getattr(blade.stackdb, "stacks", None)
    _swstacks_attr = getattr(blade.stackdb, "swstacks", None)
    _log.info(
        "shell_mesh_general entry",
        extra={
            "stage": "entry",
            "elementSize": float(elementSize),
            "forSolid": bool(forSolid),
            "includeAdhesive": bool(includeAdhesive),
            "stacks_shape": list(_stacks_attr.shape)
            if isinstance(_stacks_attr, np.ndarray)
            else (len(_stacks_attr) if _stacks_attr is not None else 0),
            "swstacks_shape": list(_swstacks_attr.shape)
            if isinstance(_swstacks_attr, np.ndarray)
            else (len(_swstacks_attr) if _swstacks_attr is not None else 0),
        },
    )

    geometry = blade.geometry
    coordinates = geometry.coordinates
    profiles = geometry.profiles
    key_points = blade.keypoints.key_points
    stacks = blade.stackdb.stacks
    swstacks = blade.stackdb.swstacks

    geomSz = coordinates.shape
    lenGeom = geomSz[0]
    numXsec = geomSz[2]
    # XSCurvePts stores fractional indices into the chord-direction
    # geometry curve at each station. Float dtype (not int) so that when
    # consecutive design keypoints are < 3 geometry-indices apart (typical
    # at root cylinder transition and tip taper), the linspace subdivision
    # below produces strictly monotonic samples instead of collisions
    # from np.round(). Fractional indices are resolved via np.interp on
    # the coordinates array.
    XSCurvePts = np.array([], dtype=float)
    assert numXsec >= 2, f"shell_mesh_general: blade has only {numXsec} cross sections"
    assert elementSize > 0, f"shell_mesh_general: elementSize must be positive, got {elementSize}"

    ## Determine the key curve points along the OML at each cross section
    for i in range(numXsec):
        keyPts = np.array([0])
        minDist = 1
        lePt = 0
        for j in range(lenGeom):
            prof = profiles[j, :, i]
            mag = np.linalg.norm(prof)
            if mag < minDist:
                minDist = mag
                lePt = j

        for j in range(5):
            kpCrd = key_points[j, :, i]
            minDist = geometry.ichord[i]
            pti = 1
            for k in range(lePt):
                ptCrd = coordinates[k, :, i]
                vec = ptCrd - kpCrd
                mag = np.linalg.norm(vec)
                if mag < minDist:
                    minDist = mag
                    pti = k
            keyPts = np.concatenate((keyPts, [pti]))
            coordinates[pti, :, i] = np.array(kpCrd)

        keyPts = np.concatenate((keyPts, [lePt]))
        for j in range(5, 10):
            kpCrd = key_points[j, :, i]
            minDist = geometry.ichord[i]
            pti = 1
            for k in range(lePt, lenGeom):
                ptCrd = coordinates[k, :, i]
                vec = ptCrd - kpCrd
                mag = np.linalg.norm(vec)
                if mag < minDist:
                    minDist = mag
                    pti = k
            keyPts = np.concatenate((keyPts, [pti]))
            coordinates[pti, :, i] = np.array(kpCrd)

        keyPts = np.concatenate((keyPts, [lenGeom - 1]))
        allPts = np.array([float(keyPts[0])])
        for j in range(0, len(keyPts) - 1):
            # 4 fractional samples between consecutive key indices. NO
            # rounding: the downstream lookup uses np.interp so fractional
            # values are well-defined, and we avoid the collision pattern
            # that np.round(linspace) produces when keyPts are < 3 apart
            # (root cylinder / tip taper regions).
            secPts = np.linspace(float(keyPts[j]), float(keyPts[j + 1]), 4)
            allPts = np.concatenate((allPts, secPts[1:4]))

        XSCurvePts = np.vstack((XSCurvePts, allPts)) if XSCurvePts.size else allPts
    rws, cls = XSCurvePts.shape

    ## Create longitudinal splines down the blade through each of the key X-section points
    # `coordinates[:, :, i]` is a (lenGeom, 3) chord-direction polyline at
    # station i. We evaluate it at the fractional indices XSCurvePts[i, :]
    # via linear interpolation. This guarantees that consecutive entries of
    # XSCurvePts produce DISTINCT 3D points whenever the underlying chord
    # curve has any extent — eliminating the duplicate-keypoint pathology
    # at the root and tip.
    _geom_idx = np.arange(lenGeom, dtype=float)

    def _interp_station(station_idx: int) -> np.ndarray:
        """Return (cls, 3) interpolated coords for station `station_idx`."""
        xp = XSCurvePts[station_idx, :]
        out = np.empty((xp.shape[0], 3), dtype=float)
        for axis in range(3):
            out[:, axis] = np.interp(xp, _geom_idx, coordinates[:, axis, station_idx])
        return out

    _station0 = _interp_station(0)
    splineX = _station0[:, 0]
    splineY = _station0[:, 1]
    splineZ = _station0[:, 2]
    for i in range(1, rws):
        _row = _interp_station(i)
        splineX = np.vstack((splineX, _row[:, 0].T))
        splineY = np.vstack((splineY, _row[:, 1].T))
        splineZ = np.vstack((splineZ, _row[:, 2].T))

    spParam = np.transpose(np.linspace(0, 1, rws))
    nSpi = rws + 2 * (rws - 1)
    spParami = np.transpose(np.linspace(0, 1, nSpi))
    splineXi = interpolator_wrap(spParam, splineX[:, 0], spParami, "pchip")
    splineYi = interpolator_wrap(spParam, splineY[:, 0], spParami, "pchip")
    splineZi = interpolator_wrap(spParam, splineZ[:, 0], spParami, "pchip")
    for i in range(1, cls):
        splineXi = np.vstack(
            [splineXi, interpolator_wrap(spParam, splineX[:, i], spParami, "pchip")]
        )
        splineYi = np.vstack(
            [splineYi, interpolator_wrap(spParam, splineY[:, i], spParami, "pchip")]
        )
        splineZi = np.vstack(
            [splineZi, interpolator_wrap(spParam, splineZ[:, i], spParami, "pchip")]
        )
    splineXi = splineXi.T
    splineYi = splineYi.T
    splineZi = splineZi.T
    ## Determine the first spanwise section that needs adhesive
    if includeAdhesive == 1:
        stPt = 0
        frstXS = 0
        while frstXS == 0 and stPt < splineXi.shape[0]:
            v1x = splineXi[stPt, 6] - splineXi[stPt, 4]
            v1y = splineYi[stPt, 6] - splineYi[stPt, 4]
            v1z = splineZi[stPt, 6] - splineZi[stPt, 4]
            v2x = splineXi[stPt, 30] - splineXi[stPt, 32]
            v2y = splineYi[stPt, 30] - splineYi[stPt, 32]
            v2z = splineZi[stPt, 30] - splineZi[stPt, 32]
            mag1 = np.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
            mag2 = np.sqrt(v2x * v2x + v2y * v2y + v2z * v2z)
            dp = (1 / (mag1 * mag2)) * (v1x * v2x + v1y * v2y + v1z * v2z)
            if dp > 0.7071:
                frstXS = stPt
            stPt = stPt + 3

        if frstXS == 0:
            frstXS = splineXi.shape[0]
    else:
        frstXS = splineXi.shape[0]

    ## Generate the mesh using the splines as surface guides
    
    ## --------------------------TEMP
    ## secNel = [1,3,15,5,8,1,1,13,5,12,3,1]
    ## secNel = [1,4,10,10,8,2,2,8,10,10,4,1]
    ## --------------------------END TEMP
    
    bladeSurf = Surface()
    ## Outer shell sections
    outShES = set()
    secList = list()
    stPt = 0
    for i in range(rws - 1):
        # if stPt < frstXS:
        #     stSec = 0
        #     endSec = 11
        #     stSp = 0
        # else:
        #     stSec = 1
        #     endSec = 10
        #     stSp = 3
        # Per-station chord regions. Normally the 12 fixed regions
        # (HP_TE_FLAT .. LP_TE_FLAT), but on the shell path thin TE_FLAT
        # strips near the tip are folded into the adjacent TE_REINF to
        # avoid high-aspect-ratio slivers that warp under NLGEOM. The
        # merge is disabled for solid seeds (forSolid) — see
        # _station_region_plan.
        plan = _station_region_plan(
            splineXi, splineYi, splineZi, stPt, elementSize,
            enable_merge=(not forSolid),
        )
        for stack_j, spec in plan:
            if spec[0] == "merge":
                shellKp = _shell_kp_merged(
                    splineXi, splineYi, splineZi, stPt, spec[1], spec[2]
                )
            else:
                shellKp = _shell_kp(splineXi, splineYi, splineZi, stPt, spec[1])
            # Per-edge element counts: see _compute_edge_nels for the trapezoidal-
            # patch caveat. The Sandia commented hint `nEl1 = secNel[j]` /
            # `nEl3 = secNel[j]` suggests opposite chord edges should share a
            # precomputed per-stack count; current code derives them per cell.
            nEl = _compute_edge_nels(
                shellKp, elementSize, region_name=stacks[stack_j, i].name
            )
            if nEl is None:
                # Degenerate patch (e.g. zero-chord at root/tip cylinder).
                # _compute_edge_nels already logged the skip.
                continue

            bladeSurf.addShellRegion(
                "quad3",
                shellKp,
                nEl,
                name=stacks[stack_j, i].name,
                elType="quad",
                meshMethod="structured",
            )
            outShES.add(stacks[stack_j, i].name)
            newSec = dict()
            newSec["type"] = "shell"
            layup = list()
            for pg in stacks[stack_j, i].plygroups:
                totThick = 0.001*pg.thickness * pg.nPlies
                ply = [pg.materialid, totThick, pg.angle]
                layup.append(ply)
            newSec["layup"] = layup
            newSec["elementSet"] = stacks[stack_j, i].name
            newSec["xDir"] = (shellKp[3,:] - shellKp[0,:]) + (shellKp[2,:] - shellKp[1,:])
            newSec["xyDir"] = (shellKp[1,:] - shellKp[0,:]) + (shellKp[2,:] - shellKp[3,:])
            secList.append(newSec)
        stPt = stPt + 3

    ## Shift the appropriate splines if the mesh is for a solid model seed
    if forSolid == 1:
        caseIndex = np.array([[9, 27, 3], [12, 24, 3], [24, 12, 8], [27, 9, 8]])
        for i in range(caseIndex.shape[0]):
            spl = caseIndex[i, 0]
            tgtSp = caseIndex[i, 1]
            sec = caseIndex[i, 2]
            stPt = 0
            for j in range(rws - 1):
                totalThick = 0
                for k in range(3):
                    tpp = 0.001 * stacks[sec, j].plygroups[k].thickness
                    npls = stacks[sec, j].plygroups[k].nPlies
                    totalThick = totalThick + tpp * npls
                for k in range(3):
                    vx = splineXi[stPt, tgtSp] - splineXi[stPt, spl]
                    vy = splineYi[stPt, tgtSp] - splineYi[stPt, spl]
                    vz = splineZi[stPt, tgtSp] - splineZi[stPt, spl]
                    magInv = 1 / np.sqrt(vx * vx + vy * vy + vz * vz)
                    ux = magInv * vx
                    uy = magInv * vy
                    uz = magInv * vz
                    splineXi[stPt, spl] = splineXi[stPt, spl] + 0.1*totalThick * ux
                    splineYi[stPt, spl] = splineYi[stPt, spl] + 0.1*totalThick * uy
                    splineZi[stPt, spl] = splineZi[stPt, spl] + 0.1*totalThick * uz
                    stPt = stPt + 1

    ## Shear web sections
    #
    # Each shear web spans HP -> LP across the cross-section.  The HP/LP
    # endpoints are located at design keypoint indices stored in
    # `blade.keypoints.web_indices[k_web] = [hp_key_idx, lp_key_idx]`.
    # XSCurvePts is built with 3 fractional samples between each consecutive
    # design keypoint, so the spline-column for design keypoint `k` is
    # `3 * k`.
    #
    # Historical orientation note: in the original 2-web implementation the
    # first web placed shellKp[0] on the HP edge and shellKp[1] on the LP
    # edge, while the second web swapped that ordering (shellKp[0]=LP,
    # shellKp[1]=HP).  This affects the `xyDir` of the shell section and
    # therefore the in-plane material orientation.  To remain
    # bit-identical for existing 2-web blades (e.g. BAR0) we preserve that
    # convention: k_web == 0 uses HP->LP; all other webs use LP->HP.
    swES = set()
    n_webs = swstacks.shape[0]
    web_indices = blade.keypoints.web_indices

    def _col_for_key(k_idx):
        """Spline-column index for design keypoint `k_idx`.

        XSCurvePts samples 3 fractional points between consecutive design
        keypoints (see XSCurvePts construction above), so the column
        offset is simply `3 * k_idx`.  Returns None when the web endpoint
        is not snapped to a design keypoint (np.nan in web_indices).
        """
        if k_idx is None:
            return None
        try:
            if np.isnan(k_idx):
                return None
        except (TypeError, ValueError):
            pass
        return 3 * int(k_idx)

    stPt = 0
    for i in range(rws - 1):
        for k_web in range(n_webs):
            sw_stack = swstacks[k_web][i]
            if not sw_stack.plygroups:
                continue

            # Resolve HP / LP spline columns for this web.  If either
            # endpoint isn't snapped to a design keypoint (np.nan in
            # web_indices), we have no spline column to read from, so we
            # skip this web at this station rather than emit garbage.
            try:
                hp_key, lp_key = web_indices[k_web]
            except (IndexError, TypeError, ValueError):
                continue
            hp_col = _col_for_key(hp_key)
            lp_col = _col_for_key(lp_key)
            if hp_col is None or lp_col is None:
                continue

            # Preserve original (k_web == 0) orientation for backward
            # compatibility; all subsequent webs use the swapped ordering
            # historically applied to web 1.
            if k_web == 0:
                col0, col1 = hp_col, lp_col
            else:
                col0, col1 = lp_col, hp_col

            shellKp = np.zeros((16, 3))
            shellKp[0, :] = np.array(
                [splineXi[stPt, col0], splineYi[stPt, col0], splineZi[stPt, col0]]
            )
            shellKp[1, :] = np.array(
                [splineXi[stPt, col1], splineYi[stPt, col1], splineZi[stPt, col1]]
            )
            shellKp[2, :] = np.array(
                [splineXi[stPt + 3, col1], splineYi[stPt + 3, col1], splineZi[stPt + 3, col1]]
            )
            shellKp[3, :] = np.array(
                [splineXi[stPt + 3, col0], splineYi[stPt + 3, col0], splineZi[stPt + 3, col0]]
            )
            shellKp[6, :] = np.array(
                [splineXi[stPt + 1, col1], splineYi[stPt + 1, col1], splineZi[stPt + 1, col1]]
            )
            shellKp[7, :] = np.array(
                [splineXi[stPt + 2, col1], splineYi[stPt + 2, col1], splineZi[stPt + 2, col1]]
            )
            shellKp[10, :] = np.array(
                [splineXi[stPt + 2, col0], splineYi[stPt + 2, col0], splineZi[stPt + 2, col0]]
            )
            shellKp[11, :] = np.array(
                [splineXi[stPt + 1, col0], splineYi[stPt + 1, col0], splineZi[stPt + 1, col0]]
            )
            shellKp[4, :] = 0.6666 * shellKp[0, :] + 0.3333 * shellKp[1, :]
            shellKp[5, :] = 0.3333 * shellKp[0, :] + 0.6666 * shellKp[1, :]
            shellKp[8, :] = 0.6666 * shellKp[2, :] + 0.3333 * shellKp[3, :]
            shellKp[9, :] = 0.3333 * shellKp[2, :] + 0.6666 * shellKp[3, :]
            shellKp[12, :] = 0.6666 * shellKp[11, :] + 0.3333 * shellKp[6, :]
            shellKp[13, :] = 0.3333 * shellKp[11, :] + 0.6666 * shellKp[6, :]
            shellKp[14, :] = 0.6666 * shellKp[7, :] + 0.3333 * shellKp[10, :]
            shellKp[15, :] = 0.3333 * shellKp[7, :] + 0.6666 * shellKp[10, :]

            # Per-edge element counts via the central helper, which also
            # forces opposite-edge equality to avoid the node-pulling bug.
            nEl = _compute_edge_nels(
                shellKp, elementSize, region_name=sw_stack.name
            )
            if nEl is not None:
                bladeSurf.addShellRegion(
                    "quad3",
                    shellKp,
                    nEl,
                    name=sw_stack.name,
                    elType="quad",
                    meshMethod="structured",
                )
                swES.add(sw_stack.name)
                newSec = dict()
                newSec["type"] = "shell"
                layup = list()
                for pg in sw_stack.plygroups:
                    totThick = 0.001*pg.thickness * pg.nPlies
                    ply = [pg.materialid, totThick, pg.angle]
                    layup.append(ply)
                newSec["layup"] = layup
                newSec["elementSet"] = sw_stack.name
                newSec["xDir"] = np.array([0.0,0.0,1.0])
                newSec["xyDir"] = (shellKp[1,:] - shellKp[0,:]) + (shellKp[2,:] - shellKp[3,:])
                secList.append(newSec)
        stPt = stPt + 3

    ## Generate Shell mesh

    print('getting blade mesh')
    shellData = bladeSurf.getSurfaceMesh()
    shellData["sections"] = secList
    
    ## Get local direction cosine orientations for individual elements
    print('getting element orientations')
    nodes = shellData["nodes"]
    elements = shellData["elements"]
    numEls = len(shellData["elements"])
    elOri = np.zeros((numEls,9),dtype=float)
    
    esi = 0
    for sec in secList:
        dirCos = get_direction_cosines(sec["xDir"],sec["xyDir"])
        es = shellData["sets"]["element"][esi]
        for ei in es["labels"]:
            eX = list()
            eY = list()
            eZ = list()
            for ndi in elements[ei]:
                if(ndi > -1):
                    eX.append(nodes[ndi,0])
                    eY.append(nodes[ndi,1])
                    eZ.append(nodes[ndi,2])
            if(len(eX) == 3):
                elType = "shell3"
            else:
                elType = "shell4"
            elCrd = np.array([eX,eY,eZ])
            elDirCos = correct_orient(dirCos,elCrd,elType)
            elOri[ei,0:3] = elDirCos[0]
            elOri[ei,3:6] = elDirCos[1]
            elOri[ei,6:9] = elDirCos[2]
        esi = esi + 1
        
    shellData["elementOrientations"] = elOri
    
    ## Get all outer shell and all shear web element sets
    
    outerLab = list()
    swLab = list()
    for es in shellData["sets"]["element"]:
        nm = es["name"]
        if(nm in outShES):
            outerLab.extend(es["labels"])
        elif(nm in swES):
            swLab.extend(es["labels"])
    outerSet = dict()
    outerSet["name"] = "allOuterShellEls"
    outerSet["labels"] = outerLab
    shellData["sets"]["element"].append(outerSet)
    swSet = dict()
    swSet["name"] = "allShearWebEls"
    swSet["labels"] = swLab
    shellData["sets"]["element"].append(swSet)
    
    ## Get root (Zmin) node set
    minZ = np.min(splineZi)
    rootLabs = list()
    lab = 0
    for nd in nodes:
        if(np.abs(nd[2] - minZ) < 0.25*elementSize):
            rootLabs.append(lab)
        lab = lab + 1
    newSet = dict()
    newSet["name"] = "RootNodes"
    newSet["labels"] = rootLabs
    try:
        shellData["sets"]["node"].append(newSet)
    except:
        nodeSets = list()
        nodeSets.append(newSet)
        shellData["sets"]["node"] = nodeSets

    ## Generate adhesive bondlines if requested
    print('getting adhesive mesh')
    if includeAdhesive == 1:
        # TE bondline. Corner-column convention (LP_outer-ish, HP_outer-ish,
        # HP_inner, LP_inner) as in the original inline code: slots 0/3 sit
        # near the TE tip, slots 1/2 sit one chord-segment inboard. The
        # +y outward face (LP face) is between slots 2 and 3; the -y
        # outward face (HP face) between slots 0 and 1.
        shellData = _emit_adhesive_volume(
            blade, shellData,
            splineXi, splineYi, splineZi,
            frstXS, elementSize,
            bond_name="TE_BOND",
            corner_cols=(4, 6, 30, 32),
            target_shell_set_substrs=["LP_TE_REINF", "HP_TE_REINF"],
        )
        # LE bondline. Mirror-image corner pattern about the LE column
        # (col 18 = "le" keypoint): slots 0/3 are one fractional column
        # off the LE on the HP/LP sides (cols 17/19); slots 1/2 are at the
        # HP/LP "a" design keypoints (cols 15/21), one chord-segment
        # inboard. Tie targets include both the bare LE chord-segments
        # (NN_NN_HP_LE / NN_NN_LP_LE) and the adjacent LE_PANEL sets so
        # tip-region LE-bond nodes (the bare LE stack tapers to zero
        # before z=tip) still find a shell face to tie to.
        shellData = _emit_adhesive_volume(
            blade, shellData,
            splineXi, splineYi, splineZi,
            frstXS, elementSize,
            bond_name="LE_BOND",
            corner_cols=(17, 15, 21, 19),
            target_shell_set_substrs=[
                ("LP_LE", "LP_LE_PANEL"),
                ("HP_LE", "HP_LE_PANEL"),
            ],
        )


    matList = list()
    for mn in blade.definition.materials:
        newMat = dict()
        mat = blade.definition.materials[mn]
        newMat['name'] = mat.name
        newMat['density'] = mat.density
        newMat['elastic'] = dict()
        newMat['elastic']['E'] = [mat.ex,mat.ey,mat.ez]
        if(mat.type == "isotropic"):
            nu = mat.prxy
            newMat['elastic']['nu'] = [nu,nu,nu]
        else:
            newMat['elastic']['nu'] = [mat.prxy,mat.prxz,mat.pryz]
        newMat['elastic']['G'] = [mat.gxy,mat.gxz,mat.gyz]
        matList.append(newMat)

    shellData['materials'] = matList

    return shellData



def solidMeshFromShell(blade, shellMesh, layerNumEls, elementSize,
                       n_normal_smoothing_iter=2,
                       layer_thickness_cap_factor=0.7,
                       untangle_max_iter=30):
    """Extrude a shell mesh through-thickness into a 3D solid mesh.

    Combines three mesh-quality treatments standard in industrial
    boundary-layer / sweep meshers (HyperMesh "Bias Style",
    ANSYS Workbench Sweep, Cubit Sweep, Pointwise T-Rex):

    1.  **Laplacian smoothing of per-node normals** before extrusion,
        to reduce normal divergence at sharp-curvature regions.
    2.  **Per-node adaptive layer thickness clamping** to prevent
        knife-edge bricks where the shell quad is locally thin.
    3.  **Post-extrusion untangling** (Knupp-style targeted node-pull)
        for any residual bad-Jacobian bricks.

    On BAR0 at ``elementSize=0.5`` with ``layerNumEls=[1,1,1]`` the
    three stages drive the bad-Jacobian count from **94 → 28 → 8 → 0**
    while moving at most ~22 nodes by an average ~4 mm in the final
    untangle pass.

    Parameters
    ----------
    blade, shellMesh, layerNumEls, elementSize : as before.
    n_normal_smoothing_iter : int, default 2
        Laplacian smoothing passes applied to the per-node averaged
        normal vectors before extrusion. At sharp-curvature regions
        (e.g. the BAR0 trailing-edge flat), adjacent nodes can have
        normals that diverge sharply, producing folded bricks with
        non-positive Jacobian after extrusion. Smoothing averages each
        node's normal with its immediate ring-1 neighbours, which on
        BAR0 at elementSize=0.5 drops the bad-Jacobian count by ~70 %
        (94 -> 28 with k=2). Set to 0 to reproduce legacy un-smoothed
        behaviour.
    layer_thickness_cap_factor : float | None, default 0.7
        At each node, cap the per-layer extrusion offset to
        ``layer_thickness_cap_factor * min(incident shell-edge length)``.
        Prevents producing knife-edge bricks at high-aspect-ratio shell
        regions (TE flat panels: in-plane 0.49 m vs 0.03 m on BAR0).
        Combined with smoothing, drops bad-Jacobian count an extra ~70 %
        (28 -> 8 at default 0.7). Trade-off: at constrained nodes the
        layer is locally thinner than the nominal ply thickness, which
        slightly under-represents through-thickness stiffness. Industry
        sweet spot 0.5-0.7. Set to ``None`` to disable the cap
        (preserves exact ply thicknesses but allows knife-edge bricks).
    untangle_max_iter : int, default 30
        Maximum iterations for the post-extrusion untangling pass
        (see ``mesh_tools.untangle_solid_mesh``). The pass moves
        TOP-face corners of bad-Jacobian bricks toward their BOTTOM-face
        counterparts (through-thickness shrink), with neighbour-
        preservation guard. Set to 0 to disable; the residual bad
        elements then need user-side filtering (as in
        ``examples/write_abaqus_solid_model.py``).
    """
    shNodes = shellMesh["nodes"]
    shElements = shellMesh["elements"]
    elSets = shellMesh["sets"]["element"]
    sectns = shellMesh["sections"]

    print('building solid mesh')
    ## Initialize 3D solid mesh from the shell mesh
    bladeMesh = Mesh3D(shNodes, shElements)
    ## Calculate unit normal vectors for all nodes
    numNds = len(shNodes)
    numShEls = len(shElements)
    nodeNorms = np.zeros((numNds, 3))
    for i in range(0, len(shElements)):
        n1 = shElements[i, 0]
        n2 = shElements[i, 1]
        n3 = shElements[i, 2]
        n4 = shElements[i, 3]
        if n4 == -1:
            v1 = shNodes[n3, :] - shNodes[n1, :]
            v2 = shNodes[n2, :] - shNodes[n1, :]
        else:
            v1 = shNodes[n4, :] - shNodes[n2, :]
            v2 = shNodes[n3, :] - shNodes[n1, :]
        v3x = v1[1] * v2[2] - v1[2] * v2[1]
        v3y = v1[2] * v2[0] - v1[0] * v2[2]
        v3z = v1[0] * v2[1] - v1[1] * v2[0]
        v3 = np.array([v3x, v3y, v3z])
        mag = np.linalg.norm(v3)
        uNorm = (1.0 / mag) * v3
        for j in range(4):
            nj = shElements[i, j]
            if nj != -1:
                nodeNorms[nj, :] = nodeNorms[nj, :] + uNorm

    for i in range(numNds):
        mag = np.linalg.norm(nodeNorms[i])
        nodeNorms[i] = (1.0 / mag) * nodeNorms[i]

    ## Smooth normals across ring-1 neighbours to reduce divergence at
    ## sharp-curvature regions (TE flat, root taper). See docstring.
    if n_normal_smoothing_iter > 0:
        nbr = [set() for _ in range(numNds)]
        for el in shElements:
            valid = [n for n in el if n != -1]
            for a in valid:
                for b in valid:
                    if a != b:
                        nbr[a].add(b)
        nbr = [list(s) for s in nbr]
        for _ in range(n_normal_smoothing_iter):
            new_norms = nodeNorms.copy()
            for i in range(numNds):
                if nbr[i]:
                    acc = nodeNorms[i].copy()
                    for n in nbr[i]:
                        acc = acc + nodeNorms[n]
                    mag = np.linalg.norm(acc)
                    if mag > 0:
                        new_norms[i] = acc / mag
            nodeNorms = new_norms

    ## Per-node max extrusion thickness from min incident shell edge —
    ## prevents knife-edge bricks at thin shell patches. See docstring.
    if layer_thickness_cap_factor is not None:
        node_max_thick = np.full(numNds, np.inf)
        for el in shElements:
            valid = [n for n in el if n != -1]
            coords = shNodes[valid]
            nC = len(valid)
            min_edge = np.inf
            for k in range(nC):
                m = (k + 1) % nC
                edge_len = np.linalg.norm(coords[m] - coords[k])
                if edge_len < min_edge:
                    min_edge = edge_len
            cap = layer_thickness_cap_factor * min_edge
            for n in valid:
                if cap < node_max_thick[n]:
                    node_max_thick[n] = cap
    else:
        node_max_thick = None

    ## Extrude shell mesh into solid mesh
    if len(layerNumEls) == 0:
        layerNumEls = np.array([1, 1, 1])

    prevLayer = shNodes.copy()
    guideNds = list()
    for i in range(0, len(layerNumEls)):
        nodeDist = np.zeros(numNds)
        nodeHitCt = np.zeros(numNds, dtype=int)
        numSec, numStat = blade.stackdb.stacks.shape
        j = 0
        for sec in sectns:
            layerThick = sec["layup"][i][1]
            #layerThick = sectns[j]["layup"][i][1]
            for el in elSets[j]["labels"]:
                for nd in shElements[el]:
                    if nd != -1:
                        nodeDist[nd] = nodeDist[nd] + layerThick
                        nodeHitCt[nd] = nodeHitCt[nd] + 1
            j = j + 1
        newLayer = np.zeros((numNds, 3))
        for j in range(0, numNds):
            if nodeHitCt[j] != 0:
                nodeDist[j] = nodeDist[j] / nodeHitCt[j]
                if node_max_thick is not None and nodeDist[j] > node_max_thick[j]:
                    nodeDist[j] = node_max_thick[j]
                newLayer[j] = prevLayer[j] + nodeDist[j] * nodeNorms[j]

        guideNds.append(newLayer)
        prevLayer = newLayer.copy()

    solidMesh = bladeMesh.createSweptMesh(
        sweepMethod="toDestNodes",
        sweepElements=layerNumEls,
        destNodes=guideNds,
        interpMethod="linear",
    )

    print('getting element sets')
    ## Construct the element set list, extrapolated from the shell model
    newSetList = list()
    newSectList = list()
    esi = 0
    for sec in sectns:
        es = elSets[esi]
        elArray = np.array(es["labels"])
        elLayer = 0
        li = 1
        for lne in layerNumEls:
            newSet = dict()
            newSet["name"] = es["name"] + "layer_" + str(li)
            newLabels = list()
            for i in range(0, lne):
                newLabels.extend(elArray + numShEls * elLayer)
                elLayer = elLayer + 1
            newSet["labels"] = newLabels
            newSetList.append(newSet)
            newSec = dict()
            newSec["type"] = "solid"
            newSec["elementSet"] = newSet["name"]
            newSec["material"] = sec["layup"][li - 1][0]
            newSec["xDir"] = sec["xDir"]
            newSec["xyDir"] = sec["xyDir"]
            newSectList.append(newSec)
            li = li + 1
        esi = esi + 1

    solidMesh["sets"] = dict()
    solidMesh["sets"]["element"] = newSetList
    solidMesh["sections"] = newSectList

    solidMesh["adhesiveNds"] = shellMesh["adhesiveNds"]
    solidMesh["adhesiveEls"] = shellMesh["adhesiveEls"]
    solidMesh["adhesiveElSet"] = shellMesh["adhesiveElSet"]
    
    numBdEls = len(solidMesh["elements"])
    totElLrs = sum(layerNumEls)
    
    ## Get orientations for all elements
    elOri = np.zeros((numBdEls,9),dtype=float)
    for li in range(0,totElLrs):
        shft = li*numShEls
        oi = 0
        for ori in shellMesh["elementOrientations"]:
            elOri[oi+shft] = ori
            oi = oi + 1
    solidMesh["elementOrientations"] = elOri
    
    ## Get extruded node and element sets
    extSets = get_extruded_sets(shellMesh, totElLrs)
    solidMesh["sets"]["node"] = extSets["node"]
    solidMesh["sets"]["element"].extend(extSets["element"])
    
    ## Get section node sets
    solidMesh = get_matching_node_sets(solidMesh)
    
    ## Get shear web edge set
    
    numBdNds = len(solidMesh["nodes"])
    ndElCt = np.zeros(numBdNds,dtype=int)
    for el in solidMesh["elements"]:
        for nd in el:
            if(nd > -1):
                ndElCt[nd] = ndElCt[nd] + 1
    
    edgeLab = list()
    for ns in solidMesh["sets"]["node"]:
        if(ns["name"] == "allShearWebEls"):
            for nd in ns["labels"]:
                if(ndElCt[nd] <= 2):
                    edgeLab.append(nd)
    
    newSet = dict()
    newSet["name"] = "shearWebEdges"
    newSet["labels"] = edgeLab
    
    solidMesh["sets"]["node"].append(newSet)
    
    print('getting shear web constraints.  This can take a few minutes...')
    ## Get tie constraints for shear webs
    
    constraints = shellMesh["constraints"]
    
    webConst = tie_2_sets_constraints(solidMesh, "shearWebEdges", "allOuterShellEls", 0.05*elementSize)
    
    constraints.extend(webConst)
    
    #print('getting adhesive constraints')
    ## Get tie constraints for adhesive
    
    # adMesh = dict()
    # adMesh["nodes"] = solidMesh["adhesiveNds"]
    # adMesh["elements"] = solidMesh["adhesiveEls"]
    # adConst = tie_2_meshes_constraints(adMesh,solidMesh,0.015*ndSp)
    
    #constraints.extend(adConst)
    
    solidMesh["constraints"] = constraints

    ## Post-extrusion untangling — Knupp-style targeted node pull on
    ## any residual bad-Jacobian bricks left after smoothing + clamp.
    if untangle_max_iter > 0:
        from pynumad.mesh_gen.mesh_tools import (
            check_all_jacobians, untangle_solid_mesh,
        )
        pre_bad = len(check_all_jacobians(solidMesh["nodes"], solidMesh["elements"]))
        if pre_bad > 0:
            new_nodes, n_rem, n_it = untangle_solid_mesh(
                solidMesh["nodes"], solidMesh["elements"],
                max_iter=untangle_max_iter,
            )
            print(f'untangle: {pre_bad} bad -> {n_rem} bad in {n_it} iter')
            solidMesh["nodes"] = new_nodes

    return solidMesh


def get_solid_mesh(blade, layerNumEls, elementSize):
    ## Edit stacks to be usable for 3D solid mesh
    blade.stackdb.edit_stacks_for_solid_mesh()
    ## Create shell mesh as seed
    ## Note the new output structure of shellMeshGeneral, as a single python dictionary  -E Anderson
    shellMesh = shell_mesh_general(blade, 1, 1, elementSize)
    print("finished shell mesh")
    solidMesh = solidMeshFromShell(blade, shellMesh, layerNumEls, elementSize)
    return solidMesh


def get_shell_mesh(blade, includeAdhesive, elementSize):
    meshData = shell_mesh_general(blade, 0, includeAdhesive, elementSize)
    return meshData

def get_root_mesh(axisPts, radius, thickness, elementSize, elLayers, config='inserts', boltRad=None, insertThk=None, adhesiveThk=None, numIns=None, tubeThk=None, tubeExtend=None, extNumEls=None):
    if(config == 'inserts'):
        ## Create insert cross section mesh
        thInc = 2.0*np.pi/numIns
        minR = radius[0] - thickness[0] ## inner radius of root
        midR = 0.5*(2.0*radius[0] - thickness[0]) ## mid-thickness radius of root
        
        bd = Boundary2D()
        x1 = (midR - boltRad)*np.cos(0.5*thInc)
        y1 = (midR - boltRad)*np.sin(0.5*thInc)
        x2 = (midR + boltRad)*np.cos(0.5*thInc)
        y2 = (midR + boltRad)*np.sin(0.5*thInc)
        kp = [[x1,y1],
              [x2,y2],
              [x1,y1]]
        nEls = int(np.ceil(2.0*np.pi*boltRad/elementSize))
        bd.addSegment('arc',kp,nEls)
        bdData = bd.getBoundaryMesh()
        
        mesher = Mesh2D(bdData['nodes'],bdData['elements'])
        boltMesh = mesher.createUnstructuredMesh('quad')
        boltMesh = make3D(boltMesh)
        # plotShellMesh(insMesh)
        
        ## Create insert cross section mesh
        
        insertRad = boltRad + insertThk
        x1 = (midR - insertRad)*np.cos(0.5*thInc)
        y1 = (midR - insertRad)*np.sin(0.5*thInc)
        x2 = (midR + insertRad)*np.cos(0.5*thInc)
        y2 = (midR + insertRad)*np.sin(0.5*thInc)
        kp = [[x1,y1],
              [x2,y2],
              [x1,y1]]
        bd.addSegment('arc',kp,nEls)
        bdData = bd.getBoundaryMesh()
        
        mesher = Mesh2D(bdData['nodes'],bdData['elements'])
        insMesh = mesher.createUnstructuredMesh('quad')
        insMesh = make3D(insMesh)
        # plotShellMesh(adhMesh)
        
        ## Create adhesive cross section mesh
        
        
        bd = Boundary2D()
        x1 = (midR - insertRad)*np.cos(0.5*thInc)
        y1 = (midR - insertRad)*np.sin(0.5*thInc)
        x2 = (midR + insertRad)*np.cos(0.5*thInc)
        y2 = (midR + insertRad)*np.sin(0.5*thInc)
        kp = [[x1,y1],
              [x2,y2],
              [x1,y1]]
        bd.addSegment('arc',kp,nEls)
        
        adhesiveRad = boltRad + insertThk + adhesiveThk
        x1 = (midR - adhesiveRad)*np.cos(0.5*thInc)
        y1 = (midR - adhesiveRad)*np.sin(0.5*thInc)
        x2 = (midR + adhesiveRad)*np.cos(0.5*thInc)
        y2 = (midR + adhesiveRad)*np.sin(0.5*thInc)
        kp = [[x1,y1],
              [x2,y2],
              [x1,y1]]
        bd.addSegment('arc',kp,nEls)
        bdData = bd.getBoundaryMesh()
        
        mesher = Mesh2D(bdData['nodes'],bdData['elements'])
        adhMesh = mesher.createUnstructuredMesh('quad')
        adhMesh = make3D(adhMesh)

        ## Create root fill material unit cell mesh
        
        bd = Boundary2D()
        x1 = (midR - adhesiveRad)*np.cos(0.5*thInc)
        y1 = (midR - adhesiveRad)*np.sin(0.5*thInc)
        x2 = (midR + adhesiveRad)*np.cos(0.5*thInc)
        y2 = (midR + adhesiveRad)*np.sin(0.5*thInc)
        kp = [[x1,y1],
              [x2,y2],
              [x1,y1]]
        bd.addSegment('arc',kp,nEls)
        
        kp = [[minR,0.],
              [radius[0],0.]]
        nEls = int(np.ceil(thickness[0]/elementSize))
        bd.addSegment('line',kp,nEls)
        
        x1 = radius[0]
        y1 = 0.
        x2 = radius[0]*np.cos(0.5*thInc)
        y2 = radius[0]*np.sin(0.5*thInc)
        x3 = radius[0]*np.cos(thInc)
        y3 = radius[0]*np.sin(thInc)
        kp = [[x1,y1],
              [x2,y2],
              [x3,y3]]
        nEls = int(np.ceil(radius[0]*thInc/elementSize))
        bd.addSegment('arc',kp,nEls)
        
        x1 = radius[0]*np.cos(thInc)
        y1 = radius[0]*np.sin(thInc)
        x2 = minR*np.cos(thInc)
        y2 = minR*np.sin(thInc)
        kp = [[x1,y1],
              [x2,y2]]
        nEls = int(np.ceil(thickness[0]/elementSize))
        bd.addSegment('line',kp,nEls)
        
        x1 = minR
        y1 = 0.
        x2 = minR*np.cos(0.5*thInc)
        y2 = minR*np.sin(0.5*thInc)
        x3 = minR*np.cos(thInc)
        y3 = minR*np.sin(thInc)
        kp = [[x1,y1],
              [x2,y2],
              [x3,y3]]
        nEls = int(np.ceil(radius[0]*thInc/elementSize))
        bd.addSegment('arc',kp,nEls)
        
        bdData = bd.getBoundaryMesh()
        
        mesher = Mesh2D(bdData['nodes'],bdData['elements'])
        fillMesh = mesher.createUnstructuredMesh('quad')
        fillMesh = make3D(fillMesh)
        # plotShellMesh(fillMesh)
        
        ## Create mesh of whole root cross section
        faceSurf = Surface()
        
        boltSets = list()
        insSets = list()
        adhSets = list()
        fillSets = list()
        for i in range(0,numIns):
            rotAng = thInc*i*180.0/np.pi
            newBolt = cp.deepcopy(boltMesh)
            newBolt = rotate_mesh(newBolt,[0.,0.,0.],[0.,0.,1.],rotAng)
            nm = 'bolt_' + str(i)
            faceSurf.addMesh(newBolt,name=nm)
            boltSets.append(nm)
            newIns = cp.deepcopy(insMesh)
            newIns = rotate_mesh(newIns,[0.,0.,0.],[0.,0.,1.],rotAng)
            nm = 'insert_' + str(i)
            faceSurf.addMesh(newIns,name=nm)
            insSets.append(nm)
            newAd = cp.deepcopy(adhMesh)
            newAd = rotate_mesh(newAd,[0.,0.,0.],[0.,0.,1.],rotAng)
            nm = 'adhesive_' + str(i)
            faceSurf.addMesh(newAd,name=nm)
            adhSets.append(nm)
            newFill = cp.deepcopy(fillMesh)
            newFill = rotate_mesh(newFill,[0.,0.,0.],[0.,0.,1.],rotAng)
            nm = 'fill_' + str(i)
            faceSurf.addMesh(newFill,name=nm)
            fillSets.append(nm)
            
        faceMesh = faceSurf.getSurfaceMesh()
        # plotShellMesh(faceMesh)
        
        tVec = [0.,0.,axisPts[0]]
        faceMesh = translate_mesh(faceMesh,tVec)
        
        nPts = len(axisPts)
        allZpts = np.linspace(axisPts[0],axisPts[nPts-1],elLayers+1)
        radFun = interpolate.interp1d(axisPts,radius,kind='linear')
        thkFun = interpolate.interp1d(axisPts,thickness,kind='linear')
        ptRad = radFun(allZpts)
        ptThk = thkFun(allZpts)
        
        rootMesher = Mesh3D(faceMesh['nodes'],faceMesh['elements'])
        
        gdNds = list()
        nEls = list()
        for i in range(1,len(allZpts)):
            newNds = faceMesh['nodes'].copy()
            thkFact = ptThk[i]/ptThk[0]
            shftDist = ptRad[i] - ptRad[0]*thkFact
            for j, nd in enumerate(newNds):
                mag = np.linalg.norm(nd[0:2])
                unitXY = (1.0/mag)*nd[0:2]
                newXY = thkFact*nd[0:2] + shftDist*unitXY
                newNds[j,0:2] = newXY
                newNds[j,2] = allZpts[i]
            gdNds.append(newNds)
            nEls.append(1)
            
        rootMesh = rootMesher.createSweptMesh('toDestNodes',nEls,destNodes=gdNds,interpMethod='linear')
        
        # rootMesher = Mesh3D(faceMesh['nodes'],faceMesh['elements'])
        # sD = axisPts[nPts-1] - axisPts[0]
        # rootMesh = rootMesher.createSweptMesh('inDirection',elLayers,sweepDistance=sD,axis=[0.,0.,1.])
        
        # plotSolidMesh(rootMesh)
        
        extSets = get_extruded_sets(faceMesh,elLayers)
        
        rootMesh['sets'] = extSets
        
        rootMesh = get_element_set_union(rootMesh,boltSets,'allBolts')
        rootMesh = get_element_set_union(rootMesh,insSets,'allInsert')
        rootMesh = get_element_set_union(rootMesh,adhSets,'allAdhesive')
        rootMesh = get_element_set_union(rootMesh,fillSets,'allFill')
        
        rootMesh = get_matching_node_sets(rootMesh)
        
        sections = list()
        
        newSec = dict()
        newSec['type'] = 'solid'
        newSec['elementSet'] = 'allBolts'
        newSec['material'] = 'boltMat'
        newSec['xDir'] = [0.,0.,1.]
        newSec['xyDir'] = [1.,0.,0.]
        sections.append(newSec)
        
        newSec = dict()
        newSec['type'] = 'solid'
        newSec['elementSet'] = 'allInsert'
        newSec['material'] = 'insertMat'
        newSec['xDir'] = [0.,0.,1.]
        newSec['xyDir'] = [1.,0.,0.]
        sections.append(newSec)
        
        newSec = dict()
        newSec['type'] = 'solid'
        newSec['elementSet'] = 'allAdhesive'
        newSec['material'] = 'adhesiveMat'
        newSec['xDir'] = [0.,0.,1.]
        newSec['xyDir'] = [1.,0.,0.]
        sections.append(newSec)
        
        newSec = dict()
        newSec['type'] = 'solid'
        newSec['elementSet'] = 'allFill'
        newSec['material'] = 'fillMat'
        newSec['xDir'] = [0.,0.,1.]
        newSec['xyDir'] = [1.,0.,0.]
        sections.append(newSec)
        
        rootMesh['sections'] = sections
        
        return rootMesh
        
    else:
        ## Create main root
        circEls = int(2.0*np.pi*radius[0]/elementSize)
        minR = radius[0] - thickness[0]
        hFillThk = 0.5*(thickness[0] - 2.0*adhesiveThk - tubeThk)
        faceSurf = Surface()
        
        ## Inner fill
        print('Inner fill')
        
        bd = Boundary2D()
        xi = 0.0
        
        y1 = -minR
        y2 = minR
        kp = [[xi,y1],
              [xi,y2],
              [xi,y1]]
        bd.addSegment('arc',kp,circEls)
        
        # y1 = -minR - hFillThk
        # y2 = minR + hFillThk
        # kp = [[xi,y1],
              # [xi,y2],
              # [xi,y1]]
        # bd.addSegment('arc',kp,circEls)
        
        bdData = bd.getBoundaryMesh()
        mesher = Mesh2D(bdData['nodes'],bdData['elements'])
        #layerMesh = mesher.createUnstructuredMesh('quad')
        sEls = int(np.ceil(hFillThk/elementSize))
        layerMesh = mesher.createSweptMesh('fromPoint',sEls,sweepDistance=hFillThk,point=[0.,0.])
        layerMesh = make3D(layerMesh)
        faceSurf.addMesh(layerMesh,name='innerFill')
        
        ## Inner adhesive
        print('Inner adhesive')
        
        bd = Boundary2D()
        xi = 0.0
        
        y1 = -minR - hFillThk
        y2 = minR + hFillThk
        kp = [[xi,y1],
              [xi,y2],
              [xi,y1]]
        bd.addSegment('arc',kp,circEls)
        
        # y1 = -minR - hFillThk - adhesiveThk
        # y2 = minR + hFillThk + adhesiveThk
        # kp = [[xi,y1],
              # [xi,y2],
              # [xi,y1]]
        # bd.addSegment('arc',kp,circEls)
        
        bdData = bd.getBoundaryMesh()
        mesher = Mesh2D(bdData['nodes'],bdData['elements'])
        #layerMesh = mesher.createUnstructuredMesh('quad')
        sEls = int(np.ceil(adhesiveThk/elementSize))
        layerMesh = mesher.createSweptMesh('fromPoint',sEls,sweepDistance=adhesiveThk,point=[0.,0.])
        layerMesh = make3D(layerMesh)
        faceSurf.addMesh(layerMesh,name='innerAdhesive')
        
        ## Tube
        print('Tube')
        
        bd = Boundary2D()
        xi = 0.0
        
        y1 = -minR - hFillThk - adhesiveThk
        y2 = minR + hFillThk + adhesiveThk
        kp = [[xi,y1],
              [xi,y2],
              [xi,y1]]
        bd.addSegment('arc',kp,circEls)
        
        # y1 = -minR - hFillThk - adhesiveThk - tubeThk
        # y2 = minR + hFillThk + adhesiveThk + tubeThk
        # kp = [[xi,y1],
              # [xi,y2],
              # [xi,y1]]
        # bd.addSegment('arc',kp,circEls)
        
        bdData = bd.getBoundaryMesh()
        mesher = Mesh2D(bdData['nodes'],bdData['elements'])
        #layerMesh = mesher.createUnstructuredMesh('quad')
        sEls = int(np.ceil(tubeThk/elementSize))
        layerMesh = mesher.createSweptMesh('fromPoint',sEls,sweepDistance=tubeThk,point=[0.,0.])
        layerMesh = make3D(layerMesh)
        faceSurf.addMesh(layerMesh,name='tube')
        
        ## Outer adhesive
        print('Outer adhesive')
        
        bd = Boundary2D()
        xi = 0.0
        
        y1 = -radius[0] + hFillThk + adhesiveThk
        y2 = -y1
        kp = [[xi,y1],
              [xi,y2],
              [xi,y1]]
        bd.addSegment('arc',kp,circEls)
        
        # y1 = -radius[0] + hFillThk
        # y2 = -y1
        # kp = [[xi,y1],
              # [xi,y2],
              # [xi,y1]]
        # bd.addSegment('arc',kp,circEls)
        
        bdData = bd.getBoundaryMesh()
        mesher = Mesh2D(bdData['nodes'],bdData['elements'])
        #layerMesh = mesher.createUnstructuredMesh('quad')
        sEls = int(np.ceil(adhesiveThk/elementSize))
        layerMesh = mesher.createSweptMesh('fromPoint',sEls,sweepDistance=adhesiveThk,point=[0.,0.])
        layerMesh = make3D(layerMesh)
        faceSurf.addMesh(layerMesh,name='outerAdhesive')
        
        ## Outer fill
        print('Outer fill')
        
        bd = Boundary2D()
        xi = 0.0
        
        y1 = -radius[0] + hFillThk
        y2 = -y1
        kp = [[xi,y1],
              [xi,y2],
              [xi,y1]]
        bd.addSegment('arc',kp,circEls)
        
        # y1 = -radius[0]
        # y2 = -y1
        # kp = [[xi,y1],
              # [xi,y2],
              # [xi,y1]]
        # bd.addSegment('arc',kp,circEls)
        
        bdData = bd.getBoundaryMesh()
        mesher = Mesh2D(bdData['nodes'],bdData['elements'])
        #layerMesh = mesher.createUnstructuredMesh('quad')
        sEls = int(np.ceil(hFillThk/elementSize))
        layerMesh = mesher.createSweptMesh('fromPoint',sEls,sweepDistance=hFillThk,point=[0.,0.])
        layerMesh = make3D(layerMesh)
        faceSurf.addMesh(layerMesh,name='outerFill')
        
        ## Build 3D mesh
        faceMesh = faceSurf.getSurfaceMesh()
        
        tVec = [0.,0.,axisPts[0]]
        faceMesh = translate_mesh(faceMesh,tVec)
        
        nPts = len(axisPts)
        allZpts = np.linspace(axisPts[0],axisPts[nPts-1],elLayers+1)
        radFun = interpolate.interp1d(axisPts,radius,kind='linear')
        thkFun = interpolate.interp1d(axisPts,thickness,kind='linear')
        ptRad = radFun(allZpts)
        ptThk = thkFun(allZpts)
        
        rootMesher = Mesh3D(faceMesh['nodes'],faceMesh['elements'])
        
        gdNds = list()
        nEls = list()
        for i in range(1,len(allZpts)):
            newNds = faceMesh['nodes'].copy()
            thkFact = ptThk[i]/ptThk[0]
            shftDist = ptRad[i] - ptRad[0]*thkFact
            for j, nd in enumerate(newNds):
                mag = np.linalg.norm(nd[0:2])
                unitXY = (1.0/mag)*nd[0:2]
                newXY = thkFact*nd[0:2] + shftDist*unitXY
                newNds[j,0:2] = newXY
                newNds[j,2] = allZpts[i]
            gdNds.append(newNds)
            nEls.append(1)
            
        rootMesh = rootMesher.createSweptMesh('toDestNodes',nEls,destNodes=gdNds,interpMethod='linear')
        
        extSets = get_extruded_sets(faceMesh,len(nEls))
        
        rootMesh['sets'] = extSets
        
        ## Build tube extension
        print('extension')
        
        faceSurf = Surface()
        bd = Boundary2D()
        xi = 0.0
        
        y1 = -minR - hFillThk - adhesiveThk
        y2 = minR + hFillThk + adhesiveThk
        kp = [[xi,y1],
              [xi,y2],
              [xi,y1]]
        bd.addSegment('arc',kp,circEls)
        
        # y1 = -minR - hFillThk - adhesiveThk - tubeThk
        # y2 = minR + hFillThk + adhesiveThk + tubeThk
        # kp = [[xi,y1],
              # [xi,y2],
              # [xi,y1]]
        # bd.addSegment('arc',kp,circEls)
        
        bdData = bd.getBoundaryMesh()
        mesher = Mesh2D(bdData['nodes'],bdData['elements'])
        #layerMesh = mesher.createUnstructuredMesh('quad')
        sEls = int(np.ceil(tubeThk/elementSize))
        layerMesh = mesher.createSweptMesh('fromPoint',sEls,sweepDistance=tubeThk,point=[0.,0.])
        layerMesh = make3D(layerMesh)
        faceSurf.addMesh(layerMesh,name='tubeExtension')
        
        faceMesh = faceSurf.getSurfaceMesh()
        
        tVec = [0.,0.,axisPts[0]]
        faceMesh = translate_mesh(faceMesh,tVec)
        
        extMesher = Mesh3D(layerMesh['nodes'],layerMesh['elements'])
        extMesh = extMesher.createSweptMesh('inDirection',extNumEls,sweepDistance=tubeExtend,axis=[0.,0.,-1.])
        
        extSets = get_extruded_sets(faceMesh,extNumEls)
        
        extMesh['sets'] = extSets
        
        fullMesh = merge_meshes(rootMesh,extMesh)
        
        sections = list()
        
        newSec = dict()
        newSec['type'] = 'solid'
        newSec['elementSet'] = 'innerFill'
        newSec['material'] = 'fillMat'
        newSec['xDir'] = [0.,0.,1.]
        newSec['xyDir'] = [1.,0.,0.]
        sections.append(newSec)
        
        newSec = dict()
        newSec['type'] = 'solid'
        newSec['elementSet'] = 'innerAdhesive'
        newSec['material'] = 'adhesiveMat'
        newSec['xDir'] = [0.,0.,1.]
        newSec['xyDir'] = [1.,0.,0.]
        sections.append(newSec)
        
        newSec = dict()
        newSec['type'] = 'solid'
        newSec['elementSet'] = 'tube'
        newSec['material'] = 'tubeMat'
        newSec['xDir'] = [0.,0.,1.]
        newSec['xyDir'] = [1.,0.,0.]
        sections.append(newSec)
        
        newSec = dict()
        newSec['type'] = 'solid'
        newSec['elementSet'] = 'outerAdhesive'
        newSec['material'] = 'adhesiveMat'
        newSec['xDir'] = [0.,0.,1.]
        newSec['xyDir'] = [1.,0.,0.]
        sections.append(newSec)
        
        newSec = dict()
        newSec['type'] = 'solid'
        newSec['elementSet'] = 'outerFill'
        newSec['material'] = 'fillMat'
        newSec['xDir'] = [0.,0.,1.]
        newSec['xyDir'] = [1.,0.,0.]
        sections.append(newSec)
        
        newSec = dict()
        newSec['type'] = 'solid'
        newSec['elementSet'] = 'tubeExtension'
        newSec['material'] = 'tubeMat'
        newSec['xDir'] = [0.,0.,1.]
        newSec['xyDir'] = [1.,0.,0.]
        sections.append(newSec)
        
        fullMesh['sections'] = sections
        
        return fullMesh