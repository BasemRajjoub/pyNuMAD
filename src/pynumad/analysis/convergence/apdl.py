"""APDL POST1 command emission for the convergence-study QoIs.

Each emitter is a pure function: takes a spec, returns an APDL string.
No file IO, no FE state — easy to unit-test with regex / string checks.

The block layout for a single ANSYS run is:

    SET,LAST                       (load the last load step / static)
    *CFOPEN, ...                   (open the CSV output)
    -- tip deflection block --
    -- per-patch blocks --
    -- per-section blocks --
    *CFCLOSE

Every emitter writes exactly one row per scalar QoI to the CSV with a
fixed schema: ``"category,name,key,value\\n"``. The parser side
recognises this schema; do not change it without updating
``parse.py`` and the tests.

Element-set substring filtering happens in PYTHON before deck emission
— callers pass the list of matching component names to
:func:`emit_patch`, not a regex/substring. APDL's ``STRSUB`` is
finicky across versions and we don't trust it for production work.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from pynumad.analysis.convergence.specs import (
    ConvergenceSpec,
    PatchSpec,
    SectionSpec,
    TipDeflectionSpec,
)


CSV_HEADER = "category,name,key,value"


def filter_element_sets(set_names: Iterable[str], substring: str) -> list[str]:
    """Return every name in ``set_names`` that contains ``substring``.

    Order-preserving and case-sensitive. Caller is responsible for
    making sure each returned name is a valid ANSYS component name (we
    don't validate length / chars here).
    """
    return [n for n in set_names if substring in n]


def emit_csv_open(csv_path: str) -> str:
    """Open the output CSV and write the header row."""
    return (
        f"*CFOPEN,{csv_path}\n"
        f"*VWRITE,\n('{CSV_HEADER}')\n"
    )


def emit_csv_close() -> str:
    """Close the output CSV opened by :func:`emit_csv_open`."""
    return "*CFCLOSE\n"


def emit_tip_deflection(spec: TipDeflectionSpec) -> str:
    """APDL block: deflection statistics over a tip-side z-band.

    Writes THREE rows per tip block:

    * ``umax_m``  — max ‖u‖ across the band (single-node, noisy in
      shells because local panel deformation can spike one node).
    * ``umean_m`` — arithmetic mean of ‖u‖ across the band (much more
      stable convergence metric; matches the IEA-22 design report's
      "tip deflection envelope average").
    * ``n_nodes`` — node count, for sanity.

    For paper-2 convergence we report ``umean_m`` as the primary
    industry-standard tip-deflection QoI. ``umax_m`` is kept for
    backward-compatibility with anyone reading the older convention.

    Note on string literals in *VWRITE
    ----------------------------------
    APDL truncates character data arguments to *VWRITE at 8 characters
    (an internal limit on character parameters). Names longer than 8
    chars (``HP_SPAR_r20``, ``L3_BOT_svm_Pa``) silently get clipped.
    To work around this we bake the name and key INTO THE FORMAT STRING
    literal — Fortran-style format literals have no length limit. The
    only variable in *VWRITE is now the numerical value.
    """
    z_lo, z_hi = spec.z_band_m
    return f"""
! ---- tip deflection block: {spec.name} ----
ALLSEL
NSEL,S,LOC,Z,{z_lo:.6f},{z_hi:.6f}
*GET,n_tip_nd,NODE,0,COUNT
*IF,n_tip_nd,GT,0,THEN
  NSORT,U,SUM,1,0,0
  *GET,tip_umax,SORT,,MAX
*ENDIF
! Component-wise mean displacements via ETABLE+SSUM on tip-band
! elements (CENT,Z selection).  ETABLE,U,SUM (the "vector magnitude"
! item) returns NEGATIVE values on this mesh — likely algebraic sum
! of components rather than magnitude as the docs claim.  Per-component
! ETABLE,U,X/Y/Z is reliable, well-documented and gives signed mean
! displacements per axis.  Magnitude is then computed in Python from
! the three component means by parse-side post-processing.
ALLSEL
ESEL,S,CENT,Z,{z_lo:.6f},{z_hi:.6f}
*GET,n_tip_el,ELEM,0,COUNT
*IF,n_tip_el,GT,0,THEN
  ETABLE,erase
  ETABLE,ux_,U,X
  ETABLE,uy_,U,Y
  ETABLE,uz_,U,Z
  SSUM
  *GET,sum_ux_,SSUM,0,ITEM,UX_
  *GET,sum_uy_,SSUM,0,ITEM,UY_
  *GET,sum_uz_,SSUM,0,ITEM,UZ_
  _tip_umx = sum_ux_ / n_tip_el
  _tip_umy = sum_uy_ / n_tip_el
  _tip_umz = sum_uz_ / n_tip_el
  *VWRITE,tip_umax
('tip,{spec.name},umax_m,',E16.8)
  *VWRITE,_tip_umx
('tip,{spec.name},umean_x_m,',E16.8)
  *VWRITE,_tip_umy
('tip,{spec.name},umean_y_m,',E16.8)
  *VWRITE,_tip_umz
('tip,{spec.name},umean_z_m,',E16.8)
  *VWRITE,n_tip_el
('tip,{spec.name},n_elements,',F12.0)
  *VWRITE,n_tip_nd
('tip,{spec.name},n_nodes,',F12.0)
*ENDIF
! Single-node tip-most QoI for mesh-independence comparison. Pick the
! node with the maximum z coordinate in the model and report its U
! components.  This is the conventional "tip-LE deflection" used by
! IEC 61400-1 acceptance tests — mesh-independent in the limit because
! every refinement still has a single tip-most node at z = max(z).
ALLSEL
*GET,z_max_,NODE,0,MXLOC,Z
NSEL,S,LOC,Z,z_max_ - 0.001,z_max_ + 0.001
*GET,n_at_max_,NODE,0,COUNT
*IF,n_at_max_,GT,0,THEN
  *GET,tip_node_,NODE,0,NUM,MIN
  *GET,tip_ux_,NODE,tip_node_,U,X
  *GET,tip_uy_,NODE,tip_node_,U,Y
  *GET,tip_uz_,NODE,tip_node_,U,Z
  tip_umagnode_ = sqrt(tip_ux_*tip_ux_+tip_uy_*tip_uy_+tip_uz_*tip_uz_)
  *VWRITE,tip_ux_
('tip,{spec.name},upoint_x_m,',E16.8)
  *VWRITE,tip_uy_
('tip,{spec.name},upoint_y_m,',E16.8)
  *VWRITE,tip_uz_
('tip,{spec.name},upoint_z_m,',E16.8)
  *VWRITE,tip_umagnode_
('tip,{spec.name},upoint_mag_m,',E16.8)
  *VWRITE,z_max_
('tip,{spec.name},upoint_z_loc_m,',E16.8)
*ENDIF
ALLSEL
"""


def emit_patch(
    spec: PatchSpec,
    element_ranges: Sequence[tuple[int, int]],
) -> str:
    """APDL block: area-weighted σ_vM over a patch.

    ``element_ranges`` is a list of ``(e_min, e_max)`` 1-indexed element
    ID intervals (one per matching pyNuMAD element-set). The caller
    builds this by:

    1. Filtering set names by ``spec.element_set_substr`` via
       :func:`filter_element_sets`.
    2. Looking up each matching set's element-label range
       (``min(labels)+1``, ``max(labels)+1``) since pyNuMAD's labels
       are 0-indexed and ANSYS deck IDs are 1-indexed.

    pyNuMAD's ANSYS deck writer does NOT emit one ANSYS component per
    pyNuMAD element-set — it just writes the element list inline. So
    we cannot use ``CMSEL,A,<name>``; we use ``ESEL,A,ELEM,,e_min,e_max``
    instead. That choice is the difference between this version of the
    function and an earlier prototype that tried CMSEL and failed at
    run-time with "Component is not defined".

    Patch selection:

    1. Union of every range in ``element_ranges`` (``ESEL,A,...``).
    2. Restrict by z-centroid band ``spec.z_band_m`` (``ESEL,R,CENT,Z,...``).

    Per (layer, surface):

    3. ETABLE the element volume (= area × layer thickness; the
       thickness factor cancels in the area-weighted mean).
    4. ETABLE σ_vM (``S,EQV``).
    5. ``ETABLE,svw_,*,svm_,vol_`` → element-wise product.
    6. ``SSUM`` then ``avg = sum(svw_) / sum(vol_)`` → area-weighted mean.
    """
    z_lo, z_hi = spec.z_band_m
    blocks = [f"\n! ---- patch block: {spec.name} ----\n"]

    if not element_ranges:
        blocks.append(f"! WARNING: no element ranges for {spec.element_set_substr!r}\n")
        return "".join(blocks)

    blocks.append("ALLSEL\nESEL,NONE\n")
    for e_min, e_max in element_ranges:
        blocks.append(f"ESEL,A,ELEM,,{e_min},{e_max}\n")
    blocks.append(f"ESEL,R,CENT,Z,{z_lo:.6f},{z_hi:.6f}\n")

    for layer in spec.layers:
        for surf in spec.surfaces:
            blocks.append(_emit_patch_layer_surface(spec, layer, surf))

    blocks.append("ALLSEL\n")
    return "".join(blocks)


def _emit_patch_layer_surface(
    spec: PatchSpec, layer: int, surface: str,
) -> str:
    surf = surface.upper()
    return f"""
*GET,n_patch,ELEM,0,COUNT
*IF,n_patch,GT,0,THEN
  LAYER,{layer}
  SHELL,{surf}
  ETABLE,erase
  ETABLE,vol_,VOLU,
  ETABLE,svm_,S,EQV
  ! Element-wise svw_ = svm_ * vol_ (ETABLE math requires SMULT, not
  ! the ETABLE,*,... syntax; that one's only for "store result-item
  ! component" and rejects '*' as the item).
  SMULT,svw_,svm_,vol_,1,1
  SSUM
  *GET,sum_sw_,SSUM,0,ITEM,SVW_
  *GET,sum_v_,SSUM,0,ITEM,VOL_
  *IF,sum_v_,GT,0,THEN
    avg_svm_ = sum_sw_ / sum_v_
  *ELSE
    avg_svm_ = 0
  *ENDIF
  ! Names baked into the format string — APDL truncates character data
  ! to 8 chars but format-string literals are unlimited (see
  ! emit_tip_deflection docstring).
  *VWRITE,avg_svm_
('patch,{spec.name},L{layer}_{surf}_svm_Pa,',E16.8)
  *VWRITE,sum_v_
('patch,{spec.name},L{layer}_{surf}_volu_m3,',E16.8)
  *VWRITE,n_patch
('patch,{spec.name},L{layer}_{surf}_n_elem,',F12.0)
*ENDIF
"""


def emit_section(spec: SectionSpec) -> str:
    """APDL block: integrated force + moment at a spanwise cut.

    .. warning::

       The ANSYS-side ``FSUM`` on a selected element set returns
       essentially zero in a converged static problem because Newton's
       3rd law cancels internal contributions at interior nodes. We
       found that the hard way on smoke tests (My ≈ 4 N·m at section
       r=20 m where the analytical answer is ~58 MN·m). The block
       below is preserved only for compatibility with anyone who
       wants to *see* that ANSYS-side FSUM = 0 — it's not used as a
       convergence metric.

       The correct section-moment computation for a static FE with
       prescribed nodal forces is to sum the applied loads outboard
       of the cut, which by equilibrium equals the internal section
       force. That computation lives in
       :mod:`pynumad.analysis.convergence.forces_src` and runs in
       pure Python on the ``forces.src`` file the runner writes for
       ANSYS — exact by construction, no FE error.

    Convergence metric for the loading pipeline becomes "do the ANSYS
    root reactions match the applied root resultant at every h?", a
    direct equilibrium check available from the existing root_reaction
    extraction in run_static_v2.py. See ``validate_root_reactions.py``.
    """
    return f"""
! ---- section cut block: {spec.name} at z={spec.z_m} m ----
ALLSEL
ESEL,S,CENT,Z,{spec.z_m:.6f},1.0e10
WPCSYS,-1,0
WPOFFS,0,0,{spec.z_m:.6f}
FSUM,RSYS,0,WP
*GET,sec_fx_,FSUM,0,ITEM,FX
*GET,sec_fy_,FSUM,0,ITEM,FY
*GET,sec_fz_,FSUM,0,ITEM,FZ
*GET,sec_mx_,FSUM,0,ITEM,MX
*GET,sec_my_,FSUM,0,ITEM,MY
*GET,sec_mz_,FSUM,0,ITEM,MZ
! Names baked into format string (APDL 8-char string-data limit).
*VWRITE,sec_fx_
('section,{spec.name},Fx_N,',E16.8)
*VWRITE,sec_fy_
('section,{spec.name},Fy_N,',E16.8)
*VWRITE,sec_fz_
('section,{spec.name},Fz_N,',E16.8)
*VWRITE,sec_mx_
('section,{spec.name},Mx_Nm,',E16.8)
*VWRITE,sec_my_
('section,{spec.name},My_Nm,',E16.8)
*VWRITE,sec_mz_
('section,{spec.name},Mz_Nm,',E16.8)
WPCSYS,-1,0
ALLSEL
"""


def patch_element_ranges_from_mesh(
    mesh: dict, substring: str,
) -> list[tuple[int, int]]:
    """For every element-set in ``mesh`` whose name contains ``substring``,
    return its ANSYS 1-indexed ``(e_min, e_max)`` range.

    pyNuMAD's mesh-dict labels are 0-indexed (the ANSYS deck writer
    adds ``+1`` when emitting EMODIF / ESEL — see
    ``analysis/ansys/write.py:1272``). We add ``+1`` here so the
    returned range can be passed straight to ``ESEL,A,ELEM,,e_min,e_max``.

    Empty sets (no labels) are silently skipped.
    """
    el_sets = mesh.get("sets", {}).get("element", []) or []
    out: list[tuple[int, int]] = []
    for s in el_sets:
        name = s.get("name", "")
        if substring not in name:
            continue
        labels = s.get("labels", [])
        if not labels:
            continue
        lo = int(min(labels)) + 1
        hi = int(max(labels)) + 1
        out.append((lo, hi))
    return out


def emit_post1(
    spec: ConvergenceSpec,
    mesh: dict,
) -> str:
    """Build the full POST1 block for the entire ConvergenceSpec.

    Parameters
    ----------
    spec
        The QoI bundle.
    mesh
        pyNuMAD mesh dict — output of ``shell_mesh_general``. We read
        ``mesh["sets"]["element"]`` to look up the (e_min, e_max)
        element-ID range for each patch's matching element-sets.
    """
    out = [emit_csv_open(spec.csv_path)]
    if spec.tip_deflection is not None:
        out.append(emit_tip_deflection(spec.tip_deflection))
    for ps in spec.patches:
        ranges = patch_element_ranges_from_mesh(mesh, ps.element_set_substr)
        out.append(emit_patch(ps, ranges))
    for ss in spec.sections:
        out.append(emit_section(ss))
    out.append(emit_csv_close())
    return "".join(out)
