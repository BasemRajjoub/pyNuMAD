"""APDL POST1 emitters for full-field extraction (paper-2 field-PCE).

Builds the APDL command block that dumps every FE-result field listed in
:mod:`pynumad.analysis.field_io.schema` to plain text files. The writer
(:mod:`pynumad.analysis.field_io.writer`) reads those text files back and
assembles a single HDF5 container.

Why text dumps and not PyMAPDL / .rst parsing
---------------------------------------------
The existing convergence pipeline already uses ``*VWRITE`` for scalar QoIs
and works reliably. PyMAPDL .rst readers add a non-trivial dep with
version-sensitive struct layouts. The extra cost of going through text is
absorbed by the writer step on the same node.

Element numbering assumption
----------------------------
pyNuMAD's ANSYS deck writer numbers elements sequentially as 1..N_elem
(see ``analysis/ansys/write.py`` ``EN`` writes). The extractor uses
``*VFILL,RAMP`` to generate element IDs, which gives correct IDs only
under that contiguous-numbering assumption. If a future deck writer
deletes/renumbers elements, this needs to switch to ``*VGET,...,ELEM,n,
ELIST`` or read the ESEL list.

Output files (in ``out_dir``)
-----------------------------
* ``field_L<NN>_<SURF>.txt`` — one per (ply, surface) tuple, rows per
  element, columns: ``elem, s11, s22, s33, s12, s13, s23, eps11, eps22,
  gamma12, gamma13, gamma23, svm`` (13 columns, all SI).
* ``disp.txt`` — one row per node, columns: ``node, ux, uy, uz, rotx,
  roty, rotz`` (7 columns).
* ``sene.txt`` — one row per element, columns: ``elem, sene`` (2
  columns; J).
* ``react.txt`` — one row, columns ``FX, FY, FZ, MX, MY, MZ`` (6
  columns; N, N·m).
* ``elem_geom.txt`` — one row per element, columns: ``elem, area,
  volume`` (3 columns).
* ``modal_freq.txt`` — one row per mode, columns: ``mode, freq_Hz,
  eff_mass_x, eff_mass_y, eff_mass_z``.
* ``mode_shape_M<NN>.txt`` — one per mode, rows per node, columns:
  ``node, ux, uy, uz, rotx, roty, rotz``.

All numeric columns are formatted with ``E20.12`` for round-trip
fidelity; element / node IDs use ``F12.0``.
"""
from __future__ import annotations

from pynumad.analysis.field_io.schema import N_SURF, SURFACES


# *VWRITE format helpers ------------------------------------------------

# All numeric fields use scientific notation with 12 significant digits.
_NUM_FMT = "E20.12"
_ID_FMT = "F12.0"


def _ascii_dir(out_dir: str) -> str:
    """Normalise ``out_dir`` to a trailing-slash POSIX path for APDL.

    APDL is sensitive to path separators on Windows but our cluster is
    Linux-only — keep it simple and assume POSIX. Strip any trailing
    slash to avoid double-slashes when we concatenate filenames.
    """
    return out_dir.rstrip("/")


def emit_static_fields(out_dir: str, n_ply_max: int) -> str:
    """Emit per-element static result fields.

    Writes one CSV-like text file per (ply, surface) tuple holding the
    six stress, five strain, and von Mises components in the material
    (laminate) coordinate system, plus ``disp``, ``sene``, ``react``,
    and ``elem_geom`` files.

    The block assumes that ``SET,LAST`` has been called and that the
    last STATIC step is active. Caller is responsible for ordering this
    block before any subsequent ``ANTYPE,MODAL`` step.
    """
    od = _ascii_dir(out_dir)
    parts: list[str] = []
    parts.append(_HEADER_STATIC)

    parts.append(_emit_alloc_arrays())
    parts.append(_emit_elem_geom_block(od))

    for ply in range(1, n_ply_max + 1):
        for surf_idx, surf in enumerate(SURFACES, start=1):
            parts.append(_emit_ply_surface_block(od, ply, surf))

    parts.append(_emit_disp_block(od))
    parts.append(_emit_sene_block(od))
    parts.append(_emit_react_block(od))
    parts.append(_FOOTER_STATIC)

    return "".join(parts)


def emit_modal_fields(out_dir: str, n_modes: int) -> str:
    """Emit modal results from a modal solve.

    Requires that a modal solution has been run (``ANTYPE,MODAL`` +
    ``SOLVE``) so that ``SET,1,imode`` cycles through modes. Writes
    ``modal_freq.txt`` and ``mode_shape_M<NN>.txt`` per mode.
    """
    od = _ascii_dir(out_dir)
    parts: list[str] = [_HEADER_MODAL]
    parts.append(_emit_alloc_nodal_arrays())
    parts.append(_emit_modal_freq_block(od, n_modes))
    for imode in range(1, n_modes + 1):
        parts.append(_emit_mode_shape_block(od, imode))
    parts.append(_FOOTER_MODAL)
    return "".join(parts)


# Internal block emitters ------------------------------------------------

_HEADER_STATIC = """
! ============================================================
! field_io: per-element static field extraction
!   schema: pynumad.analysis.field_io.schema
!   parser: pynumad.analysis.field_io.writer
! Block is self-contained: enters /POST1, calls SET,LAST, leaves
! with FINISH. Must be spliced AFTER the static SOLVE and BEFORE
! any other analysis step that would invalidate the static result.
! Elements / nodes are assumed to be numbered 1..N contiguously
! (matches the pyNuMAD deck writer).
!
! CRITICAL: RSYS,LSYS activates the LAYER coordinate system so that
! S,X / S,Y / S,Z / S,XY / S,XZ / S,YZ retrieved via ETABLE map to
! the laminate-frame (along-fiber, transverse, through-thickness)
! components needed for composite failure criteria. Without LSYS,
! ANSYS returns stresses in the element CS, which is not aligned
! with the fibre direction once plies are oriented at angle.
! Verified against IEA-22 spar cap: with RSYS,LSYS, sigma11/E1 ~ ply
! axial strain (along fibre); without it, the components are
! mis-rotated and sigma_z picks up the dominant span-direction stress.
! ============================================================
/POST1
SET,LAST
RSYS,LSYS
ALLSEL
*GET,n_elem_,ELEM,0,COUNT
*GET,n_node_,NODE,0,COUNT
"""

_FOOTER_STATIC = """
ALLSEL
FINISH
! ---- end field_io: static field extraction ----
"""

_HEADER_MODAL = """
! ============================================================
! field_io: modal field extraction
! Self-contained: enters /POST1, leaves with FINISH. Splice
! AFTER the modal SOLVE has completed and the modes are expanded
! (MXPAND with mode-shape writeout enabled).
! ============================================================
/POST1
SET,LIST
"""

_FOOTER_MODAL = """
ALLSEL
FINISH
! ---- end field_io: modal field extraction ----
"""


def _emit_alloc_arrays() -> str:
    """Allocate scratch arrays for per-element and per-node columns.

    Reused across (ply, surface) blocks — *VGET re-fills them each
    iteration. APDL arrays are statically sized so we allocate once.
    """
    return """
! Allocate scratch arrays (one element-sized column per component).
*DEL,_eid,,NOPR
*DEL,_nid,,NOPR
*DEL,_s11,,NOPR
*DEL,_s22,,NOPR
*DEL,_s33,,NOPR
*DEL,_s12,,NOPR
*DEL,_s13,,NOPR
*DEL,_s23,,NOPR
*DEL,_e11,,NOPR
*DEL,_e22,,NOPR
*DEL,_g12,,NOPR
*DEL,_g13,,NOPR
*DEL,_g23,,NOPR
*DEL,_svm,,NOPR
*DEL,_ar_,,NOPR
*DEL,_vl_,,NOPR
*DEL,_sn_,,NOPR
*DEL,_ux,,NOPR
*DEL,_uy,,NOPR
*DEL,_uz,,NOPR
*DEL,_rx,,NOPR
*DEL,_ry,,NOPR
*DEL,_rz,,NOPR
*DIM,_eid,ARRAY,n_elem_
*DIM,_s11,ARRAY,n_elem_
*DIM,_s22,ARRAY,n_elem_
*DIM,_s33,ARRAY,n_elem_
*DIM,_s12,ARRAY,n_elem_
*DIM,_s13,ARRAY,n_elem_
*DIM,_s23,ARRAY,n_elem_
*DIM,_e11,ARRAY,n_elem_
*DIM,_e22,ARRAY,n_elem_
*DIM,_g12,ARRAY,n_elem_
*DIM,_g13,ARRAY,n_elem_
*DIM,_g23,ARRAY,n_elem_
*DIM,_svm,ARRAY,n_elem_
*DIM,_ar_,ARRAY,n_elem_
*DIM,_vl_,ARRAY,n_elem_
*DIM,_sn_,ARRAY,n_elem_
*VFILL,_eid(1),RAMP,1,1
*DIM,_nid,ARRAY,n_node_
*DIM,_ux,ARRAY,n_node_
*DIM,_uy,ARRAY,n_node_
*DIM,_uz,ARRAY,n_node_
*DIM,_rx,ARRAY,n_node_
*DIM,_ry,ARRAY,n_node_
*DIM,_rz,ARRAY,n_node_
*VFILL,_nid(1),RAMP,1,1
"""


def _emit_alloc_nodal_arrays() -> str:
    """Allocate per-node scratch arrays for the modal field block.

    The static-block allocator already creates these, but the modal
    block runs in a fresh /POST1 invocation. APDL parameters do
    persist across FINISH/POST1 boundaries, but re-allocating is safe
    (the ``*DEL,...,NOPR`` lines are idempotent) and makes the modal
    block usable standalone when no static-field extraction precedes
    it.
    """
    return """
! Allocate per-node scratch arrays (idempotent; static block also
! allocates these, modal re-allocates so it can stand alone).
ALLSEL
*GET,n_node_,NODE,0,COUNT
*DEL,_nid,,NOPR
*DEL,_ux,,NOPR
*DEL,_uy,,NOPR
*DEL,_uz,,NOPR
*DEL,_rx,,NOPR
*DEL,_ry,,NOPR
*DEL,_rz,,NOPR
*DIM,_nid,ARRAY,n_node_
*DIM,_ux,ARRAY,n_node_
*DIM,_uy,ARRAY,n_node_
*DIM,_uz,ARRAY,n_node_
*DIM,_rx,ARRAY,n_node_
*DIM,_ry,ARRAY,n_node_
*DIM,_rz,ARRAY,n_node_
*VFILL,_nid(1),RAMP,1,1
"""


def _emit_elem_geom_block(out_dir: str) -> str:
    return f"""
! ---- element geometry (area, volume) ----
ALLSEL
ETABLE,erase
ETABLE,_ar_,VOLU
! Layer-1 mid-surface volume == area * layer1_thickness; layer-0 VOLU is
! the *total* element volume (sum of all layers). Use that for elem_volume.
SHELL,MID
LAYER,0
ETABLE,_vl_,VOLU
*VGET,_ar_(1),ELEM,1,ETAB,_AR_
*VGET,_vl_(1),ELEM,1,ETAB,_VL_
*CFOPEN,{out_dir}/elem_geom,txt
*VWRITE,_eid(1),_ar_(1),_vl_(1)
({_ID_FMT},2(', ',{_NUM_FMT}))
*CFCLOSE
"""


def _emit_ply_surface_block(out_dir: str, ply: int, surface: str) -> str:
    """Emit the ETABLE+VGET+VWRITE block for one (ply, surface) pair."""
    surf = surface.upper()
    # Surface short tags for filenames (2-char to keep stable widths).
    surf_tag = {"TOP": "TOP", "MID": "MID", "BOT": "BOT"}[surf]
    fname = f"field_L{ply:02d}_{surf_tag}"
    return f"""
! ---- field block: layer={ply} surface={surf_tag} ----
ALLSEL
LAYER,{ply}
SHELL,{surf}
! RSYS,LSYS reasserted per (ply, surface) block in case any earlier
! command perturbed it. With LAYER active, LSYS picks up the layer
! orientation angle automatically.
RSYS,LSYS
ETABLE,erase
ETABLE,s11_,S,X
ETABLE,s22_,S,Y
ETABLE,s33_,S,Z
ETABLE,s12_,S,XY
ETABLE,s13_,S,XZ
ETABLE,s23_,S,YZ
ETABLE,e11_,EPEL,X
ETABLE,e22_,EPEL,Y
ETABLE,e33_,EPEL,Z
ETABLE,g12_,EPEL,XY
ETABLE,g13_,EPEL,XZ
ETABLE,g23_,EPEL,YZ
ETABLE,svm_,S,EQV
*VGET,_s11(1),ELEM,1,ETAB,S11_
*VGET,_s22(1),ELEM,1,ETAB,S22_
*VGET,_s33(1),ELEM,1,ETAB,S33_
*VGET,_s12(1),ELEM,1,ETAB,S12_
*VGET,_s13(1),ELEM,1,ETAB,S13_
*VGET,_s23(1),ELEM,1,ETAB,S23_
*VGET,_e11(1),ELEM,1,ETAB,E11_
*VGET,_e22(1),ELEM,1,ETAB,E22_
*VGET,_g12(1),ELEM,1,ETAB,G12_
*VGET,_g13(1),ELEM,1,ETAB,G13_
*VGET,_g23(1),ELEM,1,ETAB,G23_
*VGET,_svm(1),ELEM,1,ETAB,SVM_
*CFOPEN,{out_dir}/{fname},txt
*VWRITE,_eid(1),_s11(1),_s22(1),_s33(1),_s12(1),_s13(1),_s23(1),_e11(1),_e22(1),_g12(1),_g13(1),_g23(1),_svm(1)
({_ID_FMT},12(', ',{_NUM_FMT}))
*CFCLOSE
"""


def _emit_disp_block(out_dir: str) -> str:
    """Per-node displacement and rotation dump.

    ``*VGET,...,NODE,n,U,X`` walks the node table starting at ``n``
    without honouring selection (verified in earlier APDL debug cycle).
    We therefore call ALLSEL and assume node numbering is contiguous
    1..N_node, matching the pyNuMAD deck writer.
    """
    return f"""
! ---- nodal displacement + rotation ----
ALLSEL
*VGET,_ux(1),NODE,1,U,X
*VGET,_uy(1),NODE,1,U,Y
*VGET,_uz(1),NODE,1,U,Z
*VGET,_rx(1),NODE,1,ROT,X
*VGET,_ry(1),NODE,1,ROT,Y
*VGET,_rz(1),NODE,1,ROT,Z
*CFOPEN,{out_dir}/disp,txt
*VWRITE,_nid(1),_ux(1),_uy(1),_uz(1),_rx(1),_ry(1),_rz(1)
({_ID_FMT},6(', ',{_NUM_FMT}))
*CFCLOSE
"""


def _emit_sene_block(out_dir: str) -> str:
    return f"""
! ---- element strain energy ----
ALLSEL
ETABLE,erase
ETABLE,_sn_,SENE
*VGET,_sn_(1),ELEM,1,ETAB,_SN_
*CFOPEN,{out_dir}/sene,txt
*VWRITE,_eid(1),_sn_(1)
({_ID_FMT},', ',{_NUM_FMT})
*CFCLOSE
"""


def _emit_react_block(out_dir: str) -> str:
    """Root reactions via FSUM on the clamped root node-set.

    The runner selects the root nodes (z ~ 0) before SOLVE and stores
    the component CM_ROOT. If CM_ROOT is missing, FSUM defaults to
    ALLSEL and is meaningless — we assert on the Python side that the
    component exists. Reactions are summed in the *global* CSYS so
    they match HAWC2's input convention.
    """
    return f"""
! ---- root reactions (sum of nodal reaction forces / moments) ----
! CMSEL with type NODE; FSUM on selected nodes returns the algebraic
! sum of forces (= reactions, since these nodes are clamped).
! Do NOT call NSLE,S afterwards - NSLE selects nodes attached to
! selected ELEMENTS and would empty the selection.
CMSEL,S,CM_ROOT,NODE
FSUM
*GET,_rfx,FSUM,0,ITEM,FX
*GET,_rfy,FSUM,0,ITEM,FY
*GET,_rfz,FSUM,0,ITEM,FZ
*GET,_rmx,FSUM,0,ITEM,MX
*GET,_rmy,FSUM,0,ITEM,MY
*GET,_rmz,FSUM,0,ITEM,MZ
ALLSEL
*CFOPEN,{out_dir}/react,txt
*VWRITE,_rfx,_rfy,_rfz,_rmx,_rmy,_rmz
({_NUM_FMT},5(', ',{_NUM_FMT}))
*CFCLOSE
"""


def _emit_modal_freq_block(out_dir: str, n_modes: int) -> str:
    """Frequencies and effective modal mass (per direction) for the
    first ``n_modes`` modes from the current modal solution."""
    parts = [f"""
! ---- modal frequencies + effective mass ----
*DEL,_mid,,NOPR
*DEL,_fhz,,NOPR
*DEL,_emx,,NOPR
*DEL,_emy,,NOPR
*DEL,_emz,,NOPR
*DIM,_mid,ARRAY,{n_modes}
*DIM,_fhz,ARRAY,{n_modes}
*DIM,_emx,ARRAY,{n_modes}
*DIM,_emy,ARRAY,{n_modes}
*DIM,_emz,ARRAY,{n_modes}
*VFILL,_mid(1),RAMP,1,1
"""]
    for imode in range(1, n_modes + 1):
        parts.append(f"""
SET,1,{imode}
*GET,_fhz({imode}),ACTIVE,,SET,FREQ
*GET,_emx({imode}),MODE,{imode},EFFM,X
*GET,_emy({imode}),MODE,{imode},EFFM,Y
*GET,_emz({imode}),MODE,{imode},EFFM,Z
""")
    parts.append(f"""
*CFOPEN,{out_dir}/modal_freq,txt
*VWRITE,_mid(1),_fhz(1),_emx(1),_emy(1),_emz(1)
({_ID_FMT},4(', ',{_NUM_FMT}))
*CFCLOSE
""")
    return "".join(parts)


def _emit_mode_shape_block(out_dir: str, imode: int) -> str:
    fname = f"mode_shape_M{imode:02d}"
    return f"""
! ---- mode shape: mode {imode} ----
SET,1,{imode}
ALLSEL
*VGET,_ux(1),NODE,1,U,X
*VGET,_uy(1),NODE,1,U,Y
*VGET,_uz(1),NODE,1,U,Z
*VGET,_rx(1),NODE,1,ROT,X
*VGET,_ry(1),NODE,1,ROT,Y
*VGET,_rz(1),NODE,1,ROT,Z
*CFOPEN,{out_dir}/{fname},txt
*VWRITE,_nid(1),_ux(1),_uy(1),_uz(1),_rx(1),_ry(1),_rz(1)
({_ID_FMT},6(', ',{_NUM_FMT}))
*CFCLOSE
"""
