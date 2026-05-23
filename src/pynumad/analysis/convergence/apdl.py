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
    """APDL block: max ‖u‖ over nodes in ``z_band_m``."""
    z_lo, z_hi = spec.z_band_m
    return f"""
! ---- tip deflection block: {spec.name} ----
ALLSEL
NSEL,S,LOC,Z,{z_lo:.6f},{z_hi:.6f}
*GET,n_tip_nd,NODE,0,COUNT
*IF,n_tip_nd,GT,0,THEN
  NSORT,U,SUM,1,0,0
  *GET,tip_umax,SORT,,MAX
  *VWRITE,'{spec.name}',tip_umax
('tip,',A,',umax_m,',E16.8)
*ENDIF
ALLSEL
"""


def emit_patch(spec: PatchSpec, matching_components: Sequence[str]) -> str:
    """APDL block: area-weighted σ_vM over a patch.

    ``matching_components`` must already be filtered to the components
    that match ``spec.element_set_substr`` — use
    :func:`filter_element_sets` to build it from the mesh's set list.

    Patch selection:

    1. Select union of every named component in ``matching_components``.
    2. Restrict by z-centroid band ``spec.z_band_m`` (intersection).

    Per (layer, surface):

    3. ETABLE the element area (``VOLU`` on a shell == area·thickness;
       we use the per-layer area via the SECTION info instead — see
       comment below).
    4. ETABLE σ_vM (``S,EQV``).
    5. SUM(σ_vM · A), SUM(A) → area-weighted mean.

    Note on element area
    ~~~~~~~~~~~~~~~~~~~~
    For SHELL181, ``ETABLE,area,VOLU`` reports the **volume** of the
    element (area × thickness). That's still a valid weight for an
    area-weighted MEAN as long as we use the same quantity for both
    numerator and denominator — the per-layer thickness cancels. So
    using VOLU as the weight is mathematically correct here.
    """
    z_lo, z_hi = spec.z_band_m
    blocks = [f"\n! ---- patch block: {spec.name} ----\n"]

    if not matching_components:
        blocks.append(f"! WARNING: no components match {spec.element_set_substr!r}\n")
        return "".join(blocks)

    blocks.append("ALLSEL\nESEL,NONE\n")
    for cname in matching_components:
        blocks.append(f"CMSEL,A,{cname}\n")
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
  ETABLE,svw_,*,svm_,vol_
  SSUM
  *GET,sum_sw_,SSUM,0,ITEM,SVW_
  *GET,sum_v_,SSUM,0,ITEM,VOL_
  *IF,sum_v_,GT,0,THEN
    avg_svm_ = sum_sw_ / sum_v_
  *ELSE
    avg_svm_ = 0
  *ENDIF
  *VWRITE,'{spec.name}','L{layer}_{surf}_svm_Pa',avg_svm_
('patch,',A,',',A,',',E16.8)
  *VWRITE,'{spec.name}','L{layer}_{surf}_volu_m3',sum_v_
('patch,',A,',',A,',',E16.8)
  *VWRITE,'{spec.name}','L{layer}_{surf}_n_elem',n_patch
('patch,',A,',',A,',',F12.0)
*ENDIF
"""


def emit_section(spec: SectionSpec) -> str:
    """APDL block: integrated force + moment at a spanwise cut.

    Strategy
    ~~~~~~~~
    Select the outboard segment (elements with centroid z ≥ z_m), shift
    the working-plane origin to ``(0, 0, z_m)``, call ``FSUM`` which
    sums element nodal-force contributions about the WP origin. The
    six scalars (Fx, Fy, Fz, Mx, My, Mz) are then written as separate
    CSV rows for parsing simplicity.
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
*VWRITE,'{spec.name}','Fx_N',sec_fx_
('section,',A,',',A,',',E16.8)
*VWRITE,'{spec.name}','Fy_N',sec_fy_
('section,',A,',',A,',',E16.8)
*VWRITE,'{spec.name}','Fz_N',sec_fz_
('section,',A,',',A,',',E16.8)
*VWRITE,'{spec.name}','Mx_Nm',sec_mx_
('section,',A,',',A,',',E16.8)
*VWRITE,'{spec.name}','My_Nm',sec_my_
('section,',A,',',A,',',E16.8)
*VWRITE,'{spec.name}','Mz_Nm',sec_mz_
('section,',A,',',A,',',E16.8)
WPCSYS,-1,0
ALLSEL
"""


def emit_post1(
    spec: ConvergenceSpec,
    mesh_set_names: Sequence[str],
) -> str:
    """Build the full POST1 block for the entire ConvergenceSpec.

    Parameters
    ----------
    spec
        The QoI bundle.
    mesh_set_names
        All element-set names present in the pyNuMAD mesh dict (e.g.
        ``[s["name"] for s in mesh["sets"]["element"]]``). Used to
        expand each patch's ``element_set_substr`` into concrete
        ``CMSEL,A,<name>`` calls.
    """
    out = [emit_csv_open(spec.csv_path)]
    if spec.tip_deflection is not None:
        out.append(emit_tip_deflection(spec.tip_deflection))
    for ps in spec.patches:
        matching = filter_element_sets(mesh_set_names, ps.element_set_substr)
        out.append(emit_patch(ps, matching))
    for ss in spec.sections:
        out.append(emit_section(ss))
    out.append(emit_csv_close())
    return "".join(out)
