"""Dataclasses that specify what to extract from an ANSYS run for a
mesh-convergence study.

There are three QoI families, each with its own dataclass:

* :class:`TipDeflectionSpec` — the universal global-stiffness convergence
  metric. ``max ‖u‖`` over nodes in a tip-side band. Always reported in
  every wind-blade FE convergence paper.

* :class:`PatchSpec` — an **area-weighted** mean of σ_vM (or σ_x, ε, …)
  over a fixed physical region of the blade. This is the **stress
  convergence metric**: spar caps, TE reinforcements, root buildup. The
  patch is defined by a chord-region tag (e.g. ``HP_SPAR``) plus a fixed
  spanwise band (e.g. ``z ∈ [18, 22]`` m) so the same physical region is
  sampled at every mesh size — adjacent elements may differ between
  refinements, but the patch boundary doesn't.

* :class:`SectionSpec` — internal force/moment at a fixed spanwise cut.
  Computed by summing element nodal-force contributions from the
  outboard half of the blade. In linear FE this is preserved by
  equilibrium so it's primarily a **validation check** (does the
  internal moment match the applied load?), not a convergence metric —
  but every paper reports it because reviewers expect it.

All three are pure data — no APDL strings, no FE state. The
:mod:`pynumad.analysis.convergence.apdl` module turns them into
ANSYS POST1 blocks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class TipDeflectionSpec:
    """Maximum total displacement magnitude over a tip-side z-band.

    Parameters
    ----------
    name : str
        Identifier used in CSV output, e.g. ``"tip"``.
    z_band_m : tuple[float, float]
        ``(z_lo, z_hi)`` — node selection range in metres along the
        blade-z axis. Typical for IEA-22: ``(135, 140)`` (last 5 m
        of a 138 m blade).
    """
    name: str
    z_band_m: tuple[float, float]


@dataclass(frozen=True)
class PatchSpec:
    """Area-weighted average over a fixed physical region.

    The patch is defined as the intersection of

    1. **element-set substring** ``element_set_substr`` — e.g.
       ``"HP_SPAR"`` matches every pyNuMAD-emitted element-set name
       containing that token (``03_05_HP_SPAR``, ``03_06_HP_SPAR``, …).
    2. **spanwise band** ``z_band_m`` — only elements with centroid
       z in that range count.

    The patch boundary is **independent of element size**: as h refines,
    the same physical region keeps the same z_band but contains more
    elements. Area-weighting (`Σ σ·A / Σ A`) means small and large
    elements contribute proportionally, removing the bias the
    arithmetic mean has when refinement adds many small elements near a
    singularity.

    Parameters
    ----------
    name : str
        Identifier used in CSV output, e.g. ``"HP_SPAR_r20"``.
    element_set_substr : str
        Substring matched against pyNuMAD-emitted element-set names to
        pick the chord region (typically a pyNuMAD chord-group tag like
        ``"HP_SPAR"``, ``"HP_TE_REINF"``, ``"LP_SPAR"``).
    z_band_m : tuple[float, float]
        ``(z_lo, z_hi)`` band along the blade-z axis in metres.
    layers : tuple[int, ...]
        SHELL layer indices to extract from (1-indexed, ANSYS
        convention). For composite spars use the load-bearing layers
        only — typically ``(3,)`` for the carbon-uniax core of the spar
        cap stack on IEA-22.
    surfaces : tuple[str, ...]
        Per-layer SHELL surfaces to extract, each ``"TOP"`` or
        ``"BOT"``. Default ``("TOP", "BOT")`` reports both.
    """
    name: str
    element_set_substr: str
    z_band_m: tuple[float, float]
    layers: tuple[int, ...] = (3,)
    surfaces: tuple[str, ...] = ("TOP", "BOT")


@dataclass(frozen=True)
class SectionSpec:
    """Internal force/moment at a spanwise section cut.

    Computed in ANSYS POST1 via element nodal-force summation on the
    outboard half of the blade about the section centroid. By
    equilibrium in linear FE this equals the moment of the external
    loads applied outboard of the section; convergence is therefore
    "trivial" (it's the same value at every h), but reviewers expect
    the comparison against the HAWC2 input to be shown.

    Parameters
    ----------
    name : str
        Identifier used in CSV output, e.g. ``"section_r20"``.
    z_m : float
        Spanwise z location of the section cut, in metres.
    band_half_width_m : float
        Half-width of the band used to identify the cutting plane
        (the nodes on the cut). Default ``0.05`` m = pick nodes within
        ±50 mm of ``z_m``.
    """
    name: str
    z_m: float
    band_half_width_m: float = 0.05


@dataclass(frozen=True)
class ConvergenceSpec:
    """Bundle of all QoIs to extract from a single ANSYS run.

    Pass this to :func:`pynumad.analysis.convergence.apdl.emit_post1`
    to get the APDL block, and to
    :func:`pynumad.analysis.convergence.parse.parse_results` to read
    the CSV back into a Python dict.
    """
    tip_deflection: TipDeflectionSpec | None = None
    patches: tuple[PatchSpec, ...] = field(default_factory=tuple)
    sections: tuple[SectionSpec, ...] = field(default_factory=tuple)
    csv_path: str = "qoi.csv"


# ---------------------------------------------------------------------------
# Convenience factories for the canonical paper-2 / IEA-22 QoI set.
# ---------------------------------------------------------------------------


def iea22_default_spec() -> ConvergenceSpec:
    """The canonical QoI set we use for the IEA-22 paper-2 convergence:

    * tip-deflection over last 5 m of span
    * 15 area-weighted patches: HP/LP_SPAR @ r=20/35/50 m,
      HP/LP_TE_REINF @ r=20/35/50 m, SW @ r=20/35/50 m
    * 5 section cuts at r = 20, 35, 50, 70, 90 m
    """
    patches: list[PatchSpec] = []
    for prefix in ("HP_SPAR", "LP_SPAR", "HP_TE_REINF", "LP_TE_REINF"):
        for r in (20.0, 35.0, 50.0):
            patches.append(PatchSpec(
                name=f"{prefix}_r{int(r)}",
                element_set_substr=prefix,
                z_band_m=(r - 2.0, r + 2.0),
                layers=(3,),
            ))
    for r in (20.0, 35.0, 50.0):
        patches.append(PatchSpec(
            name=f"SW_r{int(r)}",
            element_set_substr="SW",
            z_band_m=(r - 2.0, r + 2.0),
            layers=(1,),  # web core layer
        ))
    sections = tuple(
        SectionSpec(name=f"section_r{int(r)}", z_m=r)
        for r in (20.0, 35.0, 50.0, 70.0, 90.0)
    )
    return ConvergenceSpec(
        tip_deflection=TipDeflectionSpec(name="tip", z_band_m=(135.0, 140.0)),
        patches=tuple(patches),
        sections=sections,
        csv_path="qoi_convergence.csv",
    )
