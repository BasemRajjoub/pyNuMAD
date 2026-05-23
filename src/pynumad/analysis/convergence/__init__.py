"""Mesh-convergence study helpers for pyNuMAD-driven ANSYS analyses.

A small, focused subpackage that emits the POST1 commands needed to
extract the three QoIs typical of wind-blade FE convergence studies
and parses the results back into Python:

* ``tip_deflection`` — global stiffness-convergence metric.
* ``patches``        — area-weighted σ_vM at fixed physical regions
  (spar caps, TE reinforcements, shear webs).
* ``sections``       — internal force + moment at spanwise cuts
  (validation against the applied loading by equilibrium).

The package is split into four files so each piece can be unit-tested
without bringing the others along:

* :mod:`pynumad.analysis.convergence.specs`   — pure dataclasses.
* :mod:`pynumad.analysis.convergence.apdl`    — APDL string emitters.
* :mod:`pynumad.analysis.convergence.parse`   — CSV parser.
* :mod:`pynumad.analysis.convergence.analyze` — drift-and-verdict tables.

Typical usage in a runner script::

    from pynumad.analysis.convergence import (
        ConvergenceSpec, PatchSpec, SectionSpec, TipDeflectionSpec,
        emit_post1, parse_results, analyse, print_drift_table,
        iea22_default_spec,
    )

    # In the deck-writer (per ANSYS run):
    spec = iea22_default_spec()
    set_names = [s["name"] for s in mesh["sets"]["element"]]
    apdl_block = emit_post1(spec, set_names)
    deck.write(apdl_block)

    # After all sweep runs:
    results_by_h = {h: parse_results(out_dir(h) / "qoi_convergence.csv")
                    for h in element_sizes}
    rows = analyse(results_by_h)
    print_drift_table(rows)
"""
from __future__ import annotations

from pynumad.analysis.convergence.analyze import (
    DriftRow,
    analyse,
    print_drift_table,
)
from pynumad.analysis.convergence.apdl import (
    emit_csv_close,
    emit_csv_open,
    emit_patch,
    emit_post1,
    emit_section,
    emit_tip_deflection,
    filter_element_sets,
    patch_element_ranges_from_mesh,
)
from pynumad.analysis.convergence.forces_src import (
    read_forces_src,
    section_resultants,
    section_resultants_at,
)
from pynumad.analysis.convergence.parse import (
    ConvergenceResult,
    parse_results,
)
from pynumad.analysis.convergence.specs import (
    ConvergenceSpec,
    PatchSpec,
    SectionSpec,
    TipDeflectionSpec,
    iea22_default_spec,
)

__all__ = [
    # specs
    "ConvergenceSpec",
    "PatchSpec",
    "SectionSpec",
    "TipDeflectionSpec",
    "iea22_default_spec",
    # apdl
    "emit_csv_open",
    "emit_csv_close",
    "emit_tip_deflection",
    "emit_patch",
    "emit_section",
    "emit_post1",
    "filter_element_sets",
    "patch_element_ranges_from_mesh",
    # parse
    "ConvergenceResult",
    "parse_results",
    # forces_src
    "read_forces_src",
    "section_resultants",
    "section_resultants_at",
    # analyze
    "DriftRow",
    "analyse",
    "print_drift_table",
]
