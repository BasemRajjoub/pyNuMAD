"""Per-sample field-extraction + HDF5 container for paper-2 field PCE.

A sibling subpackage to :mod:`pynumad.analysis.convergence`. Where the
convergence pipeline emits **scalar** QoIs (patch averages, section
moments, tip deflection), this one emits **fields** — every per-element,
per-ply, per-surface stress/strain plus nodal displacements, modal
shapes, and the supporting mesh + materials metadata — into a single
HDF5 container per DoE sample.

Module map
----------

* :mod:`.schema` — single source of truth for HDF5 paths, shapes,
  dtypes, units.
* :mod:`.apdl_extract` — APDL ``*VWRITE`` emitter (static + modal).
* :mod:`.writer` — parse text dumps, assemble HDF5 (:class:`.writer.SampleMeta`,
  :func:`.writer.write_sample_h5`).
* :mod:`.reader` — typed HDF5 reader (:class:`.reader.FieldSample`,
  :func:`.reader.load_sample`, :func:`.reader.reconstruct_svm`).
"""
from __future__ import annotations

from pynumad.analysis.field_io.apdl_extract import (
    emit_modal_fields,
    emit_static_fields,
)
from pynumad.analysis.field_io.reader import (
    FieldSample,
    load_sample,
    open_sample,
    reconstruct_svm,
)
from pynumad.analysis.field_io.schema import (
    ALL_DATASETS,
    N_SURF,
    SCHEMA_VERSION,
    STRAIN_COMPONENTS,
    STRESS_COMPONENTS,
    SURFACES,
)
from pynumad.analysis.field_io.writer import SampleMeta, write_sample_h5

__all__ = [
    # schema
    "ALL_DATASETS",
    "SCHEMA_VERSION",
    "STRESS_COMPONENTS",
    "STRAIN_COMPONENTS",
    "SURFACES",
    "N_SURF",
    # APDL emitter
    "emit_static_fields",
    "emit_modal_fields",
    # writer
    "SampleMeta",
    "write_sample_h5",
    # reader
    "FieldSample",
    "load_sample",
    "open_sample",
    "reconstruct_svm",
]
