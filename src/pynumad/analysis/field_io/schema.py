"""HDF5 schema for per-sample field-PCE data containers.

Single source of truth for the on-disk layout of one ANSYS sample. Imported
by the APDL emitter, the writer, and the reader. If the writer and the
reader ever disagree on a dataset path, dtype, or shape, this module is
authoritative.

Purpose
-------
One DoE sample produces one HDF5 file holding every field that ANSYS can
emit from a SHELL181 layered-shell linear-static + modal solve. Downstream
consumers (KL decomposition, PCE fits, Tsai-Wu evaluation, fatigue,
modal-uncertainty studies) read from this file and never need to re-run
ANSYS. The schema is intentionally *generous* — we save everything once,
so we never have to re-run a 250-sample DoE because a downstream analysis
needed a different field.

Layout
------
::

    /meta                            attrs only (no datasets)
        sample_id, doe_seed, git_sha_pynumad, git_sha_repo,
        ansys_version, mesh_h, load_case, load_scale, timestamp_utc,
        schema_version, units_system="SI"

    /mesh
        nodes              (N_node, 3)            float32  m
        elem_conn          (N_elem, 4)            int32    1-indexed
        elem_section_id    (N_elem,)              int32
        elem_area          (N_elem,)              float32  m^2
        elem_volume        (N_elem,)              float32  m^3
        ply_thickness      (N_elem, N_ply_max)    float32  m       (NaN if ply absent)
        ply_angle_deg      (N_elem, N_ply_max)    float32  deg     (NaN if ply absent)
        ply_material_id    (N_elem, N_ply_max)    int32             (-1 if ply absent)

    /materials
        rv_names           (N_rv,)                str
        rv_values          (N_rv,)                float64
        baseline_cards     attrs (JSON-encoded text per material)

    /static
        /stress
            s11, s22, s33, s12, s13, s23           (N_elem, N_ply_max, N_surf)  float32  Pa
            # Components are in the LAYER coordinate system (RSYS,LSYS):
            #   s11 = along fibre direction
            #   s22 = transverse in-plane
            #   s33 = through-thickness normal (small but non-zero in SHELL181)
            #   s12 = in-plane shear (drives Tsai-Wu cross term)
            #   s13, s23 = interlaminar shears (drive delamination)
            # NOT element CS, NOT global CS - the APDL emitter forces
            # RSYS,LSYS in the static-extraction header. Composite failure
            # criteria (Tsai-Wu, Hashin, Puck, max-strain) all assume
            # this laminate frame.
        /strain
            eps11, eps22, gamma12, gamma13, gamma23 (N_elem, N_ply_max, N_surf) float32  -
            # Same laminate-frame convention as /stress.
        svm_ply                                    (N_elem, N_ply_max, N_surf)  float32  Pa
        disp                                       (N_node, 6)                  float32  m,rad
        sene                                       (N_elem,)                    float32  J
        react                                      (6,)                         float64  N, Nm
                                                                                # FX,FY,FZ,MX,MY,MZ at root

    /modal
        freq            (N_modes,)            float64  Hz
        eff_mass        (N_modes, 3)          float64  kg
        shape           (N_modes, N_node, 6)  float32  m,rad  (mass-normalised)

    /scalars                                  attrs (JSON-encoded legacy QoIs from
                                              pynumad.analysis.convergence)

Shape constants (``N_elem``, ``N_node``, ``N_ply_max``, ``N_surf``,
``N_rv``, ``N_modes``) are written as attrs on ``/meta`` so any reader can
sanity-check before allocation.

Surface order (axis -1 of ``stress``/``strain``/``svm_ply``): ``TOP, MID,
BOT`` — see :data:`SURFACES`. Ply index axis is **0-indexed in HDF5** but
the APDL ``LAYER`` command is 1-indexed; the writer converts.

Compression: every numeric dataset is ``gzip`` level 4, chunked along the
element axis (for per-element fields) or the node axis (for nodal fields).
Strings are stored as UTF-8 variable-length.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SCHEMA_VERSION = "1.0"

STRESS_COMPONENTS: tuple[str, ...] = ("s11", "s22", "s33", "s12", "s13", "s23")
STRAIN_COMPONENTS: tuple[str, ...] = (
    "eps11", "eps22", "gamma12", "gamma13", "gamma23",
)
SURFACES: tuple[str, ...] = ("TOP", "MID", "BOT")
N_SURF = 3

# SHELL181 is a Mindlin-Reissner shell with a plane-stress kinematic
# assumption (sigma_33 ~ 0 by element formulation; eps_13, eps_23 not
# solved). The transverse-shear stresses (s13, s23) are RECONSTRUCTED
# post-solve from a parabolic equilibrium ansatz, not from the
# constitutive equation - they are not honest FE outputs and should
# not feed a PCE / Sobol analysis. For SHELL281 quadratic shells or
# solid elements these would become meaningful and the skip-list
# below should be re-evaluated.
#
# The KL/PCE loader (paper2_fem/03_post_klpce/load_snapshots.py)
# consumes this constant and excludes the listed (field, components)
# from snapshot stacking by default.
SHELL181_INVALID: dict[str, tuple[str, ...]] = {
    "stress": ("s33", "s13", "s23"),
    "strain": ("gamma13", "gamma23"),
    # eps33 is never extracted - no entry needed.
}

REQUIRED_META_ATTRS: tuple[str, ...] = (
    "sample_id",
    "doe_seed",
    "git_sha_pynumad",
    "git_sha_repo",
    "ansys_version",
    "mesh_h",
    "load_case",
    "load_scale",
    "timestamp_utc",
    "schema_version",
    "units_system",
    "n_elem",
    "n_node",
    "n_ply_max",
    "n_surf",
    "n_rv",
    "n_modes",
)


Axis = Literal["elem", "node", "ply", "mode", "scalar"]


@dataclass(frozen=True)
class DatasetSpec:
    """Specification for a single HDF5 dataset.

    ``shape_axes`` lists the logical axes in order; concrete sizes come
    from the ``/meta`` attrs at write time. ``chunk_axis`` is the axis
    along which the dataset is chunked for compression — ``None`` means
    "store contiguously" (only used for tiny vectors).
    """
    path: str
    shape_axes: tuple[Axis, ...]
    dtype: str
    units: str
    chunk_axis: int | None
    description: str


# Per-element, per-ply, per-surface stress/strain components are
# materialised as 6 (resp. 5) sibling datasets at /static/stress/<comp>
# and /static/strain/<comp>. Writing six datasets instead of one (..., 6)
# array makes downstream slicing trivial and lets HDF5 compress each
# component independently — composite stresses have very different
# magnitudes per component, so per-dataset gzip is more effective.

def _ply_surf_datasets(prefix: str, comps: tuple[str, ...], units: str,
                       description: str) -> tuple[DatasetSpec, ...]:
    return tuple(
        DatasetSpec(
            path=f"{prefix}/{c}",
            shape_axes=("elem", "ply", "scalar"),  # last axis is N_surf=3
            dtype="float32",
            units=units,
            chunk_axis=0,
            description=f"{description} ({c})",
        )
        for c in comps
    )


STATIC_DATASETS: tuple[DatasetSpec, ...] = (
    *_ply_surf_datasets(
        "/static/stress", STRESS_COMPONENTS, "Pa",
        "Per-ply, per-surface Cauchy stress in material frame",
    ),
    *_ply_surf_datasets(
        "/static/strain", STRAIN_COMPONENTS, "-",
        "Per-ply, per-surface elastic strain in material frame",
    ),
    DatasetSpec(
        path="/static/svm_ply",
        shape_axes=("elem", "ply", "scalar"),
        dtype="float32",
        units="Pa",
        chunk_axis=0,
        description="Per-ply, per-surface von Mises (ANSYS diagnostic)",
    ),
    DatasetSpec(
        path="/static/disp",
        shape_axes=("node", "scalar"),
        dtype="float32",
        units="m,rad",
        chunk_axis=0,
        description="Nodal displacement [ux,uy,uz,rotx,roty,rotz]",
    ),
    DatasetSpec(
        path="/static/sene",
        shape_axes=("elem",),
        dtype="float32",
        units="J",
        chunk_axis=0,
        description="Element strain energy",
    ),
    DatasetSpec(
        path="/static/react",
        shape_axes=("scalar",),
        dtype="float64",
        units="N,Nm",
        chunk_axis=None,
        description="Root reactions [FX,FY,FZ,MX,MY,MZ]",
    ),
)


MODAL_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        path="/modal/freq",
        shape_axes=("mode",),
        dtype="float64",
        units="Hz",
        chunk_axis=None,
        description="Natural frequencies",
    ),
    DatasetSpec(
        path="/modal/eff_mass",
        shape_axes=("mode", "scalar"),
        dtype="float64",
        units="kg",
        chunk_axis=None,
        description="Effective modal mass [X, Y, Z]",
    ),
    DatasetSpec(
        path="/modal/shape",
        shape_axes=("mode", "node", "scalar"),
        dtype="float32",
        units="m,rad",
        chunk_axis=1,
        description="Mass-normalised mode shapes [ux,uy,uz,rotx,roty,rotz]",
    ),
)


MESH_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        path="/mesh/nodes",
        shape_axes=("node", "scalar"),
        dtype="float32",
        units="m",
        chunk_axis=0,
        description="Nodal coordinates",
    ),
    DatasetSpec(
        path="/mesh/elem_conn",
        shape_axes=("elem", "scalar"),
        dtype="int32",
        units="-",
        chunk_axis=0,
        description="Element connectivity (1-indexed quad nodes)",
    ),
    DatasetSpec(
        path="/mesh/elem_section_id",
        shape_axes=("elem",),
        dtype="int32",
        units="-",
        chunk_axis=0,
        description="Section ID per element (1-indexed)",
    ),
    DatasetSpec(
        path="/mesh/elem_area",
        shape_axes=("elem",),
        dtype="float32",
        units="m^2",
        chunk_axis=0,
        description="Element area",
    ),
    DatasetSpec(
        path="/mesh/elem_volume",
        shape_axes=("elem",),
        dtype="float32",
        units="m^3",
        chunk_axis=0,
        description="Element volume (area x total laminate thickness)",
    ),
    DatasetSpec(
        path="/mesh/ply_thickness",
        shape_axes=("elem", "ply"),
        dtype="float32",
        units="m",
        chunk_axis=0,
        description="Ply thickness (NaN where ply absent)",
    ),
    DatasetSpec(
        path="/mesh/ply_angle_deg",
        shape_axes=("elem", "ply"),
        dtype="float32",
        units="deg",
        chunk_axis=0,
        description="Ply orientation angle (NaN where ply absent)",
    ),
    DatasetSpec(
        path="/mesh/ply_material_id",
        shape_axes=("elem", "ply"),
        dtype="int32",
        units="-",
        chunk_axis=0,
        description="Ply material ID (-1 where ply absent)",
    ),
)


MATERIALS_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        path="/materials/rv_names",
        shape_axes=("scalar",),
        dtype="vlen-str",
        units="-",
        chunk_axis=None,
        description="Random-variable names (length N_rv)",
    ),
    DatasetSpec(
        path="/materials/rv_values",
        shape_axes=("scalar",),
        dtype="float64",
        units="mixed",
        chunk_axis=None,
        description="Random-variable values (length N_rv)",
    ),
)


ALL_DATASETS: tuple[DatasetSpec, ...] = (
    *MESH_DATASETS,
    *MATERIALS_DATASETS,
    *STATIC_DATASETS,
    *MODAL_DATASETS,
)


def axis_size(axis: Axis, n_elem: int, n_node: int, n_ply_max: int,
              n_modes: int, n_rv: int) -> int:
    """Resolve a logical axis label to its concrete size.

    The ``scalar`` axis is context-dependent (3 for vectors, 6 for nodal
    DOF, 4 for element connectivity, etc.) — callers must override the
    last-axis size when allocating those datasets. This helper handles
    only the size-from-meta axes.
    """
    return {
        "elem": n_elem,
        "node": n_node,
        "ply": n_ply_max,
        "mode": n_modes,
    }[axis]
