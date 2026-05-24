"""HDF5 reader for per-sample field containers.

Typed accessors for the layout defined in
:mod:`pynumad.analysis.field_io.schema`. Returns numpy arrays via
:class:`FieldSample`. No caching, no laziness — callers can decide
themselves whether to hold the file open (use :func:`open_sample` for
that) or fully materialise (use :func:`load_sample`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from pynumad.analysis.field_io.schema import (
    STRAIN_COMPONENTS,
    STRESS_COMPONENTS,
)


@dataclass
class FieldSample:
    """Fully materialised view of one HDF5 sample container.

    All arrays are read into memory at construction. For a 35k-element
    IEA-22 sample with 8 plies this is ~70 MB; if you have hundreds of
    samples open simultaneously, use :func:`open_sample` and read fields
    on demand instead.
    """
    meta: Mapping[str, Any]
    nodes: np.ndarray              # (N_node, 3)
    elem_conn: np.ndarray          # (N_elem, 4)
    elem_section_id: np.ndarray    # (N_elem,)
    elem_area: np.ndarray          # (N_elem,)
    elem_volume: np.ndarray        # (N_elem,)
    ply_thickness: np.ndarray      # (N_elem, N_ply_max)
    ply_angle_deg: np.ndarray      # (N_elem, N_ply_max)
    ply_material_id: np.ndarray    # (N_elem, N_ply_max)
    rv_names: list[str]
    rv_values: np.ndarray          # (N_rv,)
    baseline_materials: Mapping[str, Any] | None
    stress: dict[str, np.ndarray]  # comp -> (N_elem, N_ply, N_surf)
    strain: dict[str, np.ndarray]  # comp -> (N_elem, N_ply, N_surf)
    svm_ply: np.ndarray            # (N_elem, N_ply, N_surf)
    disp: np.ndarray               # (N_node, 6)
    sene: np.ndarray               # (N_elem,)
    react: np.ndarray              # (6,)
    freq: np.ndarray               # (N_modes,)
    eff_mass: np.ndarray           # (N_modes, 3)
    mode_shape: np.ndarray         # (N_modes, N_node, 6)
    scalars: Mapping[str, Any] | None


def load_sample(path: str | Path) -> FieldSample:
    """Read all datasets into memory and return a :class:`FieldSample`."""
    path = Path(path)
    with h5py.File(path, "r") as h5:
        meta = dict(h5["meta"].attrs.items())
        nodes = h5["/mesh/nodes"][...]
        elem_conn = h5["/mesh/elem_conn"][...]
        elem_section_id = h5["/mesh/elem_section_id"][...]
        elem_area = h5["/mesh/elem_area"][...]
        elem_volume = h5["/mesh/elem_volume"][...]
        ply_thickness = h5["/mesh/ply_thickness"][...]
        ply_angle_deg = h5["/mesh/ply_angle_deg"][...]
        ply_material_id = h5["/mesh/ply_material_id"][...]
        rv_names = [s.decode() if isinstance(s, bytes) else str(s)
                     for s in h5["/materials/rv_names"][...]]
        rv_values = h5["/materials/rv_values"][...]
        bm_raw = h5["materials"].attrs.get("baseline_cards", None)
        baseline_materials = json.loads(bm_raw) if bm_raw else None

        stress = {c: h5[f"/static/stress/{c}"][...]
                  for c in STRESS_COMPONENTS}
        strain = {c: h5[f"/static/strain/{c}"][...]
                  for c in STRAIN_COMPONENTS}
        svm_ply = h5["/static/svm_ply"][...]
        disp = h5["/static/disp"][...]
        sene = h5["/static/sene"][...]
        react = h5["/static/react"][...]

        freq = h5["/modal/freq"][...]
        eff_mass = h5["/modal/eff_mass"][...]
        mode_shape = h5["/modal/shape"][...]

        sc_raw = h5["scalars"].attrs.get("legacy_qois", None)
        scalars = json.loads(sc_raw) if sc_raw else None

    return FieldSample(
        meta=meta,
        nodes=nodes, elem_conn=elem_conn,
        elem_section_id=elem_section_id,
        elem_area=elem_area, elem_volume=elem_volume,
        ply_thickness=ply_thickness, ply_angle_deg=ply_angle_deg,
        ply_material_id=ply_material_id,
        rv_names=rv_names, rv_values=rv_values,
        baseline_materials=baseline_materials,
        stress=stress, strain=strain, svm_ply=svm_ply,
        disp=disp, sene=sene, react=react,
        freq=freq, eff_mass=eff_mass, mode_shape=mode_shape,
        scalars=scalars,
    )


def open_sample(path: str | Path) -> h5py.File:
    """Return the raw :class:`h5py.File` open for read.

    Use this when you want to read a subset of fields without paying the
    full materialisation cost. Caller owns closing the file.
    """
    return h5py.File(Path(path), "r")


def reconstruct_svm(stress_s11: np.ndarray, stress_s22: np.ndarray,
                    stress_s33: np.ndarray, stress_s12: np.ndarray,
                    stress_s13: np.ndarray, stress_s23: np.ndarray
                    ) -> np.ndarray:
    """Reconstruct von Mises from the six stress components.

    Used by the smoke-test verifier (vM round-trip check) and as a
    sanity sentinel anywhere the laminate-frame stresses are touched.
    """
    s11, s22, s33 = stress_s11, stress_s22, stress_s33
    s12, s13, s23 = stress_s12, stress_s13, stress_s23
    inner = (
        (s11 - s22) ** 2
        + (s22 - s33) ** 2
        + (s33 - s11) ** 2
        + 6.0 * (s12 ** 2 + s13 ** 2 + s23 ** 2)
    )
    return np.sqrt(0.5 * inner)
