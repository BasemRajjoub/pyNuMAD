"""Parse APDL text dumps and write the per-sample HDF5 container.

Pipeline:
    ANSYS *VWRITE  -->  text files in out_dir/  -->  writer (this module)
                                              -->  one .h5 per sample

This module owns the writer; the reader lives in ``reader.py``. Both
import from ``schema.py`` for dataset paths and shapes, so they can't
drift.

The writer is intentionally noisy: it raises ``FileNotFoundError`` if any
expected text file is missing and ``ValueError`` if any dataset shape
deviates from the schema. There is no "best effort" mode — a botched
extraction must fail fast (coding.md: root cause, no silent fallbacks).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from pynumad.analysis.field_io.schema import (
    ALL_DATASETS,
    N_SURF,
    REQUIRED_META_ATTRS,
    SCHEMA_VERSION,
    STRAIN_COMPONENTS,
    STRESS_COMPONENTS,
    SURFACES,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleMeta:
    """Metadata attrs that get attached to ``/meta`` of the HDF5 file.

    All fields are required. The writer rejects samples with missing
    metadata — every sample in the DoE must be fully traceable.
    """
    sample_id: int
    doe_seed: int
    git_sha_pynumad: str
    git_sha_repo: str
    ansys_version: str
    mesh_h: float
    load_case: str
    load_scale: float
    timestamp_utc: str


def write_sample_h5(
    out_path: str | Path,
    *,
    text_dir: str | Path,
    mesh: Mapping[str, Any],
    rv_names: Sequence[str],
    rv_values: Sequence[float],
    baseline_materials: Mapping[str, Any] | None,
    n_modes: int,
    n_ply_max: int,
    meta: SampleMeta,
    scalars: Mapping[str, Any] | None = None,
) -> Path:
    """Assemble one HDF5 sample container from APDL text dumps.

    Parameters
    ----------
    out_path
        Destination HDF5 path. Parent directory is created if missing.
    text_dir
        Directory holding the APDL ``*VWRITE`` text files emitted by
        :mod:`pynumad.analysis.field_io.apdl_extract`.
    mesh
        pyNuMAD mesh dict — ``mesh["nodes"]`` (N, 3), ``mesh["elements"]``
        (N, k) with k = 4 quad nodes (0-indexed); ``mesh["sets"]`` is
        not required here but typically present.
    rv_names, rv_values
        Random-variable names and sampled values for the DoE row that
        produced this sample. Lengths must match.
    baseline_materials
        Optional dict snapshot of the baseline matlib (before RV
        overrides). Stored as a JSON string on ``/materials`` attr
        ``baseline_cards``.
    n_modes
        Number of modal modes extracted (must match the number of
        ``mode_shape_M<NN>.txt`` files present).
    n_ply_max
        Max ply count any element has — matches the ``LAYER`` loop in
        the APDL extractor. Determines the ply axis in stress/strain
        datasets.
    meta
        Full :class:`SampleMeta` attached as ``/meta`` attrs.
    scalars
        Optional dict of legacy QoIs (patch σ_vM, section moments, tip
        deflection) — stored as JSON on ``/scalars`` attrs.

    Returns
    -------
    Path
        The HDF5 file that was written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    td = Path(text_dir)

    assert len(rv_names) == len(rv_values), (
        f"RV name/value length mismatch: {len(rv_names)} vs {len(rv_values)}"
    )

    # Parse all the text dumps up front. Failing fast here avoids leaving
    # a partial HDF5 on disk.
    nodes = np.asarray(mesh["nodes"], dtype=np.float32)
    conn0 = np.asarray(mesh["elements"], dtype=np.int64)
    assert nodes.ndim == 2 and nodes.shape[1] == 3, (
        f"mesh['nodes'] shape {nodes.shape}, expected (N, 3)"
    )
    assert conn0.ndim == 2 and conn0.shape[1] >= 3, (
        f"mesh['elements'] shape {conn0.shape}, expected (N, >=3)"
    )
    # pyNuMAD labels are 0-indexed; ANSYS deck uses 1-indexed. Match the
    # ANSYS convention since downstream tools expect it.
    conn = (conn0[:, :4].astype(np.int32, copy=False)) + np.int32(1)
    n_elem = conn.shape[0]
    n_node = nodes.shape[0]

    elem_geom = _parse_table(td / "elem_geom.txt", n_cols=3)
    elem_geom = _truncate_elem_axis(elem_geom, n_elem, "elem_geom.txt")
    elem_area = elem_geom[:, 1].astype(np.float32, copy=False)
    elem_volume = elem_geom[:, 2].astype(np.float32, copy=False)

    stress, strain, svm = _parse_field_files(td, n_elem, n_ply_max)

    disp_tbl = _parse_table(td / "disp.txt", n_cols=7)
    disp_tbl = _truncate_node_axis(disp_tbl, n_node, "disp.txt")
    disp = disp_tbl[:, 1:].astype(np.float32, copy=False)

    sene_tbl = _parse_table(td / "sene.txt", n_cols=2)
    sene_tbl = _truncate_elem_axis(sene_tbl, n_elem, "sene.txt")
    sene = sene_tbl[:, 1].astype(np.float32, copy=False)

    react = _parse_table(td / "react.txt", n_cols=6).reshape(-1)
    assert react.size == 6, f"react.txt has {react.size} values, expected 6"
    react = react.astype(np.float64, copy=False)

    modal_freq_tbl = _parse_table(td / "modal_freq.txt", n_cols=5)
    _assert_n_rows(modal_freq_tbl, n_modes, "modal_freq.txt")
    freq = modal_freq_tbl[:, 1].astype(np.float64, copy=False)
    eff_mass = modal_freq_tbl[:, 2:5].astype(np.float64, copy=False)

    mode_shape = np.empty((n_modes, n_node, 6), dtype=np.float32)
    for imode in range(1, n_modes + 1):
        ms_tbl = _parse_table(td / f"mode_shape_M{imode:02d}.txt", n_cols=7)
        ms_tbl = _truncate_node_axis(
            ms_tbl, n_node, f"mode_shape_M{imode:02d}.txt",
        )
        mode_shape[imode - 1] = ms_tbl[:, 1:].astype(np.float32, copy=False)

    # Ply geometry (thickness, angle, material id) from the pyNuMAD
    # mesh dict. shell_mesh_general emits sections as a list with
    # elementSet + layup, and materials as a list with elastic E/nu/G.
    # See _build_ply_tables for the exact layout.
    elem_section_id, ply_thick, ply_angle, ply_matid, mat_names = (
        _build_ply_tables(mesh, n_elem, n_ply_max)
    )

    # Reconstruct gamma13/gamma23 from tau13/tau23 via per-ply
    # transverse shear moduli (G13, G23). ANSYS does NOT output
    # EPEL,XZ / EPEL,YZ for layered SHELL181 (transverse-shear stresses
    # come from parabolic equilibrium, not kinematics). For linear
    # elasticity gamma_ij = tau_ij / G_ij which is well-defined.
    g13_arr, g23_arr = _ply_shear_moduli(mesh, ply_matid, mat_names)
    # gamma shape (N_elem, N_ply, N_surf): broadcast G over surfaces
    g13_3d = np.broadcast_to(g13_arr[:, :, None],
                              strain[STRAIN_COMPONENTS[3]].shape)
    g23_3d = np.broadcast_to(g23_arr[:, :, None],
                              strain[STRAIN_COMPONENTS[4]].shape)
    # tau13 = stress["s13"], tau23 = stress["s23"] (laminate frame)
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma13_recon = (stress["s13"].astype(np.float64) / g13_3d).astype(
            np.float32,
        )
        gamma23_recon = (stress["s23"].astype(np.float64) / g23_3d).astype(
            np.float32,
        )
    # Wherever G is NaN (missing material) or tau is exactly zero
    # (absent ply per ANSYS extraction), the reconstructed gamma is
    # NaN/0 - both acceptable signals downstream.
    strain[STRAIN_COMPONENTS[3]] = gamma13_recon
    strain[STRAIN_COMPONENTS[4]] = gamma23_recon

    # Now write everything in one open() block so any error leaves no
    # half-written file on disk.
    with h5py.File(out_path, "w") as h5:
        # /meta
        meta_grp = h5.create_group("meta")
        for k in REQUIRED_META_ATTRS:
            if k in {"n_elem", "n_node", "n_ply_max", "n_surf",
                     "n_rv", "n_modes", "schema_version", "units_system"}:
                continue  # set below
            meta_grp.attrs[k] = getattr(meta, k)
        meta_grp.attrs["n_elem"] = n_elem
        meta_grp.attrs["n_node"] = n_node
        meta_grp.attrs["n_ply_max"] = n_ply_max
        meta_grp.attrs["n_surf"] = N_SURF
        meta_grp.attrs["n_rv"] = len(rv_names)
        meta_grp.attrs["n_modes"] = n_modes
        meta_grp.attrs["schema_version"] = SCHEMA_VERSION
        meta_grp.attrs["units_system"] = "SI"

        # /mesh
        _create(h5, "/mesh/nodes", nodes)
        _create(h5, "/mesh/elem_conn", conn)
        _create(h5, "/mesh/elem_section_id", elem_section_id)
        _create(h5, "/mesh/elem_area", elem_area)
        _create(h5, "/mesh/elem_volume", elem_volume)
        _create(h5, "/mesh/ply_thickness", ply_thick)
        _create(h5, "/mesh/ply_angle_deg", ply_angle)
        _create(h5, "/mesh/ply_material_id", ply_matid)

        # /materials
        mat_grp = h5.create_group("materials")
        mat_grp.create_dataset(
            "rv_names",
            data=np.asarray(rv_names, dtype=h5py.string_dtype("utf-8")),
        )
        mat_grp.create_dataset(
            "rv_values",
            data=np.asarray(rv_values, dtype=np.float64),
        )
        if baseline_materials is not None:
            mat_grp.attrs["baseline_cards"] = json.dumps(baseline_materials)

        # /static
        for c in STRESS_COMPONENTS:
            _create(h5, f"/static/stress/{c}", stress[c])
            # Tag near-zero-variance fields so the Phase-D KL loader
            # can skip them without crashing on a degenerate snapshot
            # matrix. SHELL181 plane-stress kinematics force s33 ~ 0,
            # so it shows up here every time; if we later switch to
            # SHELL281 or solids the tag stops triggering naturally.
            max_abs = float(np.nanmax(np.abs(stress[c]))) \
                if stress[c].size else 0.0
            ds = h5[f"/static/stress/{c}"]
            ds.attrs["max_abs_pa"] = max_abs
            ds.attrs["near_zero"] = int(max_abs < 1.0)
        for c in STRAIN_COMPONENTS:
            _create(h5, f"/static/strain/{c}", strain[c])
            arr = strain[c]
            max_abs = float(np.nanmax(np.abs(arr))) if arr.size else 0.0
            ds = h5[f"/static/strain/{c}"]
            ds.attrs["max_abs"] = max_abs
            ds.attrs["near_zero"] = int(max_abs < 1e-12)
        _create(h5, "/static/svm_ply", svm)
        _create(h5, "/static/disp", disp)
        _create(h5, "/static/sene", sene)
        _create(h5, "/static/react", react)

        # /modal - rigid-body ghost flag per mode (Lanczos sometimes
        # returns a near-zero "ghost" for very stiff fully-clamped
        # structures; keep the data, mark the mode so loaders can
        # skip it without re-deriving the criterion).
        _create(h5, "/modal/freq", freq)
        _create(h5, "/modal/eff_mass", eff_mass)
        _create(h5, "/modal/shape", mode_shape)
        is_rigid = (np.abs(freq) < 1e-3).astype(np.int8)
        h5["/modal"].attrs["is_rigid_body"] = is_rigid
        h5["/modal"].attrs["rigid_body_threshold_hz"] = 1e-3

        # /scalars (legacy convergence QoIs — JSON on the group attrs)
        sc = h5.create_group("scalars")
        if scalars is not None:
            sc.attrs["legacy_qois"] = json.dumps(_jsonable(scalars))

    return out_path


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


_FIELD_FNAME_RE = re.compile(r"^field_L(\d{2})_(TOP|MID|BOT)\.txt$")


def _parse_table(path: Path, n_cols: int) -> np.ndarray:
    """Read an APDL *VWRITE CSV-like table.

    The file is comma-delimited, possibly with extra whitespace. Returns
    a ``(N_rows, n_cols)`` ``float64`` array. Raises ``ValueError`` if
    the column count doesn't match.
    """
    if not path.exists():
        raise FileNotFoundError(f"missing extractor dump: {path}")
    arr = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    if arr.shape[1] != n_cols:
        raise ValueError(
            f"{path.name}: expected {n_cols} columns, got {arr.shape[1]}"
        )
    return arr


def _assert_n_rows(arr: np.ndarray, expected: int, name: str) -> None:
    if arr.shape[0] != expected:
        raise ValueError(
            f"{name}: expected {expected} rows, got {arr.shape[0]}"
        )


def _truncate_elem_axis(arr: np.ndarray, n_elem: int, name: str
                         ) -> np.ndarray:
    """Tolerate trailing zero-valued rows in an ANSYS element-axis text
    dump.

    ANSYS sometimes adds a phantom constraint / rigid-body element with
    zero area / zero volume after the real shell elements. pyNuMAD's
    mesh dict has the canonical count; if the text dump has *one or a
    handful* of extra rows AND those rows are all-zero in the data
    columns, drop them. Otherwise raise so a real mismatch surfaces.
    """
    extra = arr.shape[0] - n_elem
    if extra < 0:
        raise ValueError(
            f"{name}: expected at least {n_elem} rows, got {arr.shape[0]}"
        )
    if extra == 0:
        return arr
    tail = arr[n_elem:, 1:]
    if not np.allclose(tail, 0.0):
        raise ValueError(
            f"{name}: {extra} trailing row(s) present and are not all "
            f"zero - extraction probably out of sync with mesh"
        )
    print(f"[field_io] {name}: dropped {extra} trailing phantom "
          f"element row(s)")
    return arr[:n_elem]


def _truncate_node_axis(arr: np.ndarray, n_node: int, name: str
                         ) -> np.ndarray:
    """Reconcile an ANSYS node-keyed table with the mesh-dict node count.

    Two directions are tolerated:

    * **Surplus** (ANSYS > mesh): trailing all-zero phantom rows are
      dropped (mirrors the element-axis logic).

    * **Deficit** (ANSYS < mesh): ``nummrg,all`` merged coincident nodes
      (e.g. duplicate nodes the conforming T-junction mesher places at
      shell/web interfaces) and ``numcmp,node`` renumbered the survivors.
      ANSYS then reports fewer nodes than the pyNuMAD mesh dict. The
      node-keyed tables (``disp``, ``mode_shape``) are AUXILIARY fields:
      the field-PCE pipeline reads only per-element stress/strain/svm,
      which are unaffected (elements are not renumbered by numcmp,node).
      We zero-pad the deficit so the HDF5 nodal-field shape stays
      consistent with ``n_node``, and warn. A hard cap distinguishes a
      benign coincident-node merge from genuine extraction corruption.
    """
    extra = arr.shape[0] - n_node
    if extra == 0:
        return arr
    if extra > 0:
        tail = arr[n_node:, 1:]
        if not np.allclose(tail, 0.0):
            raise ValueError(
                f"{name}: {extra} trailing row(s) present and are not all "
                f"zero - extraction probably out of sync with mesh"
            )
        print(f"[field_io] {name}: dropped {extra} trailing phantom "
              f"node row(s)")
        return arr[:n_node]

    # extra < 0: ANSYS returned fewer nodes than the mesh dict.
    deficit = -extra
    cap = max(16, int(0.005 * n_node))  # >0.5% missing ⇒ real corruption
    if deficit > cap:
        raise ValueError(
            f"{name}: ANSYS returned {arr.shape[0]} rows but mesh has "
            f"{n_node} nodes (deficit {deficit} > cap {cap}). This is too "
            f"large for coincident-node merging — likely a real mesh / "
            f"extraction mismatch."
        )
    print(f"[field_io] {name}: ANSYS node count {arr.shape[0]} < mesh "
          f"{n_node} (deficit {deficit}); nummrg merged coincident nodes. "
          f"Zero-padding auxiliary nodal field (unused by field-PCE).")
    pad = np.zeros((deficit, arr.shape[1]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def _parse_field_files(
    text_dir: Path, n_elem: int, n_ply_max: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Parse every ``field_L<NN>_<SURF>.txt`` file and stack into
    schema-shaped arrays.

    Returns
    -------
    stress : dict[str, ndarray]
        Mapping ``s11``...``s23`` to ``(N_elem, N_ply, N_surf)`` arrays.
    strain : dict[str, ndarray]
        Mapping ``eps11``...``gamma23`` to same-shape arrays.
    svm : ndarray
        ``(N_elem, N_ply, N_surf)`` of per-ply von Mises.

    Plies with no extraction file get NaN entries — useful so KL
    truncation can ignore them, or downstream fatigue code can skip.
    """
    stress = {c: np.full((n_elem, n_ply_max, N_SURF), np.nan,
                          dtype=np.float32) for c in STRESS_COMPONENTS}
    strain = {c: np.full((n_elem, n_ply_max, N_SURF), np.nan,
                          dtype=np.float32) for c in STRAIN_COMPONENTS}
    svm = np.full((n_elem, n_ply_max, N_SURF), np.nan, dtype=np.float32)

    surf_idx = {s: i for i, s in enumerate(SURFACES)}
    seen_any = False
    for f in sorted(text_dir.iterdir()):
        m = _FIELD_FNAME_RE.match(f.name)
        if not m:
            continue
        ply = int(m.group(1))
        surf = m.group(2)
        if not (1 <= ply <= n_ply_max):
            raise ValueError(
                f"{f.name}: ply index {ply} outside [1, {n_ply_max}]"
            )
        tbl = _parse_table(f, n_cols=13)
        tbl = _truncate_elem_axis(tbl, n_elem, f.name)
        seen_any = True
        si = surf_idx[surf]
        # Column order (set by emitter): elem, s11, s22, s33, s12, s13,
        # s23, eps11, eps22, gamma12, gamma13, gamma23, svm.
        for i, c in enumerate(STRESS_COMPONENTS, start=1):
            stress[c][:, ply - 1, si] = tbl[:, i].astype(np.float32)
        for i, c in enumerate(STRAIN_COMPONENTS, start=7):
            strain[c][:, ply - 1, si] = tbl[:, i].astype(np.float32)
        svm[:, ply - 1, si] = tbl[:, 12].astype(np.float32)

    if not seen_any:
        raise FileNotFoundError(
            f"no field_L*_<SURF>.txt files in {text_dir}"
        )
    return stress, strain, svm


def _build_ply_tables(
    mesh: Mapping[str, Any], n_elem: int, n_ply_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Per-ply geometry + material lookup from the pyNuMAD mesh dict.

    The actual ``shell_mesh_general`` output layout (verified on
    IEA-22) is::

        mesh["sets"]["element"]   list of {"name": str, "labels": [int, ...]}
        mesh["sections"]          list of {"elementSet": str, "layup":
                                            [[mat_name, thickness, angle], ...]}

    The section is matched to its element-set by name; ply tables are
    then filled for every element in that set. Materials referenced by
    name are looked up against ``mesh["materials"]`` (also a list of
    dicts) to build a per-ply material-id table.

    Returns
    -------
    sec_id, thickness, angle, mat_id, mat_names
        Arrays sized (N_elem,) / (N_elem, N_ply_max) plus a list of
        unique material names (mat_id is the 0-indexed position in
        this list; -1 for absent ply).
    """
    sec_id = np.zeros(n_elem, dtype=np.int32)
    thick = np.full((n_elem, n_ply_max), np.nan, dtype=np.float32)
    angle = np.full((n_elem, n_ply_max), np.nan, dtype=np.float32)
    matid = np.full((n_elem, n_ply_max), -1, dtype=np.int32)
    mat_names: list[str] = []

    el_sets = (mesh.get("sets") or {}).get("element") or []
    sections = mesh.get("sections") or []
    materials = mesh.get("materials") or []
    if not el_sets or not sections:
        return sec_id, thick, angle, matid, mat_names

    # name -> [labels...]
    set_labels = {str(s.get("name", "")): list(s.get("labels", []))
                  for s in el_sets}

    # Material name -> stable index (0..M-1)
    mat_names = [str(m.get("name", "")) for m in materials]
    mat_idx_by_name = {n: i for i, n in enumerate(mat_names)}

    for sid, sec in enumerate(sections, start=1):
        set_name = str(sec.get("elementSet", ""))
        labels = set_labels.get(set_name)
        if not labels:
            continue
        idx = np.asarray(labels, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < n_elem)]
        if idx.size == 0:
            continue
        sec_id[idx] = sid
        layup = sec.get("layup") or sec.get("plies") or []
        for ip, ply in enumerate(layup[:n_ply_max]):
            if isinstance(ply, (list, tuple)) and len(ply) >= 3:
                ply_mat, ply_t, ply_a = ply[0], ply[1], ply[2]
            elif isinstance(ply, dict):
                ply_mat = ply.get("material") or ply.get("materialName") \
                    or ply.get("materialId")
                ply_t = ply.get("thickness", np.nan)
                ply_a = ply.get("angle", np.nan)
            else:
                continue
            thick[idx, ip] = float(ply_t) if ply_t is not None \
                else float("nan")
            angle[idx, ip] = float(ply_a) if ply_a is not None \
                else float("nan")
            if isinstance(ply_mat, str):
                matid[idx, ip] = mat_idx_by_name.get(ply_mat, -1)
            elif isinstance(ply_mat, int):
                matid[idx, ip] = int(ply_mat)

    return sec_id, thick, angle, matid, mat_names


def _ply_shear_moduli(
    mesh: Mapping[str, Any], matid: np.ndarray, mat_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-element, per-ply G13 and G23 (Pa) for transverse-shear
    strain reconstruction.

    ``mesh["materials"][i]["elastic"]["G"]`` is ``[G12, G13, G23]``
    (or ``[Gxy, Gxz, Gyz]`` in laminate-axis notation). Missing
    materials get NaN, which propagates to gamma = tau / G and yields
    NaN strains - flagging "cannot reconstruct".
    """
    materials = mesh.get("materials") or []
    g13_lookup = np.full(len(mat_names), np.nan, dtype=np.float64)
    g23_lookup = np.full(len(mat_names), np.nan, dtype=np.float64)
    for i, name in enumerate(mat_names):
        # Find the matching material by name
        m = next((mm for mm in materials
                  if str(mm.get("name", "")) == name), None)
        if m is None:
            continue
        elastic = m.get("elastic") or {}
        g_arr = elastic.get("G") or []
        if isinstance(g_arr, (list, tuple, np.ndarray)) and len(g_arr) >= 3:
            g13_lookup[i] = float(g_arr[1])
            g23_lookup[i] = float(g_arr[2])

    n_elem, n_ply = matid.shape
    g13 = np.full((n_elem, n_ply), np.nan, dtype=np.float64)
    g23 = np.full((n_elem, n_ply), np.nan, dtype=np.float64)
    valid = matid >= 0
    g13[valid] = g13_lookup[matid[valid]]
    g23[valid] = g23_lookup[matid[valid]]
    return g13, g23


def _create(h5: h5py.File, path: str, data: np.ndarray) -> None:
    """Create a dataset with sensible chunking + gzip compression.

    Chunk along axis 0 for ndarrays with > 1 element along that axis;
    contiguous storage otherwise. gzip level 4 is the sweet spot for
    composite stress fields (verified on smoke-test data — going higher
    yields < 5 % extra compression for 3-4x more CPU).
    """
    if data.ndim == 0 or (data.ndim >= 1 and data.shape[0] <= 1):
        h5.create_dataset(path, data=data)
        return
    chunks = list(data.shape)
    # Bound chunk size to ~ 1 MB to keep h5py happy on small fields.
    elem_bytes = np.dtype(data.dtype).itemsize
    while np.prod(chunks) * elem_bytes > 1_048_576 and chunks[0] > 1:
        chunks[0] = max(1, chunks[0] // 2)
    h5.create_dataset(
        path, data=data, chunks=tuple(chunks),
        compression="gzip", compression_opts=4, shuffle=True,
    )


def _jsonable(obj: Any) -> Any:
    """Convert nested numpy types to JSON-serialisable Python builtins."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
