"""Unit tests for pynumad.analysis.field_io.

These tests do not touch ANSYS. They fabricate the text-dump files that
the APDL extractor would emit, run them through the writer, read them
back through the reader, and assert that:

* shapes match the schema for every dataset,
* values round-trip bit-for-bit (modulo float32 cast),
* missing extractor files raise ``FileNotFoundError`` with the filename,
* shape mismatches raise ``ValueError``,
* the APDL emitter produces well-formed strings,
* the von Mises reconstruction helper matches a hand-computed value.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from pynumad.analysis.field_io import (
    N_SURF,
    SCHEMA_VERSION,
    STRAIN_COMPONENTS,
    STRESS_COMPONENTS,
    SURFACES,
    SampleMeta,
    emit_modal_fields,
    emit_static_fields,
    load_sample,
    reconstruct_svm,
    write_sample_h5,
)


# ---------------------------------------------------------------------------
# Fixture: a tiny synthetic mesh + text dumps the writer will consume
# ---------------------------------------------------------------------------


N_ELEM = 3
N_NODE = 8
N_PLY_MAX = 2
N_MODES = 2
N_RV = 4


def _fmt_row(values, id_first: bool) -> str:
    """Match the APDL *VWRITE format used by apdl_extract."""
    out = []
    for i, v in enumerate(values):
        if i == 0 and id_first:
            out.append(f"{float(v):12.0f}")
        else:
            out.append(f"{float(v):20.12E}")
    return ",".join(out)


def _write_table(path: Path, rows, id_first=True) -> None:
    lines = [_fmt_row(r, id_first) for r in rows]
    path.write_text("\n".join(lines) + "\n")


def _make_mesh() -> dict:
    """Synthetic mesh matching the actual pyNuMAD shell_mesh_general
    layout: ``sets["element"]`` list + ``sections`` list with
    ``elementSet`` reference + ``materials`` list with elastic E/nu/G.
    """
    nodes = np.zeros((N_NODE, 3), dtype=np.float64)
    nodes[:, 0] = np.arange(N_NODE)
    elems = np.array(
        [[0, 1, 2, 3], [1, 4, 5, 2], [4, 6, 7, 5]],
        dtype=np.int64,
    )
    return {
        "nodes": nodes,
        "elements": elems,
        "sets": {
            "element": [
                {"name": "panel_all", "labels": list(range(N_ELEM))},
            ],
        },
        "sections": [
            {
                "type": "shell",
                "elementSet": "panel_all",
                "layup": [
                    ["mat_a", 0.001, 0.0],
                    ["mat_b", 0.002, 45.0],
                ],
            },
        ],
        "materials": [
            {"name": "mat_a", "density": 1900.0,
             "elastic": {"E": [40e9, 10e9, 10e9],
                         "nu": [0.3, 0.3, 0.4],
                         "G": [4e9, 3.5e9, 3.4e9]}},
            {"name": "mat_b", "density": 1600.0,
             "elastic": {"E": [140e9, 9e9, 9e9],
                         "nu": [0.31, 0.31, 0.47],
                         "G": [4.1e9, 4.1e9, 2.7e9]}},
        ],
    }


# Per-material G13 / G23 from the mesh fixture above. Used by the
# round-trip test to recompute the expected gamma13/gamma23 = tau / G.
MAT_G13 = {0: 3.5e9, 1: 4.1e9}  # mat_a then mat_b (0-indexed by writer)
MAT_G23 = {0: 3.4e9, 1: 2.7e9}


def _meta() -> SampleMeta:
    return SampleMeta(
        sample_id=42,
        doe_seed=12345,
        git_sha_pynumad="abc123",
        git_sha_repo="def456",
        ansys_version="2023R2",
        mesh_h=0.2,
        load_case="worst_flap",
        load_scale=1.0,
        timestamp_utc="2026-05-23T20:00:00Z",
    )


def _populate_text_dir(td: Path) -> dict[str, np.ndarray]:
    """Write text files in the exact format APDL would emit. Returns the
    reference arrays so tests can compare against the read-back values."""
    rng = np.random.default_rng(0)

    elem_area = rng.uniform(0.01, 0.1, size=N_ELEM)
    elem_volume = elem_area * 0.005
    rows = [(i + 1, a, v) for i, (a, v) in enumerate(zip(elem_area, elem_volume))]
    _write_table(td / "elem_geom.txt", rows)

    # Build per-(ply, surface) field tables. Use deterministic values so
    # tests can recompute expected svm from the components.
    stress = {c: np.zeros((N_ELEM, N_PLY_MAX, N_SURF), dtype=np.float32)
              for c in STRESS_COMPONENTS}
    strain = {c: np.zeros((N_ELEM, N_PLY_MAX, N_SURF), dtype=np.float32)
              for c in STRAIN_COMPONENTS}
    svm = np.zeros((N_ELEM, N_PLY_MAX, N_SURF), dtype=np.float32)

    for ply in range(1, N_PLY_MAX + 1):
        for surf_idx, surf in enumerate(SURFACES):
            rows = []
            for ie in range(N_ELEM):
                s = [rng.normal(0, 1e7) for _ in STRESS_COMPONENTS]
                e = [rng.normal(0, 1e-3) for _ in STRAIN_COMPONENTS]
                # Reference von Mises matches reconstruct_svm exactly.
                s_arr = np.array(s, dtype=np.float64)
                ref_svm = float(reconstruct_svm(*[np.array(x) for x in s]))
                for ic, c in enumerate(STRESS_COMPONENTS):
                    stress[c][ie, ply - 1, surf_idx] = s[ic]
                for ic, c in enumerate(STRAIN_COMPONENTS):
                    strain[c][ie, ply - 1, surf_idx] = e[ic]
                svm[ie, ply - 1, surf_idx] = ref_svm
                rows.append([ie + 1, *s, *e, ref_svm])
            _write_table(td / f"field_L{ply:02d}_{surf}.txt", rows)

    disp = rng.normal(0, 0.1, size=(N_NODE, 6))
    rows = [(n + 1, *disp[n]) for n in range(N_NODE)]
    _write_table(td / "disp.txt", rows)

    sene = rng.uniform(0, 100, size=N_ELEM)
    rows = [(i + 1, sene[i]) for i in range(N_ELEM)]
    _write_table(td / "sene.txt", rows)

    react = rng.normal(0, 1e5, size=6)
    _write_table(td / "react.txt", [react.tolist()], id_first=False)

    freq = np.array([1.0, 2.5])
    eff_mass = rng.uniform(0, 1000, size=(N_MODES, 3))
    rows = [(im + 1, freq[im], *eff_mass[im]) for im in range(N_MODES)]
    _write_table(td / "modal_freq.txt", rows)

    mode_shape = rng.normal(0, 1, size=(N_MODES, N_NODE, 6))
    for im in range(N_MODES):
        rows = [(n + 1, *mode_shape[im, n]) for n in range(N_NODE)]
        _write_table(td / f"mode_shape_M{im + 1:02d}.txt", rows)

    return {
        "elem_area": elem_area,
        "elem_volume": elem_volume,
        "stress": stress,
        "strain": strain,
        "svm": svm,
        "disp": disp,
        "sene": sene,
        "react": react,
        "freq": freq,
        "eff_mass": eff_mass,
        "mode_shape": mode_shape,
    }


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_writer_reader_roundtrip(self, tmp_path: Path):
        td = tmp_path / "dumps"
        td.mkdir()
        ref = _populate_text_dir(td)

        rv_names = [f"rv_{i}" for i in range(N_RV)]
        rv_values = np.linspace(0.1, 1.0, N_RV)

        h5_path = tmp_path / "sample.h5"
        write_sample_h5(
            h5_path,
            text_dir=td,
            mesh=_make_mesh(),
            rv_names=rv_names,
            rv_values=rv_values,
            baseline_materials={"E_glass": 41.0e9},
            n_modes=N_MODES,
            n_ply_max=N_PLY_MAX,
            meta=_meta(),
            scalars={"patch_HP_SPAR_r20_svm_Pa": 1.5e8},
        )
        assert h5_path.exists()
        assert h5_path.stat().st_size > 0

        s = load_sample(h5_path)

        # Meta
        assert s.meta["sample_id"] == 42
        assert s.meta["schema_version"] == SCHEMA_VERSION
        assert s.meta["n_elem"] == N_ELEM
        assert s.meta["n_ply_max"] == N_PLY_MAX
        assert s.meta["n_modes"] == N_MODES

        # Mesh
        assert s.nodes.shape == (N_NODE, 3)
        assert s.elem_conn.shape == (N_ELEM, 4)
        # 0->1 indexing applied
        assert int(s.elem_conn.min()) == 1
        np.testing.assert_allclose(s.elem_area, ref["elem_area"], atol=1e-6)
        np.testing.assert_allclose(s.elem_volume, ref["elem_volume"], atol=1e-6)

        # Materials
        assert s.rv_names == rv_names
        np.testing.assert_allclose(s.rv_values, rv_values)
        assert s.baseline_materials == {"E_glass": 41.0e9}

        # Stress: full round-trip from text file.
        for c in STRESS_COMPONENTS:
            np.testing.assert_allclose(s.stress[c], ref["stress"][c],
                                       rtol=1e-5, atol=1.0)
        # In-plane strains: round-trip from text file.
        for c in ("eps11", "eps22", "gamma12"):
            np.testing.assert_allclose(s.strain[c], ref["strain"][c],
                                       rtol=1e-5, atol=1e-8)
        # Transverse-shear strains: writer reconstructs them as
        # gamma13 = tau13 / G13, gamma23 = tau23 / G23 using per-ply
        # G from /mesh/ply_material_id + mesh["materials"]. ANSYS
        # doesn't output EPEL,XZ/YZ for layered SHELL181.
        # ply 0 -> mat_a (G13=3.5e9), ply 1 -> mat_b (G13=4.1e9).
        for ply, g13 in MAT_G13.items():
            np.testing.assert_allclose(
                s.strain["gamma13"][:, ply, :],
                s.stress["s13"][:, ply, :] / g13,
                rtol=1e-4, atol=1e-8,
            )
        for ply, g23 in MAT_G23.items():
            np.testing.assert_allclose(
                s.strain["gamma23"][:, ply, :],
                s.stress["s23"][:, ply, :] / g23,
                rtol=1e-4, atol=1e-8,
            )
        np.testing.assert_allclose(s.svm_ply, ref["svm"],
                                   rtol=1e-5, atol=1.0)

        # Nodal
        np.testing.assert_allclose(s.disp, ref["disp"].astype(np.float32),
                                   rtol=1e-5)
        np.testing.assert_allclose(s.sene, ref["sene"].astype(np.float32),
                                   rtol=1e-5)

        # Reactions
        np.testing.assert_allclose(s.react, ref["react"], rtol=1e-6)

        # Modal
        np.testing.assert_allclose(s.freq, ref["freq"])
        np.testing.assert_allclose(s.eff_mass, ref["eff_mass"])
        np.testing.assert_allclose(s.mode_shape,
                                   ref["mode_shape"].astype(np.float32),
                                   rtol=1e-5)

        # Scalars
        assert s.scalars["patch_HP_SPAR_r20_svm_Pa"] == 1.5e8


# ---------------------------------------------------------------------------
# Schema enforcement
# ---------------------------------------------------------------------------


class TestSchemaEnforcement:
    def test_missing_disp_raises(self, tmp_path: Path):
        td = tmp_path / "dumps"
        td.mkdir()
        _populate_text_dir(td)
        (td / "disp.txt").unlink()
        with pytest.raises(FileNotFoundError, match="disp.txt"):
            write_sample_h5(
                tmp_path / "x.h5", text_dir=td, mesh=_make_mesh(),
                rv_names=["a"], rv_values=[1.0],
                baseline_materials=None, n_modes=N_MODES,
                n_ply_max=N_PLY_MAX, meta=_meta(),
            )

    def test_missing_field_files_raises(self, tmp_path: Path):
        td = tmp_path / "dumps"
        td.mkdir()
        _populate_text_dir(td)
        for f in td.glob("field_L*.txt"):
            f.unlink()
        with pytest.raises(FileNotFoundError, match="field_L"):
            write_sample_h5(
                tmp_path / "x.h5", text_dir=td, mesh=_make_mesh(),
                rv_names=["a"], rv_values=[1.0],
                baseline_materials=None, n_modes=N_MODES,
                n_ply_max=N_PLY_MAX, meta=_meta(),
            )

    def test_wrong_column_count_raises(self, tmp_path: Path):
        td = tmp_path / "dumps"
        td.mkdir()
        _populate_text_dir(td)
        # Truncate one field file: keep elem ID + only 3 columns instead of 13
        f = next(td.glob("field_L01_TOP.txt"))
        lines = f.read_text().splitlines()
        bad = [",".join(line.split(",")[:4]) for line in lines]
        f.write_text("\n".join(bad) + "\n")
        with pytest.raises(ValueError, match="13 columns"):
            write_sample_h5(
                tmp_path / "x.h5", text_dir=td, mesh=_make_mesh(),
                rv_names=["a"], rv_values=[1.0],
                baseline_materials=None, n_modes=N_MODES,
                n_ply_max=N_PLY_MAX, meta=_meta(),
            )

    def test_wrong_row_count_raises(self, tmp_path: Path):
        td = tmp_path / "dumps"
        td.mkdir()
        _populate_text_dir(td)
        f = td / "sene.txt"
        lines = f.read_text().splitlines()
        f.write_text("\n".join(lines[:-1]) + "\n")  # drop one row
        with pytest.raises(ValueError, match="sene.txt"):
            write_sample_h5(
                tmp_path / "x.h5", text_dir=td, mesh=_make_mesh(),
                rv_names=["a"], rv_values=[1.0],
                baseline_materials=None, n_modes=N_MODES,
                n_ply_max=N_PLY_MAX, meta=_meta(),
            )


# ---------------------------------------------------------------------------
# HDF5 hygiene
# ---------------------------------------------------------------------------


class TestHdf5Hygiene:
    def test_modal_rigid_body_flag(self, tmp_path: Path):
        # Phase-D loader uses /modal attrs["is_rigid_body"] to skip
        # the Lanczos near-zero ghost mode. Confirm the flag fires
        # for any |freq| < 1e-3 Hz and stays 0 above that.
        td = tmp_path / "dumps"
        td.mkdir()
        ref = _populate_text_dir(td)
        # Force mode-1 to be the rigid-body ghost.
        mf = (td / "modal_freq.txt").read_text().splitlines()
        first = mf[0].split(",")
        first[1] = f"{2.3e-8:20.12E}"
        mf[0] = ",".join(first)
        (td / "modal_freq.txt").write_text("\n".join(mf) + "\n")

        h5p = tmp_path / "s.h5"
        write_sample_h5(
            h5p, text_dir=td, mesh=_make_mesh(),
            rv_names=["a"], rv_values=[1.0],
            baseline_materials=None, n_modes=N_MODES,
            n_ply_max=N_PLY_MAX, meta=_meta(),
        )
        with h5py.File(h5p, "r") as h5:
            flags = h5["/modal"].attrs["is_rigid_body"]
            assert flags[0] == 1
            assert flags[1] == 0   # mode 2 is 2.5 Hz from fixture
            assert h5["/modal"].attrs["rigid_body_threshold_hz"] == 1e-3

    def test_stress_strain_near_zero_attrs(self, tmp_path: Path):
        # Loader uses ds.attrs['near_zero'] to skip degenerate KL
        # decompositions (SHELL181 s33 ~ 0). Confirm attrs are present
        # and reflect the data magnitude.
        td = tmp_path / "dumps"
        td.mkdir()
        _populate_text_dir(td)
        h5p = tmp_path / "s.h5"
        write_sample_h5(
            h5p, text_dir=td, mesh=_make_mesh(),
            rv_names=["a"], rv_values=[1.0],
            baseline_materials=None, n_modes=N_MODES,
            n_ply_max=N_PLY_MAX, meta=_meta(),
        )
        with h5py.File(h5p, "r") as h5:
            for c in STRESS_COMPONENTS:
                ds = h5[f"/static/stress/{c}"]
                assert "max_abs_pa" in ds.attrs
                assert "near_zero" in ds.attrs
            for c in STRAIN_COMPONENTS:
                ds = h5[f"/static/strain/{c}"]
                assert "max_abs" in ds.attrs
                assert "near_zero" in ds.attrs

    def test_datasets_have_gzip(self, tmp_path: Path):
        td = tmp_path / "dumps"
        td.mkdir()
        _populate_text_dir(td)
        h5_path = tmp_path / "sample.h5"
        write_sample_h5(
            h5_path, text_dir=td, mesh=_make_mesh(),
            rv_names=["a"], rv_values=[1.0],
            baseline_materials=None, n_modes=N_MODES,
            n_ply_max=N_PLY_MAX, meta=_meta(),
        )
        with h5py.File(h5_path, "r") as h5:
            ds = h5["/static/stress/s11"]
            assert ds.compression == "gzip"
            assert ds.chunks is not None


# ---------------------------------------------------------------------------
# vM reconstruction helper
# ---------------------------------------------------------------------------


class TestReconstructSvm:
    def test_uniaxial(self):
        # σ11 = 100 MPa, all other zero -> vM = 100 MPa
        svm = reconstruct_svm(
            np.array(1e8), np.array(0.0), np.array(0.0),
            np.array(0.0), np.array(0.0), np.array(0.0),
        )
        assert abs(float(svm) - 1e8) / 1e8 < 1e-6

    def test_pure_shear(self):
        # τ12 = 50 MPa, all normals zero -> vM = sqrt(3) * τ
        svm = reconstruct_svm(
            np.array(0.0), np.array(0.0), np.array(0.0),
            np.array(5e7), np.array(0.0), np.array(0.0),
        )
        expected = np.sqrt(3.0) * 5e7
        assert abs(float(svm) - expected) / expected < 1e-6

    def test_zero(self):
        svm = reconstruct_svm(*[np.array(0.0)] * 6)
        assert float(svm) == 0.0


# ---------------------------------------------------------------------------
# APDL emitter strings
# ---------------------------------------------------------------------------


class TestApdlEmitter:
    def test_static_block_has_one_field_block_per_ply_surface(self):
        s = emit_static_fields("/tmp/out", n_ply_max=4)
        # 4 plies × 3 surfaces = 12 blocks
        assert s.count("field block: layer=") == 12
        # one disp / sene / react / elem_geom block each
        assert s.count("---- nodal displacement") == 1
        assert s.count("---- element strain energy") == 1
        assert s.count("---- root reactions") == 1
        assert s.count("---- element geometry") == 1

    def test_static_block_uses_etable_for_all_stress_components(self):
        s = emit_static_fields("/tmp/out", n_ply_max=1)
        for comp_label in ("S,X", "S,Y", "S,Z", "S,XY", "S,XZ", "S,YZ"):
            assert comp_label in s, f"missing {comp_label}"
        for comp_label in ("EPEL,X", "EPEL,Y", "EPEL,XY",
                            "EPEL,XZ", "EPEL,YZ"):
            assert comp_label in s, f"missing {comp_label}"

    def test_modal_block_writes_per_mode_shape(self):
        s = emit_modal_fields("/tmp/out", n_modes=3)
        assert "mode_shape_M01" in s
        assert "mode_shape_M02" in s
        assert "mode_shape_M03" in s
        assert "modal_freq" in s

    def test_field_filenames_use_two_digit_layer(self):
        s = emit_static_fields("/tmp/out", n_ply_max=12)
        assert "field_L01_TOP" in s
        assert "field_L12_BOT" in s

    def test_static_block_sets_layer_coord_system(self):
        # Composite stress extraction MUST use RSYS,LSYS so the
        # components map to the laminate (fibre / transverse /
        # through-thickness) frame; element CS / global CS would
        # invalidate every composite failure criterion downstream.
        s = emit_static_fields("/tmp/out", n_ply_max=2)
        assert "RSYS,LSYS" in s
        # Per-block reassertion guards against any earlier command
        # perturbing RSYS.
        assert s.count("RSYS,LSYS") >= 2 * 3 + 1  # 2 plies x 3 surfaces + header
