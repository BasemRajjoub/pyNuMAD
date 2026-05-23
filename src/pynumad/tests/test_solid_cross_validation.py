"""Cross-validation between the Abaqus and ANSYS 3D-solid writers.

Both writers consume the same ``bladeMesh`` dict from ``get_solid_mesh``.
This test suite writes both decks, parses them with small in-file
parsers, and asserts the extracted quantities match exactly (counts,
material names, element sets) or to floating-point tolerance
(material constants, orientation vectors). No FEA solver required.

A mismatch here means one writer dropped or duplicated structural data
relative to the other — a divergence bug that the deck-only unit tests
in test_ansys_solid_writer.py / write_solid_general would not catch.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pytest

from pynumad.analysis.abaqus.write import write_solid_general
from pynumad.analysis.ansys.write import write_ansys_solid_general
from pynumad.tests._mesh_cache import get_blade, get_solid_mesh_cached

# ---------------------------------------------------------------------------
# Parsers — extract structural quantities from each format
# ---------------------------------------------------------------------------

def _parse_abaqus_solid(inp_path: str) -> dict:
    """Return n_nodes, n_hex, n_wedge, mat_names (ordered list, dedup),
    mat_ex_by_name, mat_density_by_name, xdir/xydir per *Orientation,
    and elset sizes by name. Tolerant to mixed-case Abaqus keywords."""
    out = {
        "n_nodes": 0, "n_hex": 0, "n_wedge": 0,
        "mat_names": [], "mat_ex": {}, "mat_density": {},
        "xdir": [], "xydir": [],
        "elset_sizes": defaultdict(int),
    }
    with open(inp_path) as f:
        lines = [L.rstrip("\n") for L in f]

    section = None             # current *KEYWORD context
    current_mat = None
    expect_density = False
    expect_elastic = False
    current_elset = None
    expect_orient_data = False
    last_orientation_xdir = None

    for raw in lines:
        L = raw.strip()
        if not L:
            continue
        if L.startswith("*"):
            # Switch context. Lowercase token before any comma.
            head = L.split(",", 1)[0].lower().strip()
            section = head
            expect_density = False
            expect_elastic = False
            if head == "*node":
                continue
            if head == "*element":
                # *Element, type=C3D8I  or  type=C3D6
                m = re.search(r"type\s*=\s*([A-Za-z0-9]+)", L, re.I)
                etype = m.group(1).upper() if m else ""
                section = ("element", etype)
                continue
            if head == "*elset":
                m = re.search(r"elset\s*=\s*([^\s,]+)", L, re.I)
                current_elset = m.group(1) if m else None
                continue
            if head == "*material":
                m = re.search(r"name\s*=\s*([^\s,]+)", L, re.I)
                current_mat = m.group(1) if m else None
                if current_mat and current_mat not in out["mat_names"]:
                    out["mat_names"].append(current_mat)
                continue
            if head == "*density":
                expect_density = True
                continue
            if head == "*elastic":
                expect_elastic = True
                continue
            if head == "*orientation":
                expect_orient_data = True
                last_orientation_xdir = None
                continue
            # All other *keywords: leave section set for skip behaviour.
            continue

        # Data lines for the current section.
        if section == "*node":
            # id, x, y, z
            parts = [p.strip() for p in L.split(",")]
            if len(parts) >= 4:
                out["n_nodes"] += 1
            continue
        if isinstance(section, tuple) and section[0] == "element":
            etype = section[1]
            if etype.startswith("C3D8"):
                out["n_hex"] += 1
            elif etype.startswith("C3D6"):
                out["n_wedge"] += 1
            continue
        if section == "*elset" and current_elset is not None:
            # Comma-separated element IDs, possibly multiple per line.
            parts = [p.strip() for p in L.split(",") if p.strip()]
            out["elset_sizes"][current_elset] += len(parts)
            continue
        if expect_density:
            val = float(L.split(",")[0])
            out["mat_density"][current_mat] = val
            expect_density = False
            continue
        if expect_elastic:
            # First data line: 8 fields (E1, E2, E3, nu12, nu13, nu23, G12, G13)
            parts = [p.strip() for p in L.split(",") if p.strip()]
            if parts:
                out["mat_ex"][current_mat] = float(parts[0])
            expect_elastic = False
            continue
        if expect_orient_data:
            parts = [p.strip() for p in L.split(",") if p.strip()]
            if last_orientation_xdir is None and len(parts) >= 6:
                # First data line: a1, a2, a3, b1, b2, b3
                xd = [float(p) for p in parts[:3]]
                yd = [float(p) for p in parts[3:6]]
                out["xdir"].append(xd)
                out["xydir"].append(yd)
                last_orientation_xdir = xd
            else:
                # Second data line ("3, 0.") — discard, end of block.
                expect_orient_data = False
            continue

    out["xdir"] = np.array(out["xdir"]) if out["xdir"] else np.zeros((0, 3))
    out["xydir"] = np.array(out["xydir"]) if out["xydir"] else np.zeros((0, 3))
    out["elset_sizes"] = dict(out["elset_sizes"])
    return out


def _parse_ansys_solid(mac_path: str) -> dict:
    """Return analogous dict for the APDL deck. Element counts split
    hex vs degenerate-brick wedge by inspecting EN connectivity."""
    out = {
        "n_nodes": 0, "n_hex": 0, "n_wedge": 0,
        "mat_names": [], "mat_ex": {}, "mat_density": {},
        "xdir": [], "xydir": [],
        "elset_sizes": defaultdict(int),
    }
    # APDL has no named *Material blocks — we identify materials by the
    # comment line "! <name>" immediately preceding the mp,ex,<matid>
    # command, and by the matid integer assigned in insertion order.
    matid_to_name: dict[int, str] = {}
    mp_ex_by_matid: dict[int, float] = {}
    mp_dens_by_matid: dict[int, float] = {}

    # CSKP order matches section order. K commands give us the keypoint
    # coordinates, then CSKP references them to define the csys.
    kp_coords: dict[int, tuple] = {}
    cskp_records: list[tuple] = []   # (csys_id, kp_o, kp_x, kp_y)

    last_comment = None
    section_assign_lines: list[tuple[int, int]] = []   # (matid, csys_id)
    section_member_counts: list[int] = []  # parallel to sections, counts ESEL members

    # Track ESEL accumulation between EMODIF blocks (a section).
    current_esel_count = 0
    inside_esel = False

    with open(mac_path) as f:
        for raw in f:
            L = raw.strip()
            if not L:
                continue

            if L.startswith("!"):
                # Capture the most recent comment for material naming.
                last_comment = L.lstrip("!").strip()
                continue

            head = L.split(",", 1)[0].lower()

            if head == "n":
                out["n_nodes"] += 1
                continue
            if head == "en":
                # EN,eid,n1,n2,n3,n4,n5,n6,n7,n8 — degenerate brick has
                # n4==n3 and n8==n7 for wedges.
                parts = [p.strip() for p in L.split(",")]
                if len(parts) == 10:
                    n3, n4 = parts[4], parts[5]
                    n7, n8 = parts[8], parts[9]
                    if n3 == n4 and n7 == n8:
                        out["n_wedge"] += 1
                    else:
                        out["n_hex"] += 1
                continue
            if head == "k":
                # K,kp_id,x,y,z
                parts = [p.strip() for p in L.split(",")]
                if len(parts) >= 5:
                    kp_id = int(parts[1])
                    coords = (float(parts[2]), float(parts[3]), float(parts[4]))
                    kp_coords[kp_id] = coords
                continue
            if head == "cskp":
                # CSKP,csys_id,kcs,kp_o,kp_x,kp_y
                parts = [p.strip() for p in L.split(",")]
                if len(parts) >= 6:
                    csys_id = int(parts[1])
                    kp_o = int(parts[3])
                    kp_x = int(parts[4])
                    kp_y = int(parts[5])
                    cskp_records.append((csys_id, kp_o, kp_x, kp_y))
                continue
            if head == "mp":
                # mp,ex,<matid>,<val>  or  mp,dens,<matid>,<val>  ...
                parts = [p.strip() for p in L.split(",")]
                if len(parts) >= 4:
                    prop = parts[1].lower()
                    matid = int(parts[2])
                    val = float(parts[3])
                    if prop == "ex":
                        mp_ex_by_matid[matid] = val
                        # The comment immediately preceding mp,ex IS the
                        # material name (mirrors shell writer convention).
                        if matid not in matid_to_name and last_comment:
                            matid_to_name[matid] = last_comment
                    elif prop == "dens":
                        mp_dens_by_matid[matid] = val
                continue
            if head == "esel":
                # ESEL,S,ELEM,,<n> starts a new accumulation; ESEL,A
                # appends. Reset count on S, increment on either.
                op = L.split(",")[1].strip().upper() if "," in L else ""
                if op == "S":
                    current_esel_count = 1
                    inside_esel = True
                elif op == "A" and inside_esel:
                    current_esel_count += 1
                continue
            if head == "emodif":
                # EMODIF,ALL,MAT,<matid>  or  EMODIF,ALL,ESYS,<csys_id>
                parts = [p.strip() for p in L.split(",")]
                if len(parts) >= 4 and parts[1].upper() == "ALL":
                    prop = parts[2].upper()
                    val = int(parts[3])
                    if prop == "MAT":
                        section_assign_lines.append((val, None))
                    elif prop == "ESYS":
                        if section_assign_lines and section_assign_lines[-1][1] is None:
                            matid, _ = section_assign_lines[-1]
                            section_assign_lines[-1] = (matid, val)
                            # Snapshot the current ESEL count as this section's size.
                            section_member_counts.append(current_esel_count)
                            inside_esel = False
                            current_esel_count = 0
                continue

    # Resolve xDir/xyDir from CSKP keypoints (origin at KP_o, xDir along
    # KP_o → KP_x, xyDir along KP_o → KP_y).
    for _csys_id, kp_o, kp_x, kp_y in cskp_records:
        o = kp_coords.get(kp_o)
        x = kp_coords.get(kp_x)
        y = kp_coords.get(kp_y)
        if o is None or x is None or y is None:
            continue
        xd = (x[0] - o[0], x[1] - o[1], x[2] - o[2])
        yd = (y[0] - o[0], y[1] - o[1], y[2] - o[2])
        out["xdir"].append(xd)
        out["xydir"].append(yd)
    out["xdir"] = np.array(out["xdir"]) if out["xdir"] else np.zeros((0, 3))
    out["xydir"] = np.array(out["xydir"]) if out["xydir"] else np.zeros((0, 3))

    # Material name table — sorted by insertion order (matid 1..N).
    n_mats = max(mp_ex_by_matid.keys()) if mp_ex_by_matid else 0
    out["mat_names"] = [matid_to_name.get(i, f"mat_{i}")
                        for i in range(1, n_mats + 1)]
    out["mat_ex"] = {matid_to_name.get(i, f"mat_{i}"): v
                     for i, v in mp_ex_by_matid.items()}
    out["mat_density"] = {matid_to_name.get(i, f"mat_{i}"): v
                          for i, v in mp_dens_by_matid.items()}
    out["section_member_counts"] = section_member_counts
    return out


# ---------------------------------------------------------------------------
# Fixture: build both decks once per module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def both_decks(tmp_path_factory):
    import copy

    blade = get_blade()
    mesh_full = get_solid_mesh_cached(elementSize=0.5)
    # Strip the adhesive bondline before either writer call so the deck
    # comparison is apples-to-apples: Abaqus write_solid_general does
    # NOT emit adhesive (it lives in a separate *Part in the example
    # script), but the ANSYS writer's _write_ansys_adhesive() helper
    # would otherwise add extra nodes/elements. Counts only match when
    # both writers see the same mesh.
    mesh = {k: v for k, v in mesh_full.items()
            if k not in {"adhesiveNds", "adhesiveEls", "adhesiveElSet",
                         "constraints"}}
    out_dir = tmp_path_factory.mktemp("cross")
    abq_path = str(out_dir / "blade.inp")
    ans_path = str(out_dir / "blade.mac")
    # Abaqus write_solid_general mutates wedge rows in place
    # (el[0:6] = el[0:6] + 1 — known quirk of the existing writer).
    # Deep-copy before passing so the shared mesh stays clean.
    write_solid_general(abq_path, _build_mesh_dict_for_abaqus(copy.deepcopy(mesh), blade))
    write_ansys_solid_general(ans_path, blade, mesh)
    abq = _parse_abaqus_solid(abq_path)
    ans = _parse_ansys_solid(ans_path)
    return blade, mesh, abq, ans


def _build_mesh_dict_for_abaqus(mesh: dict, blade) -> dict:
    """Abaqus ``write_solid_general`` expects a ``materials`` key (list
    of dicts) inside the mesh dict; the ANSYS writer reads from
    ``blade.definition.materials`` directly. Build the Abaqus-flavoured
    dict from the canonical mesh + blade pair so both writers see the
    same underlying data."""
    mat_dicts = []
    for name, mat in blade.definition.materials.items():
        if mat.type == "isotropic":
            nu_xy = mat.prxy
            mat_dicts.append({
                "name": name, "density": float(mat.density),
                "elastic": {
                    "E": [float(mat.ex), float(mat.ex), float(mat.ex)],
                    "nu": [nu_xy, nu_xy, nu_xy],
                    "G": [float(mat.ex) / (2 * (1 + nu_xy))] * 3,
                },
            })
        else:
            mat_dicts.append({
                "name": name, "density": float(mat.density),
                "elastic": {
                    "E": [float(mat.ex), float(mat.ey), float(mat.ez)],
                    "nu": [float(mat.prxy), float(mat.prxz), float(mat.pryz)],
                    "G": [float(mat.gxy), float(mat.gxz), float(mat.gyz)],
                },
            })
    return {**mesh, "materials": mat_dicts}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_node_count_match(both_decks):
    _, mesh, abq, ans = both_decks
    expected = mesh["nodes"].shape[0]
    assert abq["n_nodes"] == expected, (
        f"Abaqus parser saw {abq['n_nodes']} nodes vs mesh {expected}"
    )
    assert ans["n_nodes"] == expected, (
        f"ANSYS parser saw {ans['n_nodes']} nodes vs mesh {expected}"
    )


def test_hex_count_match(both_decks):
    _, _, abq, ans = both_decks
    assert abq["n_hex"] == ans["n_hex"], (
        f"hex count diverges: abq={abq['n_hex']} ans={ans['n_hex']}"
    )


def test_wedge_count_match(both_decks):
    _, _, abq, ans = both_decks
    assert abq["n_wedge"] == ans["n_wedge"], (
        f"wedge count diverges: abq={abq['n_wedge']} ans={ans['n_wedge']}"
    )


def test_total_element_count_matches_mesh(both_decks):
    _, mesh, abq, ans = both_decks
    expected = mesh["elements"].shape[0]
    assert abq["n_hex"] + abq["n_wedge"] == expected, (
        f"Abaqus wrote {abq['n_hex']+abq['n_wedge']} elements vs mesh {expected}"
    )
    assert ans["n_hex"] + ans["n_wedge"] == expected, (
        f"ANSYS wrote {ans['n_hex']+ans['n_wedge']} elements vs mesh {expected}"
    )


def test_material_names_match(both_decks):
    _, _, abq, ans = both_decks
    assert set(abq["mat_names"]) == set(ans["mat_names"]), (
        f"material name sets diverge:\n  Abaqus only: "
        f"{set(abq['mat_names']) - set(ans['mat_names'])}\n  ANSYS only: "
        f"{set(ans['mat_names']) - set(abq['mat_names'])}"
    )


def test_material_ex_match(both_decks):
    _, _, abq, ans = both_decks
    common = set(abq["mat_ex"]).intersection(set(ans["mat_ex"]))
    assert common, "no common materials between Abaqus and ANSYS decks"
    for name in common:
        # Both writers use %g formatting (6 sig figs) → tolerance 1e-5
        # relative is the floor; tighter would be format-noise.
        a = abq["mat_ex"][name]
        b = ans["mat_ex"][name]
        rel = abs(a - b) / max(abs(a), abs(b), 1e-30)
        assert rel < 1e-5, (
            f"material {name} EX: abq={a:g} ans={b:g} (rel diff {rel:.2e})"
        )


def test_material_density_match(both_decks):
    _, _, abq, ans = both_decks
    common = set(abq["mat_density"]).intersection(set(ans["mat_density"]))
    assert common, "no common materials between Abaqus and ANSYS decks"
    for name in common:
        a = abq["mat_density"][name]
        b = ans["mat_density"][name]
        rel = abs(a - b) / max(abs(a), abs(b), 1e-30)
        assert rel < 1e-5, (
            f"material {name} DENS: abq={a:g} ans={b:g} (rel diff {rel:.2e})"
        )


def test_orientation_section_count_match(both_decks):
    _, mesh, abq, ans = both_decks
    expected = len(mesh["sections"])
    assert abq["xdir"].shape[0] == expected, (
        f"Abaqus has {abq['xdir'].shape[0]} orientations vs {expected} sections"
    )
    assert ans["xdir"].shape[0] == expected, (
        f"ANSYS has {ans['xdir'].shape[0]} orientations vs {expected} sections"
    )


def test_orientation_xdir_match(both_decks):
    _, _, abq, ans = both_decks
    # Order: both writers iterate sections in the same order, so row i
    # in xdir corresponds to section i in both files.
    assert abq["xdir"].shape == ans["xdir"].shape, (
        f"xdir shape mismatch: abq={abq['xdir'].shape} ans={ans['xdir'].shape}"
    )
    np.testing.assert_allclose(
        abq["xdir"], ans["xdir"], atol=1e-5, rtol=1e-5,
        err_msg="xDir vectors diverge between Abaqus *Orientation and ANSYS CSKP",
    )


def test_orientation_xydir_match(both_decks):
    _, _, abq, ans = both_decks
    assert abq["xydir"].shape == ans["xydir"].shape
    np.testing.assert_allclose(
        abq["xydir"], ans["xydir"], atol=1e-5, rtol=1e-5,
        err_msg="xyDir vectors diverge between Abaqus *Orientation and ANSYS CSKP",
    )


def test_section_member_counts_match_mesh(both_decks):
    _, mesh, _, ans = both_decks
    # ANSYS deck's section_member_counts = list of ESEL accumulation sizes
    # per EMODIF block. Compare against the source mesh's per-section
    # element-set sizes (parallel iteration order).
    expected = []
    set_by_name = {s["name"]: len(s["labels"])
                   for s in mesh["sets"]["element"]}
    for sec in mesh["sections"]:
        expected.append(set_by_name[sec["elementSet"]])
    assert ans["section_member_counts"] == expected, (
        "ANSYS ESEL-accumulated section sizes diverge from mesh element sets "
        f"(first 5 expected: {expected[:5]}, got: {ans['section_member_counts'][:5]})"
    )
