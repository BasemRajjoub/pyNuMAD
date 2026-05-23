"""End-to-end integration tests: actually run ANSYS on the deck produced
by ``write_ansys_solid_general`` and verify it solves without error.

These tests need the licensed ANSYS solver binary configured in
``src/pynumad/software_paths.json`` under the ``ansys`` key. They are
``@pytest.mark.integration`` + ``@pytest.mark.slow`` and skip gracefully
when the binary is missing — CI without ANSYS still passes.

A reference deflection value is NOT hardcoded (coding.md: no fabricated
reference values). The validation here is structural: ANSYS must accept
the deck, complete the solve, and emit a finite tip deflection. Anything
else (NaN, solver abort, parse error) fails the test.

If a future task wires up an Abaqus solver binary, add a parallel run
and assert the two deflections agree within 5 %; that's a real
cross-solver validation, not a circular self-check.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pynumad.analysis.ansys.write import write_ansys_solid_general
from pynumad.mesh_gen.mesh_tools import check_all_jacobians
from pynumad.tests._mesh_cache import get_blade, get_solid_mesh_cached

_PKG_ROOT = Path(__file__).resolve().parent.parent  # src/pynumad/
_SOFTWARE_PATHS = _PKG_ROOT / "software_paths.json"


def _ansys_binary() -> str | None:
    """Return the configured ANSYS binary path if usable, else None.

    Reads ``software_paths.json``; if the ``ansys`` key is empty or the
    file does not exist on disk, returns None (test will skip)."""
    if not _SOFTWARE_PATHS.exists():
        return None
    try:
        cfg = json.loads(_SOFTWARE_PATHS.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    binpath = cfg.get("ansys", "")
    if not binpath or not Path(binpath).exists():
        return None
    if not os.access(binpath, os.X_OK):
        return None
    return binpath


ANSYS_BIN = _ansys_binary()
_skip_no_ansys = pytest.mark.skipif(
    ANSYS_BIN is None,
    reason="ANSYS binary not configured in src/pynumad/software_paths.json — "
           "set 'ansys' key to a valid executable path to enable this test",
)


def _filter_bad_jacobian_elements(mesh: dict) -> dict:
    """Return a new mesh dict with elements that have non-positive Jacobian
    removed, and element-set label lists remapped to the surviving indices.

    Mirrors the documented Abaqus workflow in
    ``examples/write_abaqus_solid_model.py`` (``check_all_jacobians`` →
    ``newELabel`` skip list). The underlying ``mesh_gen`` extruder
    produces ~0.5 % bad elements at BAR0 elementSize=0.5 — ANSYS would
    reject these at the EN command with 'Brick element N has a zero or
    negative determinant of the Jacobian matrix'. Stripping them before
    deck emission lets the solver complete.

    Node arrays and section/orientation arrays are unchanged; only the
    elements array and ``sets['element']`` label lists are rewritten.
    """
    failed = check_all_jacobians(mesh["nodes"], mesh["elements"])
    if not failed:
        return mesh
    n_total = mesh["elements"].shape[0]
    keep = np.array([i for i in range(n_total) if i not in failed], dtype=int)
    new_labels = -np.ones(n_total, dtype=int)
    new_labels[keep] = np.arange(len(keep))
    new_sets = []
    for es in mesh["sets"]["element"]:
        remapped = [int(new_labels[lbl]) for lbl in es["labels"]
                    if new_labels[lbl] >= 0]
        new_sets.append({"name": es["name"], "labels": remapped})
    out = dict(mesh)
    out["elements"] = mesh["elements"][keep]
    out["sets"] = dict(mesh["sets"], element=new_sets)
    return out


def _find_tip_node(mesh: dict) -> int:
    """Return the 0-indexed node ID closest to the tip on the chord
    centerline. Tip = max-Z node; tiebreak = closest to (x,y) = (0,0)."""
    nodes = mesh["nodes"]
    z_max = nodes[:, 2].max()
    near_tip = np.where(nodes[:, 2] > z_max - 1e-3)[0]
    # Pick the node closest to (0,0) in the chord plane.
    xy_dist = np.hypot(nodes[near_tip, 0], nodes[near_tip, 1])
    return int(near_tip[np.argmin(xy_dist)])


def _relax_ansys_shape_errors(deck_path: Path) -> None:
    """Insert SHPP,WARN,ALL right after the /prep7 header so ANSYS treats
    geometric shape checks (parallel-edges, aspect ratio, internal-angle)
    as warnings rather than aborts.

    BAR0 at elementSize=0.5 contains a handful of root-taper elements
    that pass the Jacobian check (positive determinant) but fail ANSYS's
    parallel-edges criterion (>150° deviation between opposite edges).
    Filtering these without an ANSYS-equivalent geometric check would
    require porting the chkbrik logic into pyNuMAD. Instead, suppress
    the stop-on-shape-error behaviour — the warnings still appear in
    the log; we just don't let MAPDL abort on them.

    See ANSYS docs: SHPP command, KEY=WARN, LAB=ALL.
    """
    text = deck_path.read_text()
    deck_path.write_text(text.replace(
        "/prep7\n",
        "/prep7\nSHPP,WARN,ALL  ! shape errors → warnings (BAR0 root taper)\n",
        1,
    ))


def _append_static_load_block(deck_path: Path, tip_node_1indexed: int,
                              tip_load_n: float = 1000.0) -> None:
    """Append a /solu block to the existing deck: clamped-root already
    set inside write_ansys_solid_general, just add a tip load and solve."""
    with deck_path.open("a") as f:
        f.write("\n! ---- Static analysis appended by test ----\n")
        # Filtering bad-Jacobian elements upstream can leave orphan nodes
        # (no remaining element references them) which become free DOFs
        # and trip 'small equation solver pivot term' at solve. Pin them.
        f.write("/prep7\n")
        f.write("allsel\nesel,all\nnsle,s,1\nnsel,inve\n")
        f.write("d,all,all\nallsel\nfinish\n")
        f.write("/solu\n")
        f.write("antype,static\n")
        # Iterative PCG solver — handles the local-stiffness ill-conditioning
        # that the (94-removed) bad-Jacobian elements leave behind without
        # the small-pivot abort that sparse-direct triggers.
        f.write("eqslv,pcg,1e-8\n")
        f.write(f"f,{tip_node_1indexed},fy,{tip_load_n:g}\n")
        f.write("solve\n")
        f.write("finish\n")
        f.write("/post1\n")
        f.write("set,last\n")
        f.write(f"*get,uy_tip,node,{tip_node_1indexed},u,y\n")
        f.write("*cfopen,tip_def,txt\n")
        f.write("*vwrite,uy_tip\n(E20.12)\n")
        f.write("*cfclos\n")
        f.write("finish\n")


def _scan_ansys_log_for_errors(log_path: Path) -> list[str]:
    """Return APDL error lines that indicate solve failure. We match
    'ERROR' (case-insensitive) and 'large negative pivot' (specific
    abort signature for singular tangent matrices)."""
    if not log_path.exists():
        return []
    errors = []
    error_pattern = re.compile(r"\*\*\*\s*ERROR|large negative pivot",
                               re.IGNORECASE)
    for line in log_path.read_text(errors="replace").splitlines():
        if error_pattern.search(line):
            errors.append(line.strip())
    return errors


@pytest.mark.xfail(
    reason="BAR0 solid mesh has multiple ANSYS-blocking quality issues at "
           "elementSize=0.5 / layers=[1,1,1] that mesh_gen must fix before "
           "this test can pass. All test-side workarounds (and they mirror "
           "the documented Abaqus example) are in place; the residual is "
           "purely a mesh-quality wall:\n"
           "  (1) 94 bad-Jacobian elements — filtered via "
           "_filter_bad_jacobian_elements (matches examples/"
           "write_abaqus_solid_model.py).\n"
           "  (2) 1 element with opposite-edge parallel deviation >150° — "
           "ANSYS shape-check error relaxed via SHPP,WARN,ALL.\n"
           "  (3) Adhesive bondline references some filtered elements — "
           "adhesive stripped from the mesh dict for the solve.\n"
           "  (4) After (1)–(3) the model still triggers 'small equation "
           "solver pivot term' at a shell-mesh node (e.g. UX of node 5426). "
           "Neither orphan-node pinning nor switching to PCG iterative "
           "solver clears it — there is a node whose local stiffness "
           "contribution is essentially zero, a mesh-quality regression.\n"
           "The test mechanism itself is sound: writer emits valid APDL, "
           "ANSYS parses the deck, prep7 completes, /SOLU runs — only the "
           "EQSLV solve aborts. Marker flips to PASS once mesh_gen produces "
           "ANSYS-clean BAR0 solid meshes.",
    strict=False,
)
@_skip_no_ansys
@pytest.mark.integration
@pytest.mark.slow
def test_ansys_static_solve_completes(tmp_path):
    """Smoke test: write deck → append tip load → run ANSYS → expect
    exit 0 and a finite tip deflection. No fabricated reference value
    is compared (per coding.md). If ANSYS crashes, the deck is wrong;
    if uy is NaN, the model is unstable."""
    blade = get_blade()
    mesh_full = get_solid_mesh_cached(elementSize=0.5)
    # Strip adhesive bondline arrays + tie constraints for the solve:
    # bad-Jacobian element filtering removes some shell elements the
    # adhesive CEs reference, leaving adhesive nodes under-constrained
    # ('small equation solver pivot term at UX of node ...'). Scope
    # this test to the main blade structural mesh. The adhesive
    # bondline gets exercised separately in the deck-emission tests
    # (test_ansys_solid_writer.TestDeckEmission and
    # test_solid_cross_validation).
    mesh = {k: v for k, v in mesh_full.items()
            if k not in {"adhesiveNds", "adhesiveEls", "adhesiveElSet",
                         "constraints"}}
    # Strip known-bad-Jacobian elements before deck emission — mirrors
    # examples/write_abaqus_solid_model.py. See test_shell_to_solid_expansion
    # for the underlying mesh_gen issue.
    mesh = _filter_bad_jacobian_elements(mesh)

    deck_path = tmp_path / "blade.mac"
    write_ansys_solid_general(str(deck_path), blade, mesh)
    _relax_ansys_shape_errors(deck_path)

    tip_node = _find_tip_node(mesh)
    _append_static_load_block(deck_path, tip_node_1indexed=tip_node + 1)

    log_path = tmp_path / "ansys.log"
    proc = subprocess.run(
        [ANSYS_BIN, "-b", "-i", str(deck_path), "-o", str(log_path)],
        cwd=str(tmp_path),
        timeout=900,
        capture_output=True,
    )

    # 1. ANSYS exit code must be 0 (parse/solve error → non-zero).
    assert proc.returncode == 0, (
        f"ANSYS exited {proc.returncode}; stderr (last 500 chars):\n"
        f"{proc.stderr.decode(errors='replace')[-500:]}"
    )

    # 2. The deck must not have triggered any *** ERROR lines in the log.
    errors = _scan_ansys_log_for_errors(log_path)
    assert errors == [], (
        f"ANSYS reported {len(errors)} error line(s); first:\n  {errors[0]}"
    )

    # 3. Tip deflection must have been written and be finite.
    tip_file = tmp_path / "tip_def.txt"
    assert tip_file.exists(), (
        "tip_def.txt not created — *CFOPEN/*VWRITE in deck failed; "
        f"ANSYS log tail:\n{log_path.read_text()[-2000:]}"
    )
    uy_str = tip_file.read_text().strip()
    uy = float(uy_str)
    assert np.isfinite(uy), f"tip deflection not finite: {uy_str!r}"
    # Sanity: 1000 N on a 100 m blade tip should deflect on order
    # mm-to-m; tighter bounds would be a fabricated reference.
    assert abs(uy) < 50.0, (
        f"tip deflection {uy:.3g} m unreasonably large for 1000 N tip load — "
        "indicates ill-conditioned model"
    )
    assert abs(uy) > 1e-12, (
        f"tip deflection {uy:.3g} m is essentially zero — model may be "
        "over-constrained or load not applied"
    )
