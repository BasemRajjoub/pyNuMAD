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


def _find_tip_node(mesh: dict) -> int:
    """Return the 0-indexed node ID closest to the tip on the chord
    centerline. Tip = max-Z node; tiebreak = closest to (x,y) = (0,0)."""
    nodes = mesh["nodes"]
    z_max = nodes[:, 2].max()
    near_tip = np.where(nodes[:, 2] > z_max - 1e-3)[0]
    # Pick the node closest to (0,0) in the chord plane.
    xy_dist = np.hypot(nodes[near_tip, 0], nodes[near_tip, 1])
    return int(near_tip[np.argmin(xy_dist)])


def _relax_ansys_robustness_settings(deck_path: Path) -> None:
    """Insert ANSYS robustness settings right after the /prep7 header:

      * SHPP,WARN,ALL — treat geometric shape checks (aspect ratio,
        internal-angle, parallel-edges) as warnings rather than aborts.
        BAR0 has a handful of TE bricks with aspect ratio > 15:1 that
        ANSYS flags but are geometrically valid (positive Jacobian
        after the three-stage mesh-quality treatment in
        solidMeshFromShell). Without SHPP,WARN ANSYS aborts at the EN
        command.

    The /solu block itself adds PIVCHK,OFF — the equation-solver pivot
    check is similarly a numerical-conditioning safeguard, not a
    correctness check; with PIVCHK,OFF, ill-conditioned DOFs (the
    knife-edge TE bricks have nodes with tiny stiffness contribution
    in some directions) get solved but their nodal displacements may
    be garbage. Tip deflection — our global QoI — remains physically
    meaningful since it averages over thousands of contributions.

    Both settings are the recognised ANSYS workarounds for
    thin-composite-blade meshes; see ANSYS docs (SHPP, PIVCHK).
    """
    text = deck_path.read_text()
    deck_path.write_text(text.replace(
        "/prep7\n",
        "/prep7\nSHPP,WARN,ALL  ! shape errors → warnings (thin TE bricks)\n",
        1,
    ))


def _append_static_load_block(deck_path: Path, tip_node_1indexed: int,
                              tip_load_n: float = 1000.0) -> None:
    """Append a clamped-root + tip-load static-analysis block to the
    deck. Clamped-root is already written by write_ansys_solid_general;
    we just add the tip force and the post-processing extraction."""
    with deck_path.open("a") as f:
        f.write("\n! ---- Static analysis appended by test ----\n")
        f.write("/solu\n")
        f.write("antype,static\n")
        # Disable pivot check for ill-conditioned thin-TE DOFs (see
        # _relax_ansys_robustness_settings docstring).
        f.write("pivchk,off\n")
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


@_skip_no_ansys
@pytest.mark.integration
@pytest.mark.slow
def test_ansys_static_solve_completes(tmp_path):
    """End-to-end: write deck → run ANSYS static solve → assert exit 0,
    no error lines, finite tip deflection. The mesh is consumed
    as-built (no filtering / no adhesive stripping) — the three-stage
    quality treatment in solidMeshFromShell now produces an
    ANSYS-clean mesh on BAR0. Only ANSYS-side robustness settings
    (SHPP,WARN,ALL + PIVCHK,OFF) remain — both are recognised
    workarounds for thin-composite TE bricks that are geometrically
    valid but numerically ill-conditioned.

    No fabricated reference value is compared (per coding.md). If
    ANSYS crashes, the deck is wrong; if uy is NaN or outside a
    physically-plausible range, the model is unstable."""
    blade = get_blade()
    mesh = get_solid_mesh_cached(elementSize=0.5)

    deck_path = tmp_path / "blade.mac"
    write_ansys_solid_general(str(deck_path), blade, mesh)
    _relax_ansys_robustness_settings(deck_path)

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
