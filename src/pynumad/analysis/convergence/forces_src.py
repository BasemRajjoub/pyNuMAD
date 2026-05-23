"""Compute section internal forces / moments from an ANSYS ``forces.src``.

The ANSYS-side FSUM approach over a selected element set returns zero
in a converged static problem (Newton's 3rd law cancels internal
contributions at interior nodes). The correct definition of the
internal section force at z = r is simply the moment of all external
loads applied outboard of z = r — by the equilibrium of the outboard
segment.

We compute this in Python directly from the ``forces.src`` file that
the runner writes for ANSYS. The file is a series of

    f,<node_id>,FX,<value>
    f,<node_id>,FY,<value>
    f,<node_id>,FZ,<value>

lines (one DOF per line). Node IDs are 1-indexed ANSYS labels.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np


_F_LINE = re.compile(
    r"^\s*f\s*,\s*(\d+)\s*,\s*(FX|FY|FZ)\s*,\s*([+\-0-9.eE]+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def read_forces_src(path: str | Path, n_nodes: int) -> np.ndarray:
    """Parse a forces.src into an ``(n_nodes, 3)`` applied-load array.

    ``n_nodes`` is the global node count of the mesh; the returned
    array has zeros for any node that doesn't appear in the file.
    """
    F = np.zeros((n_nodes, 3), dtype=float)
    text = Path(path).read_text()
    for m in _F_LINE.finditer(text):
        nid = int(m.group(1)) - 1  # ANSYS 1-indexed → 0-indexed numpy row
        if nid < 0 or nid >= n_nodes:
            continue
        comp = m.group(2).upper()
        val = float(m.group(3))
        col = {"FX": 0, "FY": 1, "FZ": 2}[comp]
        F[nid, col] += val
    return F


def section_resultants(
    nodes_xyz: np.ndarray,
    applied_forces_xyz: np.ndarray,
    z_section: float,
) -> dict[str, float]:
    """Internal force + moment at section ``z = z_section`` by
    equilibrium of the outboard segment.

    Returns a dict with keys ``Fx_N, Fy_N, Fz_N, Mx_Nm, My_Nm, Mz_Nm``.
    Sign convention: positive z = outboard direction; moments are
    about the section centroid ``(0, 0, z_section)``.
    """
    z = nodes_xyz[:, 2]
    mask = z >= z_section
    if not mask.any():
        return {k: 0.0 for k in ("Fx_N", "Fy_N", "Fz_N",
                                  "Mx_Nm", "My_Nm", "Mz_Nm")}
    F = applied_forces_xyz[mask]                       # (M, 3)
    r = nodes_xyz[mask] - np.array([0.0, 0.0, z_section])  # arm
    M = np.cross(r, F)                                 # (M, 3)
    F_sum = F.sum(axis=0)
    M_sum = M.sum(axis=0)
    return {
        "Fx_N": float(F_sum[0]), "Fy_N": float(F_sum[1]), "Fz_N": float(F_sum[2]),
        "Mx_Nm": float(M_sum[0]), "My_Nm": float(M_sum[1]), "Mz_Nm": float(M_sum[2]),
    }


def section_resultants_at(
    nodes_xyz: np.ndarray,
    forces_src_path: str | Path,
    z_sections: Sequence[float],
) -> dict[float, dict[str, float]]:
    """Convenience: section resultants at every z in ``z_sections``."""
    F = read_forces_src(forces_src_path, len(nodes_xyz))
    return {float(z): section_resultants(nodes_xyz, F, z) for z in z_sections}
