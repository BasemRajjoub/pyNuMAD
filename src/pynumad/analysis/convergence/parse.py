"""Parse the CSV emitted by the APDL POST1 block.

The schema (one row per scalar QoI) is::

    category,name,key,value

with ``category`` ∈ ``{"tip", "patch", "section"}``. We aggregate rows
back into a nested dict for downstream analysis.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ConvergenceResult:
    """Parsed QoI output from a single ANSYS run.

    Fields
    ~~~~~~
    * ``tip``: ``{tip_spec_name: {key: value, ...}}``
    * ``patches``: ``{patch_spec_name: {key: value, ...}}``
    * ``sections``: ``{section_spec_name: {key: value, ...}}``

    Keys follow the APDL emitter's convention, e.g.

    * patch  → ``L3_BOT_svm_Pa``, ``L3_BOT_volu_m3``, ``L3_BOT_n_elem``
    * section → ``Fx_N``, ``Fy_N``, …, ``Mz_Nm``
    * tip    → ``umax_m``
    """
    tip: dict[str, dict[str, float]] = field(default_factory=dict)
    patches: dict[str, dict[str, float]] = field(default_factory=dict)
    sections: dict[str, dict[str, float]] = field(default_factory=dict)

    def patch_svm_pa(self, name: str, layer: int = 3, surface: str = "BOT") -> float:
        """Convenience: area-weighted σ_vM for a patch at a (layer, surface)."""
        key = f"L{layer}_{surface.upper()}_svm_Pa"
        return self.patches[name][key]

    def section_my_nm(self, name: str) -> float:
        """Convenience: bending moment My at a section cut, in N·m."""
        return self.sections[name]["My_Nm"]

    def tip_umax_m(self, name: str = "tip") -> float:
        return self.tip[name]["umax_m"]

    def tip_upoint_mag_m(self, name: str = "tip") -> float | None:
        """Magnitude of displacement at the single tip-most node.

        Mesh-independent in the limit (every refinement still has one
        node at z = z_max). Returns ``None`` if the older deck didn't
        write this row.
        """
        return self.tip.get(name, {}).get("upoint_mag_m")

    def tip_umean_m(self, name: str = "tip") -> float:
        """Magnitude of the mean displacement vector over the tip band.

        Computed as ``sqrt(<UX>² + <UY>² + <UZ>²)`` where each ``<U·>``
        is the per-element ETABLE+SSUM mean over the tip band. This is
        the canonical stable tip-deflection QoI for cantilever-blade
        FE convergence; it's signed by virtue of being a magnitude of a
        mean vector (mean cancellations show as smaller values, but
        won't flip sign).

        Backward compatibility:
        * Older runs (pre-2026-05-23) wrote a single ``umean_m`` row;
          we return that directly when present.
        * Even older runs wrote only ``umax_m`` — we return that as a
          last-resort fallback.
        """
        t = self.tip.get(name, {})
        if "umean_m" in t:
            return t["umean_m"]
        if all(k in t for k in ("umean_x_m", "umean_y_m", "umean_z_m")):
            import math
            return math.sqrt(t["umean_x_m"]**2 + t["umean_y_m"]**2 + t["umean_z_m"]**2)
        return self.tip_umax_m(name)


def parse_results(csv_path: str | Path) -> ConvergenceResult:
    """Read the four-column CSV emitted by :func:`emit_post1`."""
    result = ConvergenceResult()
    with open(csv_path, "r") as fp:
        reader = csv.reader(fp)
        try:
            header = next(reader)
        except StopIteration:
            return result
        if header[:4] != ["category", "name", "key", "value"]:
            raise ValueError(
                f"unexpected header {header!r}; expected the 4-column "
                f"convergence schema 'category,name,key,value'"
            )
        for row in reader:
            if len(row) < 4:
                continue
            cat, name, key, val_s = (s.strip() for s in row[:4])
            try:
                val = float(val_s)
            except ValueError:
                continue
            bucket = {"tip": result.tip,
                      "patch": result.patches,
                      "section": result.sections}.get(cat)
            if bucket is None:
                continue
            bucket.setdefault(name, {})[key] = val
    return result
