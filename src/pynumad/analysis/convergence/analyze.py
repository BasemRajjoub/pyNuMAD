"""Compute pair-wise drift across element sizes and verdict whether each
QoI satisfies the convergence criterion.

Inputs are a dict ``{h: ConvergenceResult}`` with monotonically
decreasing h; outputs are tables ready for printing or for the LaTeX
appendix of the paper.

We follow the wind-blade FE literature convention:

* Tip deflection: <1 % drift between successive refinements.
* Patch-averaged σ_vM: <5 % drift between successive refinements.
* Section-integrated moments: treated as a **validation** check —
  drift should be <0.1 % (equilibrium-preserved), not a convergence
  metric.
"""
from __future__ import annotations

from dataclasses import dataclass

from pynumad.analysis.convergence.parse import ConvergenceResult


# Default thresholds (per-paper-convention). Override via DriftReport.
DEFAULT_TIP_DRIFT_PCT = 1.0
DEFAULT_PATCH_DRIFT_PCT = 5.0
DEFAULT_SECTION_DRIFT_PCT = 0.1


@dataclass
class DriftRow:
    """One QoI's drift table across the h sweep.

    ``drifts`` is ordered ``(h_finer, drift_pct)`` from finest pair to
    coarsest. ``passes_at_h`` is the smallest h for which **every** drift
    at or below that h is under ``criterion_pct``; ``None`` means even
    the finest pair didn't pass.
    """
    category: str        # "tip" | "patch" | "section"
    name: str
    key: str
    values_by_h: dict[float, float]
    drifts: list[tuple[float, float]]
    criterion_pct: float
    passes_at_h: float | None


def _compute_drifts(values_by_h: dict[float, float]) -> list[tuple[float, float]]:
    """Pair-wise absolute drift, ordered fine→coarse."""
    hs = sorted(values_by_h.keys())   # finer first
    out: list[tuple[float, float]] = []
    for i in range(len(hs) - 1):
        h_fine, h_coarse = hs[i], hs[i + 1]
        v_fine = values_by_h[h_fine]
        v_coarse = values_by_h[h_coarse]
        if abs(v_coarse) < 1e-12:
            continue
        drift = 100 * abs(v_fine - v_coarse) / abs(v_coarse)
        out.append((h_fine, drift))
    return out


def _smallest_passing_h(
    drifts: list[tuple[float, float]],
    criterion_pct: float,
) -> float | None:
    """Smallest ``h_fine`` such that all drifts at h ≤ h_fine pass."""
    # drifts sorted fine→coarse; iterate coarse→fine looking for the
    # first prefix that's fully under the threshold.
    candidates = []
    for h_fine, drift in drifts:
        if drift < criterion_pct:
            candidates.append(h_fine)
    if not candidates:
        return None
    return min(candidates)


def analyse(
    results_by_h: dict[float, ConvergenceResult],
    *,
    tip_drift_pct: float = DEFAULT_TIP_DRIFT_PCT,
    patch_drift_pct: float = DEFAULT_PATCH_DRIFT_PCT,
    section_drift_pct: float = DEFAULT_SECTION_DRIFT_PCT,
) -> list[DriftRow]:
    """Compute drift tables for every (category, name, key) triple.

    Returns a flat list of :class:`DriftRow` covering tip deflection,
    every patch×(layer, surface)×field, and every section component.
    """
    rows: list[DriftRow] = []
    # tip
    for tip_name in _union_names(results_by_h, "tip"):
        for key in _union_keys(results_by_h, "tip", tip_name):
            vals = _collect(results_by_h, "tip", tip_name, key)
            drifts = _compute_drifts(vals)
            rows.append(DriftRow(
                category="tip", name=tip_name, key=key,
                values_by_h=vals, drifts=drifts,
                criterion_pct=tip_drift_pct,
                passes_at_h=_smallest_passing_h(drifts, tip_drift_pct),
            ))
    # patches — only report area-weighted σ_vM by default
    for patch_name in _union_names(results_by_h, "patches"):
        for key in _union_keys(results_by_h, "patches", patch_name):
            if not key.endswith("_svm_Pa"):
                continue  # skip volume / element-count book-keeping rows
            vals = _collect(results_by_h, "patches", patch_name, key)
            drifts = _compute_drifts(vals)
            rows.append(DriftRow(
                category="patch", name=patch_name, key=key,
                values_by_h=vals, drifts=drifts,
                criterion_pct=patch_drift_pct,
                passes_at_h=_smallest_passing_h(drifts, patch_drift_pct),
            ))
    # sections — only report the bending moments by default (My, Mx)
    for sec_name in _union_names(results_by_h, "sections"):
        for key in _union_keys(results_by_h, "sections", sec_name):
            if key not in ("Mx_Nm", "My_Nm", "Mz_Nm"):
                continue
            vals = _collect(results_by_h, "sections", sec_name, key)
            drifts = _compute_drifts(vals)
            rows.append(DriftRow(
                category="section", name=sec_name, key=key,
                values_by_h=vals, drifts=drifts,
                criterion_pct=section_drift_pct,
                passes_at_h=_smallest_passing_h(drifts, section_drift_pct),
            ))
    return rows


def _union_names(results_by_h, attr: str) -> set[str]:
    out: set[str] = set()
    for r in results_by_h.values():
        out.update(getattr(r, attr).keys())
    return out


def _union_keys(results_by_h, attr: str, name: str) -> set[str]:
    out: set[str] = set()
    for r in results_by_h.values():
        out.update(getattr(r, attr).get(name, {}).keys())
    return out


def _collect(results_by_h, attr: str, name: str, key: str) -> dict[float, float]:
    out: dict[float, float] = {}
    for h, r in results_by_h.items():
        d = getattr(r, attr).get(name, {})
        if key in d:
            out[h] = d[key]
    return out


def print_drift_table(rows: list[DriftRow]) -> None:
    """Pretty-print the drift table to stdout."""
    print(f"{'category':<8s} {'name':<22s} {'key':<22s} "
          f"{'drift_finest_%':>14s} {'criterion_%':>11s} {'passes_at_h':>12s}")
    print("-" * 95)
    for r in rows:
        d = r.drifts[0][1] if r.drifts else float("nan")
        ph = f"{r.passes_at_h:.3f}" if r.passes_at_h is not None else ">finer"
        print(f"{r.category:<8s} {r.name:<22s} {r.key:<22s} "
              f"{d:>13.2f}% {r.criterion_pct:>10.1f}% {ph:>12s}")
