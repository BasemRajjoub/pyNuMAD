"""Mesh-quality metrics and assertions for shell-element meshes.

Lifted and generalised from `paper2_fem/00_smoke_test/track_bad_region.py`.
The functions here are the **oracle** for both unit tests and runtime
instrumentation: anything that wants to know whether a mesh is "clean" uses
this module so the definition stays in one place.

Vocabulary
----------
- **mesh**: a dict with keys ``nodes`` (N x 3 array of float) and
  ``elements`` (M x K array of int, K >= 4). The first 4 columns of each
  element row are the corner-node indices of the shell quad. Slot value
  -1 marks a collapsed triangle (slot[3] == -1).
- **quad**: a 4 x 3 array of corner coordinates.
- **Jacobian flip**: the bilinear shell Jacobian changes sign within the
  element. Detected here as a sign change of the signed area of the two
  triangles that split the quad along its 0-2 diagonal, after projection
  onto the dominant element plane. This catches both bow-tie quads and
  re-entrant corners.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable

import numpy as np

__all__ = [
    "MeshQualityReport",
    "DEFAULT_MIN_JACOBIAN_RATIO",
    "DEFAULT_MAX_ASPECT_RATIO",
    "quad_edge_lengths",
    "quad_aspect_ratio",
    "quad_warp_factor",
    "quad_has_jacobian_flip",
    "quad_min_jacobian",
    "find_coincident_node_pairs",
    "analyse_mesh",
    "assert_mesh_clean",
    "assert_no_jacobian_flips",
    "assert_no_jacobian_pathology",
    "select_elements_in_z_band",
]

# ----- low-level metrics on a single quad ----------------------------------

_EPS = 1e-12


def quad_edge_lengths(corners: np.ndarray) -> np.ndarray:
    """Return the four edge lengths of a quad in walk order (0->1, 1->2, 2->3, 3->0)."""
    return np.array(
        [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    )


def quad_aspect_ratio(corners: np.ndarray) -> float:
    """max(edge)/min(edge). Clamped denominator avoids div-by-zero."""
    edges = quad_edge_lengths(corners)
    return float(edges.max() / max(edges.min(), _EPS))


def quad_warp_factor(corners: np.ndarray) -> float:
    """Out-of-plane distance from corner 3 to the plane (corner 0, 1, 2),
    normalised by mean edge length. Zero for planar quads."""
    p0, p1, p2, p3 = corners
    n = np.cross(p1 - p0, p2 - p0)
    nn = np.linalg.norm(n)
    if nn < _EPS:
        return float("nan")
    plane = n / nn
    off = abs(np.dot(p3 - p0, plane))
    avg = float(np.mean(quad_edge_lengths(corners)))
    return float(off / max(avg, _EPS))


def quad_has_jacobian_flip(corners: np.ndarray) -> bool:
    """True if the quad's signed in-plane area flips sign between its two
    diagonal-split triangles. Catches bow-tie / re-entrant geometry that
    will fail an FEA solver's Jacobian check.

    The geometry is projected onto the plane spanned by edge (0->1) and a
    direction orthogonal to it within the (0,1,2) triangle plane. Degenerate
    (zero-area) quads return True.
    """
    p0, p1, p2, p3 = corners
    n = np.cross(p1 - p0, p2 - p0)
    n_norm = np.linalg.norm(n)
    if n_norm < _EPS:
        return True
    n_hat = n / n_norm
    u = p1 - p0
    u_norm = np.linalg.norm(u)
    if u_norm < _EPS:
        return True
    u = u / u_norm
    v = np.cross(n_hat, u)
    pts = np.array([[np.dot(c - p0, u), np.dot(c - p0, v)] for c in corners])
    # signed areas of triangles (0,1,2) and (0,2,3)
    a1 = 0.5 * (
        (pts[1, 0] - pts[0, 0]) * (pts[2, 1] - pts[0, 1])
        - (pts[2, 0] - pts[0, 0]) * (pts[1, 1] - pts[0, 1])
    )
    a2 = 0.5 * (
        (pts[2, 0] - pts[0, 0]) * (pts[3, 1] - pts[0, 1])
        - (pts[3, 0] - pts[0, 0]) * (pts[2, 1] - pts[0, 1])
    )
    return bool((a1 * a2) < 0)


def quad_min_jacobian(corners: np.ndarray) -> float:
    """Scalar quality measure: ratio of signed-area of smaller diagonal
    triangle to the larger one. Negative if Jacobian flips, in [0, 1] for
    well-shaped quads (1 = square)."""
    p0, p1, p2, p3 = corners
    n = np.cross(p1 - p0, p2 - p0)
    n_norm = np.linalg.norm(n)
    if n_norm < _EPS:
        return 0.0
    n_hat = n / n_norm
    u = p1 - p0
    u_norm = np.linalg.norm(u)
    if u_norm < _EPS:
        return 0.0
    u = u / u_norm
    v = np.cross(n_hat, u)
    pts = np.array([[np.dot(c - p0, u), np.dot(c - p0, v)] for c in corners])
    a1 = 0.5 * (
        (pts[1, 0] - pts[0, 0]) * (pts[2, 1] - pts[0, 1])
        - (pts[2, 0] - pts[0, 0]) * (pts[1, 1] - pts[0, 1])
    )
    a2 = 0.5 * (
        (pts[2, 0] - pts[0, 0]) * (pts[3, 1] - pts[0, 1])
        - (pts[3, 0] - pts[0, 0]) * (pts[2, 1] - pts[0, 1])
    )
    if a1 * a2 < 0:
        return -1.0
    if max(abs(a1), abs(a2)) < _EPS:
        return 0.0
    return float(min(abs(a1), abs(a2)) / max(abs(a1), abs(a2)))


# ----- mesh-wide aggregations ----------------------------------------------


# Quality thresholds used by `MeshQualityReport.is_clean()` and by
# `assert_no_jacobian_pathology`. Chosen to roughly match ANSYS shell-element
# acceptance: an element whose smaller diagonal-triangle area is < 1e-3 of
# the larger, or whose aspect ratio is > 100, will typically trip ANSYS's
# Jacobian-ratio check at Gauss points even when no in-plane sign flip is
# present. The handoff data showed 2 such "sliver" elements at 0.80 m with
# AR ~ 1.4e3 — exactly the failure mode this threshold catches.
DEFAULT_MIN_JACOBIAN_RATIO = 1e-3
DEFAULT_MAX_ASPECT_RATIO = 100.0


@dataclass
class MeshQualityReport:
    """Aggregate quality metrics for a shell mesh."""

    n_nodes: int
    n_elements: int
    n_valid_quads: int
    n_valid_tris: int
    n_invalid_lines: int  # degenerate elements with <= 2 unique node ids
    n_jacobian_flips: int
    n_low_jacobian: int  # min_jacobian_ratio below threshold (sliver elements)
    n_severe_aspect_ratio: int  # aspect ratio above threshold
    n_coincident_node_pairs: int
    median_aspect_ratio: float
    p99_aspect_ratio: float
    max_aspect_ratio: float
    min_jacobian_ratio: float  # worst (smallest) min/max diagonal-area ratio across all quads
    median_warp_factor: float
    max_warp_factor: float
    # forensic: indices of the offending elements (capped to keep payload small)
    jacobian_flip_indices: list = field(default_factory=list)
    low_jacobian_indices: list = field(default_factory=list)
    severe_aspect_indices: list = field(default_factory=list)
    invalid_line_indices: list = field(default_factory=list)
    coincident_node_pairs: list = field(default_factory=list)  # list of (i, j)
    # the thresholds used (so reports are self-describing)
    threshold_min_jacobian_ratio: float = DEFAULT_MIN_JACOBIAN_RATIO
    threshold_max_aspect_ratio: float = DEFAULT_MAX_ASPECT_RATIO

    def is_clean(self) -> bool:
        """A mesh is 'clean' when no Jacobian flips, no slivers, no severe
        aspect ratios, no degenerate elements, no coincident nodes."""
        return (
            self.n_jacobian_flips == 0
            and self.n_low_jacobian == 0
            and self.n_severe_aspect_ratio == 0
            and self.n_invalid_lines == 0
            and self.n_coincident_node_pairs == 0
        )

    def summary(self) -> str:
        return (
            f"MeshQualityReport(nodes={self.n_nodes}, elems={self.n_elements}, "
            f"quads={self.n_valid_quads}, tris={self.n_valid_tris}, "
            f"jflips={self.n_jacobian_flips}, low_jac={self.n_low_jacobian}, "
            f"severe_AR={self.n_severe_aspect_ratio}, lines={self.n_invalid_lines}, "
            f"coincident_pairs={self.n_coincident_node_pairs}, "
            f"min_jac_ratio={self.min_jacobian_ratio:.2e}, "
            f"max_aspect={self.max_aspect_ratio:.2f}, max_warp={self.max_warp_factor:.4f})"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def find_coincident_node_pairs(
    nodes: np.ndarray, tol: float = 1e-9, max_pairs: int = 50
) -> list:
    """Return up to ``max_pairs`` pairs of node indices whose coordinates
    coincide within ``tol``. Uses a spatial hash via numpy rounding for
    O(n) average performance, not O(n^2).

    Parameters
    ----------
    nodes : (N, 3) array
    tol : float
        Coordinates rounded to ``round(c / tol) * tol`` are considered equal.
    max_pairs : int
        Truncate the returned list to bound report size.
    """
    rounded = np.round(np.asarray(nodes) / max(tol, _EPS)).astype(np.int64)
    buckets: dict[tuple, list[int]] = {}
    pairs: list[tuple[int, int]] = []
    for i, key in enumerate(map(tuple, rounded)):
        same = buckets.setdefault(key, [])
        for j in same:
            pairs.append((j, i))
            if len(pairs) >= max_pairs:
                return pairs
        same.append(i)
    return pairs


def _batch_cross_3d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Manual element-wise 3D cross product for shape ``(M, 3)`` inputs.

    Roughly 5x faster than ``np.cross`` on large batches because it skips
    numpy's general-axis broadcasting / moveaxis machinery — which dominated
    the previous implementation according to cProfile.
    """
    out = np.empty_like(a)
    out[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    out[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    out[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return out


def _batch_norm_3d(v: np.ndarray) -> np.ndarray:
    """Manual 3D L2 norm for shape ``(M, 3)`` input. Faster than
    ``np.linalg.norm(v, axis=1)`` for small last-dimension."""
    return np.sqrt(v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1] + v[:, 2] * v[:, 2])


def analyse_mesh(
    mesh: dict,
    tol_coincident: float = 1e-9,
    max_indices: int = 50,
    min_jacobian_ratio_threshold: float = DEFAULT_MIN_JACOBIAN_RATIO,
    max_aspect_ratio_threshold: float = DEFAULT_MAX_ASPECT_RATIO,
) -> MeshQualityReport:
    """Compute a `MeshQualityReport` for a pyNuMAD shell mesh dict.

    Vectorised implementation: all per-element computation done with bulk
    numpy operations on shape-``(M, ...)`` arrays. Profiling shows this is
    ~50-100x faster than the previous per-element Python loop on a 25k
    element mesh (10.5 s -> ~0.2 s).

    Accepts the standard mesh dict shape: ``{"nodes": (N, 3), "elements": (M, K)}``
    with ``K >= 4`` and ``elements[i, 3] == -1`` marking collapsed triangles.

    Thresholds:
      - ``min_jacobian_ratio_threshold``: quads whose smaller diagonal-triangle
        area divided by the larger is below this are counted as "low Jacobian"
        (sliver elements that ANSYS rejects even without a sign flip).
      - ``max_aspect_ratio_threshold``: quads whose longest/shortest edge
        ratio exceeds this are counted as "severe aspect ratio".
    """
    nodes = np.asarray(mesh["nodes"], dtype=float)
    elements = np.asarray(mesh["elements"], dtype=int)
    n_elem = elements.shape[0]
    conn = elements[:, :4]  # (M, 4)

    # -- Unique-id count per row (handles collapsed triangles via slot==-1) -
    # Strategy: sort each row, count non-negatives, count duplicates among
    # consecutive non-negative entries. unique = 4 - n_neg - n_dup.
    sorted_conn = np.sort(conn, axis=1)
    n_neg = (sorted_conn < 0).sum(axis=1)
    diffs = np.diff(sorted_conn, axis=1)
    both_valid = (sorted_conn[:, :-1] >= 0) & (sorted_conn[:, 1:] >= 0)
    n_dup = ((diffs == 0) & both_valid).sum(axis=1)
    uniq = 4 - n_neg - n_dup  # (M,)

    # -- Gather corner coordinates as (M, 4, 3) ---------------------------
    # Replace -1 sentinels with 0 to keep indexing safe; we'll mask later.
    safe_conn = np.where(conn >= 0, conn, 0)
    corners = nodes[safe_conn]  # (M, 4, 3)

    # Per-edge vectors and lengths (walk order 0->1, 1->2, 2->3, 3->0).
    e0 = corners[:, 1] - corners[:, 0]
    e1 = corners[:, 2] - corners[:, 1]
    e2 = corners[:, 3] - corners[:, 2]
    e3 = corners[:, 0] - corners[:, 3]
    n0 = _batch_norm_3d(e0)
    n1 = _batch_norm_3d(e1)
    n2 = _batch_norm_3d(e2)
    n3 = _batch_norm_3d(e3)

    # -- Aspect ratio: max edge / min edge --------------------------------
    # For full quads (uniq == 4): use the 4-walk edge lengths.
    # For triangles where slot 3 == -1 (the canonical collapsed-quad
    # sentinel produced by shell_region.py), use the 3-edge triangle walk
    # (0,1,2) so we don't include a degenerate edge in the ratio.
    edge_lens_quad = np.stack([n0, n1, n2, n3], axis=1)  # (M, 4)

    # Triangle edge lengths (corners 0,1,2): e0 already, plus 1->2 and 2->0
    e_12 = corners[:, 2] - corners[:, 1]
    e_20 = corners[:, 0] - corners[:, 2]
    edge_lens_tri = np.stack([n0, _batch_norm_3d(e_12), _batch_norm_3d(e_20)], axis=1)

    is_triangle = (conn[:, 3] == -1) | (uniq == 3)
    quad_edge_max = edge_lens_quad.max(axis=1)
    quad_edge_min = edge_lens_quad.min(axis=1)
    tri_edge_max = edge_lens_tri.max(axis=1)
    tri_edge_min = edge_lens_tri.min(axis=1)

    edge_max = np.where(is_triangle, tri_edge_max, quad_edge_max)
    edge_min = np.where(is_triangle, tri_edge_min, quad_edge_min)
    aspects = np.where(
        edge_min > _EPS,
        edge_max / np.maximum(edge_min, _EPS),
        np.nan,
    )
    # Degenerates: when fewer than 3 unique nodes, no meaningful AR.
    aspects = np.where(uniq >= 3, aspects, np.nan)
    # Keep `edge_lens` as the 4-walk version for downstream warp calc
    edge_lens = edge_lens_quad

    # -- Warp factor: out-of-plane offset of corner 3 from plane(0,1,2) ---
    n_vec = _batch_cross_3d(e0, corners[:, 2] - corners[:, 0])  # (M, 3)
    n_mag = _batch_norm_3d(n_vec)
    # Plane normal (unit); divide-by-zero guarded
    plane = np.where(
        n_mag[:, None] > _EPS, n_vec / np.maximum(n_mag[:, None], _EPS), 0.0
    )
    off = np.abs(((corners[:, 3] - corners[:, 0]) * plane).sum(axis=1))
    avg_edge = edge_lens.mean(axis=1)
    warps = np.where(
        (avg_edge > _EPS) & (uniq == 4),
        off / np.maximum(avg_edge, _EPS),
        0.0,
    )
    warps = np.where(uniq <= 2, np.nan, warps)

    # -- Jacobian flip / min-jacobian-ratio for quads (uniq == 4) ---------
    # Project corners onto plane(0,1,2) basis (u=e0/|e0|, v=normal × u),
    # compute signed area of triangles (0,1,2) and (0,2,3); sign disagreement
    # = bow-tie. We work directly in 3D using the cross product instead of
    # full 2D projection because we only need the sign and the area ratio.
    e02 = corners[:, 2] - corners[:, 0]
    e03 = corners[:, 3] - corners[:, 0]
    # a1 = 0.5 * |e0 × e02|  (always >= 0, |area| of triangle 0-1-2)
    cross_012 = _batch_cross_3d(e0, e02)
    a1 = 0.5 * _batch_norm_3d(cross_012)
    # a2_vec = e02 × e03;  a2 sign relative to n_hat tells us flip
    cross_023 = _batch_cross_3d(e02, e03)
    a2_signed = 0.5 * (cross_023 * plane).sum(axis=1)
    a2_abs = np.abs(a2_signed)

    # Detect degenerate (n_mag ~= 0): no consistent normal — treat as flipped
    degenerate_plane = n_mag < _EPS
    sign_flip = (a2_signed < 0) | degenerate_plane

    # min_jacobian_ratio = min(|a1|, |a2|) / max(|a1|, |a2|); negative when
    # signs disagree
    ab_max = np.maximum(a1, a2_abs)
    ab_min = np.minimum(a1, a2_abs)
    raw_ratio = np.where(ab_max > _EPS, ab_min / np.maximum(ab_max, _EPS), 0.0)
    min_jacs = np.where(sign_flip, -1.0, raw_ratio)

    # Only quads (uniq == 4) get a Jacobian metric
    quad_mask = uniq == 4
    jflips = np.zeros(n_elem, dtype=bool)
    jflips[quad_mask] = (min_jacs[quad_mask] < 0)
    min_jacs = np.where(quad_mask, min_jacs, np.nan)

    # -- Classification masks --------------------------------------------
    valid_min_jac = quad_mask & np.isfinite(min_jacs) & (min_jacs >= 0)
    low_jac_mask = valid_min_jac & (min_jacs < min_jacobian_ratio_threshold)
    severe_ar_mask = (uniq >= 3) & (aspects > max_aspect_ratio_threshold) & np.isfinite(aspects)
    invalid_mask = uniq <= 2

    # -- Forensic index lists (capped to max_indices for payload size) ----
    def _cap(mask):
        idx = np.flatnonzero(mask)
        return idx[:max_indices].tolist()

    jflip_idx = _cap(jflips)
    low_jac_idx = _cap(low_jac_mask)
    severe_ar_idx = _cap(severe_ar_mask)
    invalid_idx = _cap(invalid_mask)

    # -- Summary statistics on finite values ------------------------------
    finite_aspects = aspects[np.isfinite(aspects)]
    finite_warps = warps[np.isfinite(warps)]
    finite_min_jacs = min_jacs[np.isfinite(min_jacs)]

    coincident = find_coincident_node_pairs(
        nodes, tol=tol_coincident, max_pairs=max_indices
    )

    n_low_jac = int(low_jac_mask.sum())
    n_severe_ar = int(severe_ar_mask.sum())

    return MeshQualityReport(
        n_nodes=int(nodes.shape[0]),
        n_elements=int(n_elem),
        n_valid_quads=int((uniq == 4).sum()),
        n_valid_tris=int((uniq == 3).sum()),
        n_invalid_lines=int(((uniq <= 2)).sum()),
        n_jacobian_flips=int(jflips.sum()),
        n_low_jacobian=n_low_jac,
        n_severe_aspect_ratio=n_severe_ar,
        n_coincident_node_pairs=len(coincident),
        median_aspect_ratio=float(np.median(finite_aspects))
        if finite_aspects.size
        else float("nan"),
        p99_aspect_ratio=float(np.percentile(finite_aspects, 99))
        if finite_aspects.size
        else float("nan"),
        max_aspect_ratio=float(finite_aspects.max())
        if finite_aspects.size
        else float("nan"),
        min_jacobian_ratio=float(finite_min_jacs.min())
        if finite_min_jacs.size
        else float("nan"),
        median_warp_factor=float(np.median(finite_warps))
        if finite_warps.size
        else float("nan"),
        max_warp_factor=float(finite_warps.max())
        if finite_warps.size
        else float("nan"),
        jacobian_flip_indices=jflip_idx,
        low_jacobian_indices=low_jac_idx,
        severe_aspect_indices=severe_ar_idx,
        invalid_line_indices=invalid_idx,
        coincident_node_pairs=coincident,
        threshold_min_jacobian_ratio=min_jacobian_ratio_threshold,
        threshold_max_aspect_ratio=max_aspect_ratio_threshold,
    )


# ----- assertion helpers (use these from tests) -----------------------------


def assert_mesh_clean(
    mesh: dict,
    tol_coincident: float = 1e-9,
    msg_prefix: str = "",
) -> MeshQualityReport:
    """Assert that ``mesh`` has zero Jacobian flips, zero invalid-line elements,
    and zero coincident node pairs. Returns the full report so callers can
    inspect quality metrics on success.
    """
    report = analyse_mesh(mesh, tol_coincident=tol_coincident)
    if not report.is_clean():
        prefix = f"{msg_prefix}: " if msg_prefix else ""
        raise AssertionError(
            f"{prefix}{report.summary()}\n"
            f"  jflip_indices={report.jacobian_flip_indices[:10]}\n"
            f"  invalid_line_indices={report.invalid_line_indices[:10]}\n"
            f"  coincident_node_pairs={report.coincident_node_pairs[:10]}"
        )
    return report


def assert_no_jacobian_flips(mesh: dict, msg_prefix: str = "") -> MeshQualityReport:
    """Weaker check than `assert_mesh_clean`: only fail on sign-flipped Jacobian.

    Does NOT catch low-Jacobian slivers — for that use `assert_no_jacobian_pathology`.
    """
    report = analyse_mesh(mesh)
    if report.n_jacobian_flips > 0:
        prefix = f"{msg_prefix}: " if msg_prefix else ""
        raise AssertionError(
            f"{prefix}{report.n_jacobian_flips} Jacobian-flipped elements\n"
            f"  indices (first 10): {report.jacobian_flip_indices[:10]}"
        )
    return report


def assert_no_jacobian_pathology(
    mesh: dict,
    msg_prefix: str = "",
    min_jacobian_ratio_threshold: float = DEFAULT_MIN_JACOBIAN_RATIO,
    max_aspect_ratio_threshold: float = DEFAULT_MAX_ASPECT_RATIO,
) -> MeshQualityReport:
    """Strict-but-targeted Jacobian check: fail on sign flips AND low-Jacobian
    slivers AND severe aspect ratios. Anything ANSYS would reject at the
    element-Jacobian stage.

    Use this as the EARLY-EXIT smoke check at the start of a test or pipeline:
    catches the IEA-22 root-transition sliver elements (aspect > 1000, two
    near-coincident corners) that `assert_no_jacobian_flips` lets through.
    """
    report = analyse_mesh(
        mesh,
        min_jacobian_ratio_threshold=min_jacobian_ratio_threshold,
        max_aspect_ratio_threshold=max_aspect_ratio_threshold,
    )
    problems = []
    if report.n_jacobian_flips > 0:
        problems.append(
            f"{report.n_jacobian_flips} sign-flipped (indices: {report.jacobian_flip_indices[:10]})"
        )
    if report.n_low_jacobian > 0:
        problems.append(
            f"{report.n_low_jacobian} low-Jacobian slivers "
            f"(min ratio threshold={min_jacobian_ratio_threshold:.0e}, "
            f"worst={report.min_jacobian_ratio:.2e}, "
            f"indices: {report.low_jacobian_indices[:10]})"
        )
    if report.n_severe_aspect_ratio > 0:
        problems.append(
            f"{report.n_severe_aspect_ratio} severe-aspect-ratio quads "
            f"(threshold={max_aspect_ratio_threshold:g}, "
            f"worst AR={report.max_aspect_ratio:.2e}, "
            f"indices: {report.severe_aspect_indices[:10]})"
        )
    if problems:
        prefix = f"{msg_prefix}: " if msg_prefix else ""
        raise AssertionError(prefix + "Jacobian pathology — " + "; ".join(problems))
    return report


# ----- geometric region filters --------------------------------------------


def select_elements_in_z_band(
    mesh: dict, z_low: float, z_high: float
) -> np.ndarray:
    """Return a bool mask over elements whose centroid z lies in [z_low, z_high].

    Used to focus regression assertions on the root-transition zone where the
    structured-mesh bug lives.
    """
    nodes = np.asarray(mesh["nodes"], dtype=float)
    elements = np.asarray(mesh["elements"], dtype=int)
    mask = np.zeros(elements.shape[0], dtype=bool)
    for ei, conn in enumerate(elements[:, :4]):
        valid = conn[conn >= 0]
        if valid.size == 0:
            continue
        z_mid = float(nodes[valid, 2].mean())
        mask[ei] = z_low <= z_mid <= z_high
    return mask
