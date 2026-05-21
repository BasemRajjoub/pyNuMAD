"""Tiny SVG visualisations of individual mesh elements and their neighbourhoods.

Built for forensic debugging of mesh-quality failures: when a quad has a
sign-flipped Jacobian or a sliver aspect ratio, dump it to an SVG and
inspect the node winding by eye. Self-contained — no matplotlib, no
plotly, just string-building.

Usage::

    from pynumad.testing.viz import write_quad_svg, write_neighbourhood_svg
    write_quad_svg("bad_quad.svg", corners, node_ids=[1396, 1439, 1443, 1398])
    write_neighbourhood_svg("ctx.svg", mesh, centre_elem_idx=1352, radius=2)

The SVGs render to disk; open them in a browser or any SVG viewer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


_EPS = 1e-15


def _project_onto_quad_plane(corners: np.ndarray) -> tuple[np.ndarray, dict]:
    """Project a (4, 3) quad onto its dominant plane.

    Returns ``(pts_2d, meta)`` where ``pts_2d`` is the (4, 2) array of
    projected corner positions in the plane defined by edge (0->1) and a
    direction orthogonal within the (0,1,2) triangle plane. ``meta`` records
    the basis vectors so downstream code can project additional points.
    """
    p0 = corners[0]
    n = np.cross(corners[1] - p0, corners[2] - p0)
    nn = np.linalg.norm(n)
    if nn < _EPS:
        # Degenerate triangle; fall back to xy plane
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
    else:
        n_hat = n / nn
        u = corners[1] - p0
        un = np.linalg.norm(u)
        u = u / max(un, _EPS)
        v = np.cross(n_hat, u)
    pts = np.array([[np.dot(c - p0, u), np.dot(c - p0, v)] for c in corners])
    return pts, {"origin": p0, "u": u, "v": v}


def _svg_header(view_box: tuple[float, float, float, float], size_px: int = 600) -> str:
    x, y, w, h = view_box
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size_px}" '
        f'height="{size_px}" viewBox="{x} {y} {w} {h}" '
        f'font-family="monospace" font-size="0.8" '
        f'preserveAspectRatio="xMidYMid meet">\n'
    )


def _svg_footer() -> str:
    return "</svg>\n"


def _fit_viewbox(
    pts_2d: np.ndarray, padding_frac: float = 0.15
) -> tuple[float, float, float, float]:
    """Compute a viewBox that fits all 2D points with `padding_frac` margin."""
    xs, ys = pts_2d[:, 0], pts_2d[:, 1]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    w = max(x_max - x_min, _EPS)
    h = max(y_max - y_min, _EPS)
    pad = padding_frac * max(w, h)
    return (x_min - pad, y_min - pad, w + 2 * pad, h + 2 * pad)


def _quad_svg_body(
    pts_2d: np.ndarray,
    node_ids: Sequence[int] | None,
    coords_3d: np.ndarray,
    title: str,
    fill: str = "#ffe0e0",
    stroke: str = "#c00",
    stroke_width_scale: float = 0.005,
) -> str:
    """Render one quad as SVG. Vertices labeled with their node IDs and 3D
    coords; edges numbered 0/1/2/3 (the walk order); the diagonal that
    splits the quad into two triangles is dashed so you can see whether
    the two triangles have consistent orientation.
    """
    body: list[str] = []
    # Compute a reasonable stroke width relative to the viewbox extent
    extent = max(
        float(pts_2d[:, 0].max() - pts_2d[:, 0].min()),
        float(pts_2d[:, 1].max() - pts_2d[:, 1].min()),
        _EPS,
    )
    sw = stroke_width_scale * extent
    # Title (rendered as SVG text top-left)
    body.append(
        f'  <text x="{pts_2d[:, 0].min():.6g}" '
        f'y="{pts_2d[:, 1].min() - 0.4 * sw:.6g}" '
        f'font-size="{1.2 * sw:.6g}" fill="#333">{title}</text>\n'
    )
    # Fill the quad polygon
    poly = " ".join(f"{p[0]:.6g},{p[1]:.6g}" for p in pts_2d)
    body.append(
        f'  <polygon points="{poly}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{sw:.6g}" />\n'
    )
    # Diagonal 0->2 (dashed)
    body.append(
        f'  <line x1="{pts_2d[0, 0]:.6g}" y1="{pts_2d[0, 1]:.6g}" '
        f'x2="{pts_2d[2, 0]:.6g}" y2="{pts_2d[2, 1]:.6g}" '
        f'stroke="#888" stroke-width="{0.5 * sw:.6g}" '
        f'stroke-dasharray="{2*sw:.6g},{2*sw:.6g}" />\n'
    )
    # Edge labels at midpoint, slightly offset
    for i in range(4):
        a = pts_2d[i]
        b = pts_2d[(i + 1) % 4]
        mid = 0.5 * (a + b)
        body.append(
            f'  <text x="{mid[0]:.6g}" y="{mid[1]:.6g}" '
            f'fill="#444" font-size="{0.9 * sw:.6g}">e{i}</text>\n'
        )
    # Vertices: dot + label with node id and 3D coords
    for i, p in enumerate(pts_2d):
        c3 = coords_3d[i]
        nid = node_ids[i] if node_ids is not None else i
        body.append(
            f'  <circle cx="{p[0]:.6g}" cy="{p[1]:.6g}" r="{0.6 * sw:.6g}" '
            f'fill="#06c" />\n'
        )
        body.append(
            f'  <text x="{p[0]:.6g}" y="{p[1]:.6g}" '
            f'dx="{0.8 * sw:.6g}" dy="{-0.4 * sw:.6g}" '
            f'fill="#06c" font-size="{1.0 * sw:.6g}">'
            f"v{i}=n{nid}</text>\n"
        )
        body.append(
            f'  <text x="{p[0]:.6g}" y="{p[1]:.6g}" '
            f'dx="{0.8 * sw:.6g}" dy="{0.8 * sw:.6g}" '
            f'fill="#444" font-size="{0.7 * sw:.6g}">'
            f"({c3[0]:.4g},{c3[1]:.4g},{c3[2]:.4g})</text>\n"
        )
    return "".join(body)


def write_quad_svg(
    path: str | Path,
    corners: np.ndarray,
    node_ids: Sequence[int] | None = None,
    title: str = "quad",
    size_px: int = 600,
) -> None:
    """Render one quad's natural-projection to an SVG file.

    ``corners`` is a (4, 3) ndarray of 3D corner positions. ``node_ids``
    are optional integer labels (e.g. the global node ids in the mesh).
    """
    corners = np.asarray(corners, dtype=float)
    assert corners.shape == (4, 3), f"corners must be (4, 3), got {corners.shape}"
    pts_2d, _meta = _project_onto_quad_plane(corners)
    view = _fit_viewbox(pts_2d)
    out = (
        _svg_header(view, size_px=size_px)
        + _quad_svg_body(pts_2d, node_ids, corners, title)
        + _svg_footer()
    )
    Path(path).write_text(out)


def _find_neighbour_elements(
    mesh: dict, centre_idx: int, radius: int = 1
) -> list[int]:
    """Return the indices of all elements within `radius` adjacency hops of
    ``centre_idx``. Adjacency = sharing at least one node.
    """
    elements = np.asarray(mesh["elements"], dtype=int)
    # Build node -> elements map
    node_to_elems: dict[int, list[int]] = {}
    for ei in range(elements.shape[0]):
        for nid in elements[ei, :4]:
            if nid >= 0:
                node_to_elems.setdefault(int(nid), []).append(ei)
    frontier = {centre_idx}
    seen = {centre_idx}
    for _ in range(radius):
        next_frontier: set[int] = set()
        for ei in frontier:
            for nid in elements[ei, :4]:
                if nid < 0:
                    continue
                for ej in node_to_elems.get(int(nid), []):
                    if ej not in seen:
                        next_frontier.add(ej)
                        seen.add(ej)
        frontier = next_frontier
    return sorted(seen)


def write_neighbourhood_svg(
    path: str | Path,
    mesh: dict,
    centre_elem_idx: int,
    radius: int = 1,
    size_px: int = 800,
) -> None:
    """Render the centre element + all elements within ``radius`` adjacency
    hops to an SVG. The centre element is drawn in red; neighbours in
    light grey. Node IDs are labeled for the centre element only (to keep
    the picture readable)."""
    nodes = np.asarray(mesh["nodes"], dtype=float)
    elements = np.asarray(mesh["elements"], dtype=int)
    neigh = _find_neighbour_elements(mesh, centre_elem_idx, radius=radius)

    # Project the centre element's plane onto 2D, then map all neighbour
    # vertices to that same plane.
    centre_conn = elements[centre_elem_idx, :4]
    centre_corners = nodes[centre_conn[centre_conn >= 0]]
    if centre_corners.shape[0] < 3:
        # Degenerate centre; just emit the centre quad
        write_quad_svg(
            path, nodes[centre_conn], node_ids=centre_conn.tolist(), title=f"elem {centre_elem_idx} (degenerate)"
        )
        return
    # Use first three corners to define the plane
    p0 = centre_corners[0]
    n = np.cross(centre_corners[1] - p0, centre_corners[2] - p0)
    nn = np.linalg.norm(n)
    if nn < _EPS:
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
    else:
        n_hat = n / nn
        u = centre_corners[1] - p0
        u = u / max(np.linalg.norm(u), _EPS)
        v = np.cross(n_hat, u)
    project = lambda c: np.array([np.dot(c - p0, u), np.dot(c - p0, v)])

    # Compute combined view
    all_pts: list[np.ndarray] = []
    for ei in neigh:
        conn = elements[ei, :4]
        for nid in conn:
            if nid >= 0:
                all_pts.append(project(nodes[nid]))
    pts_array = np.stack(all_pts) if all_pts else np.zeros((1, 2))
    view = _fit_viewbox(pts_array)

    body: list[str] = []
    extent = max(view[2], view[3], _EPS)
    sw = 0.003 * extent

    # Title
    body.append(
        f'  <text x="{view[0]:.6g}" y="{view[1] + sw:.6g}" '
        f'font-size="{1.5 * sw:.6g}" fill="#222">'
        f"neighbourhood of elem {centre_elem_idx} (radius={radius}, {len(neigh)} elements)</text>\n"
    )

    # Draw neighbours first (grey), then centre on top (red)
    for ei in neigh:
        is_centre = ei == centre_elem_idx
        conn = elements[ei, :4]
        valid = conn[conn >= 0]
        if valid.size < 3:
            continue
        pts2d = np.stack([project(nodes[int(nid)]) for nid in valid])
        # Close polygon
        poly = " ".join(f"{p[0]:.6g},{p[1]:.6g}" for p in pts2d)
        if is_centre:
            body.append(
                f'  <polygon points="{poly}" fill="#ffe0e0" stroke="#c00" '
                f'stroke-width="{1.2 * sw:.6g}" />\n'
            )
        else:
            body.append(
                f'  <polygon points="{poly}" fill="#f5f5f5" stroke="#999" '
                f'stroke-width="{0.5 * sw:.6g}" />\n'
            )

    # Diagonal + labels on the centre element
    centre_conn_all = elements[centre_elem_idx, :4]
    centre_pts_2d = np.stack(
        [project(nodes[int(nid)]) if nid >= 0 else np.zeros(2) for nid in centre_conn_all]
    )
    body.append(
        f'  <line x1="{centre_pts_2d[0, 0]:.6g}" y1="{centre_pts_2d[0, 1]:.6g}" '
        f'x2="{centre_pts_2d[2, 0]:.6g}" y2="{centre_pts_2d[2, 1]:.6g}" '
        f'stroke="#666" stroke-width="{0.5 * sw:.6g}" '
        f'stroke-dasharray="{2 * sw:.6g},{2 * sw:.6g}" />\n'
    )
    for i, p in enumerate(centre_pts_2d):
        nid = int(centre_conn_all[i])
        body.append(
            f'  <circle cx="{p[0]:.6g}" cy="{p[1]:.6g}" r="{0.5 * sw:.6g}" fill="#06c" />\n'
        )
        body.append(
            f'  <text x="{p[0]:.6g}" y="{p[1]:.6g}" '
            f'dx="{0.6 * sw:.6g}" dy="{-0.3 * sw:.6g}" '
            f'fill="#06c" font-size="{0.9 * sw:.6g}">v{i}=n{nid}</text>\n'
        )

    out = _svg_header(view, size_px=size_px) + "".join(body) + _svg_footer()
    Path(path).write_text(out)


def dump_bad_elements(
    mesh: dict,
    indices: Iterable[int],
    out_dir: str | Path,
    radius: int = 1,
    size_px: int = 600,
) -> list[Path]:
    """Convenience: for each element index in ``indices``, write an SVG of
    its neighbourhood to ``out_dir/elem_<idx>.svg``. Returns the list of
    file paths written.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for idx in indices:
        path = out / f"elem_{int(idx):06d}.svg"
        write_neighbourhood_svg(path, mesh, int(idx), radius=radius, size_px=size_px)
        written.append(path)
    return written
