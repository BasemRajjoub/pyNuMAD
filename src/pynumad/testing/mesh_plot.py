"""Whole-blade mesh visualisation helpers.

A clean mesh (no jflips, no slivers) is *necessary* but not *sufficient* for
geometric correctness — the mesh can be topologically valid yet have
missing patches, kinked surfaces, or distorted airfoil shapes. These
plotters render the full mesh so that a human can confirm the blade
shape looks right.

Three colouring modes:

- ``problem`` (default) — bad elements (jflips / severe-AR / low-jac) in
  red, everything else grey. Best for verifying mesh-quality work.
- ``component`` — groups elements by structural component derived from
  ``mesh["sets"]["element"]`` region names (HP/LP shell panels, spar caps,
  reinforcements, shear webs, adhesive). Best for verifying that the
  blade layout looks right.
- ``region`` — one color per region (338+ regions on BAR0). Useful for
  finding gaps where degenerate-patch skipping left a hole.

Output formats:

- ``write_mesh_html(mesh, path)`` — interactive plotly HTML. Rotate, zoom,
  toggle individual components on/off. Heavier output (~1-5 MB) but most
  useful for inspection.
- ``write_mesh_png(mesh, path)`` — static three-view matplotlib PNG
  (top / side / iso). Compact, good for sharing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from pynumad.testing.mesh_quality import analyse_mesh


def _build_face_triangles(nodes: np.ndarray, elements: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split each quad into 2 triangles; pass-through triangles.

    Returns (vertices_xyz, tri_indices, src_elem_index).
    src_elem_index[t] gives the original element id that triangle t came from.
    """
    tri_indices: list[tuple[int, int, int]] = []
    src: list[int] = []
    for ei in range(elements.shape[0]):
        conn = elements[ei, :4]
        valid = [int(x) for x in conn if x >= 0]
        if len(valid) >= 4:
            tri_indices.append((valid[0], valid[1], valid[2]))
            tri_indices.append((valid[0], valid[2], valid[3]))
            src.extend([ei, ei])
        elif len(valid) == 3:
            tri_indices.append(tuple(valid[:3]))
            src.append(ei)
        # ignore <= 2 (degenerate)
    if not tri_indices:
        return nodes, np.empty((0, 3), dtype=int), np.empty(0, dtype=int)
    return nodes, np.asarray(tri_indices, dtype=int), np.asarray(src, dtype=int)


def _classify_elements(mesh: dict) -> dict[str, set[int]]:
    """Return sets of element indices keyed by problem class."""
    report = analyse_mesh(mesh)
    return {
        "jflip": set(report.jacobian_flip_indices),
        "severe_AR": set(report.severe_aspect_indices),
        "low_jac": set(report.low_jacobian_indices),
    }


# ---------------------------------------------------------------------------
# Component classification — derive structural component from region names
# ---------------------------------------------------------------------------

# Order matters: more specific patterns first
_COMPONENT_RULES = [
    ("HP_TE_FLAT",    "HP TE flat"),
    ("HP_TE_REINF",   "HP TE reinforcement"),
    ("HP_TE_PANEL",   "HP TE panel"),
    ("HP_SPAR",       "HP spar cap"),
    ("HP_LE_REINF",   "HP LE reinforcement"),
    ("HP_LE_PANEL",   "HP LE panel"),
    ("LP_TE_FLAT",    "LP TE flat"),
    ("LP_TE_REINF",   "LP TE reinforcement"),
    ("LP_TE_PANEL",   "LP TE panel"),
    ("LP_SPAR",       "LP spar cap"),
    ("LP_LE_REINF",   "LP LE reinforcement"),
    ("LP_LE_PANEL",   "LP LE panel"),
    ("SW",            "shear web"),
]


# A 13-color qualitative palette (ColorBrewer + extras) — distinct enough
# to tell adjacent components apart in 3D.
_COMPONENT_COLORS = {
    "HP TE flat":          "#1f77b4",
    "HP TE reinforcement": "#aec7e8",
    "HP TE panel":         "#ff7f0e",
    "HP spar cap":         "#2ca02c",
    "HP LE reinforcement": "#98df8a",
    "HP LE panel":         "#d62728",
    "LP TE flat":          "#9467bd",
    "LP TE reinforcement": "#c5b0d5",
    "LP TE panel":         "#8c564b",
    "LP spar cap":         "#e377c2",
    "LP LE reinforcement": "#f7b6d2",
    "LP LE panel":         "#7f7f7f",
    "shear web":           "#bcbd22",
    "other":               "#cccccc",
}


def _classify_region(name: str) -> str:
    for pat, label in _COMPONENT_RULES:
        if pat in name:
            return label
    return "other"


def _component_groups(mesh: dict) -> dict[str, list[int]]:
    """Map component label -> list of element indices, derived from
    ``mesh['sets']['element']`` region names. Returns empty dict if the
    mesh lacks an element-sets list (e.g. raw shell_mesh_general output)."""
    sets_node = mesh.get("sets")
    if not isinstance(sets_node, dict):
        return {}
    el_sets = sets_node.get("element")
    if not isinstance(el_sets, list):
        return {}
    groups: dict[str, list[int]] = {}
    for s in el_sets:
        label = _classify_region(s.get("name", ""))
        groups.setdefault(label, []).extend(int(x) for x in s.get("labels", []))
    return groups


def _region_groups(mesh: dict) -> dict[str, list[int]]:
    """Map raw region name -> list of element indices. Up to 338 groups
    on BAR0; useful for spotting gaps where degenerate-patch skip left
    holes."""
    sets_node = mesh.get("sets")
    if not isinstance(sets_node, dict):
        return {}
    el_sets = sets_node.get("element")
    if not isinstance(el_sets, list):
        return {}
    return {
        s.get("name", f"region_{i}"): [int(x) for x in s.get("labels", [])]
        for i, s in enumerate(el_sets)
    }


# ---------------------------------------------------------------------------
# HTML output (plotly)
# ---------------------------------------------------------------------------


def _group_triangles_by_element_indices(
    nodes_xyz: np.ndarray,
    tri_idx: np.ndarray,
    src: np.ndarray,
    indices: set[int],
) -> np.ndarray:
    """Subset tri_idx to triangles whose source element id is in `indices`."""
    if not indices:
        return np.empty((0, 3), dtype=int)
    mask = np.array([int(s) in indices for s in src])
    return tri_idx[mask]


def write_mesh_html(
    mesh: dict,
    path: str | Path,
    title: str = "pyNuMAD shell mesh",
    mode: str = "problem",
    show_node_dots: bool = False,
    opacity: float = 0.85,
    include_adhesive: bool = True,
) -> None:
    """Render the full mesh to an interactive plotly HTML file.

    Parameters
    ----------
    mode :
        - ``"problem"`` (default): grey body + red bad elements.
        - ``"component"``: colour by structural component (HP/LP panels,
          spar caps, reinforcements, webs, adhesive). Toggle individual
          components in the legend.
        - ``"region"``: one trace per region name (338+ on BAR0). Slow but
          useful for spotting gaps left by degenerate-patch skipping.
    include_adhesive :
        If True and ``mesh['adhesiveEls']`` is present, render adhesive
        elements as a separate trace.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError(
            "plotly is required for write_mesh_html; install it via "
            "`pip install plotly` (pyNuMAD's dev deps already include it)."
        ) from e

    nodes = np.asarray(mesh["nodes"], dtype=float)
    elements = np.asarray(mesh["elements"], dtype=int)
    nodes_xyz, tri_idx, src = _build_face_triangles(nodes, elements)
    classes = _classify_elements(mesh)
    bad_set = classes["jflip"] | classes["severe_AR"] | classes["low_jac"]

    traces = []

    if mode == "problem":
        good_mask = np.array([int(s) not in bad_set for s in src])
        for label, mask, color, op in (
            ("clean elements", good_mask, "lightgrey", opacity),
            ("bad elements", ~good_mask, "red", min(1.0, opacity + 0.1)),
        ):
            sub = tri_idx[mask]
            if sub.size:
                traces.append(
                    go.Mesh3d(
                        x=nodes_xyz[:, 0], y=nodes_xyz[:, 1], z=nodes_xyz[:, 2],
                        i=sub[:, 0], j=sub[:, 1], k=sub[:, 2],
                        color=color, opacity=op,
                        name=f"{label} ({mask.sum() // 2} of {tri_idx.shape[0] // 2})",
                        flatshading=True, showscale=False,
                    )
                )

    elif mode == "component":
        comp_groups = _component_groups(mesh)
        if not comp_groups:
            raise ValueError(
                "mode='component' requires mesh['sets']['element']; pass a "
                "mesh from get_shell_mesh (not shell_mesh_general)."
            )
        for label, indices_list in comp_groups.items():
            indices = set(indices_list)
            sub = _group_triangles_by_element_indices(nodes_xyz, tri_idx, src, indices)
            if sub.size == 0:
                continue
            traces.append(
                go.Mesh3d(
                    x=nodes_xyz[:, 0], y=nodes_xyz[:, 1], z=nodes_xyz[:, 2],
                    i=sub[:, 0], j=sub[:, 1], k=sub[:, 2],
                    color=_COMPONENT_COLORS.get(label, "#cccccc"),
                    opacity=opacity,
                    name=f"{label} ({len(indices)})",
                    flatshading=True, showscale=False,
                )
            )

    elif mode == "region":
        reg_groups = _region_groups(mesh)
        if not reg_groups:
            raise ValueError(
                "mode='region' requires mesh['sets']['element']; pass a "
                "mesh from get_shell_mesh (not shell_mesh_general)."
            )
        # cycle through a long palette to ensure adjacent regions differ
        import colorsys
        palette = [
            f"hsl({int(360 * h)}, 70%, 50%)"
            for h in np.linspace(0, 1, len(reg_groups), endpoint=False)
        ]
        for color, (label, indices_list) in zip(palette, sorted(reg_groups.items())):
            indices = set(indices_list)
            sub = _group_triangles_by_element_indices(nodes_xyz, tri_idx, src, indices)
            if sub.size == 0:
                continue
            traces.append(
                go.Mesh3d(
                    x=nodes_xyz[:, 0], y=nodes_xyz[:, 1], z=nodes_xyz[:, 2],
                    i=sub[:, 0], j=sub[:, 1], k=sub[:, 2],
                    color=color, opacity=opacity,
                    name=f"{label} ({len(indices)})",
                    flatshading=True, showscale=False,
                )
            )
    else:
        raise ValueError(f"unknown mode {mode!r}; pick 'problem' / 'component' / 'region'")

    if include_adhesive:
        adh_nodes = mesh.get("adhesiveNds")
        adh_els = mesh.get("adhesiveEls")
        if adh_nodes is not None and adh_els is not None:
            adh_n = np.asarray(adh_nodes, dtype=float)
            adh_e = np.asarray(adh_els, dtype=int)
            if adh_n.size and adh_e.size:
                # Adhesive elements are hex8 (8 nodes); render only the
                # 6 face quads as triangles to get a surface mesh.
                # For simplicity, just render the first quad face of each hex.
                tris: list[tuple[int, int, int]] = []
                for el in adh_e:
                    el_valid = [int(x) for x in el if 0 <= int(x) < adh_n.shape[0]]
                    if len(el_valid) >= 4:
                        tris.append((el_valid[0], el_valid[1], el_valid[2]))
                        tris.append((el_valid[0], el_valid[2], el_valid[3]))
                if tris:
                    tri_arr = np.asarray(tris, dtype=int)
                    traces.append(
                        go.Mesh3d(
                            x=adh_n[:, 0], y=adh_n[:, 1], z=adh_n[:, 2],
                            i=tri_arr[:, 0], j=tri_arr[:, 1], k=tri_arr[:, 2],
                            color="#000080", opacity=0.6,
                            name=f"adhesive ({adh_e.shape[0]})",
                            flatshading=True, showscale=False,
                        )
                    )

    if show_node_dots:
        traces.append(
            go.Scatter3d(
                x=nodes_xyz[:, 0], y=nodes_xyz[:, 1], z=nodes_xyz[:, 2],
                mode="markers", marker=dict(size=1, color="black"),
                name=f"nodes ({nodes_xyz.shape[0]})",
            )
        )

    bad_label = (
        f", bad: jflip={len(classes['jflip'])} "
        f"sAR={len(classes['severe_AR'])} lowJ={len(classes['low_jac'])}"
    )
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=(
            f"{title} [mode={mode}]<br><sup>nodes={nodes_xyz.shape[0]}, "
            f"elements={elements.shape[0]}{bad_label}</sup>"
        ),
        scene=dict(
            aspectmode="data",
            xaxis_title="x [m]",
            yaxis_title="y [m]",
            zaxis_title="z [m] (span)",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(str(path), include_plotlyjs="cdn")


# ---------------------------------------------------------------------------
# PNG output (matplotlib three-view)
# ---------------------------------------------------------------------------


def write_mesh_png(
    mesh: dict,
    path: str | Path,
    title: str = "pyNuMAD shell mesh",
    mode: str = "problem",
    dpi: int = 120,
    z_band: tuple[float, float] | None = None,
) -> None:
    """Render three-view (top / side / iso) PNG of the mesh.

    Parameters
    ----------
    mode :
        ``"problem"`` (default), ``"component"``, or ``"region"`` —
        same semantics as ``write_mesh_html``.
    z_band :
        Optional ``(z_min, z_max)`` tuple to render only elements whose
        centroid lies in that band. Useful for root / mid / tip closeups.
        When set, the per-axis bounds zoom to that band's extent.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError as e:
        raise ImportError("matplotlib is required for write_mesh_png") from e

    nodes = np.asarray(mesh["nodes"], dtype=float)
    elements = np.asarray(mesh["elements"], dtype=int)
    classes = _classify_elements(mesh)
    bad_set = classes["jflip"] | classes["severe_AR"] | classes["low_jac"]

    # Filter by z-band if requested
    if z_band is not None:
        z_lo, z_hi = z_band
        in_band: list[int] = []
        for ei in range(elements.shape[0]):
            conn = elements[ei, :4]
            valid = [int(x) for x in conn if x >= 0]
            if len(valid) >= 3:
                z_mid = float(nodes[valid, 2].mean())
                if z_lo <= z_mid <= z_hi:
                    in_band.append(ei)
        band_mask = set(in_band)
    else:
        band_mask = None

    def _quads_xyz(indices):
        polys = []
        for ei in indices:
            if band_mask is not None and ei not in band_mask:
                continue
            conn = elements[ei, :4]
            valid = [int(x) for x in conn if x >= 0]
            if len(valid) >= 3:
                polys.append([nodes[v] for v in valid])
        return polys

    # Build (label, color, polys) tuples per mode
    layers: list[tuple[str, str, list]] = []
    downsampled_note = ""
    max_show = 30000

    if mode == "problem":
        good_indices = [i for i in range(elements.shape[0]) if i not in bad_set]
        if len(good_indices) > max_show:
            rng = np.random.default_rng(0)
            good_indices = [good_indices[i] for i in rng.choice(len(good_indices), max_show, replace=False)]
            downsampled_note = f" (showing {max_show} of {elements.shape[0]} good)"
        layers.append(("clean", "lightgrey", _quads_xyz(good_indices)))
        layers.append(("bad", "red", _quads_xyz(sorted(bad_set))))

    elif mode == "component":
        comp = _component_groups(mesh)
        if not comp:
            raise ValueError("mode='component' requires mesh['sets']['element']")
        for label, indices_list in comp.items():
            color = _COMPONENT_COLORS.get(label, "#cccccc")
            indices = indices_list
            if len(indices) > max_show // len(comp):
                rng = np.random.default_rng(hash(label) & 0xffffffff)
                indices = [indices[i] for i in rng.choice(len(indices), max_show // len(comp), replace=False)]
            layers.append((label, color, _quads_xyz(indices)))

    elif mode == "region":
        reg = _region_groups(mesh)
        if not reg:
            raise ValueError("mode='region' requires mesh['sets']['element']")
        n = len(reg)
        for i, (label, indices_list) in enumerate(sorted(reg.items())):
            # HSV->RGB
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(i / n, 0.7, 0.85)
            color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            layers.append((label, color, _quads_xyz(indices_list)))

    else:
        raise ValueError(f"unknown mode {mode!r}")

    # Three-view figure
    fig = plt.figure(figsize=(15, 5), dpi=dpi)
    views = [
        ("top (xy)", dict(elev=90, azim=-90)),
        ("side (xz)", dict(elev=0, azim=0)),
        ("iso", dict(elev=25, azim=30)),
    ]
    bad_label = (
        f", bad: jflip={len(classes['jflip'])} "
        f"sAR={len(classes['severe_AR'])} lowJ={len(classes['low_jac'])}"
    )
    band_label = f", z∈[{z_band[0]:.1f},{z_band[1]:.1f}]m" if z_band else ""
    fig.suptitle(
        f"{title} [mode={mode}]\nnodes={nodes.shape[0]}, elements={elements.shape[0]}"
        f"{bad_label}{band_label}{downsampled_note}",
        fontsize=11,
    )

    # Compute view extent
    if z_band is not None and band_mask:
        # zoom into the band's element nodes only
        node_in_band = sorted({n for ei in band_mask for n in elements[ei, :4] if n >= 0})
        if node_in_band:
            view_nodes = nodes[np.asarray(node_in_band, dtype=int)]
            mins, maxs = view_nodes.min(axis=0), view_nodes.max(axis=0)
        else:
            mins, maxs = nodes.min(axis=0), nodes.max(axis=0)
    else:
        mins, maxs = nodes.min(axis=0), nodes.max(axis=0)
    ranges = maxs - mins
    max_range = ranges.max()
    ctr = (mins + maxs) / 2

    for idx, (name, view_kwargs) in enumerate(views):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        for label, color, polys in layers:
            if not polys:
                continue
            is_bad = (label == "bad")
            pc = Poly3DCollection(
                polys,
                facecolor=color,
                edgecolor="darkred" if is_bad else "none",
                linewidth=0.5 if is_bad else 0,
                alpha=0.95 if is_bad else 0.8,
                label=label,
            )
            ax.add_collection3d(pc)

        ax.set_xlim(ctr[0] - max_range / 2, ctr[0] + max_range / 2)
        ax.set_ylim(ctr[1] - max_range / 2, ctr[1] + max_range / 2)
        ax.set_zlim(ctr[2] - max_range / 2, ctr[2] + max_range / 2)
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(**view_kwargs)

    # Add a legend to the iso panel only (other panels would duplicate)
    if mode == "component" or mode == "problem":
        ax_legend = fig.axes[-1]
        handles = []
        from matplotlib.patches import Patch
        for label, color, polys in layers:
            if polys:
                handles.append(Patch(color=color, label=label))
        if handles:
            ax_legend.legend(
                handles=handles, loc="upper left",
                bbox_to_anchor=(1.02, 1), fontsize=7, framealpha=0.9,
            )

    fig.tight_layout()
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def dump_mesh_views(
    mesh: dict,
    out_dir: str | Path,
    base_name: str = "mesh",
    formats: Iterable[str] = ("html", "png"),
    modes: Iterable[str] = ("problem", "component"),
    closeups: bool = True,
    title: str | None = None,
) -> list[Path]:
    """Write a battery of mesh visualisations.

    Parameters
    ----------
    formats : iterable of {"html", "png"}
        Output formats. HTML is interactive plotly, PNG is static
        three-view matplotlib.
    modes : iterable of {"problem", "component", "region"}
        Colouring modes; one file per (format, mode) combination.
    closeups : bool
        If True, also write root / mid / tip PNG closeups at the first
        listed mode (default "problem"). Closeups use z-bands based on
        the actual blade extent: root = lowest 20%, mid = middle 20%
        centred at 50% span, tip = highest 20%.

    Returns
    -------
    list of paths written.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    title = title or base_name
    written: list[Path] = []

    for mode in modes:
        if "html" in formats:
            p = out / f"{base_name}_{mode}.html"
            write_mesh_html(mesh, p, title=title, mode=mode)
            written.append(p)
        if "png" in formats:
            p = out / f"{base_name}_{mode}.png"
            write_mesh_png(mesh, p, title=title, mode=mode)
            written.append(p)

    if closeups and "png" in formats:
        primary_mode = next(iter(modes))
        nodes = np.asarray(mesh["nodes"], dtype=float)
        z_min, z_max = float(nodes[:, 2].min()), float(nodes[:, 2].max())
        z_range = z_max - z_min
        bands = [
            ("root",  (z_min,                z_min + 0.20 * z_range)),
            ("mid",   (z_min + 0.40 * z_range, z_min + 0.60 * z_range)),
            ("tip",   (z_min + 0.80 * z_range, z_max)),
        ]
        for band_name, band in bands:
            p = out / f"{base_name}_{primary_mode}_zoom_{band_name}.png"
            write_mesh_png(
                mesh, p, title=f"{title} ({band_name})",
                mode=primary_mode, z_band=band,
            )
            written.append(p)

    return written
