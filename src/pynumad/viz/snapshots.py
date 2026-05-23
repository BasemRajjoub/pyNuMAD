"""PNG snapshots of a pyNuMAD blade mesh.

Functions in this module render matplotlib-based snapshots of the
shell mesh, the 3-D adhesive volumes, and per-material highlight
panels. All outputs are static PNGs — fast to generate, easy to
embed in reports, no JavaScript dependency.

Public functions
----------------
* :func:`write_overview_snapshots` — full-blade overviews from three
  camera angles + a span-band zoom strip.
* :func:`write_component_snapshots` — one PNG per material zone with
  full + root + mid + tip close-up panels.
* :func:`write_adhesive_snapshots` — adhesive-only overview + per-band
  zooms.

Each function writes its outputs into ``output_dir`` (created if
missing) and returns the list of paths it produced, so callers can
chain them or assert in tests.

Notes on rendering
~~~~~~~~~~~~~~~~~~
matplotlib's 3-D back-end is a painter's-algorithm rasteriser. Two
behaviours are worth knowing:

1. **Aspect**: a 138 m × 7 m × 6 m blade rendered at true data aspect
   becomes a thin sliver. Helper ``_set_presentation_aspect`` lets you
   exaggerate the chord/thickness axes so individual quads stay
   visible.
2. **Depth sort**: by default, every artist's mean-z is mixed with
   every other artist's; with overlapping translucent shell + opaque
   adhesive that produces a "blue-red-blue-red" interleaved pattern.
   We set ``ax.computed_zorder = False`` and put adhesive at
   ``zorder=10`` so it paints on top, giving a continuous bondline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from pynumad.viz._mesh_utils import (
    collect_adhesive_bond_indices,
    collect_chord_group_indices,
    quads_to_polygons,
    solid_outer_face_triangles,
)
from pynumad.viz._palette import (
    ADHESIVE_COLORS,
    CHORD_GROUP_TAGS,
    DEFAULT_ADHESIVE_COLOR,
    GROUP_COLORS,
    get_color,
)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _require_matplotlib():
    """Import matplotlib lazily so the viz package stays importable
    on systems without matplotlib (e.g. CI hosts that only need the
    JSON/HTML writers).
    """
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for snapshot output. "
            "Install with `pip install matplotlib`."
        ) from exc
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    return plt, Poly3DCollection


def _set_presentation_aspect(
    ax,
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi,
    chord_boost: float = 1.0,
) -> None:
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_zlim(z_lo, z_hi)
    ax.set_box_aspect((
        (x_hi - x_lo) * chord_boost,
        (y_hi - y_lo) * chord_boost,
        z_hi - z_lo,
    ))


def _shell_polys_by_group(mesh, nodes, elements):
    groups = collect_chord_group_indices(mesh)
    out: list[tuple[str, np.ndarray, str]] = []
    for tag in CHORD_GROUP_TAGS:
        if tag not in groups:
            continue
        idx = np.asarray(sorted(set(groups[tag])), int)
        polys = quads_to_polygons(nodes, elements, idx)
        if polys.size == 0:
            continue
        out.append((tag, polys, GROUP_COLORS.get(tag, "#cccccc")))
    return out


def _adhesive_polys_by_bond(mesh):
    adh_nds = np.asarray(mesh.get("adhesiveNds", []), float)
    adh_els = np.asarray(mesh.get("adhesiveEls", []), int)
    if adh_nds.size == 0 or adh_els.size == 0:
        return adh_nds, []
    bonds = collect_adhesive_bond_indices(mesh)
    out: list[tuple[str, np.ndarray, str]] = []
    for tag, ids in bonds.items():
        idx = np.asarray(sorted(set(int(i) for i in ids)), int)
        idx = idx[(idx >= 0) & (idx < len(adh_els))]
        if idx.size == 0:
            continue
        tris = solid_outer_face_triangles(adh_els[idx])
        if tris.size == 0:
            continue
        tri_verts = adh_nds[tris]   # (n_tri, 3, 3)
        color = ADHESIVE_COLORS.get(tag, DEFAULT_ADHESIVE_COLOR)
        out.append((tag, tri_verts, color))
    return adh_nds, out


def _draw_blade(
    ax,
    shell_polys,
    adh_polys,
    z_band: tuple[float, float] | None,
    *,
    shell_alpha: float = 0.20,
    highlight_tag: str | None = None,
):
    """Paint shell (faded background or highlighted) + adhesive on top."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    ax.computed_zorder = False
    zb_lo, zb_hi = z_band if z_band else (-np.inf, np.inf)

    for tag, polys, color in shell_polys:
        cz = polys[:, :, 2].mean(axis=1)
        m = (cz >= zb_lo) & (cz <= zb_hi)
        if not m.any():
            continue
        if highlight_tag is None:
            face = color; alpha = shell_alpha
        elif tag == highlight_tag:
            face = color; alpha = 0.95
        else:
            face = "#cccccc"; alpha = 0.10
        pc = Poly3DCollection(
            polys[m], facecolor=face, alpha=alpha,
            edgecolor="none", linewidth=0, shade=False,
            zorder=10 if tag == highlight_tag else 1,
        )
        ax.add_collection3d(pc)

    for tag, tri_verts, color in adh_polys:
        cz = tri_verts[:, :, 2].mean(axis=1)
        m = (cz >= zb_lo) & (cz <= zb_hi)
        if not m.any():
            continue
        pc = Poly3DCollection(
            tri_verts[m], facecolor=color, alpha=1.0,
            edgecolor="none", linewidth=0, shade=False,
            zorder=20,
        )
        ax.add_collection3d(pc)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def write_overview_snapshots(
    mesh: dict,
    output_dir: str | Path,
    *,
    include_adhesive: bool = True,
    zoom_band_size: float = 20.0,
) -> list[Path]:
    """Three-view full-blade overviews + per-span zoom snapshots.

    Parameters
    ----------
    mesh
        Output of ``shell_mesh_general``.
    output_dir
        Directory to write PNGs into (created if missing).
    include_adhesive
        If False, skip the 3-D solid adhesive volumes even when the
        mesh has them. Mainly useful for shell-only debugging.
    zoom_band_size
        Span length per zoom panel, in metres. The blade's full span
        is partitioned into ``ceil(span / zoom_band_size)`` bands.

    Returns
    -------
    list of Paths to the PNGs written, in deterministic order.
    """
    plt, _Poly = _require_matplotlib()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = np.asarray(mesh["nodes"], float)
    elements = np.asarray(mesh["elements"], int)
    shell_polys = _shell_polys_by_group(mesh, nodes, elements)
    adh_nds, adh_polys = _adhesive_polys_by_bond(mesh) if include_adhesive else (np.empty((0,3)), [])

    all_xyz = np.vstack([nodes, adh_nds]) if adh_nds.size else nodes
    x_lo, x_hi = all_xyz[:, 0].min(), all_xyz[:, 0].max()
    y_lo, y_hi = all_xyz[:, 1].min(), all_xyz[:, 1].max()
    z_lo, z_hi = all_xyz[:, 2].min(), all_xyz[:, 2].max()

    written: list[Path] = []

    # 1) Three-view overviews
    overview_views = {
        "overview_iso":     (25, -60),
        "overview_side_TE": (0, -90),
        "overview_side_LE": (0,  90),
    }
    for name, view in overview_views.items():
        fig = plt.figure(figsize=(18, 6))
        ax = fig.add_subplot(111, projection="3d")
        _draw_blade(ax, shell_polys, adh_polys, None)
        _set_presentation_aspect(ax, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
        ax.view_init(elev=view[0], azim=view[1])
        ax.set_xlabel("chord x [m]"); ax.set_ylabel("thickness y [m]"); ax.set_zlabel("span z [m]")
        ax.set_title(f"blade — {name}")
        out = out_dir / f"{name}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        written.append(out)

    # 2) Per-span zooms
    span = z_hi - z_lo
    n_bands = max(1, int(np.ceil(span / zoom_band_size)))
    band_edges = np.linspace(z_lo, z_hi, n_bands + 1)
    for k in range(n_bands):
        zb_lo, zb_hi = band_edges[k], band_edges[k + 1]
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        _draw_blade(ax, shell_polys, adh_polys, (zb_lo, zb_hi))
        _set_presentation_aspect(ax, x_lo, x_hi, y_lo, y_hi, zb_lo, zb_hi)
        ax.view_init(elev=15, azim=-60)
        ax.set_xlabel("chord x [m]"); ax.set_ylabel("thickness y [m]"); ax.set_zlabel("span z [m]")
        ax.set_title(f"zoom z=[{zb_lo:.1f}, {zb_hi:.1f}] m")
        out = out_dir / f"zoom_z{zb_lo:05.1f}_{zb_hi:05.1f}.png".replace(" ", "0")
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        written.append(out)

    return written


def write_component_snapshots(
    mesh: dict,
    output_dir: str | Path,
    *,
    zoom_bands: Sequence[tuple[str, float, float]] = (
        ("root", 5.0,  12.0),
        ("mid",  60.0, 67.0),
        ("tip",  115.0, 122.0),
    ),
) -> list[Path]:
    """Per-material-zone PNGs: one PNG per chord group with 4 panels.

    Each PNG has: full overview + one close-up per ``zoom_bands`` entry.
    Close-ups render the highlighted group in colour against a faded
    grey wash of the rest of the blade, with black quad edges so you
    can verify node sharing.

    Returns the list of PNG paths written.
    """
    plt, Poly3DCollection = _require_matplotlib()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = np.asarray(mesh["nodes"], float)
    elements = np.asarray(mesh["elements"], int)
    shell_polys = _shell_polys_by_group(mesh, nodes, elements)
    polys_by_tag = {tag: polys for tag, polys, _c in shell_polys}

    x_lo, x_hi = nodes[:, 0].min(), nodes[:, 0].max()
    y_lo, y_hi = nodes[:, 1].min(), nodes[:, 1].max()
    z_lo, z_hi = nodes[:, 2].min(), nodes[:, 2].max()

    def _zoom_panel(ax, tag, zband, view):
        ax.computed_zorder = False
        zb_lo, zb_hi = zband
        # Background: other groups, faded
        for other_tag, other_polys, _c in shell_polys:
            if other_tag == tag:
                continue
            cz = other_polys[:, :, 2].mean(axis=1)
            m = (cz >= zb_lo) & (cz <= zb_hi)
            if not m.any():
                continue
            pc = Poly3DCollection(
                other_polys[m], facecolor="#eaeaea", alpha=0.10,
                edgecolor="#bbbbbb", linewidth=0.05,
                shade=False, zorder=1,
            )
            ax.add_collection3d(pc)
        polys = polys_by_tag[tag]
        cz = polys[:, :, 2].mean(axis=1)
        m = (cz >= zb_lo) & (cz <= zb_hi)
        if not m.any():
            ax.text(0.5, 0.5, 0.5, "(no elements)", transform=ax.transAxes)
            return
        hp = polys[m]
        pc = Poly3DCollection(
            hp, facecolor=GROUP_COLORS[tag], alpha=0.95,
            edgecolor="black", linewidth=0.4, shade=False, zorder=10,
        )
        ax.add_collection3d(pc)
        verts = hp.reshape(-1, 3)
        mx_lo, mx_hi = verts[:, 0].min(), verts[:, 0].max()
        my_lo, my_hi = verts[:, 1].min(), verts[:, 1].max()
        pad_x = max(0.2, 0.05 * (mx_hi - mx_lo))
        pad_y = max(0.2, 0.05 * (my_hi - my_lo))
        ax.set_xlim(mx_lo - pad_x, mx_hi + pad_x)
        ax.set_ylim(my_lo - pad_y, my_hi + pad_y)
        ax.set_zlim(zb_lo, zb_hi)
        ax.set_box_aspect((mx_hi - mx_lo + 2*pad_x,
                           my_hi - my_lo + 2*pad_y,
                           zb_hi - zb_lo))
        ax.view_init(elev=view[0], azim=view[1])
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")

    def _full_panel(ax, tag):
        ax.computed_zorder = False
        for other_tag, other_polys, _c in shell_polys:
            if other_tag == tag:
                continue
            pc = Poly3DCollection(
                other_polys, facecolor="#cccccc", alpha=0.10,
                edgecolor="none", linewidth=0, shade=False, zorder=1,
            )
            ax.add_collection3d(pc)
        polys = polys_by_tag[tag]
        pc = Poly3DCollection(
            polys, facecolor=GROUP_COLORS[tag], alpha=0.95,
            edgecolor="black", linewidth=0.15, shade=False, zorder=10,
        )
        ax.add_collection3d(pc)
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi); ax.set_zlim(z_lo, z_hi)
        ax.set_box_aspect(((x_hi - x_lo) * 4, (y_hi - y_lo) * 4, z_hi - z_lo))
        ax.view_init(elev=20, azim=-60)

    written: list[Path] = []
    n_panels = 1 + len(zoom_bands)
    for tag in sorted(polys_by_tag.keys()):
        n_quads = polys_by_tag[tag].shape[0]
        fig = plt.figure(figsize=(6 * n_panels, 8))
        ax = fig.add_subplot(1, n_panels, 1, projection="3d")
        _full_panel(ax, tag)
        ax.set_title(f"{tag} — full")
        for i, (band_name, zb_lo, zb_hi) in enumerate(zoom_bands, start=2):
            ax = fig.add_subplot(1, n_panels, i, projection="3d")
            _zoom_panel(ax, tag, (zb_lo, zb_hi), (20, -60))
            ax.set_title(f"{tag} — {band_name} z=[{zb_lo:.0f},{zb_hi:.0f}]m")
        fig.suptitle(f"{tag}  ({n_quads} quads, colour {GROUP_COLORS[tag]})",
                     fontsize=13)
        fig.tight_layout()
        out = out_dir / f"{tag}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    return written


def write_adhesive_snapshots(
    mesh: dict,
    output_dir: str | Path,
    *,
    chord_boost: float = 8.0,
    zoom_band_size: float = 20.0,
) -> list[Path]:
    """Adhesive-only snapshots: overview + per-span zoom bands.

    The blade is so long (138 m) and the bondlines so thin (~50 mm
    chord) that a true-aspect overview compresses individual bricks
    into single pixels. ``chord_boost`` exaggerates the chord/thickness
    axes so the bondline tube is visible.
    """
    plt, Poly3DCollection = _require_matplotlib()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adh_nds, adh_polys = _adhesive_polys_by_bond(mesh)
    if len(adh_polys) == 0 or adh_nds.size == 0:
        return []

    written: list[Path] = []
    z_lo_total, z_hi_total = adh_nds[:, 2].min(), adh_nds[:, 2].max()
    x_lo, x_hi = adh_nds[:, 0].min(), adh_nds[:, 0].max()
    y_lo, y_hi = adh_nds[:, 1].min(), adh_nds[:, 1].max()

    def _draw_adhesive(ax, z_band):
        ax.computed_zorder = False
        zb_lo, zb_hi = z_band
        drawn = False
        for _tag, tri_verts, color in adh_polys:
            cz = tri_verts[:, :, 2].mean(axis=1)
            m = (cz >= zb_lo) & (cz <= zb_hi)
            if not m.any():
                continue
            pc = Poly3DCollection(
                tri_verts[m], facecolor=color, alpha=1.0,
                edgecolor=color, linewidth=0.1, shade=False, zorder=10,
            )
            ax.add_collection3d(pc)
            drawn = True
        return drawn

    # Overview
    fig = plt.figure(figsize=(8, 14))
    ax = fig.add_subplot(111, projection="3d")
    if _draw_adhesive(ax, (z_lo_total, z_hi_total)):
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi); ax.set_zlim(z_lo_total, z_hi_total)
        ax.set_box_aspect(((x_hi - x_lo) * chord_boost,
                           (y_hi - y_lo) * chord_boost,
                           z_hi_total - z_lo_total))
        ax.view_init(elev=20, azim=-60)
        ax.set_title("Adhesive bondlines — overview (chord exaggerated)")
        out = out_dir / "adhesive_overview.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        written.append(out)
    plt.close(fig)

    # Per-band zooms
    span = z_hi_total - z_lo_total
    n_bands = max(1, int(np.ceil(span / zoom_band_size)))
    band_edges = np.linspace(z_lo_total, z_hi_total, n_bands + 1)
    for k in range(n_bands):
        zb_lo, zb_hi = band_edges[k], band_edges[k + 1]
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        if _draw_adhesive(ax, (zb_lo, zb_hi)):
            ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi); ax.set_zlim(zb_lo, zb_hi)
            ax.set_box_aspect(((x_hi - x_lo) * chord_boost,
                               (y_hi - y_lo) * chord_boost,
                               zb_hi - zb_lo))
            ax.view_init(elev=15, azim=-60)
            ax.set_title(f"Adhesive z=[{zb_lo:.1f}, {zb_hi:.1f}] m")
            out = out_dir / f"adhesive_zoom_z{zb_lo:05.1f}_{zb_hi:05.1f}.png".replace(" ", "0")
            fig.savefig(out, dpi=110, bbox_inches="tight")
            written.append(out)
        plt.close(fig)

    return written
