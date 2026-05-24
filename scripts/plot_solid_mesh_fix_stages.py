"""Plot the bad-Jacobian element population at each fix stage of
``solidMeshFromShell``. Visual verification that the three-stage
treatment (normal smoothing + adaptive layer-thickness clamp +
post-extrusion untangle) cleans up the mesh and that the final mesh
looks geometrically reasonable.

Output:
    docs/dev/figs/solid_mesh_fix_stages.{pdf,png}
    docs/dev/figs/solid_mesh_fix_stages.preview.png  (low-res for AI check)

Run:
    python scripts/plot_solid_mesh_fix_stages.py

Per-stage configuration (passed via kwargs to solidMeshFromShell):
    stage 0  legacy             : n_smooth=0, cap=None, untangle=0
    stage 1  + smoothing        : n_smooth=2, cap=None, untangle=0
    stage 2  + adaptive cap     : n_smooth=2, cap=0.7,  untangle=0
    stage 3  + untangle (final) : n_smooth=2, cap=0.7,  untangle=30 (default)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pynumad.mesh_gen.mesh_gen import get_shell_mesh as _get_shell_mesh  # noqa: F401
from pynumad.mesh_gen.mesh_gen import shell_mesh_general, solidMeshFromShell
from pynumad.mesh_gen.mesh_tools import check_all_jacobians
from pynumad.objects.blade import Blade


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
YAML = REPO / "src" / "pynumad" / "tests" / "test_data" / "blades" / "blade.yaml"
OUT_DIR = REPO / "docs" / "dev" / "figs"

# --- Stage configurations ---
STAGES = [
    ("legacy", dict(n_normal_smoothing_iter=0,
                    layer_thickness_cap_factor=None,
                    untangle_max_iter=0)),
    ("+ normal smoothing", dict(n_normal_smoothing_iter=2,
                                layer_thickness_cap_factor=None,
                                untangle_max_iter=0)),
    ("+ adaptive thickness clamp", dict(n_normal_smoothing_iter=2,
                                        layer_thickness_cap_factor=0.7,
                                        untangle_max_iter=0)),
    ("+ untangle (final)", dict(n_normal_smoothing_iter=2,
                                layer_thickness_cap_factor=0.7,
                                untangle_max_iter=30)),
]


def _build_stage(blade, shell, layers, elementSize, **kwargs):
    """Build one solid mesh with the given stage kwargs; return the
    mesh + list of bad-Jacobian element indices."""
    mesh = solidMeshFromShell(blade, shell, layers, elementSize, **kwargs)
    bad = sorted(check_all_jacobians(mesh["nodes"], mesh["elements"]))
    return mesh, bad


def _brick_faces(nodes, el):
    """Return list of 6 face polygons (4 corners each) for an 8-node hex,
    or 5 faces (2 triangles + 3 quads) for a 6-node wedge."""
    is_wedge = (el[6] == -1)
    if is_wedge:
        n = [nodes[el[i]] for i in (0, 1, 2, 3, 4, 5)]
        return [
            [n[0], n[1], n[2]],        # bottom tri
            [n[3], n[4], n[5]],        # top tri
            [n[0], n[1], n[4], n[3]],  # side 1
            [n[1], n[2], n[5], n[4]],  # side 2
            [n[2], n[0], n[3], n[5]],  # side 3
        ]
    n = [nodes[el[i]] for i in range(8)]
    return [
        [n[0], n[1], n[2], n[3]],  # bottom
        [n[4], n[5], n[6], n[7]],  # top
        [n[0], n[1], n[5], n[4]],  # side y-low
        [n[1], n[2], n[6], n[5]],  # side x-high
        [n[2], n[3], n[7], n[6]],  # side y-high
        [n[3], n[0], n[4], n[7]],  # side x-low
    ]


def _zoom_box(coords, padding=0.15, min_half=None):
    """Compute axis-equal zoom box around a coord cloud with padding.

    ``min_half`` forces a minimum half-extent regardless of the cloud
    size — useful when the bad elements are tiny relative to the
    blade but you still want a sensible viewing volume."""
    if len(coords) == 0:
        return None
    mn, mx = coords.min(axis=0), coords.max(axis=0)
    ctr = 0.5 * (mn + mx)
    half = 0.5 * (mx - mn).max() * (1 + padding)
    if min_half is not None:
        half = max(half, min_half)
    return ctr - half, ctr + half


def _nearby_element_indices(mesh, ref_coords, radius):
    """Indices of elements whose centroid is within radius of any
    coordinate in ref_coords — used to draw mesh edges in the zoom
    region for geometric context."""
    nodes = mesh["nodes"]
    elements = mesh["elements"]
    ref_ctr = ref_coords.mean(axis=0)
    out = []
    for ei in range(len(elements)):
        valid = [n for n in elements[ei] if n >= 0]
        c = nodes[valid].mean(axis=0)
        if np.linalg.norm(c - ref_ctr) < radius:
            out.append(ei)
    return out


def _render_panel(ax, mesh, bad, title, zoom_box=None,
                  context_indices=None):
    """Render a panel: light grey wireframe for nearby (good) elements
    as context, plus red filled bricks for the bad ones."""
    nodes = mesh["nodes"]
    elements = mesh["elements"]
    # Context: nearby elements as light grey wireframes (no fill).
    if context_indices is not None:
        ctx_polys = []
        for ei in context_indices:
            ctx_polys.extend(_brick_faces(nodes, elements[ei]))
        ctx = Poly3DCollection(
            ctx_polys,
            facecolor=(0.85, 0.85, 0.85, 0.04),
            edgecolor=(0.3, 0.3, 0.3, 0.35),
            linewidth=0.15,
        )
        ax.add_collection3d(ctx)
    # Bad elements: solid red fill.
    if len(bad) > 0:
        bad_polys = []
        for ei in bad:
            bad_polys.extend(_brick_faces(nodes, elements[ei]))
        bad_coll = Poly3DCollection(
            bad_polys,
            facecolor=(1.0, 0.25, 0.25, 0.85),
            edgecolor="darkred",
            linewidth=0.5,
        )
        ax.add_collection3d(bad_coll)
        title_suffix = f"\n{len(bad)} bad-Jacobian element(s)"
    else:
        title_suffix = "\n0 bad-Jacobian elements ✓"
    ax.set_title(title + title_suffix, fontsize=11)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    if zoom_box is not None:
        (mn, mx) = zoom_box
        ax.set_xlim(mn[0], mx[0])
        ax.set_ylim(mn[1], mx[1])
        ax.set_zlim(mn[2], mx[2])
        try:
            ax.set_box_aspect((mx - mn))
        except AttributeError:
            pass  # older matplotlib


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    blade = Blade(str(YAML))
    blade.stackdb.edit_stacks_for_solid_mesh()
    print("building seed shell mesh …")
    shell = shell_mesh_general(blade, 1, 1, 0.5)

    results = []
    for name, kw in STAGES:
        print(f"running stage: {name}  kwargs={kw}")
        m, bad = _build_stage(blade, shell, [1, 1, 1], 0.5, **kw)
        results.append((name, m, bad))
        print(f"  -> {len(bad)} bad elements")

    # Picking a focused worst-case bad brick from stage 0, then zooming
    # all panels to a small box around it — this makes the geometric
    # change visible at brick scale rather than blade scale.
    stage0_mesh = results[0][1]
    stage0_bad = results[0][2]
    ref_brick = stage0_bad[0]  # pick the first failing element
    ref_nodes = stage0_mesh["nodes"][[
        n for n in stage0_mesh["elements"][ref_brick] if n >= 0
    ]]
    zoom = _zoom_box(ref_nodes, padding=0.20, min_half=0.6)  # ~1.2 m cube
    ctx_radius = 0.8  # m — neighbours to draw for context

    # Per-panel context indices (different per stage because mesh nodes
    # shift slightly), all centered on the same physical region.
    context_per_stage = [
        _nearby_element_indices(m, ref_nodes, ctx_radius) for _, m, _ in results
    ]

    # 1-row x 4-col panel — close-up.
    fig = plt.figure(figsize=(16, 4.8))
    for i, (name, mesh, bad) in enumerate(results):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        _render_panel(ax, mesh, bad, name, zoom_box=zoom,
                      context_indices=context_per_stage[i])
        ax.view_init(elev=18, azim=-72)

    fig.suptitle(
        "solidMeshFromShell — bad-Jacobian elements at each industry-"
        "standard fix stage (BAR0, elementSize=0.5, layerNumEls=[1,1,1])\n"
        "Close-up on the worst-case TE region (grey = good neighbours, "
        "red = bad-Jacobian bricks)",
        fontsize=11, y=1.04,
    )
    fig.tight_layout()
    pdf = OUT_DIR / "solid_mesh_fix_stages.pdf"
    png = OUT_DIR / "solid_mesh_fix_stages.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=140)
    fig.savefig(OUT_DIR / "solid_mesh_fix_stages.preview.png",
                bbox_inches="tight", dpi=70)
    plt.close(fig)
    print(f"wrote {pdf}, {png}, .preview.png")


if __name__ == "__main__":
    main()
