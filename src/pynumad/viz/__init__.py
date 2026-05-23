"""Blade-mesh visualisation for pyNuMAD.

A small, focused package for turning a pyNuMAD shell mesh into

* an interactive **HTML** view of the blade and adhesive bondlines
  (:func:`write_blade_html`), and
* one or more **PNG** snapshots — full-blade overviews, per-span
  zooms, per-material highlight panels, and adhesive-only views
  (:func:`write_overview_snapshots`, :func:`write_component_snapshots`,
  :func:`write_adhesive_snapshots`).

All of these accept the dict returned by
``pynumad.mesh_gen.shell_mesh_general``. They have no other coupling
into pyNuMAD's internals, so they can also be called on any mesh
dict that follows the same conventions:

* ``nodes`` — ``(N, 3)`` float array.
* ``elements`` — ``(M, 4+)`` int array (``-1`` pads triangles).
* ``sets.element`` — list of ``{"name": "<i>_<j>_<region>", "labels": [...]}``
  with labels **0-indexed** (the ANSYS deck writer adds ``+1`` itself).
* Optional: ``adhesiveNds``, ``adhesiveEls``, ``adhesiveBondSets`` for
  the 3-D adhesive volumes.

CLI
~~~

The package is also runnable as a module::

    python -m pynumad.viz <blade.yaml> <output_dir> [--h 0.45] [--no-adhesive]
                          [--html|--png|--all]

This loads the blade, builds the mesh, and writes whichever output
formats you asked for. ``--all`` (the default) produces the HTML +
overview PNGs + per-component PNGs + adhesive PNGs.

Heavy plotting deps (matplotlib for PNG, plotly for HTML) are
imported lazily so ``import pynumad.viz`` works on systems where
either is missing.
"""
from __future__ import annotations

from pynumad.viz._palette import (
    ADHESIVE_COLORS,
    CHORD_GROUP_TAGS,
    GROUP_COLORS,
    classify_adhesive_set_name,
    classify_set_name,
    get_color,
)
from pynumad.viz.interactive import write_blade_html
from pynumad.viz.snapshots import (
    write_adhesive_snapshots,
    write_component_snapshots,
    write_overview_snapshots,
)

__all__ = [
    # palette
    "CHORD_GROUP_TAGS",
    "GROUP_COLORS",
    "ADHESIVE_COLORS",
    "classify_set_name",
    "classify_adhesive_set_name",
    "get_color",
    # snapshot writers
    "write_overview_snapshots",
    "write_component_snapshots",
    "write_adhesive_snapshots",
    # interactive
    "write_blade_html",
]
