"""Interactive HTML plot of a pyNuMAD blade mesh.

``write_blade_html(mesh, path)`` builds a single self-contained HTML
file with a 3-D plotly view of the shell mesh and (optionally) the
3-D solid adhesive bondlines. The file embeds plotly.js inline, has
no external dependencies, and is safe to open from ``file://`` URLs
(no CORS errors from the toolbar icons).

Features
~~~~~~~~
* Click legend items to toggle individual material zones / bondlines
  on and off.
* Group click ("Spar caps" header etc.) bulk-toggles the whole
  category.
* A side panel lists every trace's name, color, triangle count, and
  node-coverage so you can sanity-check what's being drawn.
* A console-logger panel mirrors plotly's internal traces so the
  page works as a self-debugging report (no devtools needed).

Output size
~~~~~~~~~~~
The HTML embeds plotly.js (~3 MB) plus the mesh data. Expect 5-15 MB
for IEA-22 at h=0.45 m. Run with a coarser ``elementSize`` if you
need a smaller file.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np

from pynumad.viz._mesh_utils import (
    collect_adhesive_bond_indices,
    collect_chord_group_indices,
    quads_to_triangles,
    solid_outer_face_triangles,
)
from pynumad.viz._palette import (
    ADHESIVE_COLORS,
    CHORD_GROUP_TAGS,
    DEFAULT_ADHESIVE_COLOR,
    GROUP_COLORS,
    LEGEND_GROUPS,
)


def _require_plotly():
    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plotly is required for interactive HTML output. "
            "Install with `pip install plotly`."
        ) from exc
    import plotly.graph_objects as go
    return go


# Regex of external host patterns that plotly might reference and that
# Chrome blocks under file:// origin. We scrub all such URLs from the
# emitted HTML to keep the page self-contained.
_SCRUB_PATTERNS = (
    re.compile(r"https?://unpkg\.com[^\"\)\s]*"),
    re.compile(r"https?://cdn\.[a-z]+\.[a-z]+[^\"\)\s]*"),
    re.compile(r"https?://[a-z]+\.mapbox\.com[^\"\)\s]*"),
    re.compile(r"https?://cdn\.plot\.ly[^\"\)\s]*"),
    re.compile(r"https?://api\.mapbox\.com[^\"\)\s]*"),
    re.compile(r"https?://api\.tiles\.mapbox\.com[^\"\)\s]*"),
)


def _scrub_external_urls(html: str) -> tuple[str, int]:
    """Replace external URLs with ``about:blank`` so file:// pages work."""
    n = 0
    for pat in _SCRUB_PATTERNS:
        html, count = pat.subn("about:blank", html)
        n += count
    return html, n


def _build_shell_traces(mesh, nodes, elements):
    go = _require_plotly()
    groups = collect_chord_group_indices(mesh)
    traces = []
    for tag in CHORD_GROUP_TAGS:
        if tag not in groups:
            continue
        idx = np.asarray(sorted(set(groups[tag])), int)
        idx = idx[(idx >= 0) & (idx < len(elements))]
        if idx.size == 0:
            continue
        conn = elements[idx][:, :4]
        tris = quads_to_triangles(conn)
        if tris.size == 0:
            continue
        color = GROUP_COLORS.get(tag, "#cccccc")
        legendgroup = LEGEND_GROUPS.get(tag, "Other shell")
        traces.append(go.Mesh3d(
            x=nodes[:, 0], y=nodes[:, 1], z=nodes[:, 2],
            i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
            name=f"{tag} ({len(idx)} el)",
            color=color, opacity=0.9, flatshading=True,
            legendgroup=legendgroup, legendgrouptitle_text=legendgroup,
            showlegend=True,
        ))
    return traces


def _build_adhesive_traces(mesh):
    go = _require_plotly()
    adh_nds = np.asarray(mesh.get("adhesiveNds", []), float)
    adh_els = np.asarray(mesh.get("adhesiveEls", []), int)
    if adh_nds.size == 0 or adh_els.size == 0:
        return []
    bonds = collect_adhesive_bond_indices(mesh)
    traces = []
    for tag, ids in bonds.items():
        idx = np.asarray(sorted(set(int(i) for i in ids)), int)
        idx = idx[(idx >= 0) & (idx < len(adh_els))]
        if idx.size == 0:
            continue
        tris = solid_outer_face_triangles(adh_els[idx])
        if tris.size == 0:
            continue
        color = ADHESIVE_COLORS.get(tag, DEFAULT_ADHESIVE_COLOR)
        traces.append(go.Mesh3d(
            x=adh_nds[:, 0], y=adh_nds[:, 1], z=adh_nds[:, 2],
            i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
            name=f"{tag} adhesive ({len(idx)} el)",
            color=color, opacity=1.0, flatshading=False,
            lighting=dict(ambient=0.7, diffuse=0.8, specular=0.1,
                          roughness=0.5, fresnel=0.1),
            legendgroup="Adhesive bondlines",
            legendgrouptitle_text="Adhesive bondlines",
            showlegend=True,
        ))
    return traces


_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  html, body {{ margin: 0; padding: 0; height: 100%; width: 100%;
                overflow: hidden; font-family: -apple-system, system-ui, sans-serif; }}
  #plot-host, #plot-host > div, #plot-host .plotly-graph-div,
  #plot-host .js-plotly-plot {{
    width: 100% !important; height: 100% !important; }}
  #plot-host {{ position: absolute; top: 0; bottom: 0; left: 0; right: 380px; }}
  #side-panel {{ position: absolute; top: 0; right: 0; width: 380px;
                 bottom: 0; padding: 12px; overflow-y: auto;
                 background: #fafafa; border-left: 1px solid #ddd;
                 font-size: 11px; box-sizing: border-box; }}
  #side-panel h3 {{ margin: 0 0 8px 0; font-size: 13px; }}
  #side-panel table {{ width: 100%; border-collapse: collapse;
                       font-size: 10px; }}
  #side-panel th, #side-panel td {{ padding: 2px 4px;
                                    border-bottom: 1px solid #eee; }}
  #side-panel th:first-child, #side-panel td:first-child {{ text-align: left; }}
  #side-panel .group-header {{ background: #f0f0f0; font-weight: bold; }}
  #side-panel .swatch {{ display: inline-block; width: 10px; height: 10px;
                         margin-right: 4px; vertical-align: middle;
                         border: 1px solid rgba(0,0,0,0.2); }}
  #log {{ font-family: ui-monospace, monospace; font-size: 10px;
          background: #1e1e1e; color: #d4d4d4; padding: 6px;
          height: 180px; overflow-y: auto; white-space: pre; }}
  button {{ background: #007acc; color: white; border: none;
            padding: 4px 10px; cursor: pointer; font-size: 11px;
            border-radius: 3px; }}
  button:hover {{ background: #005a9e; }}
</style>
</head>
<body>
<div id="plot-host">{plot_html}</div>
<div id="side-panel">
  <h3>Trace inventory</h3>
  <table id="trace-table">
    <thead><tr><th>Trace</th><th>Tri</th><th>Nodes</th></tr></thead>
    <tbody>
{trace_rows}
    </tbody>
  </table>
  <h3 style="margin-top:14px;">Console</h3>
  <button onclick="copyLog()">copy log</button>
  <div id="log"></div>
</div>
<script>
  const logEl = document.getElementById('log');
  function logLine(s) {{
    logEl.textContent += s + '\\n';
    logEl.scrollTop = logEl.scrollHeight;
  }}
  function copyLog() {{
    navigator.clipboard.writeText(logEl.textContent);
    logLine('[log copied to clipboard]');
  }}
  window.addEventListener('DOMContentLoaded', () => {{
    const host = document.getElementById('plot-host');
    const inner = host.querySelector('.js-plotly-plot');
    logLine('blade-viz loaded; plotly element: ' + (inner ? 'OK' : 'MISSING'));
    if (inner && window.Plotly) {{
      window.Plotly.Plots.resize(inner);
      window.addEventListener('resize', () => window.Plotly.Plots.resize(inner));
    }}
    setTimeout(() => {{
      const traces = (inner && inner.data) || [];
      logLine('traces loaded: ' + traces.length);
      traces.forEach((tr, i) => {{
        logLine(`  data[${{i}}].type=${{tr.type}}  visible=${{tr.visible}}  ` +
                `n_x=${{tr.x ? tr.x.length : '?'}}  n_i=${{tr.i ? tr.i.length : '?'}}`);
      }});
    }}, 1000);
  }});
</script>
</body>
</html>"""


def write_blade_html(
    mesh: dict,
    output_path: str | Path,
    *,
    include_adhesive: bool = True,
    title: str = "pyNuMAD blade mesh",
) -> Path:
    """Write a self-contained interactive HTML view of the blade.

    Parameters
    ----------
    mesh
        Output of ``shell_mesh_general``.
    output_path
        Destination file (any existing file is overwritten).
    include_adhesive
        Whether to add the 3-D adhesive volume traces.
    title
        ``<title>`` text for the HTML page.

    Returns
    -------
    Path to the written file.
    """
    go = _require_plotly()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nodes = np.asarray(mesh["nodes"], float)
    elements = np.asarray(mesh["elements"], int)

    fig = go.Figure()
    shell_traces = _build_shell_traces(mesh, nodes, elements)
    for tr in shell_traces:
        fig.add_trace(tr)
    if include_adhesive:
        for tr in _build_adhesive_traces(mesh):
            fig.add_trace(tr)

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="chord x [m]", showbackground=True),
            yaxis=dict(title="thickness y [m]", showbackground=True),
            zaxis=dict(title="span z [m]", showbackground=True),
            aspectmode="data",
        ),
        showlegend=True,
        legend=dict(
            itemsizing="constant",
            x=1.0, y=1.0, xanchor="right", yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#666", borderwidth=1,
            groupclick="toggleitem",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        autosize=True,
    )

    config = dict(
        displayModeBar=True,
        # Drop toolbar buttons that load icons from external CDNs;
        # the remaining ones use the inline plotly icons.
        modeBarButtonsToRemove=["toImage", "sendDataToCloud", "lasso2d",
                                "select2d", "hoverClosest3d", "resetCameraDefault3d",
                                "resetCameraLastSave3d"],
        responsive=True,
    )
    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        config=config,
        div_id="blade-plot-div",
    )

    # Trace inventory rows
    rows = []
    last_group = None
    for tr in fig.data:
        n_tri = len(tr.i) if hasattr(tr, "i") and tr.i is not None else 0
        if n_tri:
            referenced = set(int(v) for v in tr.i) | set(int(v) for v in tr.j) | set(int(v) for v in tr.k)
            n_nodes_used = len(referenced)
        else:
            n_nodes_used = 0
        group = getattr(tr, "legendgroup", "") or "Other"
        if group != last_group:
            rows.append(f"      <tr class='group-header'><td colspan='3'>{group}</td></tr>")
            last_group = group
        color = getattr(tr, "color", None) or "#cccccc"
        rows.append(
            f"      <tr><td><span class='swatch' style='background:{color};'></span>"
            f"{tr.name}</td><td>{n_tri}</td><td>{n_nodes_used}</td></tr>"
        )
    trace_rows = "\n".join(rows)

    page = _PAGE_TEMPLATE.format(
        title=title,
        plot_html=plot_html,
        trace_rows=trace_rows,
    )
    page, n_scrubbed = _scrub_external_urls(page)

    output_path.write_text(page, encoding="utf-8")
    return output_path
