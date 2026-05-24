# `scripts/`

Standalone runnable scripts that build artefacts (figures, decks,
quick mesh-quality reports) for the `pce-patches` fork. Each script
is self-contained — runs with the project's pixi env, no extra args
unless documented in the file header.

| file | purpose | produces | dependencies |
|---|---|---|---|
| [`plot_solid_mesh_fix_stages.py`](plot_solid_mesh_fix_stages.py) | Render the 4-panel before/after visualisation of the three-stage mesh-quality treatment in `solidMeshFromShell` (legacy → smoothing → adaptive clamp → untangle). Close-up zoom on the worst BAR0 TE region. | `docs/dev/figs/solid_mesh_fix_stages.{pdf,png,preview.png}` | matplotlib, BAR0 test fixture |

## Run

```bash
PATH="$PATH" ~/.pixi/bin/pixi run -- python scripts/<script>.py
```

(per the `pce-patches` Pixi-on-HPC convention — see `my-style/`).
