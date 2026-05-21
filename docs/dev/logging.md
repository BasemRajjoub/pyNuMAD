# pyNuMAD logging and debug sidecar

pyNuMAD ships with structured logging built on Python's standard `logging`
module plus an optional JSONL "debug sidecar" intended for machine consumption
(grep, `jq`, IDE-driven AI assistants).

In default use, pyNuMAD is silent — a `NullHandler` is attached to the
root logger so the library does not write to stderr unless you ask it to.

## Enabling logs

### Option 1 — vanilla Python logging

```python
import logging
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
```

After this, pyNuMAD's `INFO` and above show up in stderr as ordinary text lines.

### Option 2 — JSONL sidecar (recommended for debugging mesh failures)

Set `PYNUMAD_DEBUG_LOG` to a file path **before importing pyNuMAD**:

```bash
PYNUMAD_DEBUG_LOG=/tmp/pynumad.jsonl PYNUMAD_DEBUG_LEVEL=DEBUG python my_script.py
```

`PYNUMAD_DEBUG_LEVEL` is optional (defaults to `DEBUG`). Valid values are the
standard Python logging level names: `DEBUG`, `INFO`, `WARNING`, `ERROR`.

Or enable from Python:

```python
from pynumad._logging import enable_jsonl_sidecar
enable_jsonl_sidecar("/tmp/pynumad.jsonl", level=logging.DEBUG, truncate=True)
```

## Record schema

Every line in the JSONL file is one JSON object with these stable keys:

| key | type | description |
|-----|------|-------------|
| `ts` | string | ISO-8601 timestamp (local) |
| `level` | string | `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"` |
| `logger` | string | dotted logger name, e.g. `"pynumad.mesh_gen.shell_region"` |
| `msg` | string | human-readable message |
| `module` | string | source-file basename without extension |
| `func` | string | function name where the log call lives |
| `line` | int | source line number |
| `context` | object | structured fields (varies by log site, see below) |
| `exc` | string | only when an exception was attached |

`context` is the only freeform field. Everything callers pass via `extra=`
ends up there, with numpy arrays summarised when large (`{"_kind":"ndarray", "shape":[...], "dtype":"...", "min":..., "max":...}`).

## Context fields by stage

The `context.stage` field identifies the algorithmic step the record came from.
Known stages today:

### `entry` — `pynumad.mesh_gen.mesh_gen.shell_mesh_general`

```jsonc
{
  "stage": "entry",
  "elementSize": 0.30,
  "forSolid": false,
  "includeAdhesive": false,
  "stacks_shape": [12, 101],
  "swstacks_shape": [2, 101]
}
```

### `edge_nels` — per shell-patch edge count derivation

Emitted at WARNING level whenever opposite chord/span edges have unequal counts (the trigger condition for the node-pulling bug):

```jsonc
{
  "stage": "edge_nels",
  "region_name": "09_02_LP_TE_PANEL",
  "edge_lens": [0.93, 1.41, 0.74, 1.41],
  "nEl": [4, 5, 3, 5],
  "mismatch_chord": 1,
  "mismatch_span": 0
}
```

Use `jq 'select(.context.stage=="edge_nels" and .context.mismatch_chord != 0)'` to dump every cell that fires the bug.

### `structured_quad_entry` — `ShellRegion.createShellMesh`

```jsonc
{
  "stage": "structured_quad_entry",
  "region_name": "09_02_LP_TE_PANEL",
  "regType": "quad3",
  "edgeEls": [4, 5, 3, 5],
  "edges_matched": false
}
```

### `node_pull` — one of four snap branches in `ShellRegion`

```jsonc
{
  "stage": "node_pull",
  "region_name": "09_02_LP_TE_PANEL",
  "branch": "ee2_lt_ee0",   // also: ee0_lt_ee2, ee1_lt_ee3, ee3_lt_ee1
  "edgeEls": [4, 5, 3, 5],
  "snap_row": "top",
  "snap_count": 4,
  "row_count": 5
}
```

### `merge_duplicates` — `ShellRegion` post-snap node merge

```jsonc
{
  "stage": "merge_duplicates",
  "region_name": "09_02_LP_TE_PANEL",
  "nodes_before": 30,
  "nodes_after": 29,
  "nodes_merged": 1
}
```

### `exit` — `ShellRegion.createShellMesh` returning

```jsonc
{
  "stage": "exit",
  "region_name": "09_02_LP_TE_PANEL",
  "n_nodes": 29,
  "n_elements": 20,
  "moved": true
}
```

Only emitted at DEBUG level.

## Example: which regions hit the bug at 0.30 m?

```bash
PYNUMAD_DEBUG_LOG=/tmp/iea22-30cm.jsonl PYNUMAD_DEBUG_LEVEL=INFO \
    python paper2_fem/00_smoke_test/run_modal_only.py --esize 0.30

jq -r 'select(.context.stage=="node_pull") | .context.region_name' /tmp/iea22-30cm.jsonl \
    | sort | uniq -c | sort -rn | head
```

You'll get a frequency-ranked list of the regions that fire the node-pull
code path — the suspects most likely to host warped quads.

## Common `jq` recipes

```bash
# All warnings
jq 'select(.level=="WARNING")' /tmp/pynumad.jsonl

# All node-pull events, in order
jq 'select(.context.stage=="node_pull")' /tmp/pynumad.jsonl

# Histogram of which branch fires
jq -r 'select(.context.branch != null) | .context.branch' /tmp/pynumad.jsonl \
    | sort | uniq -c | sort -rn

# Cells with the largest mismatch_chord
jq 'select(.context.stage=="edge_nels" and (.context.mismatch_chord // 0 | fabs) >= 2)' /tmp/pynumad.jsonl
```

## Adding new instrumentation

Inside pyNuMAD code:

```python
from pynumad._logging import get_logger
_log = get_logger(__name__)

_log.debug(
    "something happened",
    extra={"stage": "my_stage", "region_name": rname, "n_nodes": n},
)
```

Conventions:

- Use module-level `_log = get_logger(__name__)`; this nests under `pynumad.<module>` automatically.
- Always set `context.stage` to a short string identifier — that's how downstream tooling slices the log.
- Cheap fields only. Avoid dumping multi-thousand-node arrays; the formatter auto-summarises arrays > 64 elements but doesn't know about lists of dicts.
- DEBUG = per-element trace; INFO = stage transitions; WARNING = recoverable pathology (low Jacobian, edge mismatch); ERROR = invariant violation about to raise.
