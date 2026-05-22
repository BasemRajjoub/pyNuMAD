# Shell-mesh continuity in pyNuMAD

## TL;DR

* Every per-material chord group on the blade outer shell (HP_SPAR,
  HP_TE_REINF, HP_LE, HP_TE_PANEL, …) is a single node-connected
  component. Every element of the group shares at least one node with
  another element of the same group. This is verified by
  [test_mesh_continuity.py](../../src/pynumad/tests/test_mesh_continuity.py)
  on the BAR0 fixture and any future change that fragments a chord
  group will fail that test.
* Within a group, **most** elements also share full edges with their
  neighbours, but the larger panel regions (HP/LP_TE_PANEL,
  HP/LP_LE_PANEL) and TE_FLAT have internal T-junctions where one row
  of nodes lands in the middle of the neighbour's edge. ANSYS handles
  those via the constraint equations pyNuMAD emits in
  [`analysis/ansys/write.py`](../../src/pynumad/analysis/ansys/write.py).
* Shear webs (SW) are N physical components, one per web in the windIO
  definition. They are intentionally not part of the chord-group
  continuity test.

## Why T-junctions appear at all

[`mesh_gen.shell_mesh_general`](../../src/pynumad/mesh_gen/mesh_gen.py)
walks the blade station by station (outer loop `for i in range(rws - 1)`).
For each station it builds 12 chord regions
(`for j in range(0, 12)`) — HP_TE_FLAT, HP_TE_REINF, HP_TE_PANEL,
HP_SPAR, HP_LE_PANEL, HP_LE, LP_LE, LP_LE_PANEL, LP_SPAR, LP_TE_PANEL,
LP_TE_REINF, LP_TE_FLAT.

Each region is a [`ShellRegion("quad3", …)`](../../src/pynumad/mesh_gen/shell_region.py)
patch using 4 spline columns. The chord and spanwise element counts
come from `_compute_edge_nels(shellKp, elementSize)` and are derived
from the edge lengths of each region independently.

Because each region computes its own element counts, two adjacent
chord regions can disagree on the spanwise subdivision. When that
happens, their shared boundary in arc-length space matches but the
node densities don't, so one row's pair of nodes lands between the
neighbour's nodes — a T-junction with a hanging node.

The clean fix would be to compute the spanwise count *once per
station* across all 12 regions (the commented-out
`secNel = [1,4,10,10,8,2,2,8,10,10,4,1]` hint at
[mesh_gen.py:791](../../src/pynumad/mesh_gen/mesh_gen.py) appears to
be what was originally intended). That is a multi-day refactor that
touches every fixture-based test and is deferred until somebody
needs strict edge-connectivity (e.g. for adjoint sensitivity at
ply-level granularity).

## Pitfall: label indexing

`mesh["sets"]["element"][k]["labels"]` is **0-indexed**. The ANSYS
deck writer at
[`analysis/ansys/write.py:1272`](../../src/pynumad/analysis/ansys/write.py)
adds `+1` on the fly when emitting EMODIF cards because ANSYS uses
1-indexed element numbering.

If you read the labels directly and subtract 1 you'll be reading the
*wrong* element and may convince yourself the mesh is broken when it
isn't. Several of the snapshot/diagnostic scripts in `paper2_fem`
had this bug before the fix in commit
[6fd4fd9](https://gitlab.uni-hannover.de/b.rajjoub/pce/-/commit/6fd4fd9).

## How to inspect connectivity yourself

```python
from pynumad.testing import mesh_continuity as mc

mesh = ...  # output of shell_mesh_general

# Node-connectivity (treat sharing any node as connected; T-junctions
# count as connected). Expect 1 per chord group, N for SW.
sizes_node = mc.all_group_components(mesh, min_shared_nodes=1)

# Edge-connectivity (require a shared edge = 2 nodes). Strictly
# stronger than node-connectivity; numbers are bigger because
# T-junctions count as cuts.
sizes_edge = mc.all_group_components(mesh, min_shared_nodes=2)

for tag in sizes_node:
    print(f"{tag:<14s} node-cc={len(sizes_node[tag])}  "
          f"edge-cc={len(sizes_edge[tag])}")
```

`mc.stray_elements(mesh, group_tag)` returns 0-indexed element ids
that share no nodes with any other element of the same chord group
— a singleton island is almost always a labelling bug.

## What we currently observe (IEA-22 at h = 0.80 m)

| group | n_els | node-cc | edge-cc |
|---|---:|---:|---:|
| HP_SPAR     | 368 | 1 | 2 |
| LP_SPAR     | 366 | 1 | 2 |
| HP_TE_REINF | 188 | 1 | 1 |
| LP_TE_REINF | 188 | 1 | 1 |
| HP_LE       | 188 | 1 | 1 |
| LP_LE       | 188 | 1 | 1 |
| HP_TE_FLAT  | 108 | 1 | 3 |
| LP_TE_FLAT  | 108 | 1 | 3 |
| HP_TE_PANEL | 746 | 1 | 6 |
| LP_TE_PANEL | 682 | 1 | 7 |
| HP_LE_PANEL | 371 | 1 | 5 |
| LP_LE_PANEL | 414 | 1 | 6 |
| SW          | 968 | 3 | 14 |

Spars, TE_REINF, and LE chord strips are fully edge-connected (1
edge-cc). The wider panels and TE_FLAT have internal T-junctions; SW
has 3 webs (correct) plus internal edge-fragmentation within each
web (same T-junction reason).
