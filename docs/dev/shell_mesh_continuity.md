# Shell-mesh continuity in pyNuMAD

## TL;DR

* pyNuMAD's blade shell mesh is **T-junction-based by design** — chord
  regions (HP_SPAR, HP_TE_REINF, …) share corner nodes with their
  neighbours but rarely share full edges. ANSYS handles the
  T-junctions via the constraint equations pyNuMAD emits.
* If you plot a single material zone in isolation and count
  connected components, you get **node-connected = 1, edge-connected
  ≈ #elements** for any well-formed zone — that's normal.
* There is one **known asymmetry**: on the HP side of the trailing
  edge, the `HP_TE_REINF` zone fragments into many node-connected
  components while `LP_TE_REINF` stays at 1. Source: keypoint clamps
  in [keypoints.py:200-210](../../src/pynumad/objects/keypoints.py)
  (HP) vs [keypoints.py:240-256](../../src/pynumad/objects/keypoints.py)
  (LP).

## Why T-junctions

`mesh_gen.shell_mesh_general` walks the blade station by station
(outer loop `for i in range(rws - 1)`). For each station it builds 12
chord regions (`for j in range(0, 12)`) — HP_TE_FLAT, HP_TE_REINF,
HP_TE_PANEL, HP_SPAR, HP_LE_PANEL, HP_LE, LP_LE, LP_LE_PANEL, LP_SPAR,
LP_TE_PANEL, LP_TE_REINF, LP_TE_FLAT.

Each region is a `ShellRegion("quad3", ...)` patch using the 4 spline
columns `stSp .. stSp+3`. The chord and spanwise element counts come
from `_compute_edge_nels(shellKp, elementSize)` — they are derived
from the edge lengths of each region independently.

Because each region computes its own element counts, two adjacent
chord regions can disagree on the spanwise subdivision. When that
happens, the shared chord boundary between them gets one row of
hanging nodes — i.e. T-junctions. Same thing happens spanwise.

The fix would be to compute the spanwise count *once per station*
across all 12 regions and the chordwise count *once per region* and
propagate them — basically what the commented-out
`secNel = [1,4,10,10,8,2,2,8,10,10,4,1]` hint at
[mesh_gen.py:791](../../src/pynumad/mesh_gen/mesh_gen.py) suggests was
the original intent. That is a multi-day refactor that touches every
test using the BAR0 / IEA-22 fixtures, so we have not done it.

## What ANSYS does with the T-junctions

`write.py` emits `CE,` (constraint equation) records that tie each
hanging node to the two endpoints of the edge it lands on. The
resulting matrix is non-singular and the static / modal results
converge cleanly on the IEA-22 fixture down to `h=0.10 m`. So the
T-junctions are an aesthetic and FE-quality concern, not a
correctness blocker.

## The HP/LP TE_REINF asymmetry

At each spanwise station, both sides have a "TE-FLAT" region (0 width
when the airfoil's TE is sharp, finite width when flat-back). On the
LP side, when TE goes sharp the inner d-keypoint is clamped *outward*
(`d <= 0.96 * arclength[nf, k]`, [keypoints.py:249](../../src/pynumad/objects/keypoints.py)).
On the HP side it's clamped *inward* (`d >= 0.98 * arclength[ns, k]`,
[keypoints.py:203](../../src/pynumad/objects/keypoints.py)). Net
effect: HP TE_REINF often gets one narrow (~0.02 m chord) "pinch"
quad and one wide quad per station, where LP gets two wide quads.
The narrow pinch elements at different stations don't share nodes
with each other → many singleton node-components in HP_TE_REINF.

Concrete IEA-22 numbers at `h = 0.80 m`:

| group | total els | node-cc | narrow (chord < 0.1 m) |
|---|---|---|---|
| HP_TE_REINF | 188 | 105 | 32 |
| LP_TE_REINF | 188 |   1 |  0 |
| HP_TE_FLAT  | 107 |   1 | 69 |
| LP_TE_FLAT  | 108 |   5 | 34 |

The structural impact is small (the narrow elements are < 0.05 % of
total mass and the residual-tie CE statements handle the load path),
but if anyone fixes the keypoint clamps to make HP symmetric with LP,
they should:

1. Run [test_mesh_continuity.py](../../src/pynumad/tests/test_mesh_continuity.py)
   — `test_hp_side_fragmentation_within_baseline` will fail by being
   too lenient. Tighten the baseline.
2. Update the IEA-22 mass-/EI- regression fixture if those numbers
   shift more than ~0.5 %.

## How to look at the mesh yourself

```python
from pynumad.testing import mesh_continuity as mc

mesh = ...  # output of shell_mesh_general
sizes = mc.all_group_components(mesh, min_shared_nodes=1)
for tag, cc in sizes.items():
    print(f"{tag:<14s} {len(cc):>4d} components,  largest = {cc[0]}")
```

For edge-connectivity (treats T-junctions as cuts), pass
`min_shared_nodes=2`. The numbers will be much larger; that's
expected.

To list singleton-island elements in a specific group (almost always
a labeling bug):

```python
strays = mc.stray_elements(mesh, "HP_TE_REINF")
# 0-indexed element ids
```
