# Mesh-Quality Wall — Algorithm Trace and Root Cause

**Status:** open. Tracked by Phase-1 reproducer tests (see `mesh_quality_wall_plan.md`).
**Symptom:** Jacobian-flipped quads at the root transition of the IEA-22 blade. Flip count grows monotonically with refinement (19 @ 0.80 m → 226 @ 0.10 m). Below ≈ 0.40 m element size ANSYS aborts. Canonical reproducer: element 1571 in the 0.30 m IEA-22 mesh, at z ≈ 6.21 m, with edges 0.054 / 0.282 / 0.099 / 0.282 m.

---

## 1. Where the bug lives

Two cooperating sites:

- **Site A (per-region node-pulling):** [src/pynumad/mesh_gen/shell_region.py:88-200](../../src/pynumad/mesh_gen/shell_region.py) — `ShellRegion.createShellMesh` when `meshMethod == "structured"` and `regType in {"quad1","quad2","quad3"}`.
- **Site B (per-cell `nEl` derivation):** [src/pynumad/mesh_gen/mesh_gen.py:319-341](../../src/pynumad/mesh_gen/mesh_gen.py) — inside the spanwise/chordwise double loop in `get_shell_mesh`, each (panel `i`, stack `j`) cell computes its four edge element counts independently from local chord lengths.

The handoff documents Site A. The deeper trigger is Site B: it produces the `edgeEls` mismatch that fires Site A's bad code path.

---

## 2. Site B — how `edgeEls` ends up unequal on opposite edges

In `get_shell_mesh` ([mesh_gen.py:212-365](../../src/pynumad/mesh_gen/mesh_gen.py)):

- Outer loop `for i in range(rws - 1)` walks spanwise stations.
- Inner loop `for j in range(stSec, endSec+1)` walks chordwise stacks (0…11).
- Each iteration constructs a 16-keypoint `shellKp` patch and computes its four edge counts (lines 319-341):

```python
vec = shellKp[1] - shellKp[0];  nEl1 = ceil(|vec| / elementSize)   # chord edge at span stPt
vec = shellKp[2] - shellKp[1];  nEl2 = ceil(|vec| / elementSize)   # span edge at chord stSp+3
vec = shellKp[3] - shellKp[2];  nEl3 = ceil(|vec| / elementSize)   # chord edge at span stPt+3
vec = shellKp[0] - shellKp[3];  nEl4 = ceil(|vec| / elementSize)   # span edge at chord stSp
nEl = [nEl1, nEl2, nEl3, nEl4]
```

`nEl1` and `nEl3` are the **opposite chordwise edges** of the same cell. Because the blade tapers and the airfoil chord changes with span, `|shellKp[1] - shellKp[0]|` ≠ `|shellKp[3] - shellKp[2]|` in the root transition — the cell is trapezoidal in chord. With `ceil(/elementSize)` and a fine `elementSize`, this geometric difference becomes a discrete element-count difference: `nEl1 != nEl3`.

Sandia left a commented hint at lines 203-205 and 324, 335:

```python
## secNel = [1,3,15,5,8,1,1,13,5,12,3,1]   # line 203
## nEl1 = secNel[j]                         # line 324
## nEl3 = secNel[j]                         # line 335
```

That hint is the right shape of the fix: opposite edges should share a precomputed per-stack count, not be derived independently per cell.

### Knock-on: inter-region connectivity

Adjacent cells share one edge geometrically. With per-cell `ceil()`, two neighbours can pick **different** counts for the **same** shared edge:

- Cell (i, j)'s edge 2 (top, at `stPt+3`) and cell (i+1, j)'s edge 0 (bottom, at `stPt+3`) are the same physical edge.
- The two `ceil(|vec|/elementSize)` calls evaluate the same `vec`, so in principle they agree. Good.
- But cell (i, j)'s edge 1 (at chord `stSp+3`) and cell (i, j+1)'s edge 3 (at chord `stSp`) — also the same physical edge — re-evaluate the same vector. Same logic, they agree.

So in the **current** code, neighbour cells *do* agree on shared edges (both run the same `ceil`). The inter-region connectivity is not the failure mode that hits us first. The within-cell `nEl1 != nEl3` mismatch is what fires Site A.

A future fix that introduces `secNel[j]` must preserve neighbour-edge agreement; it is not a free degree of freedom.

---

## 3. Site A — the node-pulling that turns mismatch into bad geometry

In `ShellRegion.createShellMesh` ([shell_region.py:88-210](../../src/pynumad/mesh_gen/shell_region.py)):

### Step A1: build a uniform grid sized to the larger opposite edge

```python
ee = self.edgeEls                  # = nEl from Site B
xNodes = max(ee[0], ee[2]) + 1
yNodes = max(ee[1], ee[3]) + 1
```

`createSweptMesh` lays a `xNodes × yNodes` grid evenly over `[-1, 1]²` in natural space. So far the grid is clean.

### Step A2: snap the short-edge row onto a coarser segment

Four symmetric branches (lines 108-180). Take the `ee[2] < ee[0]` branch (the one that fires for `edgeEls = [5, 5, 4, 5]`):

```python
seg = Segment2D("line", [[-1,1],[1,1]], self.edgeEls[2])  # 5-node segment on the top edge
segNds = seg.getNodesEdges()["nodes"]                      # ee[2]+1 = 5 evenly spaced x-coords
meshNds = mData["nodes"]                                   # xNodes = 6 evenly spaced top-row x-coords
for ndi in range(totNds - xNodes, totNds):                 # iterate top row (nodes 30..35)
    minPt = nearest(meshNds[ndi], segNds)                  # snap to nearest segment node
    meshNds[ndi] = minPt
```

For `xNodes = 6` snapping to `5` segment nodes:

| top-row node x | nearest seg x | snapped to |
|---|---|---|
| -1.0 | -1.0 | seg 0 |
| -0.6 | -0.5 | seg 1 |
| -0.2 |  0.0 | seg 2 |
|  0.2 |  0.0 | seg 2 ← coincident with previous |
|  0.6 |  0.5 | seg 3 |
|  1.0 |  1.0 | seg 4 |

Two adjacent top-row nodes (nat-space x = -0.2 and 0.2) are now coincident at x = 0.0. **The interior row directly below (still at original spacing) was not pulled.**

### Step A3: dedup and triangle collapse

```python
mData = mt.mergeDuplicateNodes(mData)          # remove the coincident pair
for each element:
    if two sorted node ids equal:
        collapse quad -> triangle (slot[3] = -1, fix winding)
```

The element that straddles the merged pair (top edges share one node, bottom corners still at original spacing) becomes a triangle. The two **neighbour** elements (one to each side) are still quads but their top edges are now uneven: one side ends where the merge happened (short edge), the other extends to the next un-merged top node (long edge). That uneven top + original-spaced bottom = warped quad.

**This is element 1571.** Edges 0.054 / 0.282 / 0.099 / 0.282 in physical space → the 0.054 and 0.099 are the disturbed top corners pulled inward by the snap; the 0.282 / 0.282 are the unchanged span-direction edges. Not degenerate, not collapsible, just warped enough for ANSYS to fail its Jacobian check.

### Step A4: physical-space mapping makes it worse

```python
XYZ = self.XYZCoord(ndLst)                     # quad3: 16-keypoint biquadratic-like blend
```

The natural-space `[-1,1]²` grid is then mapped to 3D via [shell_region.py:259-470](../../src/pynumad/mesh_gen/shell_region.py) using the `quad3` 16-node Lagrange-style basis. At the root transition the keypoint spread is non-uniform (chord changes fast), so the mapping amplifies any natural-space distortion. A near-tolerable warp at `[-1,1]²` becomes a Jacobian-flipped element at `(x, y, z)`.

---

## 4. Why the count of flipped elements grows with refinement

At coarse `elementSize` (0.80 m), few cells have chord lengths in the band where `ceil()` produces `nEl1 ≠ nEl3`. At fine `elementSize` (0.10 m), many cells fall in such bands. The snap-and-merge logic fires per-cell, and each firing has a roughly constant probability of producing a warped (non-collapsed) quad. So count scales with the number of mismatched cells, which scales with refinement. The data:

| elementSize | n_total | jflips_total |
|---|---|---|
| 0.80 m | 5 079 | 19 |
| 0.60 m | 9 159 | 44 |
| 0.50 m | 10 250 | 86 |
| 0.45 m | 15 204 | 108 |
| 0.40 m | 16 518 | 107 |
| 0.35 m | 18 533 | 98 |
| 0.30 m | 26 305 | 109 |
| 0.20 m | 52 896 | 131 |
| 0.10 m | 200 297 | 226 |

The count plateaus around 100–230 because the number of "mismatched" cells saturates once `elementSize` is below the geometric features driving the chord-length differences.

---

## 5. What "fix it" looks like (candidate ranked by simplicity)

These are options — the actual choice is made in Phase 4 after the test harness is green and instruments the existing failure.

1. **Force opposite edges to match per-cell.** Inside [mesh_gen.py:319-341](../../src/pynumad/mesh_gen/mesh_gen.py):

   ```python
   nEl_chord = max(nEl1, nEl3)
   nEl_span  = max(nEl2, nEl4)
   nEl = [nEl_chord, nEl_span, nEl_chord, nEl_span]
   ```

   - Eliminates the Site A trigger entirely.
   - Slight over-refinement in trapezoidal cells (more elements than strictly needed).
   - **Inter-region check needed**: cell A's `max(nEl1, nEl3)` and neighbour cell B's `max(nEl1, nEl3)` are derived from different vectors — they may not agree on the shared edge. This is the risk that justifies test-first.

2. **Precompute `secNel[j]` per chordwise stack.** Implement the commented hint properly: before the loop, walk all spanwise stations and pick `secNel[j] = max_i ceil(chord_len_at_(i,j) / elementSize)`. Then `nEl1 = nEl3 = secNel[j]` inside the loop. Guarantees both within-cell opposite-edge equality AND inter-cell shared-edge agreement.

3. **Laplacian smoothing post-pass.** Leave the existing algorithm; add a smoothing sweep that relaxes interior nodes after the snap. Treats the symptom, not the cause; can mask other bugs.

4. **Detect and collapse / split warped quads.** After mesh build, scan for `jacobian < ε` quads and either collapse to triangles or split. Sticking-plaster.

**Working hypothesis for Phase 4:** start with (1) because it's the minimal diff; if Phase 4 acceptance tests show inter-region gaps, escalate to (2).

---

## 6. What stays the same and what doesn't

- `XYZCoord` (the physical-space mapping) is fine — it is faithful to its inputs. Don't touch.
- `mergeDuplicateNodes` and the triangle-collapse loop are fine — they handle a downstream consequence, not the cause.
- `Segment2D`, `Boundary2D`, `createSweptMesh` are fine.
- The `meshMethod == "free"` path is unaffected — different code in `Mesh2D.createUnstructuredMesh`.
- Cubit-export path is unaffected — separate code in `analysis/cubit/`.

Anything else is fair game.

---

## 7. Tests that pin this trace

These are written in Phase 1.3 / 1.4 and must continue to assert the behaviour after the fix:

- `test_shell_region_repro.py::test_edgeEls_5_5_4_5_no_jflips` — direct `ShellRegion("quad3", kp, [5,5,4,5], meshMethod="structured")` call, post-mesh assert zero Jacobian flips and zero coincident-node pairs within tolerance.
- `test_iea22_mesh_quality.py::test_iea22_at_0p30m` — full pipeline on IEA-22 at 0.30 m, assert jflip count == 0.
- `test_iea22_mesh_quality.py::test_eigenfreq_convergence` (Phase 4) — first 10 eigenfrequencies converge monotonically through 0.10 m.

References to existing reproducer artifacts (kept for forensic value, not regression):

- [paper2_fem/00_smoke_test/track_bad_region.py](../../../paper2_fem/00_smoke_test/track_bad_region.py) — geometric tracker (logic lifted into `pynumad.testing.mesh_quality`).
- [paper2_fem/00_smoke_test/plot_bad_elements.py](../../../paper2_fem/00_smoke_test/plot_bad_elements.py) — visualization.
- [paper2_fem/continue.md](../../../paper2_fem/continue.md) — original session handoff that characterised the wall.
