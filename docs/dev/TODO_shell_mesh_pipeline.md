# TODO — Shell-mesh pipeline (deferred follow-ups)

Upstream pyNuMAD work that's deferred from the paper-2 (material-RV
PCE) scope. Both items below sit in the shell-mesh / ANSYS-writer
paths.

Companion docs:

- [`shell_mesh_continuity.md`](shell_mesh_continuity.md) — node/edge
  connectivity guarantees and the trapezoidal-patch caveat.
- [`mesh_bug_trace.md`](mesh_bug_trace.md) — historical bug trail
  for the shell mesher.
- [`TODO_solid_3d_pipeline.md`](TODO_solid_3d_pipeline.md) — parallel
  TODO list for the 3-D solid pipeline.

The geometric-pathology tests in
[`src/pynumad/tests/test_pathologies.py`](../../src/pynumad/tests/test_pathologies.py)
pin the contract (D1 no Jacobian flips, D2 no infinitesimal edges,
D3 max AR < 20, D4 collapsed-triangle fraction < 5 %, D5 twist
robustness). All five tests pass on the BAR0 fixture today.

---

## 1. TE-corner pinch elements — PARTIALLY MITIGATED

**Status.** PARTIALLY MITIGATED, not fully resolved. The original
failure mode (NLGEOM crashes immediately, max AR ~28 quads of size
0.025 m × 0.7 m at HP/LP_TE_FLAT corners — documented in
[`shell_mesh_continuity.md`](shell_mesh_continuity.md)) is gone:
initial mesh max AR is now ~19 with zero Jacobian flips and zero
collapsed triangles on IEA-22 at every paper-2 element size, and all
five D1-D5 pathology tests pass on BAR0.

However, residual TE_FLAT slivers still cause NLGEOM "excessive
distortion" termination at large deflection. The paper-2 smoke run
(jobid 7271087_5, h=0.15 m, NSUBST=50, full HAWC2 worst-flap load)
converged cleanly to TIME=0.54 then died on:

- elem 5929 in `00_05_HP_TE_FLAT` at z=6.97 m (near root) —
  initial AR=1.10, well-shaped, but distorts during NL deformation
  (likely local wrinkling of the thin 0.13 m chord TE_FLAT band).
- elem 60375 in `11_60_LP_TE_FLAT` at z=83.70 m (near tip) —
  initial edges 0.021 × 0.138 m, AR=6.71 — a residual TE_FLAT
  sliver that the AR<20 D3 threshold doesn't catch but NLGEOM does.

So: substantially better than before, but not full-load-NLGEOM ready.

**What fixed it.** Combination of upstream commits on
`feat/ansys-solid-writer`:

- `d97c368` `fix(mesh_gen): force opposite-edge equality in shell-patch
  element counts` — `_compute_edge_nels` clamps opposite edges of a
  trapezoidal patch to `max(nEl_a, nEl_b)`, avoiding the buggy
  node-pulling path in `ShellRegion.createShellMesh`.
- `ad89236` `fix(mesh_gen): skip degenerate shell patches (zero-chord
  at tip/root)` — `_DEGENERATE_EDGE_RATIO` floor that drops patches
  whose shortest edge falls below ~5 % of `elementSize`.
- `db3b040` `fix(mesh_gen): skip patches with duplicate or twisted
  corner keypoints` — kills bow-tied interior elements at root/tip
  cylindrical regions.
- `15a6d94` solid-mesh post-extrusion untangler — relevant on the solid
  path, not the direct trigger here, but reaches 0 bad-Jacobian
  elements on BAR0.

**Current mesh quality on IEA-22 (paper-2's target blade).**

| `elementSize` | n_elems  | max_AR | p99_AR | n_jflips | n_collapsed_tris | n_severe_AR (>20) |
| ---           | ---:     | ---:   | ---:   | ---:     | ---:             | ---:              |
| 0.50 m        |  10 100  | 18.07  | 9.54   | 0        | 0                | 0                 |
| 0.25 m        |  36 900  | 18.19  | 5.36   | 0        | 0                | 0                 |
| 0.15 m        |  97 833  | 17.92  | 2.61   | 0        | 0                | 0                 |
| 0.10 m        | 201 166  | 19.66  | 2.07   | 0        | 0                | 0                 |

Max AR sits just under the D3 threshold of 20 at every paper-2 element
size; p99 is well below 10.

**Paper-2 validation.** Re-attempted NLGEOM at h = 0.15 m on the
current mesh with the harder of the two originally-failed configs
(NSUBST = 50, PRED OFF, CNVTOL 0.5 %). Early substeps converged in
2-4 NR iterations with AUTOTS *increasing* the step size; AUTOTS
started bisecting at substep 14 (TIME=0.48), and ANSYS terminated at
TIME=0.54 with the "excessive distortion" errors above. Tip
deflection at termination: -7.8 m (~7.8 % of span, already past the
small-strain linear-validity threshold).

sbatch / run script:
[`paper2_fem/00_smoke_test/sbatch_nlgeom_retry.sh`](../../../paper2_fem/00_smoke_test/sbatch_nlgeom_retry.sh).
First retry (jobid 7270105) was killed by a 1500 s Python subprocess
timeout in `run_static_v2.py` sized for linear static; that timeout
is now NLGEOM-aware (14400 s when `NLGEOM=1`). Second retry (jobid
7271087, with the longer timeout) is the run that produced the
TIME=0.54 ceiling above — the failure is genuinely solver-side, not
infrastructure.

Paper 2's response: run the NLGEOM appendix validation at
`LOAD_SCALE=0.5` instead of 1.0 (see
[`paper2_fem/for_later.md`](../../../paper2_fem/for_later.md) #1).
This avoids the TE_FLAT distortion entirely and is sufficient for a
framework-paper claim. Full-load NLGEOM is deferred to a revision
follow-up if and only if reviewers ask for it.

**Further AR reduction — open follow-up.** The commented-out
`secNel = [1,4,10,10,8,2,2,8,10,10,4,1]` hint at
[`mesh_gen.py:790-791`](../../src/pynumad/mesh_gen/mesh_gen.py#L790-L791)
suggests Sandia's original intent was to precompute per-station
chord-region element counts once across all 12 regions, rather than
deriving each region's edge count independently. That refactor would
likely push max AR below ~10, eliminate the residual TE_FLAT slivers
(elem 60375 et al), and unblock full-load NLGEOM.

Drivers that would justify picking it up:

- **Full-load NLGEOM 250-sample DoE on paper 2** if reviewers force
  it during revision (currently not planned — paper 2 ships
  linear-static + a 5-10 sample NLGEOM appendix validation at
  `LOAD_SCALE=0.5`).
- Adjoint sensitivity at ply-level granularity, where strict edge
  connectivity is required (T-junctions break adjoint coupling).
- SHELL281 quadratic-shell migration (#2 below).

A possibly-cheaper alternative for the NLGEOM use case: add a
per-region **minimum chord-width clamp** in `_compute_edge_nels` that
detects HP/LP_TE_FLAT patches with chord-direction edge < ~5 % of
spanwise edge and either merges them into the adjacent TE_REINF
region or emits a thicker-than-ideal element (AR ~3 rather than AR
~7). Surgical, low-risk, ~2-3 days; would not fix #2 (SHELL281) or
the adjoint use case but would unblock full-load NLGEOM on this
blade.

Cost if either path picked up: 1-2 weeks (full refactor) or 2-3 days
(surgical TE_FLAT clamp) of upstream work + full pytest regression
on BAR0 + IEA-22. **Not currently scheduled.**

---

## 2. SHELL281 (quadratic) deck output from the ANSYS writer

**What.** pyNuMAD exposes `EMID,ADD,ALL` in the APDL writer to inject
mid-side nodes for SHELL281 (quadratic shells), but it historically
errored on collapsed triangles (`slot[3] == -1` quads). Fixing this
would unlock quadratic shells, which handle distortion better than
linear and would help both NLGEOM convergence at large deflections
and refined-mesh accuracy.

**Status.** Not currently blocked. Collapsed-triangle count is 0 on
BAR0 and IEA-22 at all tested element sizes (see table above), so the
EMID crash trigger no longer exists in the input. Pickup is therefore
"add a SHELL281 write path + golden-deck regression test exercising
EMID,ADD,ALL end-to-end on BAR0" — half a day to a day of work, not
the multi-day debug it would have been on the pre-fix mesh.

**Why deferred.** Linear shells (SHELL181) are standard practice and
sufficient for paper 2's QoIs. Pick up if a future caller needs
quadratic shells (refined-mesh blade buckling, large-deflection
sensitivity, etc.).

**Cost when picked up.** 0.5-1 day in pyNuMAD's ANSYS writer.

---

## Cross-reference

- Paper 2 deferred items (analysis-side, not pyNuMAD-source-side):
  [`../../../paper2_fem/for_later.md`](../../../paper2_fem/for_later.md)
- Paper 1 follow-ups:
  [`../../../paper1_material_sensitivity/todo.md`](../../../paper1_material_sensitivity/todo.md)
