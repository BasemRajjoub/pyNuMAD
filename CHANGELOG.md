# Changelog — pyNuMAD `pce-patches` fork

Tracks the diffs on top of Sandia upstream `main`. Format: keep entries
small and cherry-pickable.

## Mesh-quality wall fix (May 2026)

### Bug fixes

- **`mesh_gen`: fractional `XSCurvePts` + interpolated chord sampling.**
  Replaces the `np.round(np.linspace(...))` step (which collided when
  consecutive design keypoints were < 3 geometry-curve indices apart at
  cylindrical root and tip taper regions) with float-valued indices
  resolved via `np.interp`. Mesh is now clean at every density from
  0.10 m to 0.80 m on IEA-22 and BAR0. ([`463a0db`])
- **`mesh_gen._compute_edge_nels`**: force opposite chord edges to share
  the larger element count (`max(nEl1, nEl3)`); eliminates the
  node-pulling code path in `ShellRegion` that was producing
  sign-flipped Jacobian quads at root transitions. ([`d97c368`])
- **`mesh_gen._compute_edge_nels`**: skip degenerate shell patches whose
  shortest edge is < 5% of `elementSize` (typical at extreme root/tip
  where chord shrinks toward zero), with `force_match_opposite=True`
  for connectivity preservation. ([`ad89236`])
- **`mesh_gen._compute_edge_nels`**: skip patches with duplicate
  keypoints (spatial-hash dedup of the 16 shellKp entries) and with
  twisted corner geometry (Jacobian-flip check on the 4 corners).
  ([`db3b040`])
- **`mesh_gen/segment2d`**: guard against `numEls == 0` in step
  generation. ([`33d63d2`])
- **`mesh_gen/shell_region`**: initialise `minPt` before nearest-node
  search. ([`4b7aa78`])
- **`ansys/write`**: off-by-one in deflection-write tip-station guard;
  use `blade.geometry.isweep` and 1-D NaN row for presweep table.
  ([`8ee39a9`], [`92f2d3e`])
- **`yaml_to_blade`**: guard against unbound spar-cap side indices.
  ([`379156f`])

### Performance

- **`testing.mesh_quality.analyse_mesh`**: vectorised — ~150x faster.
  On 25k elements: 10.5 s → 0.07 s. On 200k elements: ~80 s → 0.9 s.
  ([`dad0dc6`])
- **`mesh_tools.get_average_node_spacing`**: vectorised — ~90x faster
  triple-loop → 12 bulk numpy ops. ([`ede8f77`])
- **`mesh_tools.mergeDuplicateNodes`**: replaced custom spatial grid +
  nested Python loops with `scipy.spatial.cKDTree.query_pairs`.
  `shell_mesh_general` at 0.30 m: 7.9 s → 3.3 s. Full pytest suite:
  6:43 → 2:24 (2.8x). ([`ede8f77`])

### New modules

- **`pynumad._logging`**: structured logging with JSONL sidecar
  (`PYNUMAD_DEBUG_LOG=path.jsonl` env var), per-module loggers nested
  under `pynumad.*`, numpy-array summarisation in JSON. Silent by
  default. ([`51cb845`])
- **`pynumad.invariants`**: `check_*` / `require_*` predicates over
  edgeEls, quad corners, mesh dicts. Used by `assert`s on hot paths
  and by `raise ValueError` at module boundaries. ([`51cb845`])
- **`pynumad.testing.mesh_quality`**: shared mesh-quality oracle —
  `MeshQualityReport`, `analyse_mesh`, `assert_mesh_clean`,
  `assert_no_jacobian_pathology` (catches slivers + severe AR + flips
  that ANSYS rejects). ([`f6eac9b`])
- **`pynumad.testing.viz`**: SVG dumps of individual quads and their
  adjacency neighbourhoods for forensic debugging. ([`fb63caf`])
- **`pynumad.testing.mesh_plot`**: interactive plotly HTML and static
  matplotlib three-view PNG renderings of whole-blade meshes. Three
  colouring modes: `problem` / `component` / `region`. Root / mid /
  tip closeups. ([`d6450f0`])

### Instrumentation

- `mesh_gen.shell_mesh_general` and `ShellRegion.createShellMesh` emit
  structured INFO/WARNING/DEBUG records keyed by `context.stage`
  (`entry`, `edge_nels`, `structured_quad_entry`, `node_pull`,
  `merge_duplicates`, `exit`). See [`docs/dev/logging.md`]. ([`9739040`])

### Testing

- pytest scaffold, dev deps (`pytest`, `pytest-cov`, `hypothesis`,
  `ruff`, `mypy`, `nox`), ruff + mypy configs in `pyproject.toml`.
  ([`2be15c2`])
- 13 unit reproducer tests in `test_shell_region_repro.py` exercising
  all four node-pulling branches and pinning topology. ([`1f9a4bd`])
- 9 IEA-22 integration tests in `test_iea22_mesh_quality.py` covering
  element sizes 0.10–0.80 m. All pass post-fix. ([`1f9a4bd`])
- 24 unit tests for `_logging` + `invariants` modules. ([`e2dee86`])
- Hypothesis-driven property test scanning `edgeEls` ∈ [2, 10]⁴
  combinations. ([`791aac5`])

### Documentation

- `docs/dev/mesh_bug_trace.md` — algorithmic walkthrough of the
  mesh-quality wall, the two cooperating sites, the four node-pulling
  branches, and the fix taxonomy.
- `docs/dev/logging.md` — JSONL schema, env vars, `jq` recipes.

### Tooling

- `nox` sessions: `tests`, `tests_all`, `cov`, `lint`, `format`,
  `typecheck`, `docs`, `serve`. Uses `ruff` (replacing the
  black + flake8 + isort triple).

### Test results — IEA-22 with all fixes applied

| esize  | elements | jflips | slivers | lines | max AR |
|--------|----------|--------|---------|-------|--------|
| 0.80 m | 4 811    | 0      | 0       | 0     | 17     |
| 0.45 m | 14 788   | 0      | 0       | 0     | 16     |
| 0.30 m | 25 855   | 0      | 0       | 0     | 19     |
| 0.20 m | 52 354   | 0      | 0       | 0     | 19     |
| 0.10 m | 199 283  | 0      | 0       | 0     | 20     |

All 82 pytest cases pass. No xfails.

[`463a0db`]: ../../commit/463a0db
[`d97c368`]: ../../commit/d97c368
[`ad89236`]: ../../commit/ad89236
[`db3b040`]: ../../commit/db3b040
[`33d63d2`]: ../../commit/33d63d2
[`4b7aa78`]: ../../commit/4b7aa78
[`8ee39a9`]: ../../commit/8ee39a9
[`92f2d3e`]: ../../commit/92f2d3e
[`379156f`]: ../../commit/379156f
[`dad0dc6`]: ../../commit/dad0dc6
[`ede8f77`]: ../../commit/ede8f77
[`51cb845`]: ../../commit/51cb845
[`f6eac9b`]: ../../commit/f6eac9b
[`fb63caf`]: ../../commit/fb63caf
[`d6450f0`]: ../../commit/d6450f0
[`9739040`]: ../../commit/9739040
[`2be15c2`]: ../../commit/2be15c2
[`1f9a4bd`]: ../../commit/1f9a4bd
[`e2dee86`]: ../../commit/e2dee86
[`791aac5`]: ../../commit/791aac5
[`docs/dev/logging.md`]: docs/dev/logging.md
