# Solid-Mesh Quality Wall — Three Open `mesh_gen` Bugs

**Status:** open. Surfaced by the ANSYS solid-writer test suite
([`src/pynumad/tests/test_shell_to_solid_expansion.py`](../../src/pynumad/tests/test_shell_to_solid_expansion.py),
[`src/pynumad/tests/test_solid_solver_run.py`](../../src/pynumad/tests/test_solid_solver_run.py)).
The new ANSYS solid writer
([`write_ansys_solid_general`](../../src/pynumad/analysis/ansys/write.py))
is correct end-to-end at the deck-emission level
(11/11 cross-validation tests against the Abaqus writer pass) but
ANSYS rejects the BAR0 solid deck at solve time because of three
issues in the extruder. Until those are fixed the integration test
([`test_ansys_static_solve_completes`](../../src/pynumad/tests/test_solid_solver_run.py))
xfails.

Companion to [`mesh_bug_trace.md`](mesh_bug_trace.md), which traces
the 2D shell-mesh wall; the issues below are at the next step —
shell-to-solid extrusion in
[`solidMeshFromShell`](../../src/pynumad/mesh_gen/mesh_gen.py).

---

## Bug 1 — ~0.5 % of extruded solid elements have non-positive Jacobian

**Where:**
[`mesh_gen.py:1219-1395`](../../src/pynumad/mesh_gen/mesh_gen.py)
`solidMeshFromShell` — the through-thickness extrusion of the shell
mesh into a 3D solid mesh.

**Symptom:**
On the bundled BAR0 fixture at `elementSize=0.50` with
`layerNumEls=[1,1,1]`,
[`mesh_tools.check_all_jacobians`](../../src/pynumad/mesh_gen/mesh_tools.py)
returns **94 / 16653 elements (0.564 %)** with zero or negative
Jacobian determinant. First 10 failing element indices:
`{6907, 6908, 7112, 7113, 7114, 7115, 7116, 7117, 7118, 7329}`.

ANSYS R2023 rejects these at the `EN` command with
`*** ERROR *** Brick element N has a zero or negative determinant
of the Jacobian matrix at one of its sampling locations`.

**Workaround in use:**
The bundled
[`examples/write_abaqus_solid_model.py`](../../examples/write_abaqus_solid_model.py)
calls `check_all_jacobians` and skips the failing elements before
emitting Abaqus C3D8/C3D6 lines. The Phase 5 integration test mirrors
this via `_filter_bad_jacobian_elements` — a band-aid, not a fix.

**Reproducer (in CI):**
[`test_all_jacobians_positive`](../../src/pynumad/tests/test_shell_to_solid_expansion.py)
(`xfail strict=False`) +
[`test_jacobian_failure_rate_under_2_percent`](../../src/pynumad/tests/test_shell_to_solid_expansion.py)
(regression bound at 2 % so this can't silently get worse).

**Likely cause:**
The extruder pushes each shell node along its averaged-normal vector
([`mesh_gen.py:1262-1285`](../../src/pynumad/mesh_gen/mesh_gen.py)).
At the BAR0 root transition the normals of adjacent shell regions
diverge sharply over short distances; the per-layer offset overshoots
the next layer, producing folded or self-intersecting bricks. The
shell mesh itself is clean at this size — the wall is in the
**extrusion step only**.

---

## Bug 2 — 1 element with opposite-edge parallel deviation > 150°

**Where:** same as Bug 1; surfaces only after Bug 1 is filtered out.

**Symptom:**
After dropping the 94 bad-Jacobian elements, ANSYS still aborts with
`*** ERROR *** Brick element 16721 has a pair of opposite edges that
are 167.7 degrees away from being parallel. This exceeds the error
limit of 150 degrees.` Only one element trips this — but it's a hard
abort.

The `check_all_jacobians` heuristic only catches negative determinants;
it doesn't measure parallel-edge deviation, which is a stricter
geometric criterion ANSYS uses to detect collapsed bricks.

**Workaround in use:**
`_relax_ansys_shape_errors` in the Phase 5 test inserts
`SHPP,WARN,ALL` after `/prep7` so shape-check errors become warnings.
This lets the deck reach `/SOLU`. Not a fix — masks the underlying
geometric degeneracy.

**To fix in `mesh_gen`:**
Either (a) extend the per-element acceptance check to include an
opposite-edge parallelism test (a port of ANSYS's `chkbrik` would be
ideal but is large), or (b) tighten the extruder so it doesn't
produce ~170° edge deviations at the root transition in the first
place. Option (b) is the structural fix; option (a) is a stronger
filter.

---

## Bug 3 — Shell-mesh nodes with near-zero local stiffness contribution

**Where:** same module; surfaces only after Bugs 1+2 are worked
around AND the adhesive bondline is stripped.

**Symptom:**
With all of the above workarounds applied (Jacobian filter, SHPP relax,
adhesive arrays stripped, orphan-node pinning, PCG iterative solver),
the `/SOLU` step still aborts:

```
*** ERROR ***  CP =  23.306   TIME= 23:06:37
There is at least 1 small equation solver pivot term (e.g., at the UX
degree of freedom of node 5426).  Please check for an insufficiently
constrained model.
```

Node 5426 is a regular shell-mesh node (well within the
`n_shell_nodes = 24696` range — i.e., not adhesive). Switching to PCG
and pinning orphan nodes do not help. Diagnosis: the node sits in a
region where the surrounding (after-filter) elements contribute
essentially zero stiffness to one of its translational DOFs. The
global stiffness matrix is therefore numerically singular.

This is a more subtle quality issue than Bugs 1/2: every element
around node 5426 passes the per-element Jacobian + shape checks,
but they assemble into a near-singular matrix at one node.

**No useful workaround.**
The Phase 5 test is `xfail strict=False` until this is fixed in
`mesh_gen`.

---

## Bug 4 — `tie_2_sets_constraints` IndexError at `layerNumEls=[2,2,2]`

**Where:**
[`mesh_tools.py:606-714`](../../src/pynumad/mesh_gen/mesh_tools.py)
`tie_2_sets_constraints` — invoked from `solidMeshFromShell` at
[`mesh_gen.py:1379`](../../src/pynumad/mesh_gen/mesh_gen.py).

**Symptom:**
Doubling per-layer element counts triggers
`IndexError: index N is out of bounds for axis 0 with size N` at
`mesh_tools.py:636 fstNd = tgtNdCrd[elements[ei,0]]`. Off-by-one
in the helper that ties shear-web edges to the outer shell. Means
the BAR0 solid mesh currently only builds at the default
`layerNumEls=[1,1,1]`.

**Reproducer (in CI):**
[`test_element_count_scales_with_layer_count`](../../src/pynumad/tests/test_shell_to_solid_expansion.py)
+
[`test_node_count_grows_with_layer_count`](../../src/pynumad/tests/test_shell_to_solid_expansion.py)
— both `xfail(raises=IndexError)`.

**Likely cause:**
`tie_2_sets_constraints` builds a `tgtNdCrd` array sized to the
*shell* node count, but `elements` after extrusion uses *solid* node
indices that exceed that range. Compare the node arrays it indexes
against the extruded `solidMesh["elements"]`.

---

## Suggested order of attack

1. **Bug 4** is the most contained (off-by-one in one helper, single
   test reproducer). Fix unlocks per-layer refinement studies.
2. **Bug 1** is the highest-impact (94 elements, blocks all ANSYS
   solves). Likely the same root cause as Bugs 2 and 3 — when the
   extruder stops producing degenerate bricks at the root transition,
   all three should clear.
3. **Bugs 2 and 3** verify the Bug 1 fix landed properly: re-run the
   Phase 5 integration test, expect XFAIL → PASS.

The integration tests already capture the failure modes; no new
reproducer scripts needed. Re-run with
`pytest src/pynumad/tests/test_solid_solver_run.py -m integration`
after each fix attempt.
