# Adhesive bondline extension — implementation plan

## What we have today

`pyNuMAD/src/pynumad/mesh_gen/mesh_gen.py` emits a **single** adhesive volume:

- **TE (trailing edge)**: a swept 2D quad cross-section built from spline-keypoint
  columns `(4, 6, 30, 32)` — the four corners of the TE bondline at each
  spanwise station. After our recent fix, every node of this volume is
  tied to the LP_TE_REINF / HP_TE_REINF shell sets via constraint
  equations (`CE` records). End-to-end ANSYS runs cleanly.

What's **missing** for a physically-complete wind-blade FE model:

1. **LE bondline** — leading-edge bond between LP and HP half-shells.
2. **Shear-web-to-skin bonds** — each web has two bond surfaces (one
   against the HP spar cap, one against the LP spar cap). With three
   webs (web0, web1, web2 in IEA-22) that's 6 bond surfaces total.
3. **Spar-cap T-joint fillets** — small adhesive fillets where the spar
   cap meets the web flange. Often modelled with the web-to-skin bond.

A physically-correct blade FE has **8 bondline regions** in total
(1 TE + 1 LE + 2-per-web × 3 webs).

## What windIO actually specifies

The windIO YAML schema declares **adhesive as a material** but does NOT
declare bondline geometry explicitly. There are no
`internal_structure_2d_fem.bondlines` or
`internal_structure_2d_fem.joints` keys in the IEA-22 YAML or in BAR0.

Instead, bondline positions are **implicit in the structural topology**:

- TE bond = where `00_NN_HP_TE_FLAT` meets `11_NN_LP_TE_FLAT` at the
  trailing edge spline. Always present.
- LE bond = where `05_NN_HP_LE` meets `06_NN_LP_LE` at the leading edge
  spline. Always present (assumed for a two-half-shell blade).
- SW-to-skin bond = where each `swstacks[k_web, k_stat]` web's top/bottom
  edge meets the inner surface of the spar cap (`03_NN_HP_SPAR` for the
  HP side, `08_NN_LP_SPAR` for the LP side).

So **no YAML changes are required** to encode the bondline geometry — we
derive it from the keypoint indices that pyNuMAD already builds. What we
DO need from the YAML is:

1. An **adhesive material** entry (already required — current code uses
   `material name lower() == 'adhesive'`).
2. Optionally a **bondline thickness** override per bond type (currently
   the thickness is implicit in the geometry — e.g. for TE, it's
   `distance(spl4 → spl6)`). For an explicit thickness, the YAML would
   need to gain a new optional field; we can defer this to a follow-up.

## Implementation plan

### Phase 1 — refactor the existing TE code into a generic bondline emitter

**Goal:** replace the inline TE-only block at `mesh_gen.py:815-1035`
with a call to a new helper:

```python
def _emit_adhesive_volume(
    blade, shellData, splineXi, splineYi, splineZi,
    bond_name, corner_cols, target_shell_sets,
    elementSize, frstXS=None,
):
    """Generate one swept adhesive volume + tie constraints.

    Parameters
    ----------
    bond_name : str
        Logical name ("TE_BOND", "LE_BOND", "SW0_HP", ...). Goes into
        the element-set name and probe identifier.
    corner_cols : tuple of 4 ints
        Spline-keypoint column indices for the 4 cross-section corners
        of this bond, in (LP_outer, HP_outer, HP_inner, LP_inner) order
        so the residual-tie fallback resolves correctly.
    target_shell_sets : list of str
        Element-set names this adhesive ties to (e.g.
        ['LP_TE_REINF', 'HP_TE_REINF'] for the TE bond).
    """
```

This is a pure refactor — the TE behaviour stays bit-identical. Verify
by running `tests/test_adhesive_coverage.py` before / after.

### Phase 2 — LE bondline

Add a second `_emit_adhesive_volume` call with `corner_cols` set to the
LE spline columns (need to inspect `keypoints.py` to confirm — typically
columns 17, 19 for HP_LE/LP_LE outer + the corresponding inner inboard
columns), and `target_shell_sets = ['HP_LE', 'LP_LE']`.

Bondline thickness at LE is typically smaller than TE (often 5-10 mm vs
20-50 mm). We can keep the geometry-derived thickness for now and add a
config knob later.

Regression test: extend `test_adhesive_coverage.py` to assert that the
mesh contains at least 2 distinct bondline regions (TE + LE) when
`includeAdhesive=True`, identified by their spatial centroids.

### Phase 3 — SW-to-skin bondlines (RESOLVED: implicit via node sharing)

**Investigation outcome:** explicit SW-skin adhesive volumes are NOT
needed for the IEA-22 / BAR0 model. pyNuMAD's mesh generator merges
each web's edge nodes with the adjacent HP_SPAR / LP_SPAR shell nodes
when constructing the SW shell region. Empirically verified at
``01_30_SW`` ∩ ``03_30_HP_SPAR``: 3 of the SW's 23 nodes are also in
the spar-cap element set, providing direct kinematic coupling.

Implications:
- **Structural coupling** between web and skin is exact (zero
  compliance, no slip) — equivalent to an infinitely stiff adhesive.
- **No CE statements or 3D solid elements required** at the SW-skin
  interface.
- **Limitation**: bondline compliance / failure cannot be modelled at
  the SW-skin interface in the current shell-only representation. To
  model SW-skin adhesive separately one would need to (a) decouple the
  web's edge nodes from the skin nodes, then (b) emit a 3D adhesive
  brick volume between them with CE ties to both. This is a deeper
  refactor that touches mesh_gen.py:611-741 and is deferred until /
  unless SW-skin debonding becomes a relevant failure mode for the
  analysis.

If SW-skin bondline modelling is needed in future, the approach is:
1. Add a small geometric offset (e.g. the YAML's "Adhesive" thickness)
   between web edge nodes and skin nodes during web meshing.
2. Use ``_emit_adhesive_volume`` with a thin 4-corner cross-section
   that spans the offset.
3. CE-tie one face to the web's edge node-set and the other face to
   the spar cap shell element-set.

### Phase 4 — bondline-thickness config (optional, defer)

Add a per-bond override knob:

```yaml
internal_structure_2d_fem:
  bondlines:               # NEW, OPTIONAL
    TE: {thickness_mm: 30}
    LE: {thickness_mm: 5}
    SW: {thickness_mm: 3}
```

If the YAML omits this, fall back to the geometry-derived thickness
(current behaviour).

## Verification strategy

1. **Unit-level**: extend `test_adhesive_coverage.py` to assert
   - `n_distinct_bondlines == 1 + (1 if has_LE_bond else 0) + 2*n_webs`
   - 100 % adhesive-node CE coverage on each bondline
   - The bondline centroids cluster near their declared geometric
     positions (TE → max-x, LE → min-x, SW → web chord positions).

2. **Topology test** (new): `test_bondline_topology.py` — for each
   bondline, run a BFS through CE + element connectivity from any
   adhesive node and verify the reached set covers exactly one
   topological component.

3. **End-to-end ANSYS smoke**: `verify_adhesive_fix.py` re-run with the
   new bondlines should still print READY TO RUN ANSYS, with 100 %
   coverage but a much higher node count (LE alone adds ~30 % and each
   SW bond ~20 %).

4. **Mass + EI re-check on IEA-22**: with all 8 bondlines emitted, mass
   should rise by maybe 0.5-1.5 t (adhesive is light but accumulates),
   bringing the predicted mass closer to (or slightly above) the
   reference 82.4 t. EI distribution should not change substantially.

## Effort estimate

- Phase 1 (refactor): 2 h
- Phase 2 (LE): 3 h — column-index discovery + 1 new test
- Phase 3 (SW-to-skin × N_webs): 6 h — most code, most tests
- Phase 4 (thickness config): 2 h, deferrable
- End-to-end re-run + plot: 30 min

Total: **~1 full day** to land Phases 1-3 with regression tests.

## Risks

- Spline column conventions: the existing TE code uses columns 4, 6,
  30, 32 — these are pyNuMAD-internal indices and may not stay stable
  across versions. We should ADD a helper that maps semantic names
  ("TE LP outer", "LE inner") to column indices via `blade.keypoints`
  rather than hardcoding the literals. This avoids hard-coding integers
  that may shift if the keypoint scheme changes.

- SW-skin bond at root: the first few stations have spar caps thinner
  than the bondline thickness. Need to test on BAR0 at h=0.5 m and
  IEA-22 at h=0.45 m for robustness.

- Constraint-equation rank: each bondline adds CE records. The total CE
  count for IEA-22 at h=0.10 m with all bondlines could approach 30 k.
  ANSYS's CE assembler should handle that, but it adds solve time.
  Measure before vs after.
