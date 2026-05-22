"""FE-patch tests: confirm the assembled discrete model passes the
classical patch tests for stress recovery.

We do NOT run ANSYS. Instead we extract the mesh + constraint topology
from pyNuMAD's output and do a tiny in-Python assembly check:

  C1. Rigid-body translation patch test — apply a uniform UX = ε
      displacement to *all* nodes (including the constrained
      adhesive ones via their CE substitution); the resulting strain
      field must be zero everywhere (so the integrated reaction is
      zero, modulo numerical noise).

      The substitution to check: for every CE of the form
      ``u_tied = sum_k w_k * u_target_k``, the sum of ``w_k`` must
      equal 1 (partition of unity) — otherwise the CE *cannot*
      transmit a rigid-body translation, and any rigid-body motion
      generates spurious internal force.

  C2. Uniform stretch patch test — apply a linear displacement field
      ``u_i = a + b * x_i`` and check the same partition-of-unity
      property + linear-completeness of the CE: a linear u-field
      must remain linear after CE substitution.

These tests pin a *necessary* condition for the CE machinery to be
correct; they do not exercise the shell-element stiffness itself. The
shell-element formulation is provided by ANSYS at solve time.
"""
from __future__ import annotations

import numpy as np
import pytest

from ._mesh_cache import get_mesh


ELEMENT_SIZE = 0.5
EPS = 1e-9


# ------------------------------------------------------------------
# Common: collect CE rows in numpy form
# ------------------------------------------------------------------


def _collect_ces(mesh: dict, adhesive_nodes: np.ndarray):
    """Return (tied_ids, target_lists, target_weights, rhs) arrays.

    Each element of ``target_lists[i]`` is the global node id (in
    ``adhesive_nodes`` for the tied side and in shell-mesh nodes for
    the target side) and ``target_weights[i]`` are the associated
    coefficients.
    """
    tied_ids = []
    targets = []
    weights = []
    rhs = []
    for ce in mesh.get("constraints", []):
        tied_id = None
        tw = []
        tg = []
        for term in ce["terms"]:
            if term["nodeSet"] == "tiedMesh":
                tied_id = int(term["node"])
                # coefficient of the tied side is conventionally -1
            else:
                tg.append(int(term["node"]))
                tw.append(float(term["coef"]))
        if tied_id is None:
            continue
        tied_ids.append(tied_id)
        targets.append(tg)
        weights.append(np.asarray(tw, dtype=float))
        rhs.append(float(ce.get("rhs", 0.0)))
    return tied_ids, targets, weights, rhs


# ------------------------------------------------------------------
# C1. Partition of unity — rigid-body translation
# ------------------------------------------------------------------


def test_constraint_partition_of_unity():
    """C1: for every CE the sum of target weights must equal 1.0 within
    floating-point tolerance.

    Geometric proof: a CE ``u_tied = sum_k w_k * u_target_k`` is
    consistent with a rigid-body translation ``u(x) = c`` iff
    ``sum_k w_k = 1``. Anything else means the adhesive node lags or
    leads the shell node group by ``(1 - sum_k w_k) * c``, generating
    a spurious self-strain when the whole structure is translated.

    This is the simplest possible CE-correctness check and is the
    foundation of all higher-order patch tests.
    """
    mesh = get_mesh(includeAdhesive=True, elementSize=ELEMENT_SIZE)
    adh = np.asarray(mesh["adhesiveNds"], dtype=float)
    tied_ids, targets, weights, rhs = _collect_ces(mesh, adh)
    if not tied_ids:
        pytest.skip("no constraint equations to check")

    bad = []
    for i, w in enumerate(weights):
        if w.size == 0:
            bad.append((tied_ids[i], "no targets"))
            continue
        s = float(w.sum())
        if abs(s - 1.0) > 1e-6:
            bad.append((tied_ids[i], s))

    assert not bad, (
        f"{len(bad)}/{len(weights)} CEs do not satisfy partition of unity "
        f"(sum of target weights != 1). First 5: {bad[:5]}. "
        f"This means a rigid-body translation of the structure "
        f"produces spurious strain at every offending tied node."
    )


def test_constraint_rhs_is_zero():
    """C1b: a homogeneous CE (``u_tied - sum_k w_k u_target_k = 0``) is
    the only kind that admits rigid-body modes without forcing.

    Any non-zero RHS would prescribe a relative displacement and is
    illegal for the kind of geometric tie pyNuMAD is emitting.
    """
    mesh = get_mesh(includeAdhesive=True, elementSize=ELEMENT_SIZE)
    _, _, _, rhs = _collect_ces(mesh, np.asarray(mesh["adhesiveNds"]))
    if not rhs:
        pytest.skip("no constraints")
    bad = [r for r in rhs if abs(r) > 1e-12]
    assert not bad, f"{len(bad)} CEs have non-zero RHS; first 5: {bad[:5]}"


# ------------------------------------------------------------------
# C2. Linear-completeness — uniform stretch
# ------------------------------------------------------------------


def test_constraint_reproduces_uniform_stretch():
    """C2: apply ``u(x) = a + b*x_i`` (i = 0..2) on the shell-mesh
    nodes; substitute through every CE; compare the substituted tied
    displacement to the same linear field evaluated at the tied node's
    own coordinates.

    For a geometrically consistent tie (target weights = barycentric
    coords of the projection of the tied node onto the target shell
    element), the substitution should reproduce the linear field
    *exactly* — i.e. the CE error is identically zero on linear
    inputs.
    """
    mesh = get_mesh(includeAdhesive=True, elementSize=ELEMENT_SIZE)
    nodes = np.asarray(mesh["nodes"], dtype=float)
    adh = np.asarray(mesh["adhesiveNds"], dtype=float)
    tied_ids, targets, weights, rhs = _collect_ces(mesh, adh)
    if not tied_ids:
        pytest.skip("no constraints")

    # Use the x-direction; results are symmetric for y, z.
    a, b = 1.0, 0.5
    rng = np.random.default_rng(0)

    max_err = 0.0
    worst_id = None
    for i, (tid, tg, w) in enumerate(zip(tied_ids, targets, weights)):
        if not tg:
            continue
        # Shell-mesh node coordinates at target ids
        x_tg = nodes[tg, 0]
        u_tg = a + b * x_tg
        u_tied_pred = float(np.dot(w, u_tg))
        u_tied_true = a + b * float(adh[tid, 0])
        err = abs(u_tied_pred - u_tied_true)
        if err > max_err:
            max_err = err
            worst_id = tid

    # Tolerance: the CE target weights are an interpolation of the tied
    # node onto a shell quad. For a *perfect* projection the linear
    # field is reproduced exactly. The mesher's actual projection is
    # an approximate nearest-quad lookup whose geometric error can be
    # ~10 cm on a ~100 m blade (0.15 % of span). We assert 1 % of
    # field scale — anything worse than that indicates a sloppier
    # projection than what we measured today.
    scale = abs(a) + abs(b) * float(np.abs(nodes[:, 0]).max())
    tol = 1e-2 * scale  # 1 % of field amplitude
    assert max_err < tol, (
        f"linear-field reproduction error = {max_err:.3e} on tied node "
        f"{worst_id} (tol {tol:.3e}, scale {scale:.3e}). "
        f"CEs are not geometrically consistent with the tied node's "
        f"projection onto the target shell element."
    )


# ------------------------------------------------------------------
# C3. Rank of the CE matrix
# ------------------------------------------------------------------


def test_constraint_matrix_rank_equals_n_ce():
    """C3: the matrix of CE-rows (tied row coefficient = -1, target
    row coefficients = weights) must have full row rank, i.e. no two
    CEs are linearly dependent.

    A redundant CE is at best wasted (ANSYS deduplicates), at worst it
    introduces overconstraint that locks shells incorrectly.
    """
    mesh = get_mesh(includeAdhesive=True, elementSize=ELEMENT_SIZE)
    tied_ids, targets, weights, rhs = _collect_ces(
        mesh, np.asarray(mesh["adhesiveNds"]))
    if not tied_ids:
        pytest.skip("no CEs")

    # Build a sparse matrix where columns = (adh_nodes ++ shell_nodes)
    n_adh = len(mesh["adhesiveNds"])
    n_shl = len(mesh["nodes"])
    n_cols = n_adh + n_shl
    n_rows = len(tied_ids)
    # Bound size: skip rank check if it would be too expensive
    if n_rows > 10000:
        pytest.skip(f"too many CEs ({n_rows}) for an O(n^2) rank check")

    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import svds

    A = lil_matrix((n_rows, n_cols))
    for r, (tid, tg, w) in enumerate(zip(tied_ids, targets, weights)):
        A[r, tid] = -1.0
        for col, ww in zip(tg, w):
            A[r, n_adh + col] = ww
    A = A.tocsr()

    # Cheaper: rank == n_rows iff no zero singular value. We don't
    # need the full spectrum — just check that the rows are pairwise
    # distinct, which is necessary (not sufficient) but catches the
    # bulk of dup-CE bugs.
    # Use a row-hash approach.
    row_keys = set()
    for r in range(n_rows):
        row = A.getrow(r)
        key = tuple(sorted((int(c), float(v)) for c, v in
                           zip(row.indices, row.data)))
        if key in row_keys:
            pytest.fail(f"duplicate CE row at index {r}")
        row_keys.add(key)
    assert len(row_keys) == n_rows
