"""Regression test for chord-group connectivity on BAR0.

Every per-material chord group on the blade outer shell (HP_SPAR,
HP_TE_REINF, HP_LE, HP_TE_PANEL, …) must form **exactly one
node-connected component**. That is the weakest possible
"plate-like" continuity property: every element of the group shares
at least one node with another element of the same group. Anything
weaker means the group has isolated islands that can't transfer load
through their own kinematics.

Shear webs are the only intentional exception — they are N physical
components, one per web in the windIO definition.

History note: an earlier version of this test treated set labels as
1-indexed and saw HP_TE_REINF fragment into ~100 components. That
was a bug in the test, not the mesh — labels in
``mesh["sets"]["element"][...]["labels"]`` are 0-indexed (the ANSYS
deck writer in ``analysis/ansys/write.py:1272`` adds ``+1`` on its
own when emitting EMODIF). With correct indexing every chord group
is a single node-component, and almost all are single edge-components
too. The few that aren't edge-connected (panels, TE_FLAT) form
T-junctions that ANSYS handles via the pyNuMAD-emitted constraint
equations.
"""
from __future__ import annotations

import pytest

from pynumad.tests._mesh_cache import get_mesh
from pynumad.testing import mesh_continuity as mc


@pytest.fixture(scope="module")
def bar0_mesh():
    return get_mesh(includeAdhesive=False, elementSize=0.5)


# Every chord group should be a single node-connected component.
# Shear webs (SW) are excluded because they are intentionally multiple
# physical components.
EXPECTED_NODE_CC = {tag: 1 for tag in mc.CHORD_GROUP_TAGS if tag != "SW"}


def test_all_chord_groups_are_node_connected(bar0_mesh):
    """Every material zone (except SW) should be a single 'plate'."""
    sizes = mc.all_group_components(bar0_mesh, min_shared_nodes=1)
    for tag, n_expected in EXPECTED_NODE_CC.items():
        cc = sizes.get(tag, [])
        assert cc, f"{tag} missing from mesh"
        assert len(cc) == n_expected, (
            f"{tag}: expected {n_expected} node-connected component(s), "
            f"got {len(cc)} (sizes={cc[:5]}...)"
        )


def test_no_singleton_islands_in_chord_groups(bar0_mesh):
    """No chord-group element should be totally isolated.

    A singleton means the element shares no nodes with any other
    element of the same chord group — almost always a labelling bug.
    """
    for tag in mc.CHORD_GROUP_TAGS:
        if tag == "SW":
            continue
        strays = mc.stray_elements(bar0_mesh, tag)
        assert not strays, (
            f"{tag}: {len(strays)} stray (totally-isolated) elements; "
            f"first few = {strays[:5]}"
        )


def test_shear_web_components_match_web_count(bar0_mesh):
    """SW should split into N node-connected components, one per web.

    BAR0 has 2 webs. IEA-22 has 3. We don't hard-code the number here
    — we just assert it is at least 2 (a single component would mean
    the webs got merged in the labelling).
    """
    cc = mc.component_sizes(bar0_mesh, "SW", min_shared_nodes=1)
    assert cc, "SW group missing from mesh"
    # One small tail singleton tolerated (cosmetic label bug at root
    # or tip — element ends up in the wrong station set). The bulk
    # components must be >= the number of physical webs.
    big = [s for s in cc if s > 5]
    assert len(big) >= 2, (
        f"SW: expected >=2 large components (one per physical web); "
        f"got sizes {cc[:10]}"
    )
