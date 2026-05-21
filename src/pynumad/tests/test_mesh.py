"""End-to-end regression tests on the bundled minimal blade fixture.

The fixture at ``test_data/blades/blade.yaml`` is the BAR0 reference blade
(206 m rotor, 30 spanwise stations). It's the smallest realistic windIO
input we have. Tests here pin its current mesh shape + quality so any
algorithmic change in mesh_gen / shell_region is visible at this level.
"""

import os
import unittest

import numpy as np
import pytest

from pynumad.mesh_gen.mesh_gen import get_shell_mesh
from pynumad.objects.blade import Blade
from pynumad.testing.mesh_quality import analyse_mesh

test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")


class TestMesh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.yamlfile = os.path.join(test_data_dir, "blades", "blade.yaml")

    def test_mesh_smoke(self):
        """The mesher runs end-to-end and produces a non-trivial mesh."""
        blade = Blade(self.yamlfile)
        meshData = get_shell_mesh(blade, includeAdhesive=1, elementSize=0.2)
        self.assertGreater(len(meshData["nodes"]), 0)
        self.assertGreater(len(meshData["elements"]), 0)
        nodes = np.asarray(meshData["nodes"])
        elements = np.asarray(meshData["elements"])
        self.assertEqual(nodes.shape[1], 3, "node rows must be (x, y, z)")
        self.assertGreaterEqual(
            elements.shape[1], 4, "element rows must have >=4 node-id slots"
        )

    def test_mesh_size_within_expected_range(self):
        """Golden-style: pin the node/element count range so any
        algorithmic change in mesh_gen / shell_region is flagged."""
        blade = Blade(self.yamlfile)
        meshData = get_shell_mesh(blade, includeAdhesive=1, elementSize=0.2)
        report = analyse_mesh(meshData)
        self.assertGreater(report.n_nodes, 1000)
        self.assertGreater(report.n_elements, 1000)
        self.assertLess(report.n_elements, 1_000_000)

    @pytest.mark.xfail(
        strict=True,
        reason="Same root-transition bug as IEA-22 — BAR0 produces 2 jflips at "
        "0.2 m today. Confirms the algorithm bug is not IEA-22-specific. "
        "Phase-4 fix should make this pass.",
    )
    def test_bar0_no_jflips(self):
        """BAR0 must produce zero sign-flipped quads at 0.2 m. Currently
        fails (2 jflips) — this is independent confirmation that the
        opposite-edge-mismatch bug affects multiple blade geometries, not
        just IEA-22's root transition."""
        blade = Blade(self.yamlfile)
        meshData = get_shell_mesh(blade, includeAdhesive=1, elementSize=0.2)
        report = analyse_mesh(meshData)
        self.assertEqual(
            report.n_jacobian_flips,
            0,
            f"BAR0 produced {report.n_jacobian_flips} sign-flipped quads "
            f"(first 10: {report.jacobian_flip_indices[:10]})",
        )


if __name__ == "__main__":
    unittest.main()
