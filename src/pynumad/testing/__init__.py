"""Testing utilities for pyNuMAD.

Re-usable helpers for unit and integration tests. Not loaded by the main
package at import time — explicitly import `pynumad.testing.mesh_quality` etc.
"""

from pynumad.testing import mesh_continuity, mesh_quality

__all__ = ["mesh_continuity", "mesh_quality"]
