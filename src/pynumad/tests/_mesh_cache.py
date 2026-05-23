"""Module-level mesh cache for the new pyNuMAD test suite.

The pyNuMAD shell mesher takes ~3 s per build on the BAR0 fixture even at
``elementSize=0.5``. Spreading every test class across its own fresh build
quickly blows the 60 s budget for the full suite.

We keyed the cache on the (``yamlfile``, ``includeAdhesive``, ``elementSize``)
tuple so that tests in different files share a single rebuild when their
parameters happen to coincide. The cache deliberately lives at module scope
(not a pytest fixture) so it is shared *across* test files in the same
``pytest`` invocation.

The returned mesh dict is **shared** — tests must treat it read-only.
"""
from __future__ import annotations

import os
from typing import Tuple

# We avoid importing Blade / get_shell_mesh at import time so that
# `pytest --collect-only` is still fast.
_CACHE: dict[Tuple[str, int, float], dict] = {}
_SOLID_CACHE: dict[Tuple[str, Tuple[int, ...], float], dict] = {}
_BLADE_CACHE: dict[str, object] = {}

TEST_DATA = os.path.join(os.path.dirname(__file__), "test_data")
BAR0_YAML = os.path.join(TEST_DATA, "blades", "blade.yaml")


def get_blade(yamlfile: str = BAR0_YAML):
    """Cached ``Blade`` instance for ``yamlfile``."""
    if yamlfile not in _BLADE_CACHE:
        from pynumad.objects.blade import Blade

        _BLADE_CACHE[yamlfile] = Blade(yamlfile)
    return _BLADE_CACHE[yamlfile]


def get_mesh(includeAdhesive: bool = True, elementSize: float = 0.5,
             yamlfile: str = BAR0_YAML) -> dict:
    """Cached shell mesh for the given (yamlfile, includeAdhesive, esize) tuple.

    Returns the **shared** mesh dict — do not mutate.
    """
    key = (yamlfile, int(bool(includeAdhesive)), float(elementSize))
    if key not in _CACHE:
        from pynumad.mesh_gen.mesh_gen import get_shell_mesh

        blade = get_blade(yamlfile)
        _CACHE[key] = get_shell_mesh(
            blade, includeAdhesive=int(bool(includeAdhesive)),
            elementSize=elementSize,
        )
    return _CACHE[key]


def get_solid_mesh_cached(elementSize: float = 0.5,
                          layerNumEls=(1, 1, 1),
                          yamlfile: str = BAR0_YAML) -> dict:
    """Cached 3D solid mesh for the given (yamlfile, layers, esize) tuple.

    ``get_solid_mesh`` mutates the blade (``edit_stacks_for_solid_mesh``) so
    each cached entry uses a **fresh** Blade instance — we cannot share with
    the shell ``_BLADE_CACHE``. Returns the **shared** mesh dict; read-only.
    """
    key = (yamlfile, tuple(int(n) for n in layerNumEls), float(elementSize))
    if key not in _SOLID_CACHE:
        from pynumad.mesh_gen.mesh_gen import get_solid_mesh
        from pynumad.objects.blade import Blade

        # Fresh blade — get_solid_mesh mutates stackdb in place.
        blade = Blade(yamlfile)
        _SOLID_CACHE[key] = get_solid_mesh(
            blade, list(layerNumEls), elementSize,
        )
    return _SOLID_CACHE[key]
