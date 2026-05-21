"""In-memory cache for ``shell_mesh_general`` / ``get_shell_mesh`` outputs.

Mesh generation is deterministic for a given ``(blade, forSolid,
includeAdhesive, elementSize)`` tuple, so repeated calls with the same
arguments return identical results. For convergence sweeps and notebooks
where the same blade is meshed multiple times, this module provides a
process-local cache that turns the second-and-later calls into a dict
lookup.

Usage::

    from pynumad.mesh_gen.cache import cached_shell_mesh, cached_get_shell_mesh

    blade = pynumad.Blade(); blade.read_yaml("IEA-22.yaml")
    mesh1 = cached_shell_mesh(blade, forSolid=False, includeAdhesive=False,
                              elementSize=0.30)   # builds
    mesh2 = cached_shell_mesh(blade, forSolid=False, includeAdhesive=False,
                              elementSize=0.30)   # cache hit, instant

Cache scope is the running Python process. The key is ``(id(blade),
forSolid, includeAdhesive, elementSize)``; ``id(blade)`` is unique for
the lifetime of the Python object, so re-loading the same YAML into a
new ``Blade`` instance is a cache miss. That's intentional — the same
filename can map to different mesh inputs if blade YAML / mutations
happen between loads.

Set ``PYNUMAD_MESH_CACHE_SIZE`` env var (default 32) to bound the cache.
Call ``clear_mesh_cache()`` to drop everything (useful in long-running
notebooks where blade state is mutated in place).
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any

from pynumad._logging import get_logger
from pynumad.mesh_gen.mesh_gen import get_shell_mesh, shell_mesh_general

__all__ = [
    "cached_shell_mesh",
    "cached_get_shell_mesh",
    "clear_mesh_cache",
    "mesh_cache_stats",
]

_log = get_logger(__name__)


def _max_size() -> int:
    try:
        return max(1, int(os.environ.get("PYNUMAD_MESH_CACHE_SIZE", "32")))
    except ValueError:
        return 32


# Two separate caches because the underlying builders take different args
# and produce dicts with different keys.
_shell_mesh_cache: "OrderedDict[tuple, dict]" = OrderedDict()
_get_shell_mesh_cache: "OrderedDict[tuple, dict]" = OrderedDict()

_stats = {"hits": 0, "misses": 0}


def _bump_lru(cache: "OrderedDict[tuple, dict]", key: tuple, value: dict) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _max_size():
        cache.popitem(last=False)


def cached_shell_mesh(blade: Any, *, forSolid: bool, includeAdhesive: bool, elementSize: float) -> dict:
    """Cached wrapper around :func:`shell_mesh_general`.

    Cache key: ``(id(blade), bool(forSolid), bool(includeAdhesive),
    float(elementSize))``. If you mutate the blade in place between calls,
    invalidate the cache via :func:`clear_mesh_cache`.
    """
    key = (id(blade), bool(forSolid), bool(includeAdhesive), float(elementSize))
    cached = _shell_mesh_cache.get(key)
    if cached is not None:
        _stats["hits"] += 1
        _shell_mesh_cache.move_to_end(key)
        _log.debug(
            "cached_shell_mesh hit",
            extra={"stage": "cache_hit", "key": key, "stats": dict(_stats)},
        )
        return cached
    _stats["misses"] += 1
    _log.debug(
        "cached_shell_mesh miss",
        extra={"stage": "cache_miss", "key": key, "stats": dict(_stats)},
    )
    mesh = shell_mesh_general(
        blade,
        forSolid=forSolid,
        includeAdhesive=includeAdhesive,
        elementSize=elementSize,
    )
    _bump_lru(_shell_mesh_cache, key, mesh)
    return mesh


def cached_get_shell_mesh(blade: Any, *, includeAdhesive: int, elementSize: float) -> dict:
    """Cached wrapper around :func:`get_shell_mesh`.

    Same caching semantics as :func:`cached_shell_mesh`.
    """
    key = (id(blade), int(includeAdhesive), float(elementSize))
    cached = _get_shell_mesh_cache.get(key)
    if cached is not None:
        _stats["hits"] += 1
        _get_shell_mesh_cache.move_to_end(key)
        return cached
    _stats["misses"] += 1
    mesh = get_shell_mesh(blade, includeAdhesive=includeAdhesive, elementSize=elementSize)
    _bump_lru(_get_shell_mesh_cache, key, mesh)
    return mesh


def clear_mesh_cache() -> None:
    """Drop all cached meshes from this process."""
    _shell_mesh_cache.clear()
    _get_shell_mesh_cache.clear()
    _stats["hits"] = 0
    _stats["misses"] = 0


def mesh_cache_stats() -> dict:
    """Return a snapshot of cache hits/misses/sizes for diagnostics."""
    return {
        "hits": _stats["hits"],
        "misses": _stats["misses"],
        "shell_mesh_cache_size": len(_shell_mesh_cache),
        "get_shell_mesh_cache_size": len(_get_shell_mesh_cache),
        "max_size": _max_size(),
    }
