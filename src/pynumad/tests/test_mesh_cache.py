"""Tests for the process-local mesh cache."""

from __future__ import annotations

import os

import pytest

from pynumad.mesh_gen.cache import (
    cached_get_shell_mesh,
    cached_shell_mesh,
    clear_mesh_cache,
    mesh_cache_stats,
)
from pynumad.objects.blade import Blade

test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")


@pytest.fixture
def blade():
    return Blade(os.path.join(test_data_dir, "blades", "blade.yaml"))


def test_cached_get_shell_mesh_same_args_returns_same_object(blade):
    clear_mesh_cache()
    m1 = cached_get_shell_mesh(blade, includeAdhesive=1, elementSize=0.2)
    m2 = cached_get_shell_mesh(blade, includeAdhesive=1, elementSize=0.2)
    assert m1 is m2, "second call must return the cached object identity"
    stats = mesh_cache_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_cached_shell_mesh_different_element_size_is_miss(blade):
    from pynumad.mesh_gen.cache import _shell_mesh_cache

    clear_mesh_cache()
    cached_shell_mesh(blade, forSolid=False, includeAdhesive=False, elementSize=0.3)
    cached_shell_mesh(blade, forSolid=False, includeAdhesive=False, elementSize=0.45)
    assert mesh_cache_stats()["misses"] == 2


def test_cached_shell_mesh_different_flag_is_miss(blade):
    clear_mesh_cache()
    cached_shell_mesh(blade, forSolid=False, includeAdhesive=False, elementSize=0.3)
    cached_shell_mesh(blade, forSolid=False, includeAdhesive=True, elementSize=0.3)
    assert mesh_cache_stats()["misses"] == 2


def test_clear_mesh_cache_resets_stats(blade):
    clear_mesh_cache()
    cached_get_shell_mesh(blade, includeAdhesive=1, elementSize=0.2)
    cached_get_shell_mesh(blade, includeAdhesive=1, elementSize=0.2)
    assert mesh_cache_stats()["hits"] == 1
    clear_mesh_cache()
    assert mesh_cache_stats()["hits"] == 0
    assert mesh_cache_stats()["misses"] == 0


def test_env_var_caps_cache_size(monkeypatch, blade):
    monkeypatch.setenv("PYNUMAD_MESH_CACHE_SIZE", "1")
    clear_mesh_cache()
    cached_shell_mesh(blade, forSolid=False, includeAdhesive=False, elementSize=0.3)
    cached_shell_mesh(blade, forSolid=False, includeAdhesive=False, elementSize=0.45)
    # Size-1 cache: second insertion should have evicted the first
    assert mesh_cache_stats()["shell_mesh_cache_size"] == 1
