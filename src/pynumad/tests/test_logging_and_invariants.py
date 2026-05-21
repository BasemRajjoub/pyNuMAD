"""Tests for the logging sidecar and invariants modules.

These keep the instrumentation honest: if someone changes the JSONL schema
or the invariants API, these tests catch it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from pynumad import _logging as plog
from pynumad import invariants as inv


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_get_logger_namespacing():
    """`get_logger` nests every name under `pynumad.*`."""
    assert plog.get_logger("mesh_gen.shell_region").name == "pynumad.mesh_gen.shell_region"
    assert plog.get_logger("pynumad.foo").name == "pynumad.foo"
    assert plog.get_logger("pynumad").name == "pynumad"
    assert plog.get_logger(None).name == "pynumad"


def test_jsonl_sidecar_writes_one_record_per_log(tmp_path: Path):
    log_path = tmp_path / "trace.jsonl"
    handler = plog.enable_jsonl_sidecar(log_path, level=logging.DEBUG, truncate=True)
    try:
        log = plog.get_logger("pynumad.test_logging")
        log.info("hello", extra={"region_name": "X", "edgeEls": [5, 5, 4, 5]})
        log.warning("danger", extra={"branch": "ee2_lt_ee0"})
        log.debug("verbose", extra={"k": 1})
    finally:
        plog.disable_jsonl_sidecar()

    records = [json.loads(ln) for ln in log_path.read_text().splitlines()]
    assert len(records) == 3
    assert records[0]["level"] == "INFO"
    assert records[0]["msg"] == "hello"
    assert records[0]["context"]["region_name"] == "X"
    assert records[0]["context"]["edgeEls"] == [5, 5, 4, 5]
    assert records[1]["level"] == "WARNING"
    assert records[1]["context"]["branch"] == "ee2_lt_ee0"


def test_jsonl_schema_keys_stable(tmp_path: Path):
    """Every record has the stable top-level keys downstream tooling expects."""
    log_path = tmp_path / "schema.jsonl"
    plog.enable_jsonl_sidecar(log_path, truncate=True)
    try:
        plog.get_logger("pynumad.schema_test").info("msg", extra={"k": "v"})
    finally:
        plog.disable_jsonl_sidecar()

    rec = json.loads(log_path.read_text().strip())
    for key in ("ts", "level", "logger", "msg", "module", "func", "line"):
        assert key in rec, f"missing key {key!r} in record {rec}"


def test_jsonl_summarises_large_numpy_arrays(tmp_path: Path):
    log_path = tmp_path / "arr.jsonl"
    plog.enable_jsonl_sidecar(log_path, truncate=True)
    try:
        log = plog.get_logger("pynumad.arr_test")
        log.debug("small", extra={"arr": np.array([1.0, 2.0, 3.0])})
        log.debug("large", extra={"arr": np.arange(1000)})
    finally:
        plog.disable_jsonl_sidecar()

    recs = [json.loads(ln) for ln in log_path.read_text().splitlines()]
    # small array dumped verbatim
    assert recs[0]["context"]["arr"] == [1.0, 2.0, 3.0]
    # large array summarised
    big = recs[1]["context"]["arr"]
    assert big["_kind"] == "ndarray"
    assert big["shape"] == [1000]
    assert big["min"] == 0.0 and big["max"] == 999.0


def test_disable_sidecar_is_idempotent():
    plog.disable_jsonl_sidecar()
    plog.disable_jsonl_sidecar()  # second call must not raise


def test_default_mode_is_silent(capsys, tmp_path: Path):
    """With no sidecar enabled, library logging should not write to stderr."""
    plog.disable_jsonl_sidecar()
    plog.get_logger("pynumad.silent_test").warning("should not appear")
    captured = capsys.readouterr()
    assert captured.err == ""


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


class TestEdgeEls:
    def test_valid(self):
        ok, msg = inv.check_edge_els([5, 5, 4, 5])
        assert ok and msg == ""

    def test_wrong_length(self):
        ok, msg = inv.check_edge_els([5, 5, 4])
        assert not ok and "length 4" in msg

    def test_zero_count(self):
        ok, msg = inv.check_edge_els([5, 0, 4, 5])
        assert not ok and "<= 0" in msg

    def test_none(self):
        ok, msg = inv.check_edge_els(None)
        assert not ok

    def test_require_raises(self):
        with pytest.raises(ValueError, match="length 4"):
            inv.require_edge_els([5, 5, 4])

    def test_require_passes_silently(self):
        inv.require_edge_els([5, 5, 5, 5])  # no exception


class TestOppositeEdges:
    def test_matched(self):
        ok, _ = inv.opposite_edges_match([5, 5, 5, 5])
        assert ok

    def test_mismatched(self):
        ok, msg = inv.opposite_edges_match([5, 5, 4, 5])
        assert not ok and "ee[0]=5" in msg and "ee[2]=4" in msg


class TestQuadJacobian:
    def test_unit_square(self):
        sq = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        ok, _ = inv.quad_is_positive_jacobian(sq)
        assert ok

    def test_bow_tie(self):
        bt = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
        ok, msg = inv.quad_is_positive_jacobian(bt)
        assert not ok and "opposite signs" in msg

    def test_degenerate_triangle_first(self):
        # Three colinear points
        d = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=float)
        ok, msg = inv.quad_is_positive_jacobian(d)
        assert not ok and "degenerate" in msg

    def test_require_raises_on_bow_tie(self):
        bt = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
        with pytest.raises(ValueError, match="non-positive Jacobian"):
            inv.require_quad_positive_jacobian(bt)


class TestMeshDict:
    def _good_mesh(self):
        return {
            "nodes": np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float),
            "elements": np.array([[0, 1, 2, 3]], dtype=int),
        }

    def test_good_mesh_passes_all_checks(self):
        m = self._good_mesh()
        assert inv.check_mesh_dict(m)[0]
        assert inv.no_unreferenced_node_ids(m)[0]
        assert inv.no_negative_node_ids_except_sentinel(m)[0]
        assert inv.quad_elements_have_distinct_first_three_nodes(m)[0]

    def test_missing_key(self):
        ok, msg = inv.check_mesh_dict({"nodes": np.zeros((2, 3))})
        assert not ok and "elements" in msg

    def test_unreferenced_node_id(self):
        m = self._good_mesh()
        m["elements"] = np.array([[0, 1, 99, 3]], dtype=int)
        ok, _ = inv.no_unreferenced_node_ids(m)
        assert not ok

    def test_negative_id_other_than_sentinel(self):
        m = self._good_mesh()
        m["elements"] = np.array([[0, 1, -2, 3]], dtype=int)  # -2 is not the -1 sentinel
        ok, _ = inv.no_negative_node_ids_except_sentinel(m)
        assert not ok

    def test_minus_one_sentinel_is_ok(self):
        m = self._good_mesh()
        m["elements"] = np.array([[0, 1, 2, -1]], dtype=int)  # collapsed triangle
        ok, _ = inv.no_negative_node_ids_except_sentinel(m)
        assert ok

    def test_degenerate_first_three_nodes_caught(self):
        m = self._good_mesh()
        m["elements"] = np.array([[0, 0, 1, 2]], dtype=int)  # duplicate in first three
        ok, _ = inv.quad_elements_have_distinct_first_three_nodes(m)
        assert not ok
