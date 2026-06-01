"""Unit tests for the conforming-mesh count-coordination helpers.

These are pure-function tests on tiny synthetic spline grids — no full
mesh build — to lock the count logic before it is wired into the mesher.
"""
from __future__ import annotations

import numpy as np
import pytest

from pynumad.mesh_gen.conforming import span_count_for_station, apply_span_count


def _grid(span_lengths, n_cols=None, n_rows=4):
    """Build a synthetic spline grid where chord column c has spanwise edge
    length ``span_lengths[c]`` (rows stacked along z, evenly)."""
    span_lengths = np.asarray(span_lengths, dtype=float)
    n_cols = n_cols or span_lengths.size
    X = np.zeros((n_rows, n_cols))
    Y = np.zeros((n_rows, n_cols))
    Z = np.zeros((n_rows, n_cols))
    for c in range(n_cols):
        L = span_lengths[c % span_lengths.size]
        # rows 0..3 evenly spaced over [0, L] along z
        Z[:, c] = np.linspace(0.0, L, n_rows)
        X[:, c] = c * 0.1  # arbitrary chord offset
    return X, Y, Z


def test_span_count_uniform_takes_max_column():
    # columns with span lengths 1.0, 3.0, 2.0 -> max 3.0
    X, Y, Z = _grid([1.0, 3.0, 2.0])
    assert span_count_for_station(X, Y, Z, 0, 3, elementSize=1.0) == 3
    # coarser element size -> fewer
    assert span_count_for_station(X, Y, Z, 0, 3, elementSize=1.5) == 2  # ceil(3/1.5)
    # finer -> more
    assert span_count_for_station(X, Y, Z, 0, 3, elementSize=0.5) == 6  # ceil(3/0.5)


def test_span_count_at_least_one():
    X, Y, Z = _grid([0.01])
    assert span_count_for_station(X, Y, Z, 0, 1, elementSize=1.0) == 1


def test_span_count_offset_station():
    # 7-row grid; station starting at row 0 spans rows 0..3
    X, Y, Z = _grid([2.0], n_cols=2, n_rows=7)
    # rows 0..3 of a linspace(0,2,7): z[3]-z[0] = 3*(2/6) = 1.0
    c = span_count_for_station(X, Y, Z, 0, 2, elementSize=0.5)
    assert c == 2  # ceil(1.0/0.5)


def test_apply_span_count_overrides_span_edges_only():
    nEl = np.array([2, 5, 2, 7])
    out = apply_span_count(nEl, 4)
    assert list(out) == [2, 4, 2, 4]          # chord (0,2) unchanged; span (1,3)=4
    assert list(nEl) == [2, 5, 2, 7]          # original not mutated


def test_apply_span_count_passthrough_none():
    assert apply_span_count(None, 4) is None


# ----------------------------------------------------------------------
# Pass 2 — AR-guarded chord-count segmentation
# ----------------------------------------------------------------------
from pynumad.mesh_gen.conforming import region_chord_count_segments


def test_chord_segments_constant_width_one_segment():
    """Uniform width along span -> a single segment, count = ceil(w/elem)."""
    counts, starts = region_chord_count_segments(
        chord_widths=[1.0] * 6, elementSize=0.5, span_elem_len=0.5, ar_max=4.0)
    assert counts == [2, 2, 2, 2, 2, 2]
    assert starts == [0]


def test_chord_segments_tapering_width_segments_when_ar_violated():
    """Tapering width forces segments when AR would exceed ar_max."""
    # width drops from 1.0 -> 0.05; with elementSize=0.5, span_elem=0.5, ar_max=4
    # count for width 1.0 -> 2; chord_elem = 1.0/2 = 0.5; AR=1 (ok)
    # extending to width 0.05 with c=2: chord_elem=0.025; AR=20 (bad) -> new segment
    counts, starts = region_chord_count_segments(
        chord_widths=[1.0, 0.05], elementSize=0.5, span_elem_len=0.5, ar_max=4.0)
    assert len(starts) == 2          # two segments
    assert counts[0] != counts[1] or starts == [0, 1]


def test_chord_segments_respect_ar_max():
    """No segment may violate AR>ar_max unless geometrically unavoidable —
    i.e. the station is so narrow that even the minimum count (1) gives
    AR>ar_max. Those degenerate sliver stations are accepted with count=1
    (any smaller would mean no elements at all)."""
    widths = [1.0, 0.8, 0.4, 0.2, 0.1, 0.05]
    span_elem = 0.5
    ar_max = 4.0
    counts, starts = region_chord_count_segments(
        widths, elementSize=0.5, span_elem_len=span_elem, ar_max=ar_max)
    for i, w in enumerate(widths):
        chord_elem = w / counts[i]
        ar = span_elem / chord_elem
        if ar > ar_max + 1e-9:
            # Allowed only when even count==1 cannot satisfy AR (w too small)
            assert counts[i] == 1 and span_elem > ar_max * w, (
                f"AR={ar:.2f} at station {i} exceeds {ar_max} avoidably")


def test_chord_segments_resolution_floor():
    """chord_element_size should not under-resolve (chord_elem <= elementSize)
    for the widest station of each segment."""
    counts, starts = region_chord_count_segments(
        chord_widths=[2.0, 1.0, 0.5], elementSize=0.5,
        span_elem_len=0.5, ar_max=4.0)
    for i, w in enumerate([2.0, 1.0, 0.5]):
        assert w / counts[i] <= 0.5 + 1e-9, (
            f"chord_elem {w/counts[i]:.3f} > elementSize 0.5 at station {i}")


def test_chord_segments_single_station_ok():
    counts, starts = region_chord_count_segments(
        chord_widths=[0.3], elementSize=0.5, span_elem_len=0.5, ar_max=4.0)
    assert counts == [1]
    assert starts == [0]
