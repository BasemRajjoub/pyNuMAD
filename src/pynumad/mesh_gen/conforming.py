"""Conforming-mesh count coordination for the structured shell mesher.

The structured mesher builds each chord region of each station from its own
local edge lengths, so adjacent patches disagree on how many nodes to put
on a shared boundary and a node of one lands mid-edge of its neighbour — a
T-junction (hanging node). Those are displacement-incompatible unless tied,
and tying thousands of them over-constrains the model. The clean fix is to
*coordinate the element counts* so shared boundaries get the same count and
the nodes are genuinely shared.

This module holds the pure (FE-state-free, unit-testable) coordination
helpers. They take geometry/edge-length arrays and return element counts.

Pass 1 — span count per station
    All chord regions of a station span the same root->tip extent, so they
    can all use one spanwise count. Using the *max* spanwise edge length in
    the station only ever *refines* a region's span (shorter elements ->
    lower aspect ratio), so it never creates slivers, and it removes the
    within-station chord-boundary T-junctions caused by ``force_match_opposite``
    clamping each region to its own max.

Pass 2 (chord count across stations) is added separately with an aspect-
ratio guard (the hybrid: conform where AR stays acceptable, otherwise keep
the local count and tie that boundary with a constraint equation).
"""
from __future__ import annotations

import numpy as np


def span_count_for_station(splineXi, splineYi, splineZi, stPt, n_cols,
                           elementSize):
    """Uniform spanwise element count for every chord region of one station.

    Parameters
    ----------
    splineXi, splineYi, splineZi : ndarray
        Spline-grid coordinates (rows = spanwise samples, cols = chord
        samples).
    stPt : int
        Spanwise start row of the station; the station spans rows
        ``stPt .. stPt+3``.
    n_cols : int
        Number of chord columns to consider (e.g. 37 for the 12-region
        layout, columns 0..36).
    elementSize : float
        Target element size (m).

    Returns
    -------
    int
        ``ceil(max_chord-column spanwise-edge-length / elementSize)``,
        clamped to >= 1. Applying this to all regions makes every chord
        boundary within the station conforming.
    """
    assert elementSize > 0
    p0 = np.stack([splineXi[stPt, :n_cols],
                   splineYi[stPt, :n_cols],
                   splineZi[stPt, :n_cols]], axis=1)
    p3 = np.stack([splineXi[stPt + 3, :n_cols],
                   splineYi[stPt + 3, :n_cols],
                   splineZi[stPt + 3, :n_cols]], axis=1)
    span_lens = np.linalg.norm(p3 - p0, axis=1)
    max_len = float(np.nanmax(span_lens))
    return max(1, int(np.ceil(max_len / elementSize)))


def region_chord_counts(splineXi, splineYi, splineZi, n_stations,
                        n_regions, cols_per_region, elementSize,
                        span_elem_len=None, ar_max=4.0):
    """Pass 2 (AR-guarded): piecewise-constant chord count per region along
    span — runs of stations that share one count where AR allows, with a
    new segment (count change -> CE-tie) only where AR would be violated.

    For each region *j* this measures the chord width at every station,
    then calls :func:`region_chord_count_segments` to choose the segmented
    counts. Constant counts inside a segment make adjacent stations'
    station-boundary chord edges conforming (shared nodes — the dominant
    ~97 % T-junction source), and equal opposite chord edges within a
    region (no Jacobian flip). Only the segment-boundary stations require
    a constraint-equation tie.

    Parameters
    ----------
    span_elem_len : float, optional
        Effective spanwise element length used for AR checks. Defaults to
        ``elementSize``.

    Returns
    -------
    counts : ndarray, shape (n_stations, n_regions), int
        ``counts[i, j]`` = chord element count at station *i*, region *j*.
    segment_starts : list[list[int]]
        ``segment_starts[j]`` = station indices where region *j* opens a
        new chord-count segment (always starts with ``0``); a CE tie is
        needed between station ``segment_starts[j][k]-1`` and
        ``segment_starts[j][k]`` for ``k>=1``.
    """
    if span_elem_len is None:
        span_elem_len = elementSize
    counts = np.zeros((n_stations, n_regions), dtype=int)
    segment_starts: list[list[int]] = []
    for j in range(n_regions):
        c_lo = cols_per_region * j
        c_hi = cols_per_region * j + cols_per_region
        widths = []
        for i in range(n_stations):
            stPt = 3 * i
            a = np.array([splineXi[stPt, c_lo], splineYi[stPt, c_lo], splineZi[stPt, c_lo]])
            b = np.array([splineXi[stPt, c_hi], splineYi[stPt, c_hi], splineZi[stPt, c_hi]])
            widths.append(float(np.linalg.norm(b - a)))
        segs, starts = region_chord_count_segments(
            widths, elementSize, span_elem_len, ar_max=ar_max,
        )
        counts[:, j] = segs
        segment_starts.append(starts)
    return counts, segment_starts


def region_chord_count_segments(chord_widths, elementSize, span_elem_len,
                                ar_max=4.0):
    """AR-guarded piecewise-constant chord count for one region along span.

    For region *j*, walk the stations and greedily extend a "segment" of
    stations that can share **one** chord count without violating

      - the resolution floor (``chord_element_size <= elementSize``, so the
        widest station in the segment is not under-resolved), and
      - the aspect-ratio cap (``AR = span_elem_len/chord_element_size
        <= ar_max`` at every station, i.e. the narrowest station's chord
        element does not become a sliver).

    A segment of stations ``[a..b]`` with count ``c`` is feasible iff

      ceil(max_widths[a..b] / elementSize)  <=  c  <=  floor(ar_max * min_widths[a..b] / span_elem_len)

    The greedy picks the tightest feasible ``c`` (the resolution floor)
    when starting a segment, and extends as long as the next station keeps
    the range non-empty. When extension would break either bound, the
    segment closes and a new one starts — that station boundary is where a
    constraint-equation tie is needed (hybrid: conform where AR allows, CE
    the rest).

    Parameters
    ----------
    chord_widths : array-like of float
        Chord width of the region at each station along span (m).
    elementSize : float
        Target element size (m).
    span_elem_len : float
        Effective spanwise element length (m); typically ``~elementSize``
        once span-coordination (Pass 1) is on.
    ar_max : float
        Max acceptable element aspect ratio.

    Returns
    -------
    counts : list[int]
        ``counts[i]`` = chord count at station i (piecewise constant; >=1).
    segment_starts : list[int]
        Station indices where a new segment (count change) begins. The
        boundary at station ``segment_starts[k]`` is where a CE tie is
        needed (between station ``segment_starts[k]-1`` and
        ``segment_starts[k]``). The list always starts with 0.
    """
    assert elementSize > 0 and span_elem_len > 0 and ar_max > 0
    widths = [float(w) for w in chord_widths]
    n = len(widths)
    counts: list[int] = []
    segment_starts: list[int] = [0]
    i = 0
    while i < n:
        run_max = widths[i]
        run_min = widths[i]
        # Tightest count that resolves the widest station so far
        c = max(1, int(np.ceil(run_max / elementSize)))
        # Extend while feasible
        j = i
        while j + 1 < n:
            new_max = max(run_max, widths[j + 1])
            new_min = min(run_min, widths[j + 1])
            new_c_min = max(1, int(np.ceil(new_max / elementSize)))
            new_c_max = int(np.floor(ar_max * new_min / span_elem_len))
            if new_c_min <= new_c_max:
                run_max, run_min = new_max, new_min
                c = new_c_min
                j += 1
            else:
                break
        counts.extend([c] * (j - i + 1))
        if j + 1 < n:
            segment_starts.append(j + 1)
        i = j + 1
    return counts, segment_starts


def apply_chord_count(nEl, chord_count):
    """Return a copy of ``[chord, span, chord, span]`` with both chord
    entries (indices 0 and 2) overridden by ``chord_count``. ``None`` passes
    through."""
    if nEl is None:
        return None
    out = np.asarray(nEl, dtype=int).copy()
    out[0] = chord_count
    out[2] = chord_count
    return out


def apply_span_count(nEl, span_count):
    """Return a copy of a 4-edge count array ``[chord, span, chord, span]``
    with both span entries (indices 1 and 3) overridden by ``span_count``.

    ``nEl`` may be ``None`` (degenerate patch skipped upstream), in which
    case ``None`` is returned unchanged.
    """
    if nEl is None:
        return None
    out = np.asarray(nEl, dtype=int).copy()
    out[1] = span_count
    out[3] = span_count
    return out
