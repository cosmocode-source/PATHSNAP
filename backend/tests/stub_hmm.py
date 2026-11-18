"""Stub HMM decoder for testing ONLY.

FOR TESTING ONLY: Never use in production.
Implements the contract:
    decode_trip(candidates, probs, network, gps_xy, times, config)
    -> {
        "arc_per_point": list[int] or np.ndarray,
        "predicted_route": list[int] or np.ndarray,
        "diagnostics": {
            "fallbacks_unreachable": int,
            "fallbacks_gap": int,
        }
    }
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np
import pandas as pd


def stub_decode_trip(
    candidates: pd.DataFrame,
    probs: np.ndarray,
    network: Dict[str, Any],
    gps_xy: np.ndarray,
    times: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Stub HMM decoder that returns classifier arcs with deliberate modifications on a few points

    to test the HMM code path, HMM changed markers, and delta calculation.
    """
    gps_rows = candidates["gps_row"].to_numpy(dtype=np.int64)
    n_points = len(times)

    # Pick top candidate per point
    row_idx = np.arange(len(probs))
    order = np.lexsort((row_idx, -probs, gps_rows))
    first_in_point = np.r_[True, gps_rows[order][1:] != gps_rows[order][:-1]]
    top_indices = order[first_in_point]
    arc_ids = candidates["arc_id"].to_numpy()[top_indices].copy()

    # Deliberately modify candidate for 1-2 points if alternatives exist (to test "HMM changed" detection)
    if n_points > 5:
        # Check if point 3 has a second candidate
        p3_cands = candidates[candidates["gps_row"] == 3]["arc_id"].to_numpy()
        if len(p3_cands) > 1:
            arc_ids[3] = p3_cands[1]

    # Collapsed route
    keep = np.r_[True, arc_ids[1:] != arc_ids[:-1]]
    predicted_route = arc_ids[keep].tolist()

    return {
        "arc_per_point": arc_ids.tolist(),
        "predicted_route": predicted_route,
        "diagnostics": {
            "fallbacks_unreachable": 1,
            "fallbacks_gap": 0,
        },
    }
