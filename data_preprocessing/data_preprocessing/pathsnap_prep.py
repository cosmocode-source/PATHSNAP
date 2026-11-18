"""PathSnap preprocessing library (sections 4.1 - 4.3 of the preprocessing plan).

What this module does
---------------------
* Loads one trip (GPS track + its OWN road network) and projects everything to metres.
* Computes per-GPS-point quality flags.
* Produces per-point reference labels for TRAINING / server-side scoring by aligning
  the GPS track to the reference route with a route-constrained Viterbi (DP) search.

Index spaces (never join them by row number!)
---------------------------------------------
* ``gps_row``    - row number in the .track file (0-based).
* ``node_id``    - row number in the .nodes file; LOCAL to a trip.
* ``arc_id``     - row number in the .arcs file; LOCAL to a trip. Never a global class label.
* ``route_pos``  - position (0-based) inside the .route sequence. ``route[route_pos]`` is an arc_id.
                   The same arc_id can occur at several route_pos values (loops).

Inference safety
----------------
``load_trip`` and ``point_flags`` NEVER read the .route file. The route is only read by
``load_route`` which is used for training labels and for server-side scoring.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely.strtree import STRtree

# --------------------------------------------------------------------------------------
# CONFIG - every threshold lives here with its reason (no magic numbers elsewhere).
# Values marked [EDA] come from the team's EDA notebook (PathSnap_EDA.ipynb CONFIG cell).
# --------------------------------------------------------------------------------------
CONFIG = {
    "seed": 42,                       # fixed seed for anything random (fold assignment etc.)
    # ---- point flags -------------------------------------------------------------
    "gap_seconds": 5.0,               # [EDA] dt above this = long GPS gap; HMM transitions break here.
    "impossible_speed_mps": 50.0,     # [EDA] 180 km/h: not plausible for road traffic -> speed_jump.
    "mad_z_threshold": 5.0,           # [EDA] |speed - median| > 5*MAD (per-trip) -> robust outlier.
    "hive_radius_m": 5.0,             # [EDA] points staying within 5 m ...
    "hive_min_points": 10,            # ... for >= 10 consecutive fixes (10 s at 1 Hz) = GPS "hive".
    "stationary_step_m": 1.0,         # [EDA] step <= 1 m with dt > 0 = stationary.
    # ---- labelling (route-constrained alignment) ---------------------------------
    "align_sigma_m": 10.0,            # emission std-dev: GPS lies ~4-6 m (median) from the route, p95 ~15 m.
    "align_emission_cap_m": 50.0,     # distances beyond this carry no extra evidence (robust to outliers).
    "align_stay_cost": 0.0,           # staying on the same route_pos is free (several fixes per arc).
    "align_advance_cost": 1.0,        # cost of moving to the next route_pos.
    "align_skip_cost": 3.0,           # extra cost per additionally skipped route_pos.
    "align_base_skip": 3,             # max forward skip (arcs) in one normal 1 s step.
    "align_arcs_per_second": 2.0,     # extra skip allowance per second of dt above 1 s (short urban arcs).
    "align_gap_skip_cost": 0.1,       # skip cost per arc right after a long gap (position is unknown).
    "far_label_m": 30.0,              # label farther than this from its GPS point = 'far' quality.
    "ambiguity_margin_m": 5.0,        # [EDA] best vs next-best route arc closer than this = 'ambiguous'.
                                      # Re-checked on train: ~ the median label distance (4.8 m) = one typical GPS error.
    # far_label_m = 30 re-checked: it is the p99 label distance of well-covered train trips (29.4 m).
    # ---- sample weights (TRAIN only; team-lead decision; student rows always weight 1) ----
    "weight_far": 0.3,                # far labels: reference route may not cover the point -> trust less.
    "weight_ambiguous": 0.5,          # ambiguous labels: label arc is geometrically tied with another route arc.
    # ---- paths -------------------------------------------------------------------
    "chunk_cells": 2_000_000,         # max (points x route arcs) cells per distance chunk (memory guard).
}


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (default cwd) until a folder containing ``Dataset`` is found."""
    p = (start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "Dataset").is_dir():
            return cand
    raise FileNotFoundError("Could not find the repository root (folder containing 'Dataset').")


def list_trips(data_dir: Path, split: str) -> list[str]:
    """Return sorted trip ids of a partition folder (e.g. 'train' or 'student_test')."""
    return sorted(p.name for p in (data_dir / split).iterdir() if p.is_dir())


# --------------------------------------------------------------------------------------
# 4.1 Per-trip loader
# --------------------------------------------------------------------------------------
def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing (lon, lat). 326xx = north, 327xx = south.

    Why UTM: metre-based and accurate to well under 1 m over a trip-sized area, unlike
    degrees (Euclidean distance in degrees is not a physical distance)."""
    zone = int((lon + 180.0) // 6.0) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


@dataclass
class Trip:
    """Everything needed to process one trip WITHOUT the reference route."""
    trip_id: str
    split: str
    epsg: int
    gps_lonlat: np.ndarray   # (N,2) lon, lat; index = gps_row
    gps_t: np.ndarray        # (N,)  seconds; index = gps_row
    gps_xy: np.ndarray       # (N,2) projected metres; index = gps_row
    node_xy: np.ndarray      # (V,2) projected metres; index = node_id (local)
    arcs: np.ndarray         # (A,2) from_node_id, to_node_id; index = arc_id (local)
    arc_a: np.ndarray        # (A,2) projected start point of each arc; index = arc_id
    arc_b: np.ndarray        # (A,2) projected end point of each arc; index = arc_id
    arc_len: np.ndarray      # (A,)  metres; index = arc_id
    arc_bearing: np.ndarray  # (A,)  degrees clockwise from north [0,360); index = arc_id
    node_in_deg: np.ndarray  # (V,)  index = node_id
    node_out_deg: np.ndarray  # (V,) index = node_id
    arc_has_reverse: np.ndarray  # (A,) bool: does the opposite-direction arc exist? index = arc_id
    _tree: STRtree | None = field(default=None, repr=False)

    @property
    def n_gps(self) -> int:
        return len(self.gps_t)

    @property
    def n_arcs(self) -> int:
        return len(self.arcs)

    def arc_geoms(self) -> np.ndarray:
        """Shapely LineString per arc (index = arc_id), built from its two node coordinates."""
        return shapely.linestrings(np.stack([self.arc_a, self.arc_b], axis=1))

    def tree(self) -> STRtree:
        """STRtree over all arcs of this trip (built once, then reused). Query returns arc_id."""
        if self._tree is None:
            self._tree = STRtree(self.arc_geoms())
        return self._tree


def _read_ws(path: Path, cols: list[str], dtype) -> pd.DataFrame:
    """Read a header-less whitespace-separated raw file with an explicit schema."""
    return pd.read_csv(path, sep=r"\s+", header=None, names=cols, dtype=dtype, engine="c")


def assert_index_spaces(trip: Trip) -> None:
    """Sanity checks that the four index spaces are internally consistent."""
    assert trip.gps_xy.shape == (trip.n_gps, 2)
    assert trip.arcs.min() >= 0 and trip.arcs.max() < len(trip.node_xy), "arc endpoint outside node_id range"
    assert len(trip.arc_len) == trip.n_arcs == len(trip.arc_a) == len(trip.arc_b)
    assert np.all(trip.arc_len >= 0)


def load_trip(data_dir: Path, split: str, trip_id: str, cache_dir: Path | None = None) -> Trip:
    """Load one trip, project to UTM metres and derive arc geometry. Never reads .route.

    Args:
        data_dir: folder holding the partitions (``Dataset``).
        split: 'train' or 'student_test' (the evaluator partition must never be passed here by us).
        trip_id: e.g. '00000001'.
        cache_dir: optional folder for a ``.npz`` cache of the projected arrays.
    """
    cache = (cache_dir / f"{split}_{trip_id}.npz") if cache_dir else None
    if cache is not None and cache.exists():
        z = np.load(cache)
        return Trip(trip_id, split, int(z["epsg"]), z["gps_lonlat"], z["gps_t"], z["gps_xy"], z["node_xy"],
                    z["arcs"], z["arc_a"], z["arc_b"], z["arc_len"], z["arc_bearing"], z["node_in_deg"],
                    z["node_out_deg"], z["arc_has_reverse"])

    d = data_dir / split / trip_id
    track = _read_ws(d / f"{trip_id}.track", ["lon", "lat", "t"], {"lon": "float64", "lat": "float64", "t": "float64"})
    nodes = _read_ws(d / f"{trip_id}.nodes", ["lon", "lat"], {"lon": "float64", "lat": "float64"})
    arcs = _read_ws(d / f"{trip_id}.arcs", ["u", "v"], {"u": "int64", "v": "int64"})

    # One UTM zone per trip, chosen from the GPS centroid (the trip is local).
    epsg = utm_epsg(float(track.lon.mean()), float(track.lat.mean()))
    tf = Transformer.from_crs(4326, epsg, always_xy=True)  # always_xy => (lon, lat) order
    gx, gy = tf.transform(track.lon.to_numpy(), track.lat.to_numpy())
    nx_, ny_ = tf.transform(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    gps_xy = np.column_stack([gx, gy])
    node_xy = np.column_stack([nx_, ny_])

    arcs_np = arcs.to_numpy()                       # row = arc_id
    arc_a, arc_b = node_xy[arcs_np[:, 0]], node_xy[arcs_np[:, 1]]
    delta = arc_b - arc_a
    arc_len = np.hypot(delta[:, 0], delta[:, 1])
    arc_bearing = (np.degrees(np.arctan2(delta[:, 0], delta[:, 1])) + 360.0) % 360.0  # 0 = north, clockwise

    n_nodes = len(node_xy)
    node_out = np.bincount(arcs_np[:, 0], minlength=n_nodes)
    node_in = np.bincount(arcs_np[:, 1], minlength=n_nodes)
    keys = arcs_np[:, 0].astype(np.int64) * n_nodes + arcs_np[:, 1]
    rev_keys = arcs_np[:, 1].astype(np.int64) * n_nodes + arcs_np[:, 0]
    has_rev = np.isin(rev_keys, keys)

    trip = Trip(trip_id, split, epsg, track[["lon", "lat"]].to_numpy(), track.t.to_numpy(), gps_xy, node_xy,
                arcs_np, arc_a, arc_b, arc_len, arc_bearing, node_in, node_out, has_rev)
    assert_index_spaces(trip)
    if cache is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache, epsg=epsg, gps_lonlat=trip.gps_lonlat, gps_t=trip.gps_t, gps_xy=gps_xy, node_xy=node_xy,
                 arcs=arcs_np, arc_a=arc_a, arc_b=arc_b, arc_len=arc_len, arc_bearing=arc_bearing,
                 node_in_deg=node_in, node_out_deg=node_out, arc_has_reverse=has_rev)
    return trip


def load_route(data_dir: Path, split: str, trip_id: str, n_arcs: int) -> np.ndarray:
    """Read the reference route (arc_id sequence). ONLY for training labels / server-side scoring."""
    r = _read_ws(data_dir / split / trip_id / f"{trip_id}.route", ["arc_id"], {"arc_id": "int64"}).arc_id.to_numpy()
    assert r.min() >= 0 and r.max() < n_arcs, "route references an arc_id outside the arcs file"
    return r  # element index = route_pos, value = arc_id (local)


# --------------------------------------------------------------------------------------
# 4.2 Point flags
# --------------------------------------------------------------------------------------
def point_flags(trip: Trip, cfg: dict = CONFIG) -> pd.DataFrame:
    """Per-GPS-point quality flags (one row per gps_row). Uses only the track, never the route.

    Removable-in-training flags: exact_duplicate, invalid_coord, non_monotonic_ts.
    Keep-and-flag: stationary, in_hive, after_gap, speed_jump, speed_outlier.
    Speed uses the real time step (dt), not an assumed 1 s.
    """
    n = trip.n_gps
    lon, lat = trip.gps_lonlat[:, 0], trip.gps_lonlat[:, 1]
    t = trip.gps_t
    xy = trip.gps_xy

    dt = np.full(n, np.nan)
    dt[1:] = np.diff(t)                                    # seconds since previous gps_row
    step = np.full(n, np.nan)
    step[1:] = np.hypot(*np.diff(xy, axis=0).T)            # metres since previous gps_row
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, step / dt, np.nan)        # m/s; undefined if dt <= 0

    # --- removable-in-training flags ---
    invalid = (np.isnan(lon) | np.isnan(lat) | np.isnan(t) | (np.abs(lon) > 180) | (np.abs(lat) > 90)
               | ((lon == 0) & (lat == 0)))
    dup = pd.DataFrame({"a": lon, "b": lat, "c": t}).duplicated().to_numpy()   # later copy is the duplicate
    nonmono = np.zeros(n, bool)
    nonmono[1:] = dt[1:] <= 0                               # timestamps must strictly increase

    # --- keep-and-flag ---
    after_gap = np.zeros(n, bool)
    after_gap[1:] = dt[1:] > cfg["gap_seconds"]
    gap_len = np.where(after_gap, dt, 0.0)
    stationary = np.zeros(n, bool)
    stationary[1:] = (step[1:] <= cfg["stationary_step_m"]) & (dt[1:] > 0)
    speed_jump = np.nan_to_num(speed, nan=0.0) > cfg["impossible_speed_mps"]

    # Robust speed outlier: per-trip median/MAD (self-normalising; needs no fitted state, so it
    # cannot leak anything from another partition and works identically at inference time).
    ok = np.isfinite(speed) & ~after_gap       # gap-average speeds are not comparable to 1 Hz speeds
    if ok.sum() > 3:
        med = np.median(speed[ok])
        mad = max(np.median(np.abs(speed[ok] - med)), 1e-6)
        outlier = np.nan_to_num(np.abs(speed - med), nan=0.0) > cfg["mad_z_threshold"] * mad
    else:
        outlier = np.zeros(n, bool)

    # Hive: a window of `hive_min_points` consecutive fixes that all stay within hive_radius_m
    # of the window centroid -> every point of the window is flagged.
    w = cfg["hive_min_points"]
    in_hive = np.zeros(n, bool)
    if n >= w:
        win = np.lib.stride_tricks.sliding_window_view(xy, w, axis=0)       # (n-w+1, 2, w)
        cen = win.mean(axis=2, keepdims=True)
        maxd = np.sqrt(((win - cen) ** 2).sum(axis=1)).max(axis=1)           # (n-w+1,)
        hits = np.flatnonzero(maxd <= cfg["hive_radius_m"])
        diff = np.zeros(n + 1, int)
        np.add.at(diff, hits, 1)
        np.add.at(diff, hits + w, -1)
        in_hive = np.cumsum(diff[:-1]) > 0

    return pd.DataFrame({
        "gps_row": np.arange(n), "lon": lon, "lat": lat, "t": t, "x_m": xy[:, 0], "y_m": xy[:, 1],
        "dt_s": dt, "step_m": step, "speed_mps": speed,
        "exact_duplicate": dup, "invalid_coord": invalid, "non_monotonic_ts": nonmono,
        "removable": dup | invalid | nonmono,
        "stationary": stationary, "in_hive": in_hive, "after_gap": after_gap, "gap_len_s": gap_len,
        "speed_jump": speed_jump, "speed_outlier": outlier,
    })


def apply_training_drop(flags: pd.DataFrame, split: str) -> tuple[pd.DataFrame, dict]:
    """Drop removable rows from TRAINING trips only. Student-test (and anything else) is never altered.

    Returns (kept_rows, counts_removed_per_rule). ``gps_row`` keeps its ORIGINAL value after dropping."""
    if split != "train":
        return flags, {"exact_duplicate": 0, "invalid_coord": 0, "non_monotonic_ts": 0}
    counts = {r: int(flags[r].sum()) for r in ["exact_duplicate", "invalid_coord", "non_monotonic_ts"]}
    return flags.loc[~flags.removable].reset_index(drop=True), counts


# --------------------------------------------------------------------------------------
# 4.3 Labelling by route-constrained alignment (TRAIN / server-side only)
# --------------------------------------------------------------------------------------
def point_segment_distance(P: np.ndarray, A: np.ndarray, B: np.ndarray, chunk_cells: int = 2_000_000):
    """Distance (m) from every point to every segment, and projection fraction in [0,1].

    Args:
        P: (n,2) points.  A, B: (m,2) segment end points (in the SAME projected metres).
    Returns:
        D (n,m) float32 distances, T (n,m) float32 fractions along the segment (0 = A, 1 = B).
    """
    n, m = len(P), len(A)
    D = np.empty((n, m), np.float32)
    T = np.empty((n, m), np.float32)
    ab = B - A                                            # (m,2)
    ab2 = np.maximum((ab ** 2).sum(1), 1e-9)              # (m,) squared length (guard zero-length arcs)
    step = max(1, chunk_cells // max(m, 1))
    for s in range(0, n, step):
        p = P[s:s + step, None, :]                        # (c,1,2)
        t = np.clip(((p - A[None]) * ab[None]).sum(-1) / ab2[None], 0.0, 1.0)   # (c,m)
        c = A[None] + t[..., None] * ab[None]             # closest point on the segment
        D[s:s + step] = np.sqrt(((p - c) ** 2).sum(-1))
        T[s:s + step] = t
    return D, T


def viterbi_align(E: np.ndarray, dt: np.ndarray, after_gap: np.ndarray, cfg: dict = CONFIG) -> np.ndarray:
    """Monotonic Viterbi over route positions.

    States are ROUTE POSITIONS (so a loop that re-uses an arc_id is two different states).
    Transition (from route_pos k at the previous fix to j now) is only allowed for j >= k:
        j == k          : stay                 -> cost align_stay_cost
        j == k + d (d>=1): advance / skip d-1  -> align_advance_cost + skip_cost * (d-1)
    The allowed d grows with the elapsed time dt (a long gap can skip many arcs); after a long
    gap the per-arc skip cost is small because the vehicle's position is unknown.

    Args:
        E: (n, R) emission cost (negative log-likelihood up to a constant) per fix and route_pos.
        dt: (n,) seconds since previous fix (dt[0] unused). after_gap: (n,) bool.
    Returns:
        (n,) int route_pos chosen for every fix.
    """
    n, R = E.shape
    cost = E[0].astype(np.float64).copy()          # free start: any route_pos may begin the trip
    bp = np.zeros((n, R), np.uint16)               # best shift d back-pointer (d <= R-1 < 65536)
    for i in range(1, n):
        extra = max(dt[i] - 1.0, 0.0) * cfg["align_arcs_per_second"]
        smax = min(R - 1, cfg["align_base_skip"] + int(math.ceil(extra)))
        per_skip = cfg["align_gap_skip_cost"] if after_gap[i] else cfg["align_skip_cost"]
        new = cost + cfg["align_stay_cost"]        # d = 0 (stay)
        arg = np.zeros(R, np.uint16)
        for d in range(1, smax + 1):
            cand = cost[:R - d] + (cfg["align_advance_cost"] + per_skip * (d - 1))
            seg, aseg = new[d:], arg[d:]           # views: writing to them updates new / arg
            better = cand < seg
            seg[better] = cand[better]
            aseg[better] = d
        cost = new + E[i]
        bp[i] = arg
    pos = np.empty(n, np.int64)
    j = int(np.argmin(cost))
    pos[-1] = j
    for i in range(n - 1, 0, -1):
        j -= int(bp[i, j])
        pos[i - 1] = j
    return pos


def align_to_route(trip: Trip, route: np.ndarray, pts: pd.DataFrame, cfg: dict = CONFIG) -> pd.DataFrame:
    """Label each kept GPS fix with a route_pos / arc_id using route-constrained Viterbi.

    Args:
        trip: loaded trip. route: arc_id sequence (index = route_pos).
        pts: flag table of the fixes to label (must contain gps_row, dt_s, after_gap).
    Returns:
        DataFrame (one row per fix) with gps_row, route_pos, arc_id (local), label_distance_m,
        margin_all_m, margin_nonadj_m, nearest_route_pos, nearest_arc_id, label_quality.
    """
    rows = pts.gps_row.to_numpy()                               # gps_row
    P = trip.gps_xy[rows]
    A, B = trip.arc_a[route], trip.arc_b[route]                 # geometry of route[route_pos]
    D, _ = point_segment_distance(P, A, B, cfg["chunk_cells"])  # (n, R) metres
    sig, cap = cfg["align_sigma_m"], cfg["align_emission_cap_m"]
    E = (np.minimum(D, cap) ** 2 / (2.0 * sig ** 2)).astype(np.float32)   # Gaussian NLL, capped
    # dt between consecutive KEPT fixes (recomputed, so it stays right if training rows were dropped).
    dt = np.r_[np.nan, np.diff(trip.gps_t[rows])]
    gap = np.r_[False, dt[1:] > cfg["gap_seconds"]]
    pos = viterbi_align(E, dt, gap, cfg)

    n, R = D.shape
    idx = np.arange(n)
    label_d = D[idx, pos].astype(np.float64)

    # Ambiguity margins: distance of the NEXT-BEST route arc minus the label's distance.
    Dm = D.copy()
    Dm[idx, pos] = np.inf
    margin_all = Dm.min(axis=1) - label_d                       # includes the adjacent route arcs
    for off in (-1, 1):                                         # exclude route neighbours (share a node)
        ok = (pos + off >= 0) & (pos + off < R)
        Dm[idx[ok], pos[ok] + off] = np.inf
    margin_nonadj = Dm.min(axis=1) - label_d
    del Dm

    nearest_pos = D.argmin(axis=1)
    quality = np.where(label_d > cfg["far_label_m"], "far",
                       np.where(margin_nonadj < cfg["ambiguity_margin_m"], "ambiguous", "clean"))
    return pd.DataFrame({
        "gps_row": rows, "route_pos": pos, "arc_id": route[pos], "label_distance_m": label_d,
        "margin_all_m": margin_all, "margin_nonadj_m": margin_nonadj,
        "nearest_route_pos": nearest_pos, "nearest_arc_id": route[nearest_pos], "label_quality": quality,
    })


def save_config(cfg: dict, path: Path) -> None:
    """Write the config (JSON) next to the code so every number can be traced."""
    path.write_text(json.dumps(cfg, indent=2, default=float), encoding="utf-8")
