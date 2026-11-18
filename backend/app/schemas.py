"""Pydantic schemas for the PathSnap API.

Never defines or returns any reference route, reference labels, or per-point correctness flags.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# --- Health & Trip Catalog ---

class HealthResponse(BaseModel):
    status: str = "ok"
    model_name: str
    model_version: str
    hmm_available: bool
    n_trips: int
    uptime_s: float


class TripInfo(BaseModel):
    trip_id: str
    n_gps_points: int
    n_nodes: int
    n_arcs: int


# --- Prediction Requests & Jobs ---

class PredictRequest(BaseModel):
    trip_id: str
    refresh: bool = False


class PredictResponse(BaseModel):
    job_id: str


class SafeErrorBody(BaseModel):
    code: str
    message: str
    stage: Optional[str] = None
    trip_id: Optional[str] = None
    request_id: Optional[str] = None
    hint: Optional[str] = ""


class JobStatusResponse(BaseModel):
    status: str = Field(..., description="queued | running | done | failed")
    stage: str
    stage_index: int
    n_stages: int
    elapsed_s: float
    result: Optional[Dict[str, Any]] = None
    error: Optional[SafeErrorBody] = None


# --- Result Schema (Section 3.7) ---

class ModelMeta(BaseModel):
    name: str
    version: str


class RouteGeometry(BaseModel):
    arc_ids: List[int]
    n_arcs: int
    segments: List[List[List[float]]]  # [[[lat, lon], ...], ...]


class RoutesOutput(BaseModel):
    classifier: RouteGeometry
    hmm: Optional[RouteGeometry] = None


class AggregateMetrics(BaseModel):
    top1_all: float = Field(..., description="Top-1 accuracy vs reference route across all points")
    top1_in_cands: float = Field(..., description="Top-1 accuracy vs reference route for points with label in candidates")
    top1_clean: float = Field(..., description="Top-1 accuracy vs reference route on clean-labelled points")
    candidate_miss_rate: float = Field(..., description="Share of points where reference arc was not in candidate set")
    route_continuity: float = Field(..., description="Share of consecutive predicted arcs connected in road network")
    sequence_arc_precision: float = Field(..., description="Arc-level precision vs reference route")
    sequence_arc_recall: float = Field(..., description="Arc-level recall vs reference route")
    sequence_arc_f1: float = Field(..., description="Arc-level F1 score vs reference route")
    sequence_length_weighted_mismatch: float = Field(..., description="Length-weighted route mismatch fraction vs reference route")


class DeltaMetrics(BaseModel):
    top1_all: float
    top1_in_cands: float
    top1_clean: float
    candidate_miss_rate: float
    route_continuity: float
    sequence_arc_precision: float
    sequence_arc_recall: float
    sequence_arc_f1: float
    sequence_length_weighted_mismatch: float


class MetricsBlock(BaseModel):
    classifier: AggregateMetrics
    hmm: Optional[AggregateMetrics] = None
    delta: Optional[DeltaMetrics] = None


class ErrorSummaryCounts(BaseModel):
    classifier: int
    hmm: Optional[int] = None


class HMMFallbacks(BaseModel):
    unreachable_candidates: int = 0
    long_gaps: int = 0


class LabelQualityShare(BaseModel):
    clean: float
    ambiguous: float
    far: float


class ErrorSummary(BaseModel):
    n_points: int
    n_correct: ErrorSummaryCounts
    n_incorrect: ErrorSummaryCounts
    broken_transitions: ErrorSummaryCounts
    hmm_fallbacks: HMMFallbacks
    n_points_hmm_changed: int
    label_quality_share: LabelQualityShare


class PerPointComparison(BaseModel):
    hmm_changed_indices: List[int]
    after_gap_indices: List[int]
    stationary_indices: List[int]
    classifier_arc_ids: List[int]
    hmm_arc_ids: Optional[List[int]] = None


class DataQualityContext(BaseModel):
    n_candidates_mean: float
    n_candidates_max: int
    share_after_long_gap: float
    share_stationary: float


class TripResult(BaseModel):
    trip_id: str
    model: ModelMeta
    hmm_used: bool
    from_cache: bool
    created_at: str
    total_runtime_s: float
    timings_s: Dict[str, float]
    routes: RoutesOutput
    gps: List[List[float]]
    display_downsample_step: int
    metrics: MetricsBlock
    error_summary: ErrorSummary
    per_point_comparison: PerPointComparison
    data_quality_context: DataQualityContext
    warnings: List[str]
