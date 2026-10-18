from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TriageRequest(BaseModel):
    patient_id: Optional[str] = None
    age: Optional[int] = 35
    sex: Optional[str] = "M"
    bp_sys: Optional[float] = None
    bp_dia: Optional[float] = None
    pulse: Optional[float] = None
    temp_c: Optional[float] = None
    spo2: Optional[float] = None
    complaint_text: Optional[str] = ""
    conditions: Optional[List[str]] = Field(default_factory=list)
    medications: Optional[List[str]] = Field(default_factory=list)


class RedFlagItem(BaseModel):
    id: str
    label: str
    matched_text: str


class RuleFiredItem(BaseModel):
    id: str
    name: str
    detail: str


class VitalStatusItem(BaseModel):
    value: Optional[float] = None
    status: str  # "normal" | "abnormal" | "critical" | "invalid" | "missing"


class ViewsItem(BaseModel):
    vitals_level: Optional[int] = None
    text_level: Optional[int] = None


class PipelineStageItem(BaseModel):
    stage_number: int
    title: str
    status: str  # "passed" | "fired" | "not_reached" | "decided"
    result: str
    detail: Optional[str] = None


class TriageResponse(BaseModel):
    triage_level: Optional[int]
    level_label: str
    primary_department: Optional[str]
    suggested_specialty: Optional[str]
    source_layer: str  # "rule" | "screen" | "fusion" | "abstain"
    abstain_reason: Optional[str]  # null | "needs_remeasure" | "model_disagreement" | "high_conformal_uncertainty"
    conformal_set: List[int]
    p_emergency: float
    tau: float
    red_flags: List[RedFlagItem]
    rules_fired: List[RuleFiredItem]
    vitals: Dict[str, VitalStatusItem]
    views: ViewsItem
    latency_ms: float
    pipeline_stages: Optional[List[PipelineStageItem]] = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str
