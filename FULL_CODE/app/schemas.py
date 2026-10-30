from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class TriageRequest(BaseModel):
    # ── Internal / Phase-1 field names ───────────────────────────────────────
    patient_id: Optional[str] = None
    age: Optional[int] = None          # internal name; also accepts age_years
    sex: Optional[str] = "M"
    bp_sys: Optional[float] = None     # internal name; also accepts bp_systolic
    bp_dia: Optional[float] = None     # internal name; also accepts bp_diastolic
    pulse: Optional[float] = None      # internal name; also accepts pulse_bpm
    temp_c: Optional[float] = None
    spo2: Optional[float] = None       # internal name; also accepts spo2_pct
    complaint_text: Optional[str] = "" # internal name; also accepts chief_complaint
    conditions: Optional[List[str]] = Field(default_factory=list)
    medications: Optional[List[str]] = Field(default_factory=list)

    # ── Phase-2 / hackathon alias fields (accepted but not stored directly) ──
    visit_id: Optional[str] = None     # alias for patient_id
    age_years: Optional[int] = None    # alias for age
    bp_systolic: Optional[float] = None   # alias for bp_sys
    bp_diastolic: Optional[float] = None  # alias for bp_dia
    pulse_bpm: Optional[float] = None     # alias for pulse
    spo2_pct: Optional[float] = None      # alias for spo2
    chief_complaint: Optional[str] = None # alias for complaint_text
    pregnant: Optional[str] = None        # passed through to raw_record

    @model_validator(mode="after")
    def _resolve_phase2_aliases(self) -> "TriageRequest":
        """Resolve Phase-2 column name aliases into internal field names."""
        if self.visit_id and not self.patient_id:
            self.patient_id = self.visit_id
        if self.age_years is not None and self.age is None:
            self.age = self.age_years
        if self.age is None:
            self.age = 35
        if self.bp_systolic is not None and self.bp_sys is None:
            self.bp_sys = self.bp_systolic
        if self.bp_diastolic is not None and self.bp_dia is None:
            self.bp_dia = self.bp_diastolic
        if self.pulse_bpm is not None and self.pulse is None:
            self.pulse = self.pulse_bpm
        if self.spo2_pct is not None and self.spo2 is None:
            self.spo2 = self.spo2_pct
        if self.chief_complaint is not None and not self.complaint_text:
            self.complaint_text = self.chief_complaint
        return self


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
