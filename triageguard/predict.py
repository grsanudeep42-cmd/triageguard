"""
triageguard/predict.py
─────────────────────────────────────────────────────────────────────────────
Phase 7 – Master inference pipeline and policy engine for TriageGuard.

POLICY HIERARCHY (Strict Order)
-------------------------------
1. Vitals rule forces a level -> output it (source="rule").
2. Red flag present -> soft guard: never output L4/L5 unless vitals are
   clearly normal (floor at L3).
3. Emergency screen p_em >= tau -> output emergency level (source="screen"),
   choosing L1 vs L2 by minimum expected cost over {1, 2}.
4. Abstain check:
   - needs_remeasure is True -> ABSTAIN (reason="needs_remeasure")
   - model disagreement >= 2 levels -> ABSTAIN (reason="model_disagreement")
   - conformal set size >= 3 levels -> ABSTAIN (reason="high_conformal_uncertainty")
5. Otherwise:
   - select minimum expected cost level within the conformal set C(x)
     (subject to red-flag floor at L3 if vitals are not clearly normal).
6. Department routing:
   - primary_department = "Emergency/Casualty" if level in {1, 2}, else suggested_specialty.
   - suggested_specialty is always provided.
7. Performance: latency tracked per patient excluding initial model loading.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from triageguard.cleaning import CleanResult, clean_record
from triageguard.conformal import ConformalPredictor
from triageguard.data_adapter import (
    adapt_schema,
    extract_red_flag_count,
    extract_vitals_features,
    load_config,
)
from triageguard.fusion import StackerFusion, build_cost_matrix, select_min_cost_level
from triageguard.model_dept import DepartmentModel
from triageguard.model_text import TextModel
from triageguard.model_vitals import VitalsModel
from triageguard.rules import apply_rules

_ROOT = Path(__file__).resolve().parent.parent
_MODEL_DIR = _ROOT / "models"


@dataclass
class TriagePrediction:
    patient_id: str
    triage_level: Optional[int]
    primary_department: str
    suggested_specialty: str
    red_flags: List[str]
    source_layer: str  # "rule", "screen", "fusion", "abstain"
    conformal_set: List[int]
    abstain_reason: Optional[str]
    warnings: List[str]
    latency_ms: float
    p_emergency: float


class TriagePipeline:
    """
    End-to-end TriageGuard inference pipeline.
    Loads models once, executes policy hierarchy with sub-millisecond per-patient latency.
    """

    def __init__(
        self,
        model_vitals: VitalsModel,
        model_text: TextModel,
        model_dept: DepartmentModel,
        fusion: StackerFusion,
        conformal: ConformalPredictor,
        cfg: Optional[Dict] = None,
    ):
        self.cfg = cfg or load_config()
        self.model_vitals = model_vitals
        self.model_text = model_text
        self.model_dept = model_dept
        self.fusion = fusion
        self.conformal = conformal
        self.cost_matrix = build_cost_matrix(self.cfg)

    @classmethod
    def load(cls, model_dir: Optional[Union[str, Path]] = None, cfg: Optional[Dict] = None) -> "TriagePipeline":
        md = Path(model_dir) if model_dir else _MODEL_DIR
        vitals = VitalsModel.load(md / "model_vitals.joblib")
        text = TextModel.load(md / "model_text.joblib")
        dept = DepartmentModel.load(md / "model_dept.joblib")
        fusion = StackerFusion.load(md / "fusion.joblib")
        conformal = ConformalPredictor.load(md / "conformal.joblib")
        return cls(vitals, text, dept, fusion, conformal, cfg)

    def _are_vitals_clearly_normal(self, cr: CleanResult) -> bool:
        """
        True only if all 5 vitals are present, valid (FLAG_OK), and strictly within
        benign ambulatory normal ranges.
        """
        if any([
            cr.bp_sys is None, cr.bp_dia is None, cr.pulse is None,
            cr.temp_c is None, cr.spo2 is None,
        ]):
            return False

        if any([
            cr.bp_sys_flag != "ok", cr.bp_dia_flag != "ok", cr.pulse_flag != "ok",
            cr.temp_c_flag != "ok", cr.spo2_flag != "ok",
        ]):
            return False

        return (
            100.0 <= cr.bp_sys <= 135.0 and
            60.0 <= cr.bp_dia <= 88.0 and
            60.0 <= cr.pulse <= 95.0 and
            36.2 <= cr.temp_c <= 37.4 and
            96.0 <= cr.spo2 <= 100.0
        )

    def predict_one(self, raw_record: Dict) -> TriagePrediction:
        """
        Run the complete triage policy on one raw intake record.
        Measures execution latency excluding model loading.
        """
        t0 = time.perf_counter()
        patient_id = str(raw_record.get("patient_id", "UNKNOWN"))
        age = int(raw_record.get("age", 35))

        # ── 1. Phase 1 Input Cleaning ─────────────────────────────────────────
        cr = clean_record(raw_record, self.cfg)
        warnings = list(cr.warnings)

        # ── 2. Phase 3 Rules Layer (FORCED vitals rules + SOFT red flags) ──────
        rr = apply_rules(cr, age=age)
        red_flags = list(rr.red_flags_detected)

        # ── 3. Base Models Evaluation ─────────────────────────────────────────
        # Prepare single-row DataFrame for models
        single_df = pd.DataFrame([{
            "bp_sys_c": cr.bp_sys,
            "bp_dia_c": cr.bp_dia,
            "pulse_c": cr.pulse,
            "temp_c_c": cr.temp_c,
            "spo2_c": cr.spo2,
            "bp_sys_flag": cr.bp_sys_flag,
            "bp_dia_flag": cr.bp_dia_flag,
            "pulse_flag": cr.pulse_flag,
            "temp_c_flag": cr.temp_c_flag,
            "spo2_flag": cr.spo2_flag,
            "age": age,
            "sex": raw_record.get("sex", "M"),
            "n_invalid_vitals": cr.n_invalid_vitals,
            "n_missing_vitals": cr.n_missing_vitals,
            "data_quality_score": cr.data_quality_score,
            "complaint_clean": cr.complaint_text,
            "n_red_flags": len(red_flags),
        }])

        X_vit = extract_vitals_features(single_df)
        p_vit_multi = self.model_vitals.predict_proba(X_vit)
        p_vit_em = self.model_vitals.predict_emergency_proba(X_vit)

        texts = [cr.complaint_text]
        p_txt_multi = self.model_text.predict_proba(texts)
        p_txt_em = self.model_text.predict_emergency_proba(texts)

        # ── 4. Phase 5 Fusion & Calibration ───────────────────────────────────
        fused_multi, calib_p_em = self.fusion.predict_fused(
            p_vit_multi, p_txt_multi, p_vit_em, p_txt_em, single_df
        )
        fused_probs = fused_multi[0]
        p_em = float(calib_p_em[0])

        # Conformal Level Set
        conformal_set = self.conformal.predict_level_set(fused_probs)

        # Model top-1 levels
        vit_level = int(np.argmax(p_vit_multi[0]) + 1)
        txt_level = int(np.argmax(p_txt_multi[0]) + 1)
        model_disagree = abs(vit_level - txt_level) >= 2

        # Department routing
        specialties = self.model_dept.predict_specialty(single_df)
        suggested_specialty = specialties[0] if specialties else "General Medicine"

        # ── 5. Hierarchical Policy Execution ─────────────────────────────────
        final_level: Optional[int] = None
        source_layer: str = "fusion"
        abstain_reason: Optional[str] = None

        # Policy Step 1: Vitals Rule forces a level
        if rr.forced_level is not None:
            final_level = int(rr.forced_level)
            source_layer = "rule"

        # Policy Step 3: Emergency Screen fires (p_em >= tau)
        elif p_em >= self.conformal.state.operating_tau:
            # Choose L1 vs L2 by min expected cost
            final_level = select_min_cost_level(
                fused_probs, allowed_levels=[1, 2], cost_matrix=self.cost_matrix
            )
            source_layer = "screen"

        # Policy Step 4: Abstain Checks
        elif cr.needs_remeasure:
            final_level = None
            source_layer = "abstain"
            abstain_reason = "needs_remeasure"
        elif model_disagree:
            final_level = None
            source_layer = "abstain"
            abstain_reason = "model_disagreement"
        elif len(conformal_set) >= 3:
            final_level = None
            source_layer = "abstain"
            abstain_reason = "high_conformal_uncertainty"

        # Policy Step 5: Min-expected-cost level within conformal set
        else:
            candidate_levels = list(conformal_set)
            # Policy Step 2 check: Soft Red Flag Floor at L3
            # If red flag is present and vitals are not clearly normal, floor at L3
            if rr.no_downgrade_below_L3 and not self._are_vitals_clearly_normal(cr):
                filtered = [lvl for lvl in candidate_levels if lvl <= 3]
                candidate_levels = filtered if filtered else [3]

            final_level = select_min_cost_level(
                fused_probs, allowed_levels=candidate_levels, cost_matrix=self.cost_matrix
            )
            source_layer = "fusion"

        # Primary department assignment
        if final_level is not None and final_level in (1, 2):
            primary_dept = "Emergency/Casualty"
        else:
            primary_dept = suggested_specialty

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        return TriagePrediction(
            patient_id=patient_id,
            triage_level=final_level,
            primary_department=primary_dept,
            suggested_specialty=suggested_specialty,
            red_flags=red_flags,
            source_layer=source_layer,
            conformal_set=conformal_set,
            abstain_reason=abstain_reason,
            warnings=warnings,
            latency_ms=round(latency_ms, 2),
            p_emergency=round(p_em, 4),
        )

    def predict_batch(self, df: pd.DataFrame) -> List[TriagePrediction]:
        """Process a batch of records, returning a list of TriagePrediction objects."""
        return [self.predict_one(row.to_dict()) for _, row in df.iterrows()]


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="TriageGuard clinical inference CLI")
    parser.add_argument("--input", type=str, help="Path to input JSON or CSV file")
    parser.add_argument("--model_dir", type=str, default=str(_MODEL_DIR), help="Directory containing trained models")
    args = parser.parse_args()

    pipeline = TriagePipeline.load(args.model_dir)

    if args.input:
        in_path = Path(args.input)
        if in_path.suffix.lower() == ".json":
            with open(in_path) as f:
                data = json.load(f)
            records = data if isinstance(data, list) else [data]
            preds = [pipeline.predict_one(r) for r in records]
            print(json.dumps([asdict(p) for p in preds], indent=2))
        else:
            df = pd.read_csv(in_path)
            preds = pipeline.predict_batch(df)
            out_df = pd.DataFrame([asdict(p) for p in preds])
            print(out_df.to_string())
    else:
        print("TriageGuard interactive mode. Provide a record dictionary.")


if __name__ == "__main__":
    main()
