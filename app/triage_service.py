import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from triageguard.cleaning import (
    FLAG_DECIMAL_SLIP,
    FLAG_MISSING,
    FLAG_OK,
    FLAG_OUT_OF_RANGE,
    CleanResult,
    clean_record,
)
from triageguard.data_adapter import extract_vitals_features
from triageguard.predict import TriagePipeline, TriagePrediction
from triageguard.rules import _fuzzy_phrase_match, _load_redflags_cfg, _load_rules_cfg, apply_rules
from app.schemas import (
    PipelineStageItem,
    RedFlagItem,
    RuleFiredItem,
    TriageRequest,
    TriageResponse,
    ViewsItem,
    VitalStatusItem,
)

LEVEL_LABELS = {
    1: "Resuscitation",
    2: "Emergent",
    3: "Urgent",
    4: "Less urgent",
    5: "Non-urgent",
}


class TriageService:
    def __init__(self, pipeline: TriagePipeline):
        self.pipeline = pipeline
        self.rules_cfg = _load_rules_cfg()
        self.redflags_cfg = _load_redflags_cfg()
        self.rules_by_id = {r["id"]: r for r in self.rules_cfg.get("rules", [])}
        self.redflags_by_id = {rf["id"]: rf for rf in self.redflags_cfg.get("red_flags", [])}
        self.redflags_by_name = {rf["name"]: rf for rf in self.redflags_cfg.get("red_flags", [])}

    def _extract_matched_red_flags(self, complaint_text: str, detected_names: List[str]) -> List[RedFlagItem]:
        if not complaint_text or not detected_names:
            return []
        
        text_words = re.findall(r"\b[a-z0-9]+\b", complaint_text.lower())
        items: List[RedFlagItem] = []

        for name in detected_names:
            rf = self.redflags_by_name.get(name)
            if not rf:
                items.append(RedFlagItem(id=name, label=name, matched_text=name))
                continue
            
            rf_id = rf["id"]
            patterns = []
            for lang_key in ("patterns_en", "patterns_tel", "patterns_hin"):
                patterns.extend(rf.get(lang_key, []))

            matched_phrase = ""
            for p in patterns:
                p_toks = p.lower().split()
                k = len(p_toks)
                if k == 0 or len(text_words) < k:
                    continue
                found = False
                for i in range(len(text_words) - k + 1):
                    sub = text_words[i : i + k]
                    if _fuzzy_phrase_match(p, sub):
                        matched_phrase = " ".join(sub)
                        found = True
                        break
                if found:
                    break

            # If not found via exact sub-tokens, fall back to the first matching pattern or name
            if not matched_phrase:
                matched_phrase = name.replace("_", " ")

            items.append(RedFlagItem(id=rf_id, label=name, matched_text=matched_phrase))

        return items

    def _extract_rules_fired(self, triggered_ids: List[str], cr: CleanResult) -> List[RuleFiredItem]:
        items: List[RuleFiredItem] = []
        for rid in triggered_ids:
            rule = self.rules_by_id.get(rid)
            if not rule:
                items.append(RuleFiredItem(id=rid, name=rid, detail=f"Rule {rid} triggered"))
                continue

            desc = rule.get("description", "").strip()
            # Clean name from description (take first clause before "—" or ":")
            name = desc.split("—")[0].split(":")[0].strip() or rule.get("name", rid)
            
            # Format custom condition detail
            conds = rule.get("conditions", {})
            cond_strs = []
            for k, val in conds.items():
                if k == "spo2_lt":
                    actual = f"{cr.spo2:.1f}%" if cr.spo2 is not None else "missing"
                    cond_strs.append(f"SpO₂ {actual} is below {val}%")
                elif k == "bp_sys_lt":
                    actual = f"{cr.bp_sys:.0f}" if cr.bp_sys is not None else "missing"
                    cond_strs.append(f"Systolic BP {actual} is below {val} mmHg")
                elif k == "pulse_gt":
                    actual = f"{cr.pulse:.0f}" if cr.pulse is not None else "missing"
                    cond_strs.append(f"Pulse {actual} is above {val} bpm")
                elif k == "pulse_lt":
                    actual = f"{cr.pulse:.0f}" if cr.pulse is not None else "missing"
                    cond_strs.append(f"Pulse {actual} is below {val} bpm")
                elif k == "temp_c_ge":
                    actual = f"{cr.temp_c:.1f}°C" if cr.temp_c is not None else "missing"
                    cond_strs.append(f"Temp {actual} is ≥ {val}°C")
                elif k == "temp_c_lt":
                    actual = f"{cr.temp_c:.1f}°C" if cr.temp_c is not None else "missing"
                    cond_strs.append(f"Temp {actual} is < {val}°C")
                else:
                    cond_strs.append(f"{k}: {val}")

            detail = "; ".join(cond_strs) if cond_strs else desc
            items.append(RuleFiredItem(id=rid, name=name, detail=detail))
        return items

    def _classify_vitals(
        self,
        raw_record: Dict[str, Any],
        cr: CleanResult,
        triggered_ids: List[str],
    ) -> Dict[str, VitalStatusItem]:
        vitals: Dict[str, VitalStatusItem] = {}
        flags = {
            "bp_sys": cr.bp_sys_flag,
            "bp_dia": cr.bp_dia_flag,
            "pulse": cr.pulse_flag,
            "temp_c": cr.temp_c_flag,
            "spo2": cr.spo2_flag,
        }
        cleaned_vals = {
            "bp_sys": cr.bp_sys,
            "bp_dia": cr.bp_dia,
            "pulse": cr.pulse,
            "temp_c": cr.temp_c,
            "spo2": cr.spo2,
        }

        # Check which vitals caused critical rules to fire
        critical_vitals = set()
        for rid in triggered_ids:
            rule = self.rules_by_id.get(rid, {})
            if int(rule.get("forced_max_level", 2)) == 1:
                for cond_key in rule.get("conditions", {}).keys():
                    v_name = cond_key.rsplit("_", 1)[0]
                    if v_name in flags:
                        critical_vitals.add(v_name)

        for v_name, flag in flags.items():
            raw_val = raw_record.get(v_name)
            val = cleaned_vals[v_name]

            if flag == FLAG_MISSING or raw_val is None or (isinstance(raw_val, float) and math.isnan(raw_val)):
                vitals[v_name] = VitalStatusItem(value=None, status="missing")
            elif flag in (FLAG_OUT_OF_RANGE, FLAG_DECIMAL_SLIP):
                vitals[v_name] = VitalStatusItem(value=raw_val, status="invalid")
            elif v_name in critical_vitals or (v_name == "spo2" and val is not None and val < 85.0):
                vitals[v_name] = VitalStatusItem(value=val, status="critical")
            else:
                # Normal range check (adult ambulatory benchmark)
                is_normal = False
                if val is not None:
                    if v_name == "bp_sys":
                        is_normal = 100.0 <= val <= 135.0
                    elif v_name == "bp_dia":
                        is_normal = 60.0 <= val <= 88.0
                    elif v_name == "pulse":
                        is_normal = 60.0 <= val <= 95.0
                    elif v_name == "temp_c":
                        is_normal = 36.2 <= val <= 37.4
                    elif v_name == "spo2":
                        is_normal = 96.0 <= val <= 100.0
                
                vitals[v_name] = VitalStatusItem(
                    value=val,
                    status="normal" if is_normal else "abnormal",
                )

        return vitals

    def triage(self, req: TriageRequest) -> TriageResponse:
        raw_record = {
            "patient_id": req.patient_id or "PT-DEMO",
            "age": req.age if req.age is not None else 35,
            "sex": req.sex or "M",
            "bp_sys": req.bp_sys,
            "bp_dia": req.bp_dia,
            "pulse": req.pulse,
            "temp_c": req.temp_c,
            "spo2": req.spo2,
            "complaint_text": req.complaint_text or "",
            "known_conditions": "; ".join(req.conditions) if req.conditions else "none",
            "medications": "; ".join(req.medications) if req.medications else "none",
        }

        # 1. Clean input
        cr = clean_record(raw_record, self.pipeline.cfg)
        age = int(raw_record["age"])

        # 2. Evaluate rules
        rr = apply_rules(cr, age=age)

        # 3. Two models top-1 levels
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
            "n_red_flags": len(rr.red_flags_detected),
        }])

        X_vit = extract_vitals_features(single_df)
        p_vit_multi = self.pipeline.model_vitals.predict_proba(X_vit)
        p_txt_multi = self.pipeline.model_text.predict_proba([cr.complaint_text])

        vit_level = int(np.argmax(p_vit_multi[0]) + 1)
        txt_level = int(np.argmax(p_txt_multi[0]) + 1)

        # 4. Official pipeline prediction
        pred: TriagePrediction = self.pipeline.predict_one(raw_record)

        # 5. Extract red flags with matched text
        red_flags_items = self._extract_matched_red_flags(req.complaint_text or "", rr.red_flags_detected)

        # 6. Extract rules fired
        rules_fired_items = self._extract_rules_fired(rr.triggered_rule_ids, cr)

        # 7. Classify vitals
        vitals_status = self._classify_vitals(raw_record, cr, rr.triggered_rule_ids)

        # 8. Tau & Emergency threshold
        tau = float(self.pipeline.conformal.state.operating_tau)

        # 9. Pipeline stages for orchestrated UI
        invalid_or_missing = cr.n_invalid_vitals + cr.n_missing_vitals
        s1_status = "fired" if cr.needs_remeasure else "passed"
        s1_result = f"{invalid_or_missing} of 5 vitals invalid or missing" if invalid_or_missing > 0 else "All 5 vitals valid"

        s2_status = "fired" if rr.triggered_rule_ids else "passed"
        if rules_fired_items:
            s2_result = f"{rules_fired_items[0].id} fired: {rules_fired_items[0].detail}"
        else:
            s2_result = "No safety rules fired"

        disagree = abs(vit_level - txt_level) >= 2
        s3_status = "fired" if disagree else "passed"
        if disagree:
            if pred.source_layer == "rule":
                s3_result = f"Vitals view L{vit_level}, text view L{txt_level}, disagree (Overridden by the safety rule)"
            elif pred.source_layer == "screen":
                s3_result = f"Vitals view L{vit_level}, text view L{txt_level}, disagree (Overridden by the emergency screen)"
            else:
                s3_result = f"Vitals view L{vit_level}, text view L{txt_level}, disagree"
        else:
            s3_result = f"Vitals view L{vit_level}, text view L{txt_level}, agree"

        screen_fired = pred.p_emergency >= tau
        s4_status = "fired" if screen_fired else "passed"
        s4_result = f"Emergency probability {pred.p_emergency:.3f}, threshold {tau:.3f}"

        # Assign "decided" status to the deciding stage
        deciding_stage = 3
        if pred.source_layer == "rule":
            deciding_stage = 2
            s2_status = "decided"
        elif pred.source_layer == "screen":
            deciding_stage = 4
            s4_status = "decided"
        elif pred.source_layer == "abstain":
            if pred.abstain_reason == "needs_remeasure":
                deciding_stage = 1
                s1_status = "decided"
            elif pred.abstain_reason == "model_disagreement":
                deciding_stage = 3
                s3_status = "decided"
            else:
                deciding_stage = 4
                s4_status = "decided"
        elif pred.source_layer == "fusion":
            deciding_stage = 3
            s3_status = "decided"

        stages = [
            PipelineStageItem(stage_number=1, title="Check data", status=s1_status, result=s1_result),
            PipelineStageItem(stage_number=2, title="Safety rules", status=s2_status, result=s2_result),
            PipelineStageItem(stage_number=3, title="Two models", status=s3_status, result=s3_result),
            PipelineStageItem(stage_number=4, title="Emergency screen", status=s4_status, result=s4_result),
        ]

        level_label = LEVEL_LABELS.get(pred.triage_level, "Nurse now") if pred.triage_level is not None else "Nurse now"

        return TriageResponse(
            triage_level=pred.triage_level,
            level_label=level_label,
            primary_department=pred.primary_department,
            suggested_specialty=pred.suggested_specialty,
            source_layer=pred.source_layer,
            abstain_reason=pred.abstain_reason,
            conformal_set=pred.conformal_set,
            p_emergency=pred.p_emergency,
            tau=round(tau, 4),
            red_flags=red_flags_items,
            rules_fired=rules_fired_items,
            vitals=vitals_status,
            views=ViewsItem(vitals_level=vit_level, text_level=txt_level),
            latency_ms=pred.latency_ms,
            pipeline_stages=stages,
        )
