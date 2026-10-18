import glob
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from fastapi import APIRouter, HTTPException, Request

from app.schemas import HealthResponse, TriageRequest, TriageResponse
from app.triage_service import LEVEL_LABELS, TriageService
from triageguard.predict import TriagePipeline

router = APIRouter(prefix="/api", tags=["api"])

# Module-level cached seed queue
_cached_seed_queue: Optional[List[Dict[str, Any]]] = None


def get_triage_service(request: Request) -> TriageService:
    service: Optional[TriageService] = getattr(request.app.state, "triage_service", None)
    if not service:
        raise HTTPException(status_code=503, detail="Triage service not initialized")
    return service


@router.get("/health", response_model=HealthResponse)
def health_check(request: Request) -> HealthResponse:
    service: Optional[TriageService] = getattr(request.app.state, "triage_service", None)
    models_loaded = service is not None and service.pipeline is not None
    return HealthResponse(
        status="ok",
        models_loaded=models_loaded,
        version="1.0.0",
    )


@router.post("/triage", response_model=TriageResponse)
def triage_patient(req: TriageRequest, request: Request) -> TriageResponse:
    service = get_triage_service(request)
    return service.triage(req)


@router.get("/demo-cases")
def get_demo_cases() -> List[Dict[str, Any]]:
    demo_path = Path("results/demo_cases.json")
    if not demo_path.exists():
        return []

    try:
        with open(demo_path, "r", encoding="utf-8") as f:
            raw_cases = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load demo cases: {e}")

    cases = []
    # Generic iteration over all keys
    for k, v in raw_cases.items():
        vitals = v.get("cleaned_vitals") or {}
        # Clean NaN values in vitals
        clean_vitals = {}
        for vk, vv in vitals.items():
            if vv is not None and isinstance(vv, float) and math.isnan(vv):
                clean_vitals[vk] = None
            else:
                clean_vitals[vk] = vv

        complaint = v.get("combined_gamed_complaint") or v.get("complaint_text") or ""

        cases.append({
            "id": k,
            "scenario": v.get("scenario", k),
            "description": v.get("description", ""),
            "selection_method": v.get("selection_method"),
            "original_complaint_text": v.get("original_complaint_text"),
            "injected_dramatic_phrase": v.get("injected_dramatic_phrase"),
            "combined_gamed_complaint": v.get("combined_gamed_complaint"),
            "input": {
                "patient_id": v.get("patient_id"),
                "age": v.get("age"),
                "sex": v.get("sex"),
                "bp_sys": clean_vitals.get("bp_sys"),
                "bp_dia": clean_vitals.get("bp_dia"),
                "pulse": clean_vitals.get("pulse"),
                "temp_c": clean_vitals.get("temp_c"),
                "spo2": clean_vitals.get("spo2"),
                "complaint_text": complaint,
                "conditions": [],
                "medications": [],
            },
            "expected_prediction": v.get("triageguard_prediction"),
            "ground_truth": {
                "true_level": v.get("true_clean_level"),
                "true_department": v.get("true_department"),
            },
        })

    return cases


@router.get("/rules")
def get_rules() -> Dict[str, Any]:
    rules_path = Path("rules.yaml")
    redflags_path = Path("redflags.yaml")

    rules_data = []
    if rules_path.exists():
        with open(rules_path, "r", encoding="utf-8") as f:
            r_cfg = yaml.safe_load(f) or {}
            rules_list = r_cfg.get("rules", [])
            for r in rules_list:
                conds = r.get("conditions", {})
                cond_strs = []
                thresh_str = ""
                for ck, cv in conds.items():
                    if ck == "spo2_lt":
                        cond_strs.append(f"SpO₂ < {cv}%")
                        thresh_str = f"< {cv}%"
                    elif ck == "bp_sys_lt":
                        cond_strs.append(f"Systolic BP < {cv} mmHg")
                        thresh_str = f"< {cv} mmHg"
                    elif ck == "pulse_gt":
                        cond_strs.append(f"Pulse > {cv} bpm")
                        thresh_str = f"> {cv} bpm"
                    elif ck == "pulse_lt":
                        cond_strs.append(f"Pulse < {cv} bpm")
                        thresh_str = f"< {cv} bpm"
                    elif ck == "temp_c_ge":
                        cond_strs.append(f"Temp ≥ {cv}°C")
                        thresh_str = f"≥ {cv}°C"
                    elif ck == "temp_c_lt":
                        cond_strs.append(f"Temp < {cv}°C")
                        thresh_str = f"< {cv}°C"
                    else:
                        cond_strs.append(f"{ck}: {cv}")
                        thresh_str = str(cv)

                trigger_text = "; ".join(cond_strs) if cond_strs else r.get("description", "")
                rules_data.append({
                    "id": r["id"],
                    "description": r.get("description", "").strip(),
                    "plain_language_trigger": trigger_text,
                    "threshold": thresh_str,
                    "threshold_label": "Placeholder pending the official primer",
                    "forced_level": int(r.get("forced_max_level", 1)),
                    "applies_age_band": r.get("applies_age_band", "all"),
                    "source": r.get("source", "PLACEHOLDER pending official triage primer"),
                })

    redflags_data = []
    if redflags_path.exists():
        with open(redflags_path, "r", encoding="utf-8") as f:
            rf_cfg = yaml.safe_load(f) or {}
            rf_list = rf_cfg.get("red_flags", [])
            for rf in rf_list:
                all_patterns = []
                for lang in ("patterns_en", "patterns_tel", "patterns_hin"):
                    all_patterns.extend(rf.get(lang, []))
                
                # Check validation status if available from JSON results
                redflags_data.append({
                    "id": rf["id"],
                    "name": rf["name"],
                    "patterns_en": rf.get("patterns_en", []),
                    "patterns_tel": rf.get("patterns_tel", []),
                    "patterns_hin": rf.get("patterns_hin", []),
                    "all_patterns": all_patterns,
                    "escalate_to_max_level": rf.get("escalate_to_max_level", 2),
                    "status": "Validated" if rf["id"] in ("RF001", "RF002", "RF008") else "Not validated",
                })

    # Read aggregate firing stats from results JSON if present
    firing_stats = {}
    layer_decomp_path = Path("results/layer_escalation_decomposition.json")
    if layer_decomp_path.exists():
        try:
            with open(layer_decomp_path, "r", encoding="utf-8") as f:
                firing_stats["layer_escalation_decomposition"] = json.load(f)
        except Exception:
            pass

    return {
        "rules": rules_data,
        "red_flags": redflags_data,
        "firing_stats": firing_stats,
    }


@router.get("/results")
def get_results() -> Dict[str, Any]:
    results_dir = Path("results")
    if not results_dir.exists():
        return {}

    aggregated: Dict[str, Any] = {}
    for fpath in results_dir.glob("*.json"):
        key = fpath.stem
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                aggregated[key] = json.load(f)
        except Exception:
            pass

    return aggregated


@router.get("/seed-queue")
def get_seed_queue(request: Request) -> List[Dict[str, Any]]:
    global _cached_seed_queue
    if _cached_seed_queue is not None:
        return _cached_seed_queue

    service = get_triage_service(request)
    splits_path = Path("data/splits.csv")
    intake_path = Path("data/synthetic_intake.csv")

    if not splits_path.exists() or not intake_path.exists():
        return []

    try:
        splits = pd.read_csv(splits_path)
        intake = pd.read_csv(intake_path)
        val_ids = set(splits[splits["split"] == "val"]["patient_id"])
        val_df = intake[intake["patient_id"].isin(val_ids)]

        sampled = val_df.sample(n=300, random_state=42)
        records = []
        for _, row in sampled.iterrows():
            r = row.to_dict()
            clean_r = {}
            for k, val in r.items():
                if isinstance(val, float) and math.isnan(val):
                    clean_r[k] = None
                else:
                    clean_r[k] = val
            p = service.pipeline.predict_one(clean_r)
            records.append({"row": clean_r, "pred": p})

        target_mix = [2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, None]
        selected = []
        used_ids = set()

        for target_lvl in target_mix:
            for item in records:
                pid = item["row"]["patient_id"]
                if pid not in used_ids and item["pred"].triage_level == target_lvl:
                    selected.append(item)
                    used_ids.add(pid)
                    break

        # Format 12 waiting patients with FCFS position and priority re-ordering
        # Base simulated arrival minutes waited: 4m to 35m
        simulated_waits = [28, 14, 32, 9, 21, 15, 7, 25, 11, 18, 5, 30]
        queue_items = []
        for idx, item in enumerate(selected):
            row = item["row"]
            pred = item["pred"]
            fcfs_rank = idx + 1
            waited_min = simulated_waits[idx % len(simulated_waits)]
            waited_sec = 15
            waited_total_sec = waited_min * 60 + waited_sec
            waited_display = f"{waited_min:02d}:{waited_sec:02d}"

            lvl = pred.triage_level
            lvl_label = f"L{lvl} {LEVEL_LABELS.get(lvl, '')}" if lvl is not None else "Nurse now"

            dept = pred.primary_department if lvl is not None else None

            queue_items.append({
                "token": f"T-{101 + idx}",
                "patient_id": row["patient_id"],
                "age": row["age"],
                "sex": row["sex"],
                "complaint_snippet": str(row.get("complaint_text", "")).strip()[:70],
                "triage_level": lvl,
                "level_label": lvl_label,
                "department": dept,
                "source_layer": pred.source_layer,
                "abstain_reason": pred.abstain_reason,
                "fcfs_rank": fcfs_rank,
                "waited_seconds": waited_total_sec,
                "waited_display": waited_display,
                "arrival_time": row.get("arrival_time", "2025-01-25T10:00:00"),
                "input": {
                    "patient_id": row["patient_id"],
                    "age": row["age"],
                    "sex": row["sex"],
                    "bp_sys": row.get("bp_sys"),
                    "bp_dia": row.get("bp_dia"),
                    "pulse": row.get("pulse"),
                    "temp_c": row.get("temp_c"),
                    "spo2": row.get("spo2"),
                    "complaint_text": row.get("complaint_text", ""),
                },
            })

        _cached_seed_queue = queue_items
        return queue_items
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate seed queue: {e}")
