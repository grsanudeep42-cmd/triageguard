import json
import math
import sys
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

def sanitize_val(val):
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    return val

def run_verification():
    demo_cases_path = Path("results/demo_cases.json")
    if not demo_cases_path.exists():
        print(f"Error: {demo_cases_path} does not exist!")
        sys.exit(1)

    with open(demo_cases_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 80)
    print("TRIAGEGUARD DEMO CASES VERIFICATION AGAINST /api/triage")
    print("=" * 80)

    all_matched = True
    fields_to_check = [
        "triage_level",
        "source_layer",
        "abstain_reason",
        "primary_department",
        "conformal_set",
    ]

    total_cases = len(data)
    print(f"Loaded {total_cases} benchmark demo cases from {demo_cases_path}.\n")

    with TestClient(app) as client:
        for idx, (case_key, case_val) in enumerate(data.items(), 1):
            vitals = case_val.get("cleaned_vitals") or case_val.get("raw_vitals") or {}
            complaint = case_val.get("combined_gamed_complaint") or case_val.get("complaint_text") or case_val.get("complaint") or ""

            payload = {
                "patient_id": case_val.get("patient_id"),
                "age": case_val.get("age"),
                "sex": case_val.get("sex"),
                "bp_sys": sanitize_val(vitals.get("bp_sys")),
                "bp_dia": sanitize_val(vitals.get("bp_dia")),
                "pulse": sanitize_val(vitals.get("pulse")),
                "temp_c": sanitize_val(vitals.get("temp_c")),
                "spo2": sanitize_val(vitals.get("spo2")),
                "complaint_text": complaint,
                "conditions": [],
                "medications": [],
            }

            expected = case_val.get("expected_prediction") or case_val.get("triageguard_prediction") or {}

            resp = client.post("/api/triage", json=payload)
            if resp.status_code != 200:
                print(f"[{idx}/{total_cases}] {case_key}: HTTP {resp.status_code} ERROR: {resp.text}")
                all_matched = False
                continue

            actual = resp.json()
            case_matched = True
            diffs = []

            for field in fields_to_check:
                exp_val = expected.get(field)
                act_val = actual.get(field)

                # normalize None and null
                if exp_val != act_val:
                    diffs.append((field, exp_val, act_val))
                    case_matched = False
                    all_matched = False

            status_str = "MATCH" if case_matched else "DIFF"
            print(f"[{idx}/{total_cases}] {case_key} -> {status_str}")
            print(f"     Complaint: \"{complaint}\"")
            print(f"     Input vitals: bp={payload['bp_sys']}/{payload['bp_dia']}, pulse={payload['pulse']}, temp={payload['temp_c']}, spo2={payload['spo2']}")
            
            for field in fields_to_check:
                exp_val = expected.get(field)
                act_val = actual.get(field)
                match_marker = "✓" if exp_val == act_val else "✗ DIFF"
                print(f"       • {field:<20}: expected={exp_val!r:<22} actual={act_val!r:<22} [{match_marker}]")

            if diffs:
                print("     DISCREPANCIES DETECTED:")
                for field, exp_v, act_v in diffs:
                    print(f"       ! {field}: expected {exp_v!r} != actual {act_v!r}")
            print("-" * 80)

    print("\n" + "=" * 80)
    if all_matched:
        print(f"ALL {total_cases} CASES MATCH 100% EXACTLY ACROSS ALL 5 REQUIRED FIELDS.")
    else:
        print("VERIFICATION FAILED: Discrepancies found above.")
    print("=" * 80)

    if not all_matched:
        sys.exit(1)

if __name__ == "__main__":
    run_verification()
