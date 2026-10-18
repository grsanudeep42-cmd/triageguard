import json
import math
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client: TestClient):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True
    assert data["version"] == "1.0.0"


def test_triage_standard_case(client: TestClient):
    payload = {
        "age": 35,
        "sex": "M",
        "bp_sys": 120.0,
        "bp_dia": 80.0,
        "pulse": 72.0,
        "temp_c": 36.8,
        "spo2": 98.0,
        "complaint_text": "mild headache for one day",
    }
    resp = client.post("/api/triage", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "triage_level" in data
    assert "source_layer" in data
    assert "p_emergency" in data
    assert "tau" in data
    assert "vitals" in data
    assert "views" in data
    assert "latency_ms" in data
    assert data["vitals"]["spo2"]["status"] == "normal"
    assert data["vitals"]["pulse"]["status"] == "normal"


def test_triage_critical_hypoxia_rule(client: TestClient):
    payload = {
        "age": 45,
        "sex": "F",
        "bp_sys": 110.0,
        "bp_dia": 70.0,
        "pulse": 85.0,
        "temp_c": 36.5,
        "spo2": 82.0,  # Below 85% -> R001 Level 1
        "complaint_text": "shortness of breath",
    }
    resp = client.post("/api/triage", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["triage_level"] == 1
    assert data["source_layer"] == "rule"
    assert any(r["id"] == "R001" for r in data["rules_fired"])
    assert data["vitals"]["spo2"]["status"] == "critical"


def test_triage_abstain_needs_remeasure(client: TestClient):
    payload = {
        "age": 40,
        "sex": "M",
        "bp_sys": 120.0,
        "bp_dia": 80.0,
        "pulse": None,  # missing
        "temp_c": None,  # missing
        "spo2": None,  # missing -> >= 2 missing vitals
        "complaint_text": "routine checkup",
    }
    resp = client.post("/api/triage", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["triage_level"] is None
    assert data["source_layer"] == "abstain"
    assert data["abstain_reason"] == "needs_remeasure"


def test_triage_demo_cases_match(client: TestClient):
    demo_path = Path("results/demo_cases.json")
    if not demo_path.exists():
        pytest.skip("results/demo_cases.json not found")

    with open(demo_path) as f:
        cases = json.load(f)

    for case_id, c in cases.items():
        v = c.get("cleaned_vitals") or {}
        payload = {
            "patient_id": c.get("patient_id"),
            "age": c.get("age"),
            "sex": c.get("sex"),
            "complaint_text": c.get("combined_gamed_complaint") or c.get("complaint_text"),
            "bp_sys": None if v.get("bp_sys") is not None and math.isnan(v.get("bp_sys")) else v.get("bp_sys"),
            "bp_dia": None if v.get("bp_dia") is not None and math.isnan(v.get("bp_dia")) else v.get("bp_dia"),
            "pulse": None if v.get("pulse") is not None and math.isnan(v.get("pulse")) else v.get("pulse"),
            "temp_c": None if v.get("temp_c") is not None and math.isnan(v.get("temp_c")) else v.get("temp_c"),
            "spo2": None if v.get("spo2") is not None and math.isnan(v.get("spo2")) else v.get("spo2"),
        }
        resp = client.post("/api/triage", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        exp = c["triageguard_prediction"]

        assert data["triage_level"] == exp["triage_level"], f"{case_id} level mismatch"
        assert data["source_layer"] == exp["source_layer"], f"{case_id} source_layer mismatch"
        assert data["abstain_reason"] == exp["abstain_reason"], f"{case_id} abstain_reason mismatch"
        assert data["primary_department"] == exp["primary_department"], f"{case_id} department mismatch"


def test_demo_cases_endpoint(client: TestClient):
    resp = client.get("/api/demo-cases")
    assert resp.status_code == 200
    cases = resp.json()
    assert isinstance(cases, list)
    assert len(cases) >= 6
    for c in cases:
        assert "id" in c
        assert "scenario" in c
        assert "input" in c
        assert "expected_prediction" in c
        assert "ground_truth" in c


def test_rules_endpoint(client: TestClient):
    resp = client.get("/api/rules")
    assert resp.status_code == 200
    data = resp.json()
    assert "rules" in data
    assert "red_flags" in data
    assert len(data["rules"]) > 0
    assert len(data["red_flags"]) > 0
    first_rule = data["rules"][0]
    assert "id" in first_rule
    assert "plain_language_trigger" in first_rule
    assert "threshold" in first_rule
    assert "forced_level" in first_rule


def test_results_endpoint(client: TestClient):
    resp = client.get("/api/results")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "demo_cases" in data or "cost_comparison" in data or len(data) > 0


def test_seed_queue_endpoint(client: TestClient):
    resp = client.get("/api/seed-queue")
    assert resp.status_code == 200
    queue = resp.json()
    assert isinstance(queue, list)
    assert len(queue) == 12
    for p in queue:
        assert "token" in p
        assert "patient_id" in p
        assert "age" in p
        assert "sex" in p
        assert "complaint_snippet" in p
        assert "level_label" in p
        assert "waited_display" in p
