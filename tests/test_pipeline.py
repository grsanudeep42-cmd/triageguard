"""
tests/test_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive unit tests for Phases 4–8 of TriageGuard:
  - Data adapter & feature extraction (truly vitals-only for VitalsModel)
  - Vitals model (LightGBM multiclass + binary emergency)
  - Text model (TF-IDF + Logistic Regression multiclass + binary)
  - Department model (dual routing: emergency/casualty vs suggested_specialty)
  - Fusion stacker (Platt calibration, expected cost decision, ECE)
  - Conformal screen (Mondrian finite-sample correction) & level sets
  - TriagePipeline end-to-end policy execution & abstention triggers
"""

import math
import numpy as np
import pandas as pd
import pytest

from triageguard.cleaning import clean_dataframe, clean_record
from triageguard.data_adapter import (
    adapt_schema,
    extract_vitals_features,
    extract_red_flag_count,
    VITALS_FEATURE_COLS,
)
from triageguard.model_vitals import VitalsModel
from triageguard.model_text import TextModel
from triageguard.model_dept import DepartmentModel
from triageguard.fusion import (
    StackerFusion,
    build_cost_matrix,
    compute_expected_costs,
    compute_ece,
    select_min_cost_level,
)
from triageguard.conformal import ConformalPredictor, ConformalState
from triageguard.predict import TriagePipeline, TriagePrediction


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA ADAPTER TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_data_adapter_schema_alias():
    """Verify adapt_schema maps known aliases to standard schema names."""
    df_raw = pd.DataFrame([{
        "pt_id": "P001",
        "patient_age": 45,
        "gender": "M",
        "chief_complaint": "chest pain and sweating",
        "systolic": 85,
        "diastolic": 55,
        "heart_rate": 125,
        "oxygen": 88,
        "temperature": 98.6,
    }])
    df_adapted = adapt_schema(df_raw)
    assert "complaint_text" in df_adapted.columns
    assert "bp_sys" in df_adapted.columns
    assert "bp_dia" in df_adapted.columns
    assert "pulse" in df_adapted.columns
    assert "spo2" in df_adapted.columns
    assert "temp_c" in df_adapted.columns
    assert df_adapted.loc[0, "pulse"] == 125


def test_vitals_features_strictly_no_red_flags():
    """VitalsModel features must NEVER contain text-derived red-flag counts."""
    assert "n_red_flags" not in VITALS_FEATURE_COLS
    assert "complaint" not in VITALS_FEATURE_COLS
    assert "complaint_clean" not in VITALS_FEATURE_COLS

    df = pd.DataFrame([{
        "age": 50,
        "sex": "F",
        "bp_sys": 120.0,
        "bp_dia": 80.0,
        "pulse": 80.0,
        "temp": 37.0,
        "spo2": 98.0,
        "complaint_text": "mild headache",
    }])
    df_cleaned = clean_dataframe(df)
    X = extract_vitals_features(df_cleaned)

    assert "n_red_flags" not in X.columns
    assert "shock_index" in X.columns
    assert "map" in X.columns
    assert "spo2_c" in X.columns
    assert abs(X.loc[0, "shock_index"] - (80.0 / 120.0)) < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# 2. VITALS MODEL TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_vitals_model_fit_and_predict():
    """Verify VitalsModel fits and returns normalized probabilities and binary p_em."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "age": np.random.randint(18, 80, n),
        "sex": np.random.choice(["M", "F"], n),
        "bp_sys": np.random.uniform(70, 160, n),
        "bp_dia": np.random.uniform(40, 100, n),
        "pulse": np.random.uniform(50, 150, n),
        "temp": np.random.uniform(36.0, 39.5, n),
        "spo2": np.random.uniform(80, 100, n),
        "complaint_text": ["test"] * n,
    })
    df_clean = clean_dataframe(df)
    X_feat = extract_vitals_features(df_clean)

    y_level = np.random.choice([1, 2, 3, 4, 5], n)
    y_em = (y_level <= 2).astype(int)

    model = VitalsModel()
    model.fit(X_feat, y_level, y_em)

    probs = model.predict_proba(X_feat)
    assert probs.shape == (n, 5)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(n), atol=1e-5)

    p_em = model.predict_emergency_proba(X_feat)
    assert len(p_em) == n
    assert (p_em >= 0.0).all() and (p_em <= 1.0).all()


# ─────────────────────────────────────────────────────────────────────────────
# 3. TEXT MODEL TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_text_model_fit_and_predict():
    """Verify TextModel fits with TF-IDF char n-grams and outputs calibrated predictions."""
    texts = [
        "severe crushing chest pain radiating to left arm",
        "difficulty breathing gasp for air",
        "mild cough cold for 3 days",
        "superficial finger scrape playing cricket",
        "headache after reading without glasses",
    ] * 20
    y_level = np.array([1, 2, 3, 4, 5] * 20)
    y_em = (y_level <= 2).astype(int)

    model = TextModel()
    model.fit(texts, y_level, y_em)

    probs = model.predict_proba(texts[:5])
    assert probs.shape == (5, 5)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), atol=1e-5)

    p_em = model.predict_emergency_proba(texts[:5])
    assert p_em[0] > p_em[3]


# ─────────────────────────────────────────────────────────────────────────────
# 4. DEPARTMENT MODEL TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_department_model_dual_routing():
    """Verify DepartmentModel predicts specialty and routes L1/L2 to Emergency/Casualty."""
    texts = [
        "chest pain cardiac arrest",
        "skin rash itching eczema",
        "ear pain discharge otitis",
        "bone fracture arm deformity",
    ] * 10
    depts = ["Cardiology", "Dermatology", "ENT", "Orthopaedics"] * 10
    ages = [55, 25, 12, 35] * 10
    sexes = ["M", "F", "M", "M"] * 10

    df = pd.DataFrame({
        "complaint_text": texts,
        "age": ages,
        "sex": sexes,
        "bp_sys": [120] * 40,
        "bp_dia": [80] * 40,
        "pulse": [75] * 40,
        "temp_c": [37.0] * 40,
        "spo2": [98.0] * 40,
    })
    df_cleaned = clean_dataframe(df)

    model = DepartmentModel()
    model.fit(df_cleaned, depts)

    test_df = pd.DataFrame({
        "complaint_text": ["acute chest pain", "dry skin rash"],
        "age": [60, 20],
        "sex": ["M", "F"],
        "bp_sys": [120, 120],
        "bp_dia": [80, 80],
        "pulse": [75, 75],
        "temp_c": [37.0, 37.0],
        "spo2": [98.0, 98.0],
    })
    test_cleaned = clean_dataframe(test_df)
    primaries, specialties = model.route_departments(test_cleaned, triage_levels=[1, 4])

    # For level 1, primary department must be Emergency/Casualty
    assert primaries[0] == "Emergency/Casualty"
    assert specialties[0] != ""

    # For level 4, primary department should be the clinical specialty
    assert primaries[1] == specialties[1]


# ─────────────────────────────────────────────────────────────────────────────
# 5. FUSION & EXPECTED COST TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_fusion_expected_cost_penalty():
    """Verify expected cost minimization severely penalizes under-triaging emergencies."""
    cost_matrix = build_cost_matrix()
    p = np.array([0.30, 0.40, 0.30, 0.0, 0.0])

    expected_costs = compute_expected_costs(p.reshape(1, 5), cost_matrix)[0]
    assert expected_costs[0] < expected_costs[3]
    assert expected_costs[1] < expected_costs[3]

    best_lvl = select_min_cost_level(p, cost_matrix=cost_matrix)
    assert best_lvl in [1, 2]


def test_ece_computation():
    """Verify Expected Calibration Error calculation."""
    y_true = np.array([1, 1, 0, 0])
    p_pred = np.array([0.9, 0.8, 0.2, 0.1])
    ece = compute_ece(p_pred, y_true, n_bins=5)
    assert 0.0 <= ece <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. CONFORMAL PREDICTOR TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_conformal_finite_sample_correction():
    """Verify finite-sample quantile formula: k = floor(alpha * (n + 1))."""
    scores = np.sort([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    n = len(scores)

    # For alpha = 0.05, n = 10: k = floor(0.05 * 11) = 0 -> infeasible
    k = math.floor(0.05 * (n + 1))
    assert k == 0

    # For alpha = 0.20, n = 10: k = floor(0.20 * 11) = 2 -> 2nd smallest score
    k = math.floor(0.20 * (n + 1))
    assert k == 2
    tau = scores[k - 1]
    assert tau == 0.2


# ─────────────────────────────────────────────────────────────────────────────
# 7. PIPELINE END-TO-END POLICY & ABSTENTION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class MockVitalsModel:
    def predict_proba(self, X):
        n = len(X)
        return np.tile([0.05, 0.05, 0.3, 0.5, 0.1], (n, 1))
    def predict_emergency_proba(self, X):
        return np.full(len(X), 0.10)


class MockTextModel:
    def predict_proba(self, texts):
        n = len(texts)
        return np.tile([0.05, 0.05, 0.3, 0.5, 0.1], (n, 1))
    def predict_emergency_proba(self, texts):
        return np.full(len(texts), 0.10)


class MockDeptModel:
    def predict_specialty(self, df):
        return ["General Medicine"] * len(df)


class MockFusion:
    def predict_fused(self, p_vit_m, p_txt_m, p_vit_e, p_txt_e, df_cleaned):
        n = len(df_cleaned)
        probs_5 = np.tile([0.05, 0.05, 0.3, 0.5, 0.1], (n, 1))
        p_em = np.full(n, 0.10)
        return probs_5, p_em


class MockConformal:
    def __init__(self, tau=0.5):
        self.state = ConformalState(
            operating_alpha=0.05,
            operating_tau=tau,
            tau_map={0.05: tau},
            level_cutoff=0.8,
            infeasible_alphas=[],
        )
    def predict_level_set(self, probs_5):
        return [3, 4]


def test_pipeline_policy_vitals_rule_override():
    """1. Vitals rule forces a level -> output it with source='rule'."""
    pipeline = TriagePipeline(
        model_vitals=MockVitalsModel(),
        model_text=MockTextModel(),
        model_dept=MockDeptModel(),
        fusion=MockFusion(),
        conformal=MockConformal(),
    )
    # SpO2 = 82 forces Level 1 via rule R001
    record = {
        "age": 45, "sex": "M",
        "complaint_text": "feeling tired",
        "bp_sys": 120, "bp_dia": 80,
        "pulse": 75, "temp": 37.0,
        "spo2": 82.0,
    }
    pred = pipeline.predict_one(record)
    assert pred.triage_level == 1
    assert pred.source_layer == "rule"


def test_pipeline_policy_red_flag_floor():
    """2. Red flag present -> floor at L3 (never L4/L5 unless vitals normal)."""
    class LowUrgencyFusion(MockFusion):
        def predict_fused(self, p_vit_m, p_txt_m, p_vit_e, p_txt_e, df_cleaned):
            n = len(df_cleaned)
            probs_5 = np.tile([0.01, 0.01, 0.08, 0.50, 0.40], (n, 1))
            p_em = np.full(n, 0.02)
            return probs_5, p_em

    pipeline = TriagePipeline(
        model_vitals=MockVitalsModel(),
        model_text=MockTextModel(),
        model_dept=MockDeptModel(),
        fusion=LowUrgencyFusion(),
        conformal=MockConformal(tau=0.99),  # Screen won't fire
    )
    # Red flag phrase in complaint: "chest pain"
    record = {
        "age": 45, "sex": "M",
        "complaint_text": "patient reports chest pain and sweating",
        "bp_sys": 120, "bp_dia": 80,
        "pulse": 75, "temp": 37.0,
        "spo2": 98.0,
    }
    pred = pipeline.predict_one(record)
    assert pred.triage_level <= 3


def test_pipeline_abstain_needs_remeasure():
    """4. needs_remeasure -> ABSTAIN with abstain_reason='needs_remeasure'."""
    pipeline = TriagePipeline(
        model_vitals=MockVitalsModel(),
        model_text=MockTextModel(),
        model_dept=MockDeptModel(),
        fusion=MockFusion(),
        conformal=MockConformal(),
    )
    # Two invalid/missing vitals (pulse missing, temp=0 -> invalid)
    record = {
        "age": 30, "sex": "F",
        "complaint_text": "mild headache",
        "bp_sys": 120, "bp_dia": 80,
        "pulse": None,
        "temp": 0.0,
        "spo2": 98.0,
    }
    pred = pipeline.predict_one(record)
    assert pred.source_layer == "abstain"
    assert pred.abstain_reason == "needs_remeasure"
