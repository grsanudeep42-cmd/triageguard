#!/usr/bin/env python3
"""
scripts/predict_phase2.py
─────────────────────────────────────────────────────────────────────────────
P1-C: TriageGuard Phase 2 Submission Generator
─────────────────────────────────────────────────────────────────────────────

Generates submission/H1_predictions.csv for the TalentAP Mega Hackathon 2026
Phase 2 test set.

Usage (from FULL_CODE/):
    .venv/bin/python scripts/predict_phase2.py

Output:
    submission/H1_predictions.csv   — 8,872 rows ready for submission
    results/predict_phase2_report.json

Scoring:
    .venv/bin/python evaluate.py \\
        --submission submission/H1_predictions.csv \\
        --truth data/phase2/train_labels.csv \\
        --out results/score_phase2.json
"""

from __future__ import annotations

import json
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure FULL_CODE/ is on sys.path so triageguard package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
_ROOT        = Path(__file__).resolve().parent.parent
_MODEL_DIR   = _ROOT / "models"
_DATA_DIR    = _ROOT / "data"
_PHASE2_DIR  = _DATA_DIR / "phase2"
_SUB_DIR     = _ROOT / "submission"
_RESULTS_DIR = _ROOT / "results"

_SUB_DIR.mkdir(exist_ok=True)
_RESULTS_DIR.mkdir(exist_ok=True)

# ── Submission constants (from evaluate.py) ─────────────────────────────────
DEPARTMENTS = [
    "General Medicine", "Paediatrics", "Obstetrics & Gynaecology",
    "General Surgery", "Orthopaedics", "ENT", "Ophthalmology", "Dermatology",
    "Psychiatry", "Dental", "Cardiology", "Pulmonology/Chest & TB",
    "Emergency/Casualty",
]
_DEPT_SET = set(DEPARTMENTS)

RED_FLAG_CODES = [
    "RF_CHEST_PAIN_CARDIAC", "RF_STROKE_SIGNS", "RF_RESP_DISTRESS",
    "RF_ALTERED_SENSORIUM", "RF_SEIZURE", "RF_SEVERE_BLEEDING",
    "RF_PREGNANCY_BLEEDING", "RF_ECLAMPSIA_SIGNS", "RF_REDUCED_FETAL_MOVEMENT",
    "RF_POISONING", "RF_SNAKEBITE", "RF_SCORPION_STING", "RF_SEVERE_DEHYDRATION",
    "RF_HIGH_FEVER_INFANT", "RF_CHILD_DANGER_SIGNS", "RF_SEPSIS_SIGNS",
    "RF_MENINGITIS_SIGNS", "RF_DENGUE_WARNING", "RF_SUICIDAL",
    "RF_ACUTE_PSYCHOSIS_AGITATION", "RF_TRAUMA_MAJOR", "RF_ANAPHYLAXIS",
    "RF_DKA_SIGNS", "RF_HYPOGLYCAEMIA", "RF_HEAT_STROKE", "RF_ACUTE_ABDOMEN",
    "RF_THUNDERCLAP_HEADACHE", "RF_TESTICULAR_TORSION", "RF_EYE_EMERGENCY",
    "RF_BURNS_MAJOR",
]
_RF_SET = set(RED_FLAG_CODES)

# ABSTAIN → evaluate.py scores as level 5 (worst case), but still valid.
# We only abstain when the pipeline says so.


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def sep(title: str) -> None:
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Red-Flag multi-label classifier (trained on Phase 2 labels)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_flags(x) -> List[str]:
    """Parse a semicolon-separated red-flag string into a list of RF codes."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return []
    return [t.strip().upper() for t in s.replace(",", ";").split(";") if t.strip()]


class RedFlagClassifier:
    """
    Multi-label TF-IDF + Logistic Regression classifier.

    Trained on (chief_complaint, known_conditions) → RF codes from Phase 2
    training labels.  Only predicts codes that appear in RED_FLAG_CODES.
    Threshold tuned per-label to maximise F1 on a held-out slice.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.mlb = MultiLabelBinarizer(classes=RED_FLAG_CODES)
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=60_000,
            sublinear_tf=True,
            min_df=2,
        )
        self.clf = OneVsRestClassifier(
            LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                random_state=random_state,
                class_weight="balanced",
            ),
            n_jobs=4,
        )
        self.thresholds: Dict[str, float] = {code: 0.80 for code in RED_FLAG_CODES}
        self._fitted = False

    def _build_text(self, df: pd.DataFrame) -> pd.Series:
        """Combine complaint + conditions into a single text field."""
        cc   = df.get("chief_complaint",   df.get("complaint_text",   pd.Series([""] * len(df), index=df.index))).fillna("")
        cond = df.get("known_conditions",  pd.Series([""] * len(df), index=df.index)).fillna("")
        meds = df.get("current_medications", pd.Series([""] * len(df), index=df.index)).fillna("")
        return cc + " " + cond + " " + meds

    def fit(self, intake: pd.DataFrame, labels: pd.DataFrame) -> "RedFlagClassifier":
        """
        Fit on intake_train + train_labels.
        Use 80% to train, 20% to tune per-label thresholds.
        """
        merged = intake.merge(labels[["visit_id", "red_flags"]], on="visit_id", how="inner")

        # Shuffle with fixed seed
        merged = merged.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        texts   = self._build_text(merged)
        y_lists = merged["red_flags"].apply(_parse_flags).tolist()

        # 80/20 split for threshold tuning
        n_train = int(len(merged) * 0.8)
        texts_tr,  texts_val  = texts[:n_train],  texts[n_train:]
        y_lists_tr, y_lists_val = y_lists[:n_train], y_lists[n_train:]

        # Fit MLB on all labels first so both splits share same column space
        self.mlb.fit(y_lists_tr + y_lists_val)
        Y_tr  = self.mlb.transform(y_lists_tr)
        Y_val = self.mlb.transform(y_lists_val)

        X_tr  = self.vectorizer.fit_transform(texts_tr)
        X_val = self.vectorizer.transform(texts_val)

        self.clf.fit(X_tr, Y_tr)

        # Tune thresholds per label on val split
        proba_val = self.clf.predict_proba(X_val)  # (n_val, n_labels)
        for i, code in enumerate(RED_FLAG_CODES):
            y_true_col = Y_val[:, i]
            if y_true_col.sum() == 0:
                self.thresholds[code] = 0.80
                continue  # no positives in val → keep default
            best_f1, best_thr = 0.0, 0.80
            for thr in np.arange(0.70, 0.98, 0.02):
                preds = (proba_val[:, i] >= thr).astype(int)
                tp = int((preds * y_true_col).sum())
                fp = int((preds * (1 - y_true_col)).sum())
                fn = int(((1 - preds) * y_true_col).sum())
                prec = tp / (tp + fp + 1e-9)
                rec  = tp / (tp + fn + 1e-9)
                f1   = 2 * prec * rec / (prec + rec + 1e-9)
                if f1 > best_f1:
                    best_f1, best_thr = f1, float(thr)
            self.thresholds[code] = max(0.80, best_thr)

        self._fitted = True
        print(f"  RedFlagClassifier: trained on {n_train} rows, tuned on {len(merged)-n_train} rows")
        return self

    def predict(self, df: pd.DataFrame) -> List[List[str]]:
        """Return per-row list of RF codes."""
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        texts = self._build_text(df)
        X     = self.vectorizer.transform(texts)
        proba = self.clf.predict_proba(X)   # (n, 30)
        results = []
        for row_proba in proba:
            codes = [
                code for i, code in enumerate(RED_FLAG_CODES)
                if row_proba[i] >= self.thresholds[code]
            ]
            results.append(codes)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Load pipeline
# ─────────────────────────────────────────────────────────────────────────────

def load_pipeline():
    """Load TriagePipeline from models/."""
    from triageguard.predict import TriagePipeline
    print("  Loading TriagePipeline …", end=" ", flush=True)
    t0 = time.perf_counter()
    pipeline = TriagePipeline.load(_MODEL_DIR)
    print(f"done ({(time.perf_counter()-t0)*1000:.0f} ms)")
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Main prediction loop
# ─────────────────────────────────────────────────────────────────────────────

def predict_dataframe(
    df: pd.DataFrame,
    pipeline,
    rf_clf: RedFlagClassifier,
    text_model,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run the unified prediction pipeline with all competition optimizations:
    1. Base pipeline evaluation with calibrated tau.
    2. Smart abstention on truly unassessable records.
    3. Emergency keyword escalation to Level 2.
    4. Confirmed red flag escalation to Level 2.
    5. Aggressive L2 bump for high-risk vitals (cardiac / shock / low SpO2).
    6. Shift routine over-predicted Level 5 to Level 4.
    """
    from triageguard.data_adapter import adapt_schema
    visit_ids = df["visit_id"].tolist()
    adapted_df = adapt_schema(df.copy())

    t_start = time.perf_counter()
    preds = pipeline.predict_batch(adapted_df)
    t_pred = time.perf_counter() - t_start

    rf_preds = rf_clf.predict(df)

    cardiac_set = {
        "RF_CHEST_PAIN_CARDIAC", "RF_RESP_DISTRESS",
        "RF_ALTERED_SENSORIUM", "RF_STROKE_SIGNS",
        "RF_SEPSIS_SIGNS", "RF_ANAPHYLAXIS", "RF_DKA_SIGNS",
    }

    boilerplate = {
        "?", "na", "-", "check", "same", "opd", "dikhana hai", "opd ticket",
        "pt lang not understood", "dr ko dikhana hai", "....", "doctor se milna hai",
        "xx", "--", "patient not able to explain", "nil", "none", "n/a", "blank",
        "ticket", "parchi", "follow up", "dikhao", "check up", "normal", "theek hai",
        ".", "---", "..", "??",
    }

    emerg_keywords = [
        "insecticide", "poison", "zehar", "visham", "scorpion", "bichhu",
        "snake", "saanp", "paambu", "kalai kaat", "suicid", "khudkushi",
        "jaan dena", "jaan nahi", "lakwa", "paralysis", "vision loss",
        "roshni chali gayi", "proshrab kom", "peshab kam",
    ]

    records = []
    level_counts: Dict[str, int] = defaultdict(int)
    dept_counts: Dict[str, int] = defaultdict(int)
    abstain_count = 0
    fallback_count = 0
    rf_flagged = 0

    for vid, pred, rfs, adapted_row in zip(visit_ids, preds, rf_preds, adapted_df.itertuples(index=False)):
        level = pred.triage_level
        dept = pred.primary_department
        p_em = pred.p_emergency

        raw_complaint = str(getattr(adapted_row, "complaint_text", "") or "").strip()
        cc_lower = raw_complaint.lower()

        # Count missing/invalid vitals
        v_missing = 0
        for attr in ["bp_sys", "bp_dia", "pulse", "temp_c", "spo2"]:
            val = getattr(adapted_row, attr, None)
            if val is None or pd.isna(val) or val <= 0:
                v_missing += 1

        is_boilerplate = (
            cc_lower in boilerplate
            or len(cc_lower) <= 2
            or any(kw in cc_lower for kw in ["not able to explain", "not understood", "doctor se milna", "dr ko dikhana"])
        )

        # Smart abstention on truly unassessable
        is_unassessable = (v_missing >= 3 and is_boilerplate)
        if is_unassessable:
            level = None

        # Text-model fallback for other abstains
        if level is None and not is_unassessable:
            if len(raw_complaint) > 5:
                try:
                    df_fallback = pd.DataFrame([{"patient_id": vid, "visit_id": vid, "complaint_text": raw_complaint}])
                    txt_proba = text_model.predict_proba(df_fallback)[0]
                    txt_em = text_model.predict_emergency_proba(df_fallback)[0]
                    if float(txt_em) >= 0.40:
                        level = 2
                    else:
                        level = int(np.argmax(txt_proba)) + 1
                    p_em = float(txt_em)
                    if level in (1, 2):
                        dept = "Emergency/Casualty"
                    fallback_count += 1
                except Exception:
                    pass

        # Clinical emergency keywords -> Level 2
        has_emerg_kw = any(kw in cc_lower for kw in emerg_keywords)
        if has_emerg_kw and (level is None or level > 2):
            level = 2
            dept = "Emergency/Casualty"

        # Valid red flags -> Level 2
        valid_rfs = [rf for rf in rfs if rf in _RF_SET]
        if valid_rfs and (level is not None and level > 2):
            level = 2
            dept = "Emergency/Casualty"

        # Change 3: Aggressive L2 bump post-processing
        if level == 3 and pred.source_layer != "rule":
            pulse_v = getattr(adapted_row, "pulse", 0) or 0
            bp_sys_v = getattr(adapted_row, "bp_sys", 0) or 0
            spo2_v = getattr(adapted_row, "spo2", 100) or 100
            shock_index = (pulse_v / bp_sys_v) if (pulse_v > 0 and bp_sys_v > 0) else 0

            conf_val = round(float(p_em if level in (1, 2) else 1.0 - p_em), 4)
            if conf_val < 0.65:
                has_cardiac_flag = bool(set(valid_rfs) & cardiac_set)
                if has_cardiac_flag or shock_index > 0.9 or spo2_v < 95:
                    level = 2
                    dept = "Emergency/Casualty"

        # Shift routine over-predicted Level 5s to Level 4
        if level == 5:
            level = 4

        # Validate department against allowed list
        if dept not in _DEPT_SET:
            dept = "General Medicine"

        # Format level
        if level is None:
            abstain_count += 1
            level_str = "ABSTAIN"
        else:
            level_str = str(int(level))

        # Format red flags
        if valid_rfs:
            rf_flagged += 1
            rf_str = ";".join(valid_rfs)
        else:
            rf_str = np.nan

        conf = round(float(p_em if level in (1, 2) else 1.0 - p_em), 4)
        conf = max(0.0, min(1.0, conf))

        level_counts[level_str] += 1
        dept_counts[dept] += 1

        records.append({
            "visit_id": vid,
            "triage_level": level_str,
            "department": dept,
            "red_flags": rf_str,
            "confidence": conf,
        })

    sub_df = pd.DataFrame(records, columns=["visit_id", "triage_level", "department", "red_flags", "confidence"])
    stats = {
        "total": len(sub_df),
        "abstain_count": abstain_count,
        "fallback_count": fallback_count,
        "rf_flagged": rf_flagged,
        "level_counts": level_counts,
        "dept_counts": dept_counts,
        "t_pred": t_pred,
    }
    return sub_df, stats


def validate_submission_df(sub_df: pd.DataFrame, orig_df: pd.DataFrame) -> None:
    errors = []
    if len(sub_df) != len(orig_df):
        errors.append(f"Row count mismatch: {len(sub_df)} vs {len(orig_df)}")

    missing_vids = set(orig_df["visit_id"]) - set(sub_df["visit_id"])
    if missing_vids:
        errors.append(f"Missing visit_ids: {list(missing_vids)[:5]}")

    bad_depts = sub_df[sub_df["department"].notna() & ~sub_df["department"].isin(_DEPT_SET)]
    if len(bad_depts):
        errors.append(f"{len(bad_depts)} rows have invalid departments")

    for _, r in sub_df.iterrows():
        rfs = _parse_flags(r["red_flags"])
        bad = [rf for rf in rfs if rf not in _RF_SET]
        if bad:
            errors.append(f"visit_id {r['visit_id']}: invalid RF codes: {bad}")
            break

    bad_conf = sub_df[sub_df["confidence"].notna() & ((sub_df["confidence"] < 0) | (sub_df["confidence"] > 1))]
    if len(bad_conf):
        errors.append(f"{len(bad_conf)} rows have confidence out of [0,1]")

    if errors:
        print("❌  VALIDATION ERRORS:")
        for e in errors:
            print(f"    {e}")
        sys.exit(1)
    else:
        print("  ✓ All validation checks passed")


def main() -> None:
    t_start = time.perf_counter()

    sep("STEP 0 — LOAD PHASE 2 DATA")
    test_path = _PHASE2_DIR / "intake_test.csv"
    train_path = _PHASE2_DIR / "intake_train.csv"
    labels_path = _PHASE2_DIR / "train_labels.csv"

    if not test_path.exists():
        print(f"ERROR: {test_path} not found", file=sys.stderr)
        sys.exit(1)

    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)
    train_labels = pd.read_csv(labels_path)
    print(f"  intake_test.csv  : {len(test_df):,} rows")
    print(f"  intake_train.csv : {len(train_df):,} rows")

    sep("STEP 1 — TRAIN RED-FLAG CLASSIFIER (PRECISION FLOOR 0.80)")
    rf_clf = RedFlagClassifier(random_state=42)
    rf_clf.fit(train_df, train_labels)

    sep("STEP 2 — LOAD TRIAGE PIPELINE & TEXT MODEL")
    pipeline = load_pipeline()
    from triageguard.model_text import TextModel
    text_model = TextModel.load(_MODEL_DIR / "model_text.joblib")

    sep("STEP 3 — GENERATE PREDICTIONS FOR TEST SET (8,872 rows)")
    test_sub, test_stats = predict_dataframe(test_df, pipeline, rf_clf, text_model)
    validate_submission_df(test_sub, test_df)

    # Save test submission
    out_path = _SUB_DIR / "H1_predictions.csv"
    test_sub.to_csv(out_path, index=False)
    # Also save to root submission
    root_sub = _ROOT.parent / "submission" / "H1_predictions.csv" if (_ROOT.name == "FULL_CODE") else (_ROOT / "submission" / "H1_predictions.csv")
    if root_sub.parent.exists():
        test_sub.to_csv(root_sub, index=False)
    print(f"  ✓ Saved {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    if root_sub.exists():
        print(f"  ✓ Synced {root_sub}")

    sep("STEP 4 — GENERATE PREDICTIONS FOR TRAIN SELF-CHECK (21,455 rows)")
    train_sub, train_stats = predict_dataframe(train_df, pipeline, rf_clf, text_model)
    validate_submission_df(train_sub, train_df)

    train_out_path = _SUB_DIR / "H1_train_check.csv"
    train_sub.to_csv(train_out_path, index=False)
    root_train_sub = _ROOT.parent / "submission" / "H1_train_check.csv" if (_ROOT.name == "FULL_CODE") else (_ROOT / "submission" / "H1_train_check.csv")
    if root_train_sub.parent.exists():
        train_sub.to_csv(root_train_sub, index=False)
    print(f"  ✓ Saved {train_out_path} ({train_out_path.stat().st_size / 1024:.1f} KB)")
    if root_train_sub.exists():
        print(f"  ✓ Synced {root_train_sub}")

    # Summary
    sep("STEP 5 — PREDICTION SUMMARY")
    print(f"  TEST SET (8,872 rows):")
    print(f"    Abstains: {test_stats['abstain_count']:,} | Flagged: {test_stats['rf_flagged']:,}")
    print(f"    Levels: {dict(test_stats['level_counts'])}")
    print(f"  TRAIN SET (21,455 rows):")
    print(f"    Abstains: {train_stats['abstain_count']:,} | Flagged: {train_stats['rf_flagged']:,}")
    print(f"    Levels: {dict(train_stats['level_counts'])}")

    sep("STEP 6 — OFFICIAL EVALUATION ON TRAIN SELF-CHECK")
    from evaluate import load_submission, load_truth, score
    sub_eval = load_submission(train_out_path)
    truth_eval = load_truth(labels_path)
    res = score(sub_eval, truth_eval)

    print()
    print("═" * 70)
    print("  EVALUATION RESULTS (TRAIN SELF-CHECK)")
    print("═" * 70)
    print(f"  PRIMARY SCORE:           {res['primary_score']:.6f}   (Baseline: 0.6014)")
    print(f"  Emergency Recall:        {res['emergency_recall']:.6f}   (Baseline: 0.9895)")
    print(f"  Normalised Safety:       {res['normalised_safety_score']:.6f}   (Baseline: 0.4352)")
    print(f"  Department Accuracy:     {res['department_accuracy']:.6f}   (Baseline: 0.7676)")
    print(f"  Red-Flag Micro F1:       {res['red_flag_micro_f1']:.6f}   (Baseline: 0.6561)")
    print(f"  Red-Flag Precision:      {res['red_flag_precision']:.6f}   (Baseline: 0.5059)")
    print(f"  Red-Flag Recall:         {res['red_flag_recall']:.6f}   (Baseline: 0.9331)")
    print(f"  Missed Emergencies:      {res['n_missed_emergencies']}            (Baseline: 9)")
    print(f"  Team Safety Cost:        {res['team_cost']:.1f}       (Baseline: 18064.5)")
    print("═" * 70)

    elapsed = time.perf_counter() - t_start
    print(f"\n  Pipeline completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
