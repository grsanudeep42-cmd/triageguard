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
            n_jobs=-1,
        )
        self.thresholds: Dict[str, float] = {code: 0.3 for code in RED_FLAG_CODES}
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
                continue  # no positives in val → keep default
            best_f1, best_thr = 0.0, 0.3
            for thr in np.arange(0.05, 0.95, 0.05):
                preds = (proba_val[:, i] >= thr).astype(int)
                tp = int((preds * y_true_col).sum())
                fp = int((preds * (1 - y_true_col)).sum())
                fn = int(((1 - preds) * y_true_col).sum())
                prec = tp / (tp + fp + 1e-9)
                rec  = tp / (tp + fn + 1e-9)
                f1   = 2 * prec * rec / (prec + rec + 1e-9)
                if f1 > best_f1:
                    best_f1, best_thr = f1, float(thr)
            self.thresholds[code] = best_thr

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

def main() -> None:
    t_start = time.perf_counter()

    # ── Step 0: Load test data ───────────────────────────────────────────────
    sep("STEP 0 — LOAD TEST DATA")
    test_path = _PHASE2_DIR / "intake_test.csv"
    if not test_path.exists():
        print(f"ERROR: {test_path} not found", file=sys.stderr)
        sys.exit(1)

    test_df = pd.read_csv(test_path)
    print(f"  intake_test.csv : {len(test_df):,} rows × {len(test_df.columns)} cols")
    print(f"  Columns         : {test_df.columns.tolist()}")

    # Keep original visit_ids for output — do NOT let adapt_schema rename them in output
    visit_ids = test_df["visit_id"].tolist()
    print(f"  visit_id sample : {visit_ids[:3]}")

    # ── Step 1: Train RedFlagClassifier on Phase 2 training data ────────────
    sep("STEP 1 — TRAIN RED-FLAG CLASSIFIER ON PHASE 2 LABELS")
    train_intake = pd.read_csv(_PHASE2_DIR / "intake_train.csv")
    train_labels = pd.read_csv(_PHASE2_DIR / "train_labels.csv")

    rf_clf = RedFlagClassifier(random_state=42)
    rf_clf.fit(train_intake, train_labels)

    # ── Step 2: Load main triage pipeline ───────────────────────────────────
    sep("STEP 2 — LOAD TRIAGE PIPELINE")
    pipeline = load_pipeline()

    # ── Step 3: Run predictions in batch ────────────────────────────────────
    sep("STEP 3 — GENERATE PREDICTIONS")
    print(f"  Predicting {len(test_df):,} patients …")

    # CRITICAL: predict_one() expects INTERNAL column names (bp_sys, pulse, spo2 …)
    # but intake_test.csv uses Phase 2 names (bp_systolic, pulse_bpm, spo2_pct …).
    # Run adapt_schema on the full DataFrame BEFORE passing rows to predict_batch.
    # This mirrors how retrain_phase2.py pre-adapts data before cleaning.
    from triageguard.data_adapter import adapt_schema
    print("  Running adapt_schema on test DataFrame …", end=" ", flush=True)
    adapted_test_df = adapt_schema(test_df.copy())
    print(f"done  (cols: {adapted_test_df.columns.tolist()[:6]} …)")

    # Predict triage levels / departments via TriagePipeline
    t_pred_start = time.perf_counter()
    preds = pipeline.predict_batch(adapted_test_df)
    t_pred_sec = time.perf_counter() - t_pred_start
    print(f"  Triage predictions done: {t_pred_sec:.1f}s  ({t_pred_sec/len(test_df)*1000:.2f} ms/patient)")

    # Predict red flags via our supervised classifier
    rf_preds = rf_clf.predict(test_df)
    print(f"  Red-flag predictions done.")

    # ── Step 4: Assemble submission ─────────────────────────────────────────
    sep("STEP 4 — ASSEMBLE SUBMISSION")

    # Text-model fallback for abstain patients: use TextModel directly when
    # needs_remeasure fires (≥2 missing/invalid vitals) but complaint text exists.
    # The text model was trained on Phase 2 labels and has AUC=0.94 for emergency.
    from triageguard.model_text import TextModel
    from triageguard.fusion import build_cost_matrix, select_min_cost_level
    import joblib as jb

    _text_model = TextModel.load(_MODEL_DIR / "model_text.joblib")
    _cost_matrix = build_cost_matrix(pipeline.cfg)

    # Build a lookup: visit_id → abstain_reason
    abstain_lookup = {
        vid: p for vid, p in zip(visit_ids, preds) if p.triage_level is None
    }

    fallback_count = 0

    records = []
    level_counts: Dict[str, int] = defaultdict(int)
    dept_counts:  Dict[str, int] = defaultdict(int)
    abstain_count = 0
    rf_flagged    = 0

    for vid, pred, rfs, adapted_row in zip(visit_ids, preds, rf_preds, adapted_test_df.itertuples(index=False)):
        level = pred.triage_level   # None = ABSTAIN
        dept  = pred.primary_department
        p_em  = pred.p_emergency

        # Text-model fallback: if abstained AND complaint text exists, predict level
        # from text model alone rather than leaving as ABSTAIN.
        if level is None:
            complaint = str(getattr(adapted_row, "complaint_text", "") or "").strip()
            if len(complaint) > 5:
                try:
                    txt_proba = _text_model.predict_proba(pd.DataFrame([{"complaint_text": complaint}]))[0]
                    txt_em    = _text_model.predict_emergency_proba(pd.DataFrame([{"complaint_text": complaint}]))[0]
                    # Use argmax (NOT cost matrix) — cost matrix always picks L2 when vitals missing
                    # because the 20× emergency penalty dominates. Argmax gives realistic L3/L4/L5
                    # for most patients; text em-probability still escalates true emergencies.
                    if float(txt_em) >= 0.40:
                        # Strong emergency signal in text → escalate
                        fallback_level = 2
                    else:
                        # Use multiclass argmax — classes are [1,2,3,4,5]
                        fallback_level = int(np.argmax(txt_proba)) + 1
                    level = fallback_level
                    p_em  = float(txt_em)
                    # Re-derive department: L1/L2 → Emergency/Casualty, else use dept model
                    if level in (1, 2):
                        dept = "Emergency/Casualty"
                    # else keep existing dept from pipeline (specialty prediction was still done)
                    fallback_count += 1
                except Exception:
                    pass   # keep as ABSTAIN on error

        # Validate department against allowed list
        if dept not in _DEPT_SET:
            dept = "General Medicine"   # safe fallback

        # Level: None → "ABSTAIN" (evaluate.py requires this exact string)
        if level is None:
            abstain_count += 1
            level_str = "ABSTAIN"
        else:
            level_str = str(int(level))

        # Red flags: validate against allowed set; semicolon-join or NaN
        valid_rfs = [rf for rf in rfs if rf in _RF_SET]
        if valid_rfs:
            rf_flagged += 1
            rf_str = ";".join(valid_rfs)
        else:
            rf_str = np.nan   # NaN = no red flags (matches sample_submission.csv)

        # Confidence: use p_emergency as proxy for L1/L2; else 1 - p_emergency
        conf = round(float(p_em if level in (1, 2) else 1.0 - p_em), 4)
        conf = max(0.0, min(1.0, conf))

        level_counts[level_str] += 1
        dept_counts[dept] += 1

        records.append({
            "visit_id":     vid,
            "triage_level": level_str,
            "department":   dept,
            "red_flags":    rf_str,
            "confidence":   conf,
        })

    sub_df = pd.DataFrame(records, columns=["visit_id", "triage_level", "department", "red_flags", "confidence"])

    # ── Step 5: Validate submission ─────────────────────────────────────────
    sep("STEP 5 — VALIDATE SUBMISSION")
    errors = []

    # visit_id completeness
    if len(sub_df) != len(test_df):
        errors.append(f"Row count mismatch: {len(sub_df)} vs {len(test_df)}")

    missing_vids = set(visit_ids) - set(sub_df["visit_id"])
    if missing_vids:
        errors.append(f"Missing visit_ids: {list(missing_vids)[:5]}")

    # department validation
    bad_depts = sub_df[sub_df["department"].notna() & ~sub_df["department"].isin(_DEPT_SET)]
    if len(bad_depts):
        errors.append(f"{len(bad_depts)} rows have invalid departments: {bad_depts['department'].unique()[:3].tolist()}")

    # red flag validation
    for _, r in sub_df.iterrows():
        rfs = _parse_flags(r["red_flags"])
        bad = [rf for rf in rfs if rf not in _RF_SET]
        if bad:
            errors.append(f"visit_id {r['visit_id']}: invalid RF codes: {bad}")
            break

    # confidence validation
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

    # ── Step 6: Print summary stats ─────────────────────────────────────────
    sep("STEP 6 — SUMMARY")
    total = len(sub_df)
    print(f"  Total predictions : {total:,}")
    print(f"  Abstain (ABSTAIN) : {abstain_count:,}  ({abstain_count/total*100:.1f}%)")
    print(f"  Text fallbacks    : {fallback_count:,}  ({fallback_count/total*100:.1f}%  — abstains resolved via text model)")
    print(f"  Patients flagged  : {rf_flagged:,}  ({rf_flagged/total*100:.1f}%)")
    print()
    print("  Level distribution:")
    for lvl in ["1", "2", "3", "4", "5", "ABSTAIN"]:
        cnt = level_counts.get(lvl, 0)
        if cnt:
            print(f"    L{lvl}: {cnt:5,}  ({cnt/total*100:.1f}%)")
    print()
    print("  Top departments:")
    for dept, cnt in sorted(dept_counts.items(), key=lambda x: -x[1])[:8]:
        print(f"    {dept:<35s}: {cnt:5,}  ({cnt/total*100:.1f}%)")

    # ── Step 7: Save ─────────────────────────────────────────────────────────
    sep("STEP 7 — SAVE")
    out_path = _SUB_DIR / "H1_predictions.csv"
    sub_df.to_csv(out_path, index=False)
    print(f"  ✓ Saved {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")

    # Save report
    elapsed = time.perf_counter() - t_start
    report = {
        "timestamp":         pd.Timestamp.utcnow().isoformat(),
        "n_predictions":     total,
        "abstain_count":     abstain_count,
        "abstain_rate":      round(abstain_count / total, 4),
        "rf_flagged":        rf_flagged,
        "rf_flagged_rate":   round(rf_flagged / total, 4),
        "level_distribution": level_counts,
        "dept_distribution":  dept_counts,
        "elapsed_sec":       round(elapsed, 2),
    }
    report_path = _RESULTS_DIR / "predict_phase2_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  ✓ Saved {report_path}")
    print(f"\n  Total elapsed: {elapsed:.1f}s")

    # ── Step 8: Score on train labels (to estimate submission score) ─────────
    sep("STEP 8 — SCORE ON TRAIN LABELS (OPTIONAL ESTIMATE)")
    # We can't score test labels, but we score on the val-equivalent
    # by running evaluate.py --submission against train_labels.
    # This tests the scorer format only; the score is NOT the test score.
    print("  Run the following to get your estimated score:")
    print()
    print("  .venv/bin/python evaluate.py \\")
    print("      --submission submission/H1_predictions.csv \\")
    print("      --truth data/phase2/train_labels.csv \\")
    print("      --out results/score_phase2.json")
    print()
    print("  ⚠  This scores predictions for 8,872 TEST patients against 21,455 TRAIN labels.")
    print("     Matched rows (~8,872) contribute; unmatched rows get worst-case score.")
    print("     True test score comes from organiser evaluation on hidden test labels.")
    print()
    print("═" * 70)
    print("  PREDICTION COMPLETE")
    print("═" * 70)


if __name__ == "__main__":
    main()
