#!/usr/bin/env python3
"""
scripts/recalibrate_tau.py
─────────────────────────────────────────────────────────────────────────────
Recalibrate the conformal emergency screen threshold τ WITHOUT retraining.

Steps:
  1. Reconstruct calib split (seed=42, same as retrain_phase2.py)
  2. Run existing fusion model → get p_em on calib split
  3. Compute τ at α = 0.01, 0.02, 0.03, 0.05 from calib emergency scores
  4. Print tradeoff table (recall/precision at each τ on calib split)
  5. For α=0.02 and α=0.03: run full train-set predictions in-memory,
     save CSVs, score with evaluate.py
  6. Pick best α → update conformal.joblib + conformal_threshold.json
  7. Regenerate test submission (H1_predictions.csv)
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triageguard.data_adapter import adapt_schema, extract_vitals_features, load_config
from triageguard.cleaning import clean_record
from triageguard.conformal import ConformalPredictor
from triageguard.fusion import StackerFusion, build_cost_matrix, select_min_cost_level
from triageguard.model_text import TextModel
from triageguard.model_vitals import VitalsModel
from triageguard.predict import TriagePipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

_ROOT      = Path(__file__).resolve().parent.parent
_MODEL_DIR = _ROOT / "models"
_PHASE2    = _ROOT / "data" / "phase2"
_SUB_DIR   = _ROOT / "submission"
_SUB_DIR.mkdir(exist_ok=True)

# ── constants ────────────────────────────────────────────────────────────────
SEED = 42
RED_FLAG_CODES = [
    "RF_CHEST_PAIN_CARDIAC","RF_STROKE_SIGNS","RF_RESP_DISTRESS","RF_ALTERED_SENSORIUM",
    "RF_SEIZURE","RF_SEVERE_BLEEDING","RF_PREGNANCY_BLEEDING","RF_ECLAMPSIA_SIGNS",
    "RF_REDUCED_FETAL_MOVEMENT","RF_POISONING","RF_SNAKEBITE","RF_SCORPION_STING",
    "RF_SEVERE_DEHYDRATION","RF_HIGH_FEVER_INFANT","RF_CHILD_DANGER_SIGNS","RF_SEPSIS_SIGNS",
    "RF_MENINGITIS_SIGNS","RF_DENGUE_WARNING","RF_SUICIDAL","RF_ACUTE_PSYCHOSIS_AGITATION",
    "RF_TRAUMA_MAJOR","RF_ANAPHYLAXIS","RF_DKA_SIGNS","RF_HYPOGLYCAEMIA","RF_HEAT_STROKE",
    "RF_ACUTE_ABDOMEN","RF_THUNDERCLAP_HEADACHE","RF_TESTICULAR_TORSION","RF_EYE_EMERGENCY",
    "RF_BURNS_MAJOR",
]
_RF_SET = set(RED_FLAG_CODES)
DEPARTMENTS = [
    "General Medicine","Paediatrics","Obstetrics & Gynaecology","General Surgery",
    "Orthopaedics","ENT","Ophthalmology","Dermatology","Psychiatry","Dental",
    "Cardiology","Pulmonology/Chest & TB","Emergency/Casualty",
]
_DEPT_SET = set(DEPARTMENTS)

def sep(title):
    print("\n" + "═"*70)
    print(f"  {title}")
    print("═"*70)

def _parse_flags(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return []
    s = str(x).strip()
    if not s or s.lower() == "nan": return []
    return [t.strip().upper() for t in s.replace(",",";").split(";") if t.strip()]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Reconstruct calib split & compute p_em
# ─────────────────────────────────────────────────────────────────────────────

sep("STEP 1 — RECONSTRUCT CALIB SPLIT & COMPUTE τ TRADEOFF")

print("  Loading Phase 2 data …")
intake = pd.read_csv(_PHASE2 / "intake_train.csv")
labels = pd.read_csv(_PHASE2 / "train_labels.csv")
merged = intake.merge(labels, on="visit_id", how="inner")
print(f"  Merged: {len(merged):,} rows")

# Exact same split as retrain_phase2.py
rest_df, train_df = train_test_split(
    merged, test_size=0.70, random_state=SEED, stratify=merged["triage_level"]
)
calib_df, val_df = train_test_split(
    rest_df, test_size=0.50, random_state=SEED, stratify=rest_df["triage_level"]
)
print(f"  Splits → train:{len(train_df):,}  calib:{len(calib_df):,}  val:{len(val_df):,}")

# Adapt + clean calib
print("  Adapting calib schema …")
calib_adapted = adapt_schema(calib_df.copy())

# Clean row-by-row (needed for vitals features)
cfg = load_config()
from triageguard.cleaning import clean_dataframe
calib_clean = clean_dataframe(calib_adapted, cfg)

# Restore label columns that cleaning might drop
for col in ["triage_level","true_emergency","visit_id","department","red_flags","assessable"]:
    if col in calib_df.columns and col not in calib_clean.columns:
        calib_clean[col] = calib_df[col].values

y_calib_em    = calib_clean["true_emergency"].values.astype(int)
y_calib_level = calib_clean["triage_level"].values.astype(int)

# Load existing models
print("  Loading models …")
model_vitals = VitalsModel.load(_MODEL_DIR / "model_vitals.joblib")
model_text   = TextModel.load(_MODEL_DIR / "model_text.joblib")
fusion       = StackerFusion.load(_MODEL_DIR / "fusion.joblib")
conformal    = ConformalPredictor.load(_MODEL_DIR / "conformal.joblib")

# Get calib predictions
X_calib_vit = extract_vitals_features(calib_clean)
complaint_col = calib_clean.get("complaint_clean", calib_clean.get("complaint_text",
                pd.Series([""] * len(calib_clean)))).fillna("")
p_calib_vit_em    = model_vitals.predict_emergency_proba(X_calib_vit)
p_calib_vit_multi = model_vitals.predict_proba(X_calib_vit)
p_calib_txt_em    = model_text.predict_emergency_proba(complaint_col)
p_calib_txt_multi = model_text.predict_proba(complaint_col)

fused_calib_multi, calib_em_fused = fusion.predict_fused(
    p_calib_vit_multi, p_calib_txt_multi, p_calib_vit_em, p_calib_txt_em, calib_clean
)

# Compute τ for multiple alphas including 0.03
em_mask = y_calib_em.astype(bool)
calib_em_scores = np.sort(calib_em_fused[em_mask])
n_em = len(calib_em_scores)
print(f"  Calib emergencies: {n_em}")

CANDIDATE_ALPHAS = [0.01, 0.02, 0.03, 0.05]
tau_map_full = {}
for alpha in CANDIDATE_ALPHAS:
    k = int(math.floor(alpha * (n_em + 1)))
    if k == 0:
        tau_map_full[alpha] = 0.0
    else:
        tau_map_full[alpha] = float(calib_em_scores[k - 1])

# Compute recall + precision on calib split for each τ
print()
print(f"  {'α':<7} {'τ':<12} {'calib_recall':>13} {'calib_precision':>16}")
print(f"  {'-'*7} {'-'*12} {'-'*13} {'-'*16}")
tradeoff_rows = []
for alpha in CANDIDATE_ALPHAS:
    tau = tau_map_full[alpha]
    screen = calib_em_fused >= tau
    tp = int(np.sum(screen & em_mask))
    fp = int(np.sum(screen & ~em_mask))
    fn = int(np.sum(~screen & em_mask))
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    marker = "  ← current" if alpha == conformal.state.operating_alpha else ""
    print(f"  {alpha:<7.3f} {tau:<12.6f} {recall*100:>12.1f}% {precision*100:>15.1f}%{marker}")
    tradeoff_rows.append({"alpha": alpha, "tau": tau, "recall": recall, "precision": precision})

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Helper: run prediction on a dataset with a given τ
# ─────────────────────────────────────────────────────────────────────────────

def build_rf_classifier(intake_df, labels_df):
    """Train the RedFlagClassifier on Phase 2 training data."""
    merged_rf = intake_df.merge(labels_df[["visit_id","red_flags"]], on="visit_id", how="inner")
    cc   = merged_rf.get("chief_complaint",   merged_rf.get("complaint_text",   pd.Series([""] * len(merged_rf)))).fillna("")
    cond = merged_rf.get("known_conditions",  pd.Series([""] * len(merged_rf))).fillna("")
    meds = merged_rf.get("current_medications", pd.Series([""] * len(merged_rf))).fillna("")
    texts = (cc + " " + cond + " " + meds).tolist()
    y_lists = merged_rf["red_flags"].apply(_parse_flags).tolist()
    n_tr = int(len(merged_rf) * 0.8)
    mlb = MultiLabelBinarizer(classes=RED_FLAG_CODES)
    mlb.fit(y_lists)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), max_features=60_000,
                          sublinear_tf=True, min_df=2)
    clf = OneVsRestClassifier(LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                                                  random_state=SEED, class_weight="balanced"), n_jobs=-1)
    Y_tr = mlb.transform(y_lists[:n_tr])
    X_tr = vec.fit_transform(texts[:n_tr])
    clf.fit(X_tr, Y_tr)
    # Per-label threshold tuning on held-out 20%
    X_val_rf = vec.transform(texts[n_tr:])
    Y_val_rf  = mlb.transform(y_lists[n_tr:])
    thresholds = {code: 0.3 for code in RED_FLAG_CODES}
    proba_val = clf.predict_proba(X_val_rf)
    for i, code in enumerate(RED_FLAG_CODES):
        y_col = Y_val_rf[:, i]
        if y_col.sum() == 0: continue
        best_f1, best_thr = 0.0, 0.3
        for thr in np.arange(0.05, 0.95, 0.05):
            preds = (proba_val[:, i] >= thr).astype(int)
            tp_ = int((preds * y_col).sum()); fp_ = int((preds * (1-y_col)).sum()); fn_ = int(((1-preds)*y_col).sum())
            p_ = tp_/(tp_+fp_+1e-9); r_ = tp_/(tp_+fn_+1e-9)
            f1_ = 2*p_*r_/(p_+r_+1e-9)
            if f1_ > best_f1: best_f1, best_thr = f1_, float(thr)
        thresholds[code] = best_thr
    return vec, clf, mlb, thresholds


def run_predictions(input_df, tau_value, pipeline_obj, text_model_obj, cost_mat,
                    rf_vec, rf_clf, rf_mlb, rf_thresholds, label=""):
    """Run full prediction on input_df with a specific τ, return DataFrame of records."""
    # Temporarily patch tau in the pipeline
    original_tau = pipeline_obj.conformal.state.operating_tau
    pipeline_obj.conformal.state.operating_tau = tau_value

    visit_ids = input_df["visit_id"].tolist()
    adapted = adapt_schema(input_df.copy())

    t0 = time.perf_counter()
    preds = pipeline_obj.predict_batch(adapted)
    elapsed = time.perf_counter() - t0
    print(f"  {label}: {len(preds):,} preds in {elapsed:.1f}s  (τ={tau_value:.6f})")

    # Red flags
    cc_col   = adapted.get("complaint_text", pd.Series([""] * len(adapted))).fillna("")
    cond_col = adapted.get("known_conditions", pd.Series([""] * len(adapted))).fillna("")
    meds_col = adapted.get("current_medications", pd.Series([""] * len(adapted))).fillna("")
    rf_texts = (cc_col + " " + cond_col + " " + meds_col).tolist()
    X_rf = rf_vec.transform(rf_texts)
    rf_proba = rf_clf.predict_proba(X_rf)
    rf_preds = []
    for row_p in rf_proba:
        codes = [code for i, code in enumerate(RED_FLAG_CODES) if row_p[i] >= rf_thresholds[code]]
        rf_preds.append(codes)

    # Assemble
    records = []
    level_counts = defaultdict(int)
    abstain_count = 0; fallback_count = 0

    for vid, pred, rfs, adapted_row in zip(visit_ids, preds, rf_preds, adapted.itertuples(index=False)):
        level = pred.triage_level
        dept  = pred.primary_department
        p_em  = pred.p_emergency

        # Text fallback for abstains
        if level is None:
            complaint = str(getattr(adapted_row, "complaint_text", "") or "").strip()
            if len(complaint) > 5:
                try:
                    txt_proba = text_model_obj.predict_proba(pd.DataFrame([{"complaint_text": complaint}]))[0]
                    txt_em    = text_model_obj.predict_emergency_proba(pd.DataFrame([{"complaint_text": complaint}]))[0]
                    level = 2 if float(txt_em) >= 0.40 else int(np.argmax(txt_proba)) + 1
                    p_em  = float(txt_em)
                    if level in (1, 2): dept = "Emergency/Casualty"
                    fallback_count += 1
                except: pass

        if dept not in _DEPT_SET: dept = "General Medicine"

        if level is None:
            abstain_count += 1; level_str = "ABSTAIN"
        else:
            level_str = str(int(level))

        valid_rfs = [rf for rf in rfs if rf in _RF_SET]
        rf_str = ";".join(valid_rfs) if valid_rfs else np.nan
        conf = round(float(p_em if level in (1, 2) else 1.0 - p_em), 4)
        conf = max(0.0, min(1.0, conf))

        level_counts[level_str] += 1
        records.append({"visit_id": vid, "triage_level": level_str,
                        "department": dept, "red_flags": rf_str, "confidence": conf})

    pipeline_obj.conformal.state.operating_tau = original_tau  # restore
    sub = pd.DataFrame(records, columns=["visit_id","triage_level","department","red_flags","confidence"])
    total = len(sub)
    l12 = sum(level_counts.get(x,0) for x in ["1","2"])
    print(f"    L1+L2: {l12:,} ({l12/total*100:.1f}%)  ABSTAIN: {abstain_count}  fallback: {fallback_count}")
    return sub


def score_submission(csv_path: Path, truth_path: Path) -> dict:
    """Run evaluate.py and return parsed JSON result."""
    out_path = csv_path.with_suffix(".score.json")
    result = subprocess.run(
        [sys.executable, "evaluate.py",
         "--submission", str(csv_path),
         "--truth", str(truth_path),
         "--out", str(out_path)],
        capture_output=False,
        text=True,
    )
    with open(out_path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Build RF classifier (once, shared across runs)
# ─────────────────────────────────────────────────────────────────────────────

sep("STEP 2 — BUILD RF CLASSIFIER & LOAD PIPELINE")
train_intake = pd.read_csv(_PHASE2 / "intake_train.csv")
train_labels = pd.read_csv(_PHASE2 / "train_labels.csv")

print("  Training RedFlagClassifier …")
rf_vec, rf_clf, rf_mlb, rf_thresholds = build_rf_classifier(train_intake, train_labels)
print("  Loading pipeline …")
pipeline = TriagePipeline.load(_MODEL_DIR)
text_model_loaded = TextModel.load(_MODEL_DIR / "model_text.joblib")
cost_mat = build_cost_matrix(pipeline.cfg)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Run train predictions for α=0.02 and α=0.03
# ─────────────────────────────────────────────────────────────────────────────

sep("STEP 3 — TRAIN-SET PREDICTIONS AT α=0.02 AND α=0.03")

truth_path = _PHASE2 / "train_labels.csv"
scores = {}

for alpha in [0.01, 0.02, 0.03]:
    tau = tau_map_full[alpha]
    csv_out = _SUB_DIR / f"H1_train_check_a{int(alpha*100):02d}.csv"

    print(f"\n  ─── α={alpha:.2f}  τ={tau:.6f} ───")
    sub_df = run_predictions(
        train_intake, tau, pipeline, text_model_loaded, cost_mat,
        rf_vec, rf_clf, rf_mlb, rf_thresholds, label=f"α={alpha}"
    )
    sub_df.to_csv(csv_out, index=False)
    print(f"  Saved → {csv_out.name}")

    print(f"  Scoring …")
    sc = score_submission(csv_out, truth_path)
    scores[alpha] = sc
    print(f"  primary_score={sc['primary_score']:.4f}  "
          f"recall={sc['emergency_recall']:.4f}  "
          f"safety={sc['normalised_safety_score']:.4f}  "
          f"dept={sc['department_accuracy']:.4f}  "
          f"rf_f1={sc['red_flag_micro_f1']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Print full evaluate.py output (already printed above inline)
# Now print comparison table
# ─────────────────────────────────────────────────────────────────────────────

sep("STEP 4 — FULL EVALUATE.PY OUTPUTS")
for alpha in [0.02, 0.03]:
    print(f"\n  ──── α={alpha:.2f} evaluate.py output ────")
    csv_out = _SUB_DIR / f"H1_train_check_a{int(alpha*100):02d}.csv"
    score_out = csv_out.with_suffix(".score.json")
    with open(score_out) as f:
        sc = json.load(f)
    total = sc.get("n_scored", 0)
    l12_count = sum(scores[alpha].get(f"n_{x}", 0) for x in [])  # from CSV
    sub_tmp = pd.read_csv(csv_out)
    l12_count = (sub_tmp["triage_level"].isin(["1","2"])).sum()
    print(f"  primary_score           : {sc['primary_score']:.6f}")
    print(f"  normalised_safety_score : {sc['normalised_safety_score']:.6f}")
    print(f"  emergency_recall        : {sc['emergency_recall']:.6f}  ({sc['n_scored'] - sc['n_missed_emergencies']}/{sc['n_true_emergencies']})")
    print(f"  department_accuracy     : {sc['department_accuracy']:.6f}")
    print(f"  red_flag_micro_f1       : {sc['red_flag_micro_f1']:.6f}")
    print(f"  n_abstain               : {sc['n_abstain']}")
    print(f"  n_over_triage           : {sc['n_over_triage']}")
    print(f"  n_under_triage          : {sc['n_under_triage_non_emergency']}")
    cost_b = sc.get("cost_breakdown", {})
    print(f"  cost breakdown → missed_em={cost_b.get('missed_emergencies',0):.0f}  "
          f"under={cost_b.get('under_triage_non_emergency',0):.0f}  "
          f"over={cost_b.get('over_triage',0):.0f}  "
          f"abstain={cost_b.get('abstain',0):.0f}")
    print(f"  L1+L2 predicted         : {l12_count:,}  ({l12_count/len(sub_tmp)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Comparison table
# ─────────────────────────────────────────────────────────────────────────────

sep("STEP 5 — COMPARISON TABLE")

BASELINE = {
    "primary_score": 0.4645, "normalised_safety_score": 0.420,
    "emergency_recall": 0.554, "department_accuracy": 0.649, "red_flag_micro_f1": 0.365
}

def pct_l12(alpha):
    csv_out = _SUB_DIR / f"H1_train_check_a{int(alpha*100):02d}.csv"
    sub_tmp = pd.read_csv(csv_out)
    n = (sub_tmp["triage_level"].isin(["1","2"])).sum()
    return f"{n/len(sub_tmp)*100:.1f}%"

def over_cost(alpha):
    return scores[alpha].get("cost_breakdown", {}).get("over_triage", 0)

print()
print("  ╔═══════════════════════════════════════════════════════════════════╗")
print("  ║         THRESHOLD COMPARISON — TRAIN SET                         ║")
print("  ╠═══════════════════════════════════════════════════════════════════╣")
print("  ║  Metric             │ Baseline │ α=0.01(now) │ α=0.02 │ α=0.03  ║")
print("  ╠═══════════════════════════════════════════════════════════════════╣")
for key, label in [
    ("primary_score", "primary_score     "),
    ("normalised_safety_score", "safety_score      "),
    ("emergency_recall", "emergency_recall  "),
    ("department_accuracy", "department_acc    "),
    ("red_flag_micro_f1", "red_flag_f1       "),
]:
    b = f"{BASELINE[key]:.4f}"
    v01 = f"{scores[0.01][key]:.4f}"
    v02 = f"{scores[0.02][key]:.4f}"
    v03 = f"{scores[0.03][key]:.4f}"
    print(f"  ║  {label}│ {b:>8} │   {v01:>8}  │ {v02:>6} │ {v03:>6}  ║")

print(f"  ║  {'L1+L2 predicted   '} │ {'—':>8} │   {pct_l12(0.01):>8}  │ {pct_l12(0.02):>6} │ {pct_l12(0.03):>6}  ║")
print(f"  ║  {'over-triage cost  '} │ {'—':>8} │   {over_cost(0.01):>8.0f}  │ {over_cost(0.02):>6.0f} │ {over_cost(0.03):>6.0f}  ║")
print("  ╚═══════════════════════════════════════════════════════════════════╝")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Pick winner, update conformal, regenerate test submission
# ─────────────────────────────────────────────────────────────────────────────

sep("STEP 6 — PICK WINNER & REGENERATE TEST SUBMISSION")

best_alpha = max([0.01, 0.02, 0.03], key=lambda a: scores[a]["primary_score"])
best_tau   = tau_map_full[best_alpha]
best_score = scores[best_alpha]["primary_score"]

print(f"  Best α = {best_alpha:.2f}  (τ = {best_tau:.6f}  primary_score = {best_score:.4f})")

# Update conformal.joblib
conformal.state.operating_alpha = float(best_alpha)
conformal.state.operating_tau   = float(best_tau)
# Add 0.03 to tau_map if not present
if 0.03 not in conformal.state.tau_map:
    conformal.state.tau_map[0.03] = float(tau_map_full[0.03])
conformal.save(_MODEL_DIR / "conformal.joblib")
print("  ✓ Updated models/conformal.joblib")

# Update conformal_threshold.json
threshold_data = {
    "conformal_alpha": best_alpha,
    "conformal_tau":   best_tau,
    "recall_floor":    0.9,
    "candidate_alphas": CANDIDATE_ALPHAS,
    "tau_map": {str(a): tau_map_full[a] for a in CANDIDATE_ALPHAS},
}
with open(_MODEL_DIR / "conformal_threshold.json", "w") as f:
    json.dump(threshold_data, f, indent=2)
print("  ✓ Updated models/conformal_threshold.json")

# Regenerate test submission
sep("STEP 7 — REGENERATE TEST SUBMISSION")
test_intake = pd.read_csv(_PHASE2 / "intake_test.csv")
test_visit_ids = test_intake["visit_id"].tolist()

print(f"  Running test predictions ({len(test_intake):,} rows) at best τ={best_tau:.6f} …")

# Reload pipeline with updated conformal
pipeline_final = TriagePipeline.load(_MODEL_DIR)
sub_test = run_predictions(
    test_intake, best_tau, pipeline_final, text_model_loaded, cost_mat,
    rf_vec, rf_clf, rf_mlb, rf_thresholds, label="test"
)
sub_test.to_csv(_SUB_DIR / "H1_predictions.csv", index=False)
total_test = len(sub_test)
l12_test = (sub_test["triage_level"].isin(["1","2"])).sum()
print(f"  ✓ Saved submission/H1_predictions.csv  ({total_test:,} rows)")
print(f"  L1+L2: {l12_test:,}  ({l12_test/total_test*100:.1f}%)")
print(f"  ABSTAIN: {(sub_test['triage_level']=='ABSTAIN').sum()}")

# Format validate
sep("STEP 8 — FORMAT VALIDATION")
result = subprocess.run(
    [sys.executable, "evaluate.py",
     "--submission", "submission/H1_predictions.csv",
     "--truth", str(truth_path)],
    capture_output=True, text=True
)
# Print only the non-score lines (format check only, not score — different visit_ids)
for line in result.stdout.split("\n") + result.stderr.split("\n"):
    if "FORMAT ERROR" in line or "Allowed" in line or "ERROR" in line:
        print(f"  ❌ {line}")
if result.returncode == 0:
    print("  ✓ No format errors")
elif result.returncode == 2:
    print("  ❌ FORMAT ERROR — check output above")
else:
    print("  ✓ Scorer ran (score meaningless — different visit_ids, expected)")

sep("DONE")
best_rec = scores[best_alpha]["emergency_recall"]
print(f"  Best α chosen: {best_alpha:.2f}  τ={best_tau:.6f}")
print(f"  Train primary_score: {best_score:.4f}")
print(f"  Emergency recall:    {best_rec:.4f}")

if best_score > 0.70:
    rec = "SUBMIT AS-IS"
elif best_score >= 0.55:
    rec = "SUBMIT AS-IS (good improvement over baseline)"
else:
    rec = "RETRAIN WITH α=0.02 URGENT"

print(f"\n  primary_score is {best_score:.2f} — recommend {rec}")
print()
