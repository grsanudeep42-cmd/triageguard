#!/usr/bin/env python3
"""
scripts/retrain_phase2.py
─────────────────────────────────────────────────────────────────────────────
Retrain all TriageGuard models on official Phase 2 data.

Architecture is UNCHANGED — only the training data changes.
  • model_vitals  → LightGBM multiclass + binary (monotone constraints)
  • model_text    → char_wb TF-IDF + LogisticRegression
  • model_dept    → char_wb TF-IDF + tabular features + LogisticRegression
  • fusion        → 5-fold OOF stacker + Platt calibration
  • conformal     → Mondrian emergency screen

Data:
  FULL_CODE/data/phase2/intake_train.csv   (21,455 visits)
  FULL_CODE/data/phase2/train_labels.csv   (21,455 labelled)

Split:  70% train / 15% calib / 15% val  — stratified on triage_level, seed=42

Usage:
  cd FULL_CODE && python scripts/retrain_phase2.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent   # FULL_CODE/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from triageguard.cleaning import clean_dataframe
from triageguard.conformal import ConformalPredictor
from triageguard.data_adapter import (
    adapt_schema,
    extract_red_flag_count,
    extract_vitals_features,
    load_config,
)
from triageguard.fusion import StackerFusion, build_cost_matrix, compute_ece
from triageguard.model_dept import DepartmentModel
from triageguard.model_text import TextModel
from triageguard.model_vitals import VitalsModel

SEED = 42
_DATA_DIR   = _ROOT / "data" / "phase2"
_MODEL_DIR  = _ROOT / "models"
_RESULTS_DIR = _ROOT / "results"

# Recall floor for conformal alpha selection (90% — realistic for new data)
RECALL_FLOOR = 0.90


# ─────────────────────────────────────────────────────────────────────────────
def sep(title: str) -> None:
    bar = "═" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


# ─────────────────────────────────────────────────────────────────────────────
def load_and_prepare() -> pd.DataFrame:
    """Load, adapt, merge, and clean the phase 2 dataset."""
    sep("STEP 0 — LOAD & INSPECT PHASE 2 DATA")

    intake = pd.read_csv(_DATA_DIR / "intake_train.csv")
    labels = pd.read_csv(_DATA_DIR / "train_labels.csv")

    print(f"intake_train.csv : {len(intake):,} rows × {len(intake.columns)} cols")
    print(f"train_labels.csv : {len(labels):,} rows × {len(labels.columns)} cols")
    print(f"Intake columns   : {intake.columns.tolist()}")
    print(f"Labels columns   : {labels.columns.tolist()}")

    # Adapt Phase 2 column names → canonical TriageGuard names
    # (visit_id→patient_id, age_years→age, pulse_bpm→pulse, etc.)
    intake_adapted = adapt_schema(intake)
    print(f"\nAfter adapt_schema: {intake_adapted.columns.tolist()}")

    # Merge on patient_id (after alias: visit_id → patient_id)
    id_col = "patient_id" if "patient_id" in intake_adapted.columns else "visit_id"
    lbl_id_col = "patient_id" if "patient_id" in labels.columns else "visit_id"

    # Ensure labels use same id name
    if lbl_id_col != id_col:
        labels = labels.rename(columns={lbl_id_col: id_col})

    merged = intake_adapted.merge(
        labels[[id_col, "triage_level", "department", "true_emergency",
                "red_flags", "assessable"]],
        on=id_col,
        how="inner",
    )
    print(f"\nMerged rows : {len(merged):,}")

    # Triage level distribution
    print("\ntriage_level distribution:")
    for lvl in [1, 2, 3, 4, 5]:
        n = (merged["triage_level"] == lvl).sum()
        print(f"  L{lvl}: {n:5d} ({n/len(merged)*100:.2f}%)")

    print(f"\nassessable == 0 (ABSTAIN ground truth): {(merged['assessable'] == 0).sum()}")
    print(f"true_emergency: {merged['true_emergency'].value_counts().to_dict()}")

    # Check department alignment
    allowed_depts = pd.read_csv(_DATA_DIR / "departments.csv")["department"].tolist()
    label_depts = merged["department"].dropna().unique()
    mismatches = [d for d in label_depts if d not in allowed_depts]
    if mismatches:
        print(f"\n⚠ DEPARTMENT MISMATCH (not in departments.csv): {mismatches}")
    else:
        print(f"\n✓ All {len(label_depts)} label departments are in the allowed list.")

    return merged, allowed_depts


# ─────────────────────────────────────────────────────────────────────────────
def split_data(merged: pd.DataFrame):
    """Stratified 70/15/15 split on triage_level."""
    sep("STEP 1 — STRATIFIED TRAIN / CALIB / VAL SPLIT (70/15/15)")

    # First split: 70% train, 30% temp
    train_df, temp_df = train_test_split(
        merged,
        test_size=0.30,
        random_state=SEED,
        stratify=merged["triage_level"],
    )
    # Second split: 50/50 on the 30% → 15% calib, 15% val
    calib_df, val_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["triage_level"],
    )

    for name, df in [("train", train_df), ("calib", calib_df), ("val", val_df)]:
        em = df["true_emergency"].sum()
        print(f"  {name:5s} : {len(df):6,} rows  |  emergencies: {int(em):4d}"
              f"  ({em/len(df)*100:.1f}%)")

    return (
        train_df.reset_index(drop=True),
        calib_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
def clean_split(df: pd.DataFrame, cfg: dict, name: str) -> pd.DataFrame:
    """Run clean_dataframe on a split, add complaint_clean + n_red_flags."""
    print(f"  Cleaning {name} ({len(df):,} rows) …", end="  ", flush=True)
    cleaned = clean_dataframe(df, cfg)
    cleaned["n_red_flags"] = extract_red_flag_count(cleaned["complaint_clean"])
    # Preserve label columns if lost in clean_dataframe
    for col in ["triage_level", "true_emergency", "department", "assessable"]:
        if col not in cleaned.columns and col in df.columns:
            cleaned[col] = df[col].values
    print(f"done ({len(cleaned):,} rows)")
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config()

    # ── Step 0: Load & merge ─────────────────────────────────────────────────
    merged, allowed_depts = load_and_prepare()

    # ── Step 1: Split ────────────────────────────────────────────────────────
    train_raw, calib_raw, val_raw = split_data(merged)

    # ── Step 2: Clean all splits ─────────────────────────────────────────────
    sep("STEP 2 — CLEAN ALL SPLITS")
    train_df = clean_split(train_raw, cfg, "train")
    calib_df = clean_split(calib_raw, cfg, "calib")
    val_df   = clean_split(val_raw,   cfg, "val")

    # Use triage_level as y_level, true_emergency as y_is_emergency
    y_train_level = train_df["triage_level"].values.astype(int)
    y_train_em    = train_df["true_emergency"].values.astype(int)
    y_calib_level = calib_df["triage_level"].values.astype(int)
    y_calib_em    = calib_df["true_emergency"].values.astype(int)
    y_val_level   = val_df["triage_level"].values.astype(int)
    y_val_em      = val_df["true_emergency"].values.astype(int)

    # ── Step 3: Train base models ─────────────────────────────────────────────
    sep("STEP 3 — TRAIN BASE MODELS ON TRAIN SPLIT")

    # 3a. Vitals model
    print("Fitting model_vitals (LightGBM, monotone constraints) …")
    X_train_vit = extract_vitals_features(train_df)
    model_vitals = VitalsModel(random_state=SEED)
    model_vitals.fit(X_train_vit, y_train_level, y_train_em)
    model_vitals.save(_MODEL_DIR / "model_vitals.joblib")
    print("  ✓ Saved model_vitals.joblib")

    # 3b. Text model
    print("Fitting model_text (IndicBERT embeddings + LightGBM) …")
    id_col = "patient_id" if "patient_id" in train_df.columns else "visit_id"
    train_texts = pd.Series(
        train_df.get("complaint_clean", train_df.get("complaint_text", "")).values,
        index=train_df[id_col].astype(str).values,
    )
    calib_texts = pd.Series(
        calib_df.get("complaint_clean", calib_df.get("complaint_text", "")).values,
        index=calib_df[id_col].astype(str).values,
    )
    val_texts = pd.Series(
        val_df.get("complaint_clean", val_df.get("complaint_text", "")).values,
        index=val_df[id_col].astype(str).values,
    )

    model_text = TextModel(random_state=SEED)
    model_text.fit(
        train_texts,
        y_train_level,
        y_train_em,
    )
    model_text.save(_MODEL_DIR / "model_text.joblib")
    print("  ✓ Saved model_text.joblib")

    # 3c. Department model — verify department labels
    print("Fitting model_dept (TF-IDF + tabular, 13 departments) …")
    dept_labels = train_df["department"].fillna("General Medicine")
    mismatched = dept_labels[~dept_labels.isin(allowed_depts)]
    if len(mismatched) > 0:
        print(f"  ⚠ {len(mismatched)} rows have department not in allowed list:")
        print(f"    {mismatched.value_counts().to_dict()}")
    model_dept = DepartmentModel(random_state=SEED)
    model_dept.fit(train_df, dept_labels)
    model_dept.save(_MODEL_DIR / "model_dept.joblib")
    print("  ✓ Saved model_dept.joblib")

    # ── Step 4: Calib-split predictions for fusion ────────────────────────────
    sep("STEP 4 — 5-FOLD OOF FUSION & PLATT CALIBRATION")

    # AUC gate on calib split (use calib as a proxy for val here)
    X_calib_vit = extract_vitals_features(calib_df)
    p_calib_vit_em  = model_vitals.predict_emergency_proba(X_calib_vit)
    p_calib_txt_em  = model_text.predict_emergency_proba(calib_texts)
    auc_vitals = roc_auc_score(y_calib_em, p_calib_vit_em)
    auc_text   = roc_auc_score(y_calib_em, p_calib_txt_em)
    print(f"  model_vitals calib AUC : {auc_vitals:.4f}")
    print(f"  model_text   calib AUC : {auc_text:.4f}")
    if auc_vitals > 0.99:
        print(f"  ⚠ AUC {auc_vitals:.4f} > 0.99 — check for leakage!")

    # 5-fold OOF fusion on train split
    print("Generating 5-fold OOF stacker features on train …")
    fusion = StackerFusion(random_state=SEED)
    fusion.fit_oof(
        train_df,
        y_level=y_train_level,
        y_is_emergency=y_train_em,
        model_vitals=model_vitals,
        model_text=model_text,
        n_splits=5,
    )
    print("  ✓ OOF stacker fit.")

    # Val-split predictions for Platt calibration
    X_val_vit = extract_vitals_features(val_df)
    p_val_vit_multi = model_vitals.predict_proba(X_val_vit)
    p_val_vit_em    = model_vitals.predict_emergency_proba(X_val_vit)
    p_val_txt_multi = model_text.predict_proba(val_texts)
    p_val_txt_em = model_text.predict_emergency_proba(val_texts)

    X_val_stack = fusion.build_stack_features(
        p_val_vit_multi, p_val_txt_multi, p_val_vit_em, p_val_txt_em, val_df
    )
    raw_val_em = fusion.stacker_binary.predict_proba(X_val_stack)[:, 1]

    # Platt calibration on val split
    ece_before, ece_after = fusion.calibrate(raw_val_em, y_val_em)
    print(f"  ECE before calibration : {ece_before:.4f}")
    print(f"  ECE after calibration  : {ece_after:.4f}")

    fused_val_multi, calib_val_em = fusion.predict_fused(
        p_val_vit_multi, p_val_txt_multi, p_val_vit_em, p_val_txt_em, val_df
    )
    auc_fused = roc_auc_score(y_val_em, calib_val_em)
    print(f"  Fused stacker val AUC  : {auc_fused:.4f}")

    fusion.save(_MODEL_DIR / "fusion.joblib")
    print("  ✓ Saved fusion.joblib")

    # ── Step 5: Conformal calibration on calib split ──────────────────────────
    sep("STEP 5 — CONFORMAL CALIBRATION (calib split) + ALPHA SELECTION")

    p_calib_vit_multi = model_vitals.predict_proba(X_calib_vit)
    p_calib_txt_multi = model_text.predict_proba(calib_texts)
    fused_calib_multi, calib_em_cal = fusion.predict_fused(
        p_calib_vit_multi, p_calib_txt_multi, p_calib_vit_em, p_calib_txt_em, calib_df
    )

    conformal = ConformalPredictor(
        candidate_alphas=[0.005, 0.01, 0.02, 0.05],
        level_alpha=0.10,
        recall_floor_on_val=RECALL_FLOOR,
    )
    conformal.calibrate_on_calib(
        p_em_calib=calib_em_cal,
        p_multi_calib=fused_calib_multi,
        y_is_emergency_calib=y_calib_em,
        y_level_calib=y_calib_level,
    )

    cost_matrix = build_cost_matrix(cfg)
    best_alpha, best_tau, val_tradeoff = conformal.select_operating_alpha_on_val(
        p_em_val=calib_val_em,
        p_multi_val=fused_val_multi,
        y_is_emergency_val=y_val_em,
        y_level_val=y_val_level,
        cost_matrix=cost_matrix,
    )

    print("\nConformal alpha trade-off table:")
    print(val_tradeoff[["alpha", "tau", "val_recall", "val_escalation_rate",
                         "total_val_cost"]].to_string(index=False))
    print(f"\n✓ Operating alpha* = {best_alpha}  →  τ = {best_tau:.4f}")

    conformal.save(_MODEL_DIR / "conformal.joblib")
    # Also save τ to JSON for downstream reference
    tau_json = {
        "conformal_alpha": float(best_alpha),
        "conformal_tau":   float(best_tau),
        "recall_floor":    RECALL_FLOOR,
        "candidate_alphas": [0.005, 0.01, 0.02, 0.05],
        "tau_map":         {str(k): (v if v is not None else None)
                            for k, v in conformal.state.tau_map.items()},
    }
    tau_path = _MODEL_DIR / "conformal_threshold.json"
    with open(tau_path, "w", encoding="utf-8") as f:
        json.dump(tau_json, f, indent=2)
    print(f"  ✓ Saved conformal.joblib and conformal_threshold.json")

    # ── Step 6: Evaluate on val split ─────────────────────────────────────────
    sep("STEP 6 — EVALUATE ON VAL SPLIT")

    # Emergency recall (using conformal screen at τ)
    screen_fires = calib_val_em >= best_tau
    em_mask_val  = y_val_em.astype(bool)
    em_recall    = np.sum(screen_fires & em_mask_val) / max(1, np.sum(em_mask_val))

    # Exact level prediction via cost-minimisation
    pred_levels = []
    abstained   = []
    for i in range(len(val_df)):
        p5 = fused_val_multi[i]
        if screen_fires[i]:
            lvl = 2   # Emergency → force to L2 (most conservative safe choice)
        else:
            # Level set check
            level_set = conformal.predict_level_set(p5)
            if len(level_set) > 3:          # too uncertain → abstain
                abstained.append(i)
                lvl = None
            else:
                from triageguard.fusion import select_min_cost_level
                lvl = select_min_cost_level(p5, allowed_levels=level_set,
                                            cost_matrix=cost_matrix)
        pred_levels.append(lvl)

    abstain_mask  = np.array([l is None for l in pred_levels])
    abstain_rate  = abstain_mask.mean()
    pred_levels_a = np.array([l if l is not None else 4 for l in pred_levels])  # fill for metrics
    level_acc     = float(np.mean(pred_levels_a[~abstain_mask] == y_val_level[~abstain_mask]))

    # Department accuracy
    val_dept_preds, _ = model_dept.route_departments(val_df, pred_levels_a.tolist())
    true_depts   = val_df["department"].fillna("General Medicine").tolist()
    dept_acc     = float(np.mean([p == t for p, t in zip(val_dept_preds, true_depts)]))

    # ECE
    ece_val = compute_ece(calib_val_em, y_val_em)

    # Print summary table
    W = 56
    sep_line = "═" * W
    print(f"\n╔{sep_line}╗")
    print(f"║{'PHASE 2 RETRAIN VALIDATION RESULTS':^{W}}║")
    print(f"╠{sep_line}╣")
    print(f"║{'  Metric':<34}│{'  Value':<{W-34}}║")
    print(f"╠{sep_line}╣")
    print(f"║  {'Emergency recall (L1+L2)':<32}│  {em_recall*100:5.1f}%{'':<{W-34-7}}║")
    print(f"║  {'Exact level accuracy':<32}│  {level_acc*100:5.1f}%{'':<{W-34-7}}║")
    print(f"║  {'Department accuracy':<32}│  {dept_acc*100:5.1f}%{'':<{W-34-7}}║")
    print(f"║  {'Abstain rate':<32}│  {abstain_rate*100:5.1f}%{'':<{W-34-7}}║")
    print(f"║  {'ECE after calibration':<32}│  {ece_val:.4f}{'':<{W-34-6}}║")
    print(f"║  {'Conformal threshold τ':<32}│  {best_tau:.4f}{'':<{W-34-6}}║")
    print(f"╚{sep_line}╝")

    # Per-level breakdown (non-abstained)
    print("\nPer-level breakdown (val split, non-abstained rows):")
    non_abs_mask = ~abstain_mask
    if non_abs_mask.sum() > 0:
        report = classification_report(
            y_val_level[non_abs_mask],
            pred_levels_a[non_abs_mask],
            labels=[1, 2, 3, 4, 5],
            target_names=["L1", "L2", "L3", "L4", "L5"],
            zero_division=0,
        )
        print(report)

    # Warn if targets not met
    if em_recall < 0.85:
        print(f"⚠ WARNING: Emergency recall {em_recall:.1%} < 85% target!")
        print("  Try re-running — or lower RECALL_FLOOR to 0.85 and set alpha=0.005")
    if dept_acc < 0.80:
        print(f"⚠ WARNING: Department accuracy {dept_acc:.1%} < 80% target")
    if abstain_rate > 0.25:
        print(f"⚠ WARNING: Abstain rate {abstain_rate:.1%} > 25% — model is over-abstaining")

    # ── Step 7: Save training report ──────────────────────────────────────────
    sep("STEP 7 — SAVE TRAINING REPORT")

    level_dist_train = {
        str(lvl): int((y_train_level == lvl).sum()) for lvl in [1, 2, 3, 4, 5]
    }

    report_data = {
        "timestamp":               datetime.now(timezone.utc).isoformat(),
        "data_source":             "data/phase2/intake_train.csv + train_labels.csv",
        "n_total":                 len(merged),
        "n_train":                 len(train_df),
        "n_calib":                 len(calib_df),
        "n_val":                   len(val_df),
        "seed":                    SEED,
        "emergency_recall_val":    round(float(em_recall), 4),
        "exact_level_accuracy_val": round(float(level_acc), 4),
        "department_accuracy_val": round(float(dept_acc), 4),
        "abstain_rate_val":        round(float(abstain_rate), 4),
        "ece":                     round(float(ece_val), 6),
        "auc_vitals_calib":        round(float(auc_vitals), 4),
        "auc_text_calib":          round(float(auc_text), 4),
        "auc_fused_val":           round(float(auc_fused), 4),
        "ece_before_calibration":  round(float(ece_before), 6),
        "ece_after_calibration":   round(float(ece_after), 6),
        "conformal_alpha":         float(best_alpha),
        "conformal_tau":           float(best_tau),
        "recall_floor_used":       RECALL_FLOOR,
        "level_distribution_train": level_dist_train,
        "models_saved": [
            "model_vitals.joblib",
            "model_text.joblib",
            "model_dept.joblib",
            "fusion.joblib",
            "conformal.joblib",
        ],
    }

    report_path = _RESULTS_DIR / "retrain_phase2_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"✓ Training report saved to {report_path}")

    # Model sizes
    print("\nModel files on disk:")
    for name in ["model_vitals.joblib", "model_text.joblib", "model_dept.joblib",
                 "fusion.joblib", "conformal.joblib"]:
        p = _MODEL_DIR / name
        if p.exists():
            size_kb = p.stat().st_size / 1024
            print(f"  {name:<28}  {size_kb:8.1f} KB")

    sep("RETRAIN COMPLETE")
    print(f"  Emergency recall : {em_recall*100:.1f}%")
    print(f"  Dept accuracy    : {dept_acc*100:.1f}%")
    print(f"  Abstain rate     : {abstain_rate*100:.1f}%")
    print(f"  τ (conformal)    : {best_tau:.4f}")


if __name__ == "__main__":
    main()
