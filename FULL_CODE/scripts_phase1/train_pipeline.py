#!/usr/bin/env python3
"""
scripts/train_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Train all TriageGuard models using strict data discipline:
• train (60%): fits model_vitals, model_text, model_dept, and generates 5-fold OOF features for stacker.
• val (10%): evaluates base-model AUC gates, tunes probability calibration (Platt scaling), and picks operating alpha.
• calib (15%): calibrates Mondrian conformal thresholds tau_alpha and level sets.
• test (15%): NEVER touched here.

GATES
-----
• Halts if model_vitals AUC > 0.99 or model_text AUC > 0.99 on val.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from triageguard.cleaning import clean_dataframe
from triageguard.conformal import ConformalPredictor
from triageguard.data_adapter import (
    extract_red_flag_count,
    extract_vitals_features,
    load_config,
    load_data,
)
from triageguard.fusion import StackerFusion, build_cost_matrix
from triageguard.model_dept import DepartmentModel
from triageguard.model_text import TextModel
from triageguard.model_vitals import VitalsModel

_MODEL_DIR = _ROOT / "models"


def sep(title: str) -> None:
    bar = "═" * 78
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def run_training() -> None:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()

    sep("STEP 1: LOADING & CLEANING DATA SPLITS")
    df_all = load_data(cfg, include_labels=True)

    # Clean the dataset using Phase 1 clean_dataframe
    print(f"Cleaning {len(df_all):,} intake records...")
    df_all_cleaned = clean_dataframe(df_all, cfg)

    # Add red-flag count
    df_all_cleaned["n_red_flags"] = extract_red_flag_count(df_all_cleaned["complaint_clean"])

    # Split into train, val, calib, test
    train_df = df_all_cleaned[df_all_cleaned["split"] == "train"].copy().reset_index(drop=True)
    val_df   = df_all_cleaned[df_all_cleaned["split"] == "val"].copy().reset_index(drop=True)
    calib_df = df_all_cleaned[df_all_cleaned["split"] == "calib"].copy().reset_index(drop=True)
    test_df  = df_all_cleaned[df_all_cleaned["split"] == "test"].copy().reset_index(drop=True)

    print(f"Train split : {len(train_df):,} rows ({train_df['is_emergency'].sum()} emergencies)")
    print(f"Val split   : {len(val_df):,} rows ({val_df['is_emergency'].sum()} emergencies)")
    print(f"Calib split : {len(calib_df):,} rows ({calib_df['is_emergency'].sum()} emergencies)")
    print(f"Test split  : {len(test_df):,} rows (UNTOUCHED)")

    # ── STEP 2: FIT BASE MODELS ON TRAIN ──────────────────────────────────────
    sep("STEP 2: TRAINING BASE MODELS ON TRAIN SPLIT")

    # 1. Vitals model (TRULY VITALS-ONLY)
    print("Fitting model_vitals (LightGBM with monotone constraints)...")
    X_train_vit = extract_vitals_features(train_df)
    model_vitals = VitalsModel(random_state=42)
    model_vitals.fit(X_train_vit, train_df["clean_triage_level"], train_df["is_emergency"])
    model_vitals.save(_MODEL_DIR / "model_vitals.joblib")
    print("✓ Saved model_vitals.joblib")

    # 2. Text model (char_wb TF-IDF + LogisticRegression)
    print("Fitting model_text (char_wb TF-IDF + LogisticRegression)...")
    model_text = TextModel(random_state=42)
    model_text.fit(train_df["complaint_clean"], train_df["clean_triage_level"], train_df["is_emergency"])
    model_text.save(_MODEL_DIR / "model_text.joblib")
    print("✓ Saved model_text.joblib")

    # 3. Department router
    print("Fitting model_dept (12-department router)...")
    model_dept = DepartmentModel(random_state=42)
    model_dept.fit(train_df, train_df["department"])
    model_dept.save(_MODEL_DIR / "model_dept.joblib")
    print("✓ Saved model_dept.joblib")

    # ── STEP 3: BASE-MODEL AUC GATES ON VAL ───────────────────────────────────
    sep("STEP 3: EVALUATING BASE-MODEL AUC GATES ON VAL SPLIT")
    X_val_vit = extract_vitals_features(val_df)
    p_val_vit_em = model_vitals.predict_emergency_proba(X_val_vit)
    p_val_txt_em = model_text.predict_emergency_proba(val_df["complaint_clean"])

    y_val_em = val_df["is_emergency"].values.astype(int)
    auc_vitals = roc_auc_score(y_val_em, p_val_vit_em)
    auc_text   = roc_auc_score(y_val_em, p_val_txt_em)

    print(f"  model_vitals Val AUC : {auc_vitals:.4f}")
    print(f"  model_text   Val AUC : {auc_text:.4f}")

    if auc_vitals > 0.99:
        print(f"✗ GATE FAILURE: model_vitals Val AUC {auc_vitals:.4f} > 0.99 indicates leakage!")
        sys.exit(1)
    if auc_text > 0.99:
        print(f"✗ GATE FAILURE: model_text Val AUC {auc_text:.4f} > 0.99 indicates leakage!")
        sys.exit(1)
    print("✓ PASS: Both base-model AUCs are strictly <= 0.99.")

    # ── STEP 4: 5-FOLD OOF FUSION & PLATT CALIBRATION ─────────────────────────
    sep("STEP 4: 5-FOLD OOF FUSION & PLATT CALIBRATION")
    print("Generating 5-fold cross-validation out-of-fold stacker features...")
    fusion = StackerFusion(random_state=42)
    fusion.fit_oof(
        train_df,
        y_level=train_df["clean_triage_level"].values,
        y_is_emergency=train_df["is_emergency"].values,
        model_vitals=model_vitals,
        model_text=model_text,
        n_splits=5,
    )
    print("✓ Stacker fit on OOF train predictions.")

    # Compute raw predictions on val
    p_val_vit_multi = model_vitals.predict_proba(X_val_vit)
    p_val_txt_multi = model_text.predict_proba(val_df["complaint_clean"])

    # Raw stacker emergency prob on val
    X_val_stack = fusion.build_stack_features(
        p_val_vit_multi, p_val_txt_multi, p_val_vit_em, p_val_txt_em, val_df
    )
    raw_val_em = fusion.stacker_binary.predict_proba(X_val_stack)[:, 1]

    # Platt calibration on val
    ece_before, ece_after = fusion.calibrate(raw_val_em, y_val_em)
    print(f"  Platt Calibration on Val Split:")
    print(f"    ECE Before Calibration : {ece_before:.4f}")
    print(f"    ECE After Calibration  : {ece_after:.4f}")

    fused_val_multi, calib_val_em = fusion.predict_fused(
        p_val_vit_multi, p_val_txt_multi, p_val_vit_em, p_val_txt_em, val_df
    )
    auc_fused = roc_auc_score(y_val_em, calib_val_em)
    print(f"  Fused Stacker Val AUC    : {auc_fused:.4f} (Reported)")

    fusion.save(_MODEL_DIR / "fusion.joblib")
    print("✓ Saved fusion.joblib")

    # ── STEP 5: CONFORMAL CALIBRATION ON CALIB & OPERATING ALPHA ON VAL ───────
    sep("STEP 5: CONFORMAL CALIBRATION & OPERATING ALPHA SELECTION")
    print("Evaluating calibration split for Mondrian emergency screen & level sets...")
    X_calib_vit = extract_vitals_features(calib_df)
    p_calib_vit_multi = model_vitals.predict_proba(X_calib_vit)
    p_calib_txt_multi = model_text.predict_proba(calib_df["complaint_clean"])
    p_calib_vit_em    = model_vitals.predict_emergency_proba(X_calib_vit)
    p_calib_txt_em    = model_text.predict_emergency_proba(calib_df["complaint_clean"])

    fused_calib_multi, calib_em = fusion.predict_fused(
        p_calib_vit_multi, p_calib_txt_multi, p_calib_vit_em, p_calib_txt_em, calib_df
    )

    conformal = ConformalPredictor(
        candidate_alphas=cfg.get("conformal", {}).get("candidate_alphas", [0.005, 0.01, 0.02, 0.05]),
        level_alpha=cfg.get("conformal", {}).get("level_set_alpha", 0.10),
        recall_floor_on_val=cfg.get("conformal", {}).get("recall_floor_on_val", 0.95),
    )
    conformal.calibrate_on_calib(
        p_em_calib=calib_em,
        p_multi_calib=fused_calib_multi,
        y_is_emergency_calib=calib_df["is_emergency"].values,
        y_level_calib=calib_df["clean_triage_level"].values,
    )

    cost_matrix = build_cost_matrix(cfg)
    best_alpha, best_tau, val_tradeoff = conformal.select_operating_alpha_on_val(
        p_em_val=calib_val_em,
        p_multi_val=fused_val_multi,
        y_is_emergency_val=y_val_em,
        y_level_val=val_df["clean_triage_level"].values,
        cost_matrix=cost_matrix,
    )

    print("\nMondrian Conformal Screening Trade-Off Table on VAL Split:")
    print(val_tradeoff.to_string(index=False))

    print(f"\n✓ Operating Alpha Selected on VAL: alpha* = {best_alpha} -> tau = {best_tau:.4f}")
    conformal.save(_MODEL_DIR / "conformal.joblib")
    print("✓ Saved conformal.joblib")

    sep("TRAINING PIPELINE COMPLETE — READY FOR TEST EVALUATION")


if __name__ == "__main__":
    run_training()
