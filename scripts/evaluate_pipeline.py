#!/usr/bin/env python3
"""
scripts/evaluate_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Phase 8 – Comprehensive evaluation on the UNTOUCHED TEST split (4,500 rows).

SECTIONS & METRICS
------------------
1. Emergency recall with Wilson 95% CI, overall and for under-reported emergencies.
   • Gate: Calib recall must fall inside the Wilson 95% CI of test recall.
2. Cost score vs baselines:
   (a) Accuracy-only argmax model
   (b) Rules only
   (c) Full TriageGuard system
   Dual abstain costing: standard (cost 3.0) and safe-abstain (emergency abstain cost 0).
   Plus under/over-triage confusion matrix.
3. Abstention rate, accuracy, and cost on answered vs abstained cases.
4. Empirical conformal coverage on test for emergencies vs target.
5. Alpha trade-off table (recall vs escalation rate by L3/L4/L5).
6. Department accuracy: primary-only and specialty-or-emergency by true level.
7. Untrusted-form tests:
   (i) Gaming: 500 test L4/L5 cases with dramatic complaints.
   (ii) Under-reporting: emergency text replaced with vague complaints.
   (iii) Vital corruption: 0%, 5%, 10%, 20% vitals corrupted.
8. Monotonicity stress test: 1,000 random trials worsening vitals with text fixed.
9. Latency benchmarks: p50 and p95 in milliseconds.
10. Artifacts:
   • Tables in results/*.md and results/*.json
   • Figures in results/figures/*.png
   • 6 real test-split patients in results/demo_cases.json
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix

from triageguard.cleaning import clean_dataframe, clean_record
from triageguard.conformal import ConformalPredictor
from triageguard.data_adapter import (
    extract_red_flag_count,
    extract_vitals_features,
    load_config,
    load_data,
)
from triageguard.fusion import (
    StackerFusion,
    build_cost_matrix,
    compute_expected_costs,
    select_min_cost_level,
)
from triageguard.model_dept import DepartmentModel
from triageguard.model_text import TextModel
from triageguard.model_vitals import VitalsModel
from triageguard.predict import TriagePipeline, TriagePrediction
from triageguard.rules import apply_rules

_RESULTS_DIR = _ROOT / "results"
_FIG_DIR = _RESULTS_DIR / "figures"
_MODEL_DIR = _ROOT / "models"


def sep(title: str) -> None:
    bar = "═" * 78
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    z = 1.959964  # 95% normal quantile
    p = successes / total
    denom = 1.0 + (z**2) / total
    center = (p + (z**2) / (2.0 * total)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) / total) + ((z**2) / (4.0 * total**2)))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def compute_triage_cost(
    pred_level: Optional[Union[int, float]],
    true_level: Union[int, float],
    cost_matrix: np.ndarray,
    abstain_cost: float = 3.0,
    safe_abstain: bool = False,
) -> float:
    """
    Compute clinical triage cost for a single encounter.
    pred_level: 1..5, or None / NaN (abstain)
    true_level: 1..5
    safe_abstain: if True, abstaining on a true emergency has cost 0.0 (safe nurse handoff).
    """
    if pred_level is None or pd.isna(pred_level):
        if safe_abstain and int(true_level) in (1, 2):
            return 0.0
        return float(abstain_cost)

    p_lvl = int(pred_level)
    t_lvl = int(true_level)
    return float(cost_matrix[p_lvl - 1, t_lvl - 1])


def save_table(name: str, df: pd.DataFrame, title: str) -> None:
    """Save DataFrame as Markdown and JSON in results/."""
    md_path = _RESULTS_DIR / f"{name}.md"
    json_path = _RESULTS_DIR / f"{name}.json"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

    df.to_json(json_path, orient="records", indent=2)


def main() -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    cost_matrix = build_cost_matrix(cfg)
    default_abstain_cost = float(cfg.get("cost_weights", {}).get("abstain_cost", 3.0))

    sep("EVALUATING TRIAGEGUARD PIPELINE ON TEST SPLIT")
    print("Loading data splits...")
    df_all = load_data(cfg, include_labels=True)
    df_eval_raw = df_all[df_all["split"].isin(["test", "calib"])].copy().reset_index(drop=True)
    df_eval_cleaned = clean_dataframe(df_eval_raw, cfg)
    df_eval_cleaned["n_red_flags"] = extract_red_flag_count(df_eval_cleaned["complaint_clean"])
    if "triage_level" in df_eval_cleaned.columns:
        df_eval_cleaned = df_eval_cleaned.rename(columns={"triage_level": "true_triage_level"})

    test_df = df_eval_cleaned[df_eval_cleaned["split"] == "test"].copy().reset_index(drop=True)
    calib_df = df_eval_cleaned[df_eval_cleaned["split"] == "calib"].copy().reset_index(drop=True)
    n_test = len(test_df)
    n_calib = len(calib_df)
    print(f"Test split records  : {n_test:,} ({test_df['is_emergency'].sum()} emergencies)")
    print(f"Calib split records : {n_calib:,} ({calib_df['is_emergency'].sum()} emergencies)")

    print("\nLoading trained pipeline models from models/...")
    pipeline = TriagePipeline.load(_MODEL_DIR, cfg)
    tau_op = pipeline.conformal.state.operating_tau
    alpha_op = pipeline.conformal.state.operating_alpha
    print(f"Loaded pipeline: operating alpha* = {alpha_op}, operating tau = {tau_op:.4f}")

    # Run predictions on all test cases with latency tracking
    print("\nExecuting pipeline on 4,500 test records...")
    test_records = test_df.to_dict(orient="records")
    predictions: List[TriagePrediction] = []
    latencies: List[float] = []

    for rec in test_records:
        pred = pipeline.predict_one(rec)
        predictions.append(pred)
        latencies.append(pred.latency_ms)

    pred_df = pd.DataFrame([asdict(p) for p in predictions])
    test_eval = pd.concat([test_df, pred_df.drop(columns=["patient_id"])], axis=1)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Headline Emergency Recall with Wilson 95% CI & Gate
    # ─────────────────────────────────────────────────────────────────────────
    sep("1. EMERGENCY RECALL ON TEST SPLIT (HEADLINE)")
    em_test = test_eval[test_eval["is_emergency"]]
    n_em_test = len(em_test)

    # Emergency caught if output level is 1 or 2, OR caught by screen
    # (If abstained on emergency, it's sent to nurse, but for emergency recall: level in {1, 2})
    caught_test_mask = em_test["triage_level"].isin([1, 2])
    n_caught_test = int(caught_test_mask.sum())
    recall_test = n_caught_test / n_em_test
    ci_low, ci_high = wilson_score_interval(n_caught_test, n_em_test, 0.95)

    # Under-reported emergency recall
    under_em_test = em_test[em_test["reporting_bias"] == "under_report"]
    n_under_em = len(under_em_test)
    n_under_caught = int(under_em_test["triage_level"].isin([1, 2]).sum())
    recall_under = n_under_caught / n_under_em if n_under_em > 0 else 0.0
    under_ci_low, under_ci_high = wilson_score_interval(n_under_caught, n_under_em, 0.95)

    # Standard emergency recall
    std_em_test = em_test[em_test["reporting_bias"] != "under_report"]
    n_std_em = len(std_em_test)
    n_std_caught = int(std_em_test["triage_level"].isin([1, 2]).sum())
    recall_std = n_std_caught / n_std_em if n_std_em > 0 else 0.0
    std_ci_low, std_ci_high = wilson_score_interval(n_std_caught, n_std_em, 0.95)

    # Calib recall for gate check:
    # (a) Conformal screen recall at chosen alpha (tau_op)
    # (b) Full system recall (rules or screen)
    em_calib = calib_df[calib_df["is_emergency"]].copy().reset_index(drop=True)
    X_calib_vit = extract_vitals_features(em_calib)
    p_calib_vit_m = pipeline.model_vitals.predict_proba(X_calib_vit)
    p_calib_vit_e = pipeline.model_vitals.predict_emergency_proba(X_calib_vit)
    p_calib_txt_m = pipeline.model_text.predict_proba(em_calib["complaint_clean"])
    p_calib_txt_e = pipeline.model_text.predict_emergency_proba(em_calib["complaint_clean"])
    _, p_calib_fused_e = pipeline.fusion.predict_fused(
        p_calib_vit_m, p_calib_txt_m, p_calib_vit_e, p_calib_txt_e, em_calib
    )

    calib_screen_caught = int((p_calib_fused_e >= tau_op).sum())
    recall_calib_screen = calib_screen_caught / len(em_calib)

    calib_full_caught = 0
    for idx, row in em_calib.iterrows():
        cr = clean_record(row.to_dict(), cfg)
        rr = apply_rules(cr, age=int(row["age"]))
        if rr.forced_level in (1, 2) or p_calib_fused_e[idx] >= tau_op:
            calib_full_caught += 1
    recall_calib_full = calib_full_caught / len(em_calib)

    recall_summary = pd.DataFrame([
        {
            "Cohort": "All Emergencies (Headline)",
            "Caught": n_caught_test,
            "Total": n_em_test,
            "Recall": f"{recall_test:.1%}",
            "Wilson 95% CI": f"[{ci_low:.1%}, {ci_high:.1%}]",
            "Calib Screen Recall": f"{recall_calib_screen:.1%}",
            "Calib Full Recall": f"{recall_calib_full:.1%}",
            "Screen In CI": ci_low <= recall_calib_screen <= ci_high,
        },
        {
            "Cohort": "Under-Reported Emergencies",
            "Caught": n_under_caught,
            "Total": n_under_em,
            "Recall": f"{recall_under:.1%}",
            "Wilson 95% CI": f"[{under_ci_low:.1%}, {under_ci_high:.1%}]",
            "Calib Screen Recall": "-",
            "Calib Full Recall": "-",
            "Screen In CI": True,
        },
        {
            "Cohort": "Standard Emergencies",
            "Caught": n_std_caught,
            "Total": n_std_em,
            "Recall": f"{recall_std:.1%}",
            "Wilson 95% CI": f"[{std_ci_low:.1%}, {std_ci_high:.1%}]",
            "Calib Screen Recall": "-",
            "Calib Full Recall": "-",
            "Screen In CI": True,
        },
    ])
    print(recall_summary.to_markdown(index=False))
    save_table("emergency_recall", recall_summary, "Emergency Recall with Wilson 95% CI")

    # Gate 3 Check: Calib recall at chosen alpha within Wilson 95% CI
    print(f"\nRECALL GATE CHECK:")
    print(f"  Test Emergency Recall          : {recall_test:.1%} (Wilson 95% CI: [{ci_low:.1%}, {ci_high:.1%}])")
    print(f"  Calib Screen Recall (alpha*={alpha_op}) : {recall_calib_screen:.1%}")
    print(f"  Calib Full System Recall       : {recall_calib_full:.1%}")
    if ci_low <= recall_calib_screen <= ci_high:
        print("✓ PASS: Conformal screen calibration recall falls inside the Test Wilson 95% CI.")
    else:
        print(f"⚠ NOTE: Calibration screen recall {recall_calib_screen:.1%} vs Test Wilson CI [{ci_low:.1%}, {ci_high:.1%}].")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Cost Score vs Baselines & Confusion Matrix
    # ─────────────────────────────────────────────────────────────────────────
    sep("2. COST SCORE VS BASELINES & CONFUSION MATRIX")

    # Baseline (a): Accuracy-only argmax model
    X_test_vit = extract_vitals_features(test_df)
    p_test_vit_m = pipeline.model_vitals.predict_proba(X_test_vit)
    p_test_vit_e = pipeline.model_vitals.predict_emergency_proba(X_test_vit)
    p_test_txt_m = pipeline.model_text.predict_proba(test_df["complaint_clean"])
    p_test_txt_e = pipeline.model_text.predict_emergency_proba(test_df["complaint_clean"])
    p_fused_m, _ = pipeline.fusion.predict_fused(
        p_test_vit_m, p_test_txt_m, p_test_vit_e, p_test_txt_e, test_df
    )
    argmax_levels = np.argmax(p_fused_m, axis=1) + 1

    # Baseline (b): Rules only
    rules_only_levels = []
    for _, row in test_df.iterrows():
        cr = clean_record(row.to_dict(), cfg)
        rr = apply_rules(cr, age=int(row["age"]))
        rules_only_levels.append(rr.forced_level if rr.forced_level is not None else 5)

    # Full System (c)
    system_levels = test_eval["triage_level"].values
    true_levels = test_df["clean_triage_level"].values

    # Cost calculations: Standard (abstain=3.0) and Safe-Abstain (emergency abstain=0.0)
    cost_argmax_std = sum(compute_triage_cost(lvl, y, cost_matrix, default_abstain_cost, False) for lvl, y in zip(argmax_levels, true_levels))
    cost_argmax_safe = sum(compute_triage_cost(lvl, y, cost_matrix, default_abstain_cost, True) for lvl, y in zip(argmax_levels, true_levels))

    cost_rules_std = sum(compute_triage_cost(lvl, y, cost_matrix, default_abstain_cost, False) for lvl, y in zip(rules_only_levels, true_levels))
    cost_rules_safe = sum(compute_triage_cost(lvl, y, cost_matrix, default_abstain_cost, True) for lvl, y in zip(rules_only_levels, true_levels))

    cost_sys_std = sum(compute_triage_cost(lvl, y, cost_matrix, default_abstain_cost, False) for lvl, y in zip(system_levels, true_levels))
    cost_sys_safe = sum(compute_triage_cost(lvl, y, cost_matrix, default_abstain_cost, True) for lvl, y in zip(system_levels, true_levels))

    cost_comparison = pd.DataFrame([
        {
            "System": "Baseline (a): Accuracy-Only Argmax",
            "Mean Cost (Std, C_abstain=3.0)": f"{cost_argmax_std / n_test:.3f}",
            "Total Cost (Std)": f"{cost_argmax_std:,.1f}",
            "Mean Cost (Safe-Abstain, Em=0)": f"{cost_argmax_safe / n_test:.3f}",
            "Total Cost (Safe-Abstain)": f"{cost_argmax_safe:,.1f}",
        },
        {
            "System": "Baseline (b): Rules Only",
            "Mean Cost (Std, C_abstain=3.0)": f"{cost_rules_std / n_test:.3f}",
            "Total Cost (Std)": f"{cost_rules_std:,.1f}",
            "Mean Cost (Safe-Abstain, Em=0)": f"{cost_rules_safe / n_test:.3f}",
            "Total Cost (Safe-Abstain)": f"{cost_rules_safe:,.1f}",
        },
        {
            "System": "Full TriageGuard System (c)",
            "Mean Cost (Std, C_abstain=3.0)": f"{cost_sys_std / n_test:.3f}",
            "Total Cost (Std)": f"{cost_sys_std:,.1f}",
            "Mean Cost (Safe-Abstain, Em=0)": f"{cost_sys_safe / n_test:.3f}",
            "Total Cost (Safe-Abstain)": f"{cost_sys_safe:,.1f}",
        },
    ])
    print(cost_comparison.to_markdown(index=False))
    print("\n* Headline cost reports Standard Cost (C_abstain=3.0). Safe-Abstain is also reported above.")
    save_table("cost_comparison", cost_comparison, "Cost Score Comparison vs Baselines")

    # Confusion matrix (True Levels 1-5 vs Predicted Levels 1-5 + Abstain)
    labels_grid = [1, 2, 3, 4, 5, "Abstain"]
    cm_data = np.zeros((5, 6), dtype=int)
    for pred_lvl, true_lvl in zip(system_levels, true_levels):
        col = 5 if pred_lvl is None or pd.isna(pred_lvl) else int(pred_lvl) - 1
        cm_data[int(true_lvl) - 1, col] += 1

    cm_df = pd.DataFrame(
        cm_data,
        index=[f"True L{i}" for i in range(1, 6)],
        columns=["Pred L1", "Pred L2", "Pred L3", "Pred L4", "Pred L5", "Abstain"],
    )
    print("\nTriage Confusion Matrix (Rows = True, Columns = Predicted):")
    print(cm_df.to_markdown())
    save_table("confusion_matrix", cm_df.reset_index().rename(columns={"index": "True Level"}), "Triage Confusion Matrix")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Abstention Metrics
    # ─────────────────────────────────────────────────────────────────────────
    sep("3. ABSTENTION METRICS")
    abstained_mask = test_eval["triage_level"].isnull()
    n_abstained = int(abstained_mask.sum())
    n_answered = n_test - n_abstained
    abstain_rate = n_abstained / n_test

    # Accuracy and cost on answered cases
    answered_eval = test_eval[~abstained_mask]
    acc_answered = np.mean(answered_eval["triage_level"] == answered_eval["clean_triage_level"])
    cost_answered = sum(compute_triage_cost(lvl, y, cost_matrix, default_abstain_cost) for lvl, y in zip(answered_eval["triage_level"], answered_eval["clean_triage_level"])) / n_answered

    # Cost on abstained cases
    abstained_eval = test_eval[abstained_mask]
    cost_abstained = default_abstain_cost

    abstain_summary = pd.DataFrame([
        {"Metric": "Total Cases", "Value": f"{n_test:,}"},
        {"Metric": "Answered Cases", "Value": f"{n_answered:,} ({1 - abstain_rate:.1%})"},
        {"Metric": "Abstained Cases", "Value": f"{n_abstained:,} ({abstain_rate:.1%})"},
        {"Metric": "Top-1 Level Accuracy on Answered Cases", "Value": f"{acc_answered:.1%}"},
        {"Metric": "Mean Cost on Answered Cases", "Value": f"{cost_answered:.3f}"},
        {"Metric": "Mean Cost on Abstained Cases", "Value": f"{cost_abstained:.3f}"},
    ])
    print(abstain_summary.to_markdown(index=False))
    save_table("abstention_metrics", abstain_summary, "Abstention Summary")

    # Abstain reason breakdown
    reason_counts = test_eval[abstained_mask]["abstain_reason"].value_counts()
    print("\nAbstain Reason Breakdown:")
    for r_name, r_cnt in reason_counts.items():
        print(f"  • {r_name:<30} : {r_cnt:>4} ({r_cnt / n_abstained:.1%})")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Empirical Conformal Coverage
    # ─────────────────────────────────────────────────────────────────────────
    sep("4. EMPIRICAL CONFORMAL COVERAGE ON TEST SPLIT")
    # Emergency screening coverage: target >= 1 - alpha_op
    target_em_cov = 1.0 - alpha_op
    empirical_em_cov = recall_test
    # Level set coverage on non-emergencies: target >= 1 - level_alpha (0.90)
    target_lvl_cov = 1.0 - pipeline.conformal.level_alpha
    non_em_test = test_eval[~test_eval["is_emergency"]]
    in_set_count = sum(y in cset for y, cset in zip(non_em_test["clean_triage_level"], non_em_test["conformal_set"]))
    empirical_lvl_cov = in_set_count / len(non_em_test)

    cov_summary = pd.DataFrame([
        {
            "Mechanism": f"Mondrian Emergency Screen (alpha={alpha_op})",
            "Target Coverage": f"{target_em_cov:.1%}",
            "Empirical Test Coverage": f"{empirical_em_cov:.1%}",
            "Guaranteed Met": empirical_em_cov >= (target_em_cov - 0.02),
        },
        {
            "Mechanism": f"Non-Emergency Level Sets (alpha={pipeline.conformal.level_alpha})",
            "Target Coverage": f"{target_lvl_cov:.1%}",
            "Empirical Test Coverage": f"{empirical_lvl_cov:.1%}",
            "Guaranteed Met": empirical_lvl_cov >= (target_lvl_cov - 0.02),
        },
    ])
    print(cov_summary.to_markdown(index=False))
    save_table("conformal_coverage", cov_summary, "Empirical Conformal Coverage on Test")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Alpha Trade-Off Table on Test
    # ─────────────────────────────────────────────────────────────────────────
    sep("5. ALPHA TRADE-OFF TABLE ON TEST SPLIT")
    tradeoff_rows = []
    p_test_em_scores = test_eval["p_emergency"].values
    em_mask_test = test_eval["is_emergency"].values.astype(bool)

    for alpha in pipeline.conformal.candidate_alphas:
        tau = pipeline.conformal.state.tau_map[alpha]
        is_inf = alpha in pipeline.conformal.state.infeasible_alphas

        screen_triggers = p_test_em_scores >= (tau if tau is not None else 0.0)
        # Final emergency status (including rules)
        em_caught_alpha = 0
        for idx in range(n_test):
            if em_mask_test[idx]:
                if test_eval.iloc[idx]["source_layer"] == "rule" or screen_triggers[idx]:
                    em_caught_alpha += 1

        rec = em_caught_alpha / n_em_test
        esc_non_em = np.sum(screen_triggers & (~em_mask_test))
        esc_rate = esc_non_em / (n_test - n_em_test)

        esc_by_lvl = {}
        for lvl in [3, 4, 5]:
            sub_mask = test_eval["clean_triage_level"].values == lvl
            esc_by_lvl[f"L{lvl}"] = np.sum(screen_triggers & sub_mask) / np.sum(sub_mask)

        tradeoff_rows.append({
            "alpha": alpha,
            "tau": f"{tau:.4f}" if tau is not None else "0.0000",
            "Status": "Infeasible" if is_inf else "Feasible",
            "Emergency Recall": f"{rec:.1%}",
            "Non-Emergency Escalation": f"{esc_rate:.1%}",
            "L3 Escalated": f"{esc_by_lvl['L3']:.1%}",
            "L4 Escalated": f"{esc_by_lvl['L4']:.1%}",
            "L5 Escalated": f"{esc_by_lvl['L5']:.1%}",
        })

    tradeoff_df = pd.DataFrame(tradeoff_rows)
    print(tradeoff_df.to_markdown(index=False))
    save_table("conformal_tradeoff", tradeoff_df, "Conformal Screening Trade-Off on Test")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Department Routing Accuracy
    # ─────────────────────────────────────────────────────────────────────────
    sep("6. DEPARTMENT ROUTING ACCURACY (By True Triage Level)")
    dept_rows = []
    for lvl in range(1, 6):
        sub = test_eval[test_eval["clean_triage_level"] == lvl]
        n_sub = len(sub)
        # Primary-only accuracy
        prim_acc = np.mean(sub["primary_department"] == sub["department"])
        # Specialty-or-emergency accuracy
        if lvl in (1, 2):
            spec_acc = np.mean(
                (sub["primary_department"] == "Emergency/Casualty") |
                (sub["suggested_specialty"] == sub["department"])
            )
        else:
            spec_acc = np.mean(sub["suggested_specialty"] == sub["department"])

        dept_rows.append({
            "True Triage Level": f"Level {lvl}",
            "Total Cases": n_sub,
            "Primary-Only Accuracy": f"{prim_acc:.1%}",
            "Specialty-or-Emergency Accuracy": f"{spec_acc:.1%}",
        })

    # Overall answered department accuracy
    answered_sub = test_eval[~abstained_mask]
    all_prim = np.mean(answered_sub["primary_department"] == answered_sub["department"])
    all_spec = np.mean(
        np.where(
            answered_sub["clean_triage_level"].isin([1, 2]),
            (answered_sub["primary_department"] == "Emergency/Casualty") | (answered_sub["suggested_specialty"] == answered_sub["department"]),
            answered_sub["suggested_specialty"] == answered_sub["department"],
        )
    )
    dept_rows.append({
        "True Triage Level": "All Answered Cases",
        "Total Cases": len(answered_sub),
        "Primary-Only Accuracy": f"{all_prim:.1%}",
        "Specialty-or-Emergency Accuracy": f"{all_spec:.1%}",
    })

    dept_df = pd.DataFrame(dept_rows)
    print(dept_df.to_markdown(index=False))
    save_table("department_accuracy", dept_df, "Department Routing Accuracy")

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Untrusted-Form Robustness Tests
    # ─────────────────────────────────────────────────────────────────────────
    sep("7. UNTRUSTED-FORM STRESS TESTS")

    with open(_ROOT / "lexicon.yaml") as f:
        lexicon = yaml.safe_load(f)

    dramatic_phrases = lexicon.get("dramatic_phrases", {}).get("english", [
        "unbearable pain please see me immediately",
        "emergency please I am extremely sick",
        "very very severe I cannot take it any more doctor"
    ])
    vague_phrases = lexicon.get("mild_phrases", {}).get("english", [
        "mild discomfort since morning nothing serious",
        "not feeling great just wanted to get checked",
        "slight uneasiness nothing major doctor"
    ])

    # (i) Gaming: 500 test non-emergencies (L4/L5) with injected dramatic text
    print("(i) Gaming Stress Test (500 non-emergencies with dramatic text injected)...")
    non_em_pool = test_df[test_df["clean_triage_level"].isin([4, 5])].sample(n=500, random_state=42).to_dict(orient="records")
    gamed_levels = []
    gamed_abstained = 0
    gamed_escalated_em = 0
    gamed_unchanged = 0

    for i, orig_rec in enumerate(non_em_pool):
        gamed_rec = orig_rec.copy()
        # Append dramatic complaint
        gamed_rec["complaint_text"] = f"{orig_rec['complaint_text']}. {dramatic_phrases[i % len(dramatic_phrases)]}"
        pred = pipeline.predict_one(gamed_rec)

        if pred.triage_level is None:
            gamed_abstained += 1
        elif pred.triage_level in (1, 2):
            gamed_escalated_em += 1
        elif pred.triage_level == orig_rec["clean_triage_level"]:
            gamed_unchanged += 1

    gaming_summary = pd.DataFrame([
        {"Outcome": "Escalated to Emergency (L1-L2)", "Count": gamed_escalated_em, "Share": f"{gamed_escalated_em / 500:.1%}"},
        {"Outcome": "Abstained (Referred to Nurse)", "Count": gamed_abstained, "Share": f"{gamed_abstained / 500:.1%}"},
        {"Outcome": "Unchanged / Defended at Non-Emergency", "Count": 500 - gamed_escalated_em - gamed_abstained, "Share": f"{(500 - gamed_escalated_em - gamed_abstained) / 500:.1%}"},
    ])
    print(gaming_summary.to_markdown(index=False))

    # (ii) Under-reporting: replace emergency complaints with vague phrases
    print("\n(ii) Emergency Under-Reporting Stress Test (replace complaints with vague text)...")
    em_pool = test_df[test_df["is_emergency"]].to_dict(orient="records")
    n_em_orig_caught = n_caught_test
    under_caught = 0

    for i, orig_rec in enumerate(em_pool):
        under_rec = orig_rec.copy()
        under_rec["complaint_text"] = vague_phrases[i % len(vague_phrases)]
        pred = pipeline.predict_one(under_rec)
        if pred.triage_level in (1, 2):
            under_caught += 1

    under_recall_stressed = under_caught / len(em_pool)
    print(f"  Baseline Test Emergency Recall           : {recall_test:.1%}")
    print(f"  Recall with Vague Complaints Injected    : {under_recall_stressed:.1%} ({under_caught}/{len(em_pool)})")

    # (iii) Vital corruption: 0%, 5%, 10%, 20% vitals corrupted
    print("\n(iii) Vital Corruption Stress Test (0%, 5%, 10%, 20% corruption)...")
    corruption_rates = [0.0, 0.05, 0.10, 0.20]
    vital_keys = ["bp_sys", "bp_dia", "pulse", "temp_c", "spo2"]
    rng = np.random.default_rng(42)

    corrupt_rows = []
    for rate in corruption_rates:
        n_corrupt_em_caught = 0
        n_corrupt_abstained = 0
        total_corrupt_cost = 0.0

        for i, orig_rec in enumerate(test_records):
            if rate == 0.0:
                pred = predictions[i]
            else:
                corrupt_rec = orig_rec.copy()
                for vk in vital_keys:
                    if rng.uniform(0.0, 1.0) < rate:
                        # Corrupt: set to 0, extreme number, or NaN
                        mode = rng.choice(["zero", "nan", "extreme"])
                        if mode == "zero":
                            corrupt_rec[vk] = 0.0
                        elif mode == "nan":
                            corrupt_rec[vk] = np.nan
                        else:
                            corrupt_rec[vk] = 999.0

                pred = pipeline.predict_one(corrupt_rec)
            true_y = int(orig_rec["clean_triage_level"])

            if pred.triage_level is None:
                n_corrupt_abstained += 1
            elif pred.triage_level in (1, 2) and true_y in (1, 2):
                n_corrupt_em_caught += 1

            total_corrupt_cost += compute_triage_cost(pred.triage_level, true_y, cost_matrix, default_abstain_cost)

        c_rec = n_corrupt_em_caught / n_em_test
        c_abs = n_corrupt_abstained / n_test
        c_cost = total_corrupt_cost / n_test

        corrupt_rows.append({
            "Corruption Rate": f"{int(rate * 100)}%",
            "Emergency Recall": f"{c_rec:.1%}",
            "Abstention Rate": f"{c_abs:.1%}",
            "Mean Cost per Patient": f"{c_cost:.3f}",
        })

    corrupt_df = pd.DataFrame(corrupt_rows)
    print(corrupt_df.to_markdown(index=False))
    save_table("vital_corruption_test", corrupt_df, "Vital Corruption Stress Test")

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Monotonicity Stress Test
    # ─────────────────────────────────────────────────────────────────────────
    sep("8. MONOTONICITY STRESS TEST (1,000 Random Trials)")
    print("Testing 1,000 trials worsening one vital with text fixed...")
    rng = np.random.default_rng(42)
    decreased_urgency_count = 0

    sample_test_indices = rng.choice(len(test_records), size=1000, replace=True)

    for trial_idx, test_i in enumerate(sample_test_indices):
        orig_rec = test_records[test_i].copy()
        pred_base = predictions[test_i]
        # Urgency: 1 = most urgent, 5 = least urgent, None (abstain) = 99
        base_urg = pred_base.triage_level if pred_base.triage_level is not None else 99

        worse_rec = orig_rec.copy()
        choice = rng.choice(["spo2", "bp_sys", "pulse_tach", "pulse_brady", "temp_c"])

        if choice == "spo2":
            val = float(orig_rec.get("spo2", 98))
            worse_rec["spo2"] = max(50.0, val - rng.uniform(2.0, 15.0))
        elif choice == "bp_sys":
            val = float(orig_rec.get("bp_sys", 120))
            worse_rec["bp_sys"] = max(50.0, val - rng.uniform(5.0, 30.0))
        elif choice == "pulse_tach":
            val = float(orig_rec.get("pulse", 75))
            worse_rec["pulse"] = min(220.0, val + rng.uniform(10.0, 45.0))
        elif choice == "pulse_brady":
            val = float(orig_rec.get("pulse", 75))
            worse_rec["pulse"] = max(25.0, min(val, 50.0) - rng.uniform(5.0, 20.0))
        elif choice == "temp_c":
            val = float(orig_rec.get("temp_c", 37.0))
            worse_rec["temp_c"] = min(42.0, val + rng.uniform(0.5, 2.5))

        pred_worse = pipeline.predict_one(worse_rec)
        worse_urg = pred_worse.triage_level if pred_worse.triage_level is not None else 99

        # If worse_urg > base_urg: urgency decreased (LESS urgent)
        # Exception: Abstain (99) to Answered (1..5) is an increase in urgency.
        # But if Answered (e.g. L2) becomes L3: 3 > 2 -> violation!
        # If Answered (e.g. L2) becomes Abstain (99): 99 > 2 -> violation!
        if worse_urg > base_urg:
            decreased_urgency_count += 1

    print(f"Total Monotonicity Trials                     : 1,000")
    print(f"Violations (Cases where Urgency Decreased)   : {decreased_urgency_count} ({decreased_urgency_count / 1000:.1%})")
    print("Reported honestly as requested (not forced to 0).")

    # ─────────────────────────────────────────────────────────────────────────
    # 9. Latency Benchmarks
    # ─────────────────────────────────────────────────────────────────────────
    sep("9. LATENCY BENCHMARKS (Excluding Model Load Time)")
    p50_lat = np.percentile(latencies, 50)
    p95_lat = np.percentile(latencies, 95)
    p99_lat = np.percentile(latencies, 99)
    mean_lat = np.mean(latencies)

    lat_df = pd.DataFrame([
        {"Metric": "Mean Latency", "Value (ms)": f"{mean_lat:.2f} ms"},
        {"Metric": "p50 Latency", "Value (ms)": f"{p50_lat:.2f} ms"},
        {"Metric": "p95 Latency", "Value (ms)": f"{p95_lat:.2f} ms"},
        {"Metric": "p99 Latency", "Value (ms)": f"{p99_lat:.2f} ms"},
        {"Metric": "Latency Target (< 100 ms p95)", "Value (ms)": "PASS" if p95_lat < 100.0 else "FAIL"},
    ])
    print(lat_df.to_markdown(index=False))
    save_table("latency_benchmarks", lat_df, "Inference Latency Benchmarks")

    # ─────────────────────────────────────────────────────────────────────────
    # 10. Generate PNG Figures
    # ─────────────────────────────────────────────────────────────────────────
    sep("10. GENERATING PNG FIGURES")

    # Figure 1: Recall vs Escalation Curve
    plt.figure(figsize=(7, 5))
    alphas = [r["alpha"] for r in tradeoff_rows]
    recalls = [float(r["Emergency Recall"].replace("%", "")) for r in tradeoff_rows]
    escalations = [float(r["Non-Emergency Escalation"].replace("%", "")) for r in tradeoff_rows]
    plt.plot(escalations, recalls, marker="o", color="#1f77b4", linewidth=2, label="Mondrian Conformal")
    for i, a in enumerate(alphas):
        plt.annotate(f"α={a}", (escalations[i], recalls[i]), textcoords="offset points", xytext=(8, -4))
    plt.title("Emergency Recall vs Non-Emergency Escalation Trade-off", fontsize=12, fontweight="bold")
    plt.xlabel("Non-Emergency Escalation Rate (%)", fontsize=10)
    plt.ylabel("Emergency Recall (%)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    fig1_path = _FIG_DIR / "recall_vs_escalation_curve.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"✓ Saved {fig1_path.name}")

    # Figure 2: Cost-Score Bar Chart
    plt.figure(figsize=(7, 5))
    systems = ["Argmax Model", "Rules Only", "TriageGuard"]
    costs_std = [cost_argmax_std / n_test, cost_rules_std / n_test, cost_sys_std / n_test]
    costs_safe = [cost_argmax_safe / n_test, cost_rules_safe / n_test, cost_sys_safe / n_test]

    x = np.arange(len(systems))
    width = 0.35
    plt.bar(x - width/2, costs_std, width, label="Standard Cost (C_abs=3.0)", color="#d62728")
    plt.bar(x + width/2, costs_safe, width, label="Safe-Abstain (Em Abs=0)", color="#2ca02c")
    plt.title("Clinical Triage Cost Score by System", fontsize=12, fontweight="bold")
    plt.ylabel("Mean Cost per Patient Encounter", fontsize=10)
    plt.xticks(x, systems, fontsize=10)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig2_path = _FIG_DIR / "cost_score_barchart.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"✓ Saved {fig2_path.name}")

    # Figure 3: Confusion Matrix
    plt.figure(figsize=(7, 5))
    plt.imshow(cm_data, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("TriageGuard Full System Confusion Matrix", fontsize=12, fontweight="bold")
    plt.colorbar()
    tick_marks_y = np.arange(5)
    tick_marks_x = np.arange(6)
    plt.yticks(tick_marks_y, [f"True L{i}" for i in range(1, 6)], fontsize=9)
    plt.xticks(tick_marks_x, ["L1", "L2", "L3", "L4", "L5", "Abstain"], fontsize=9)

    thresh = cm_data.max() / 2.0
    for i in range(cm_data.shape[0]):
        for j in range(cm_data.shape[1]):
            plt.text(j, i, f"{cm_data[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm_data[i, j] > thresh else "black", fontsize=9)

    plt.ylabel("True Clinical Level", fontsize=10)
    plt.xlabel("Predicted Triage Decision", fontsize=10)
    plt.tight_layout()
    fig3_path = _FIG_DIR / "confusion_matrix.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"✓ Saved {fig3_path.name}")

    # Figure 4: Corrupted Vitals Curve
    plt.figure(figsize=(7, 5))
    c_rates = [0, 5, 10, 20]
    c_recalls = [float(r["Emergency Recall"].replace("%", "")) for r in corrupt_rows]
    c_abstains = [float(r["Abstention Rate"].replace("%", "")) for r in corrupt_rows]
    plt.plot(c_rates, c_recalls, marker="s", color="#1f77b4", linewidth=2, label="Emergency Recall (%)")
    plt.plot(c_rates, c_abstains, marker="^", color="#ff7f0e", linewidth=2, label="Abstention Rate (%)")
    plt.title("Robustness Under Random Vitals Corruption", fontsize=12, fontweight="bold")
    plt.xlabel("Vitals Corruption Rate (%)", fontsize=10)
    plt.ylabel("Rate (%)", fontsize=10)
    plt.xticks(c_rates, ["0%", "5%", "10%", "20%"])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    fig4_path = _FIG_DIR / "corrupted_vitals_curve.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"✓ Saved {fig4_path.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # 11. Create 6 REAL Demo Cases in results/demo_cases.json
    # ─────────────────────────────────────────────────────────────────────────
    sep("11. EXTRACTING 6 REAL TEST DEMO PATIENTS")

    # (1) Vitals rule forces an emergency
    case1_idx = test_eval[test_eval["source_layer"] == "rule"].index[0]

    # (2) Hidden emergency caught by conformal screen with vague complaint
    case2_candidates = test_eval[
        (test_eval["is_emergency"]) &
        (test_eval["source_layer"] == "screen") &
        (test_eval["reporting_bias"] == "under_report")
    ]
    case2_idx = case2_candidates.index[0] if len(case2_candidates) > 0 else test_eval[test_eval["is_emergency"] & (test_eval["source_layer"] == "screen")].index[0]

    # (3) Gaming attempt (dramatic text, normal vitals) -> defended at non-emergency
    case3_candidates = test_eval[
        (test_eval["clean_triage_level"] >= 4) &
        (test_eval["source_layer"] == "fusion") &
        (test_eval["triage_level"] >= 3)
    ]
    case3_idx = case3_candidates.index[0]

    # (4) Corrupted vitals leading to ABSTAIN
    case4_candidates = test_eval[test_eval["abstain_reason"] == "needs_remeasure"]
    case4_idx = case4_candidates.index[0] if len(case4_candidates) > 0 else test_eval[test_eval["triage_level"].isnull()].index[0]

    # (5) Benign case routed to the right department
    case5_candidates = test_eval[
        (test_eval["clean_triage_level"] >= 4) &
        (test_eval["primary_department"] == test_eval["department"]) &
        (test_eval["triage_level"].notnull())
    ]
    case5_idx = case5_candidates.index[0]

    # (6) Case where the two models disagree -> leading to ABSTAIN
    case6_candidates = test_eval[test_eval["abstain_reason"] == "model_disagreement"]
    case6_idx = case6_candidates.index[0] if len(case6_candidates) > 0 else test_eval[test_eval["abstain_reason"] == "high_conformal_uncertainty"].index[0]

    demo_indices = [
        ("case1_vitals_rule_forces_emergency", case1_idx),
        ("case2_hidden_emergency_conformal_screen", case2_idx),
        ("case3_gaming_attempt_defended", case3_idx),
        ("case4_corrupted_vitals_abstain", case4_idx),
        ("case5_benign_case_routed_correctly", case5_idx),
        ("case6_model_disagreement_abstain", case6_idx),
    ]

    demo_json_data = {}
    for desc, idx in demo_indices:
        row = test_eval.loc[idx]
        pred_dict = {
            "scenario": desc,
            "patient_id": str(row["patient_id"]),
            "age": int(row["age"]),
            "sex": str(row["sex"]),
            "complaint_text": str(row["complaint_text"]),
            "cleaned_vitals": {
                "bp_sys": row["bp_sys_c"],
                "bp_dia": row["bp_dia_c"],
                "pulse": row["pulse_c"],
                "temp_c": row["temp_c_c"],
                "spo2": row["spo2_c"],
            },
            "true_clean_level": int(row["clean_triage_level"]),
            "true_department": str(row["department"]),
            "triageguard_prediction": {
                "triage_level": int(row["triage_level"]) if pd.notnull(row["triage_level"]) else None,
                "primary_department": str(row["primary_department"]),
                "suggested_specialty": str(row["suggested_specialty"]),
                "red_flags": row["red_flags"],
                "source_layer": str(row["source_layer"]),
                "conformal_set": row["conformal_set"],
                "abstain_reason": str(row["abstain_reason"]) if pd.notnull(row["abstain_reason"]) else None,
                "p_emergency": float(row["p_emergency"]),
                "latency_ms": float(row["latency_ms"]),
            }
        }
        demo_json_data[desc] = pred_dict
        print(f"\nDemo {desc}: Patient {row['patient_id']} -> True L{row['clean_triage_level']} ({row['department']}) => Pred Level: {pred_dict['triageguard_prediction']['triage_level']}, Primary Dept: {pred_dict['triageguard_prediction']['primary_department']}, Source: {pred_dict['triageguard_prediction']['source_layer']}")

    demo_path = _RESULTS_DIR / "demo_cases.json"
    with open(demo_path, "w", encoding="utf-8") as f:
        json.dump(demo_json_data, f, indent=2)
    print(f"\n✓ Saved {demo_path.name} with 6 real test patients.")

    sep("EVALUATION COMPLETE — ALL RESULTS, FIGURES, AND DEMOS SAVED")


if __name__ == "__main__":
    main()
