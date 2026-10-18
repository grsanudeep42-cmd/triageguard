#!/usr/bin/env python3
"""
scripts/compute_operating_points.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive evaluation of Operating Points, Layer Ablation with Costs,
Validation Benchmarks, and Policy Optimization.
Outputs:
  - results/operating_points.json
  - results/operating_points.md
Does NOT modify models, thresholds, or default configurations.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import scipy.stats as stats
import yaml

from triageguard.cleaning import clean_dataframe, clean_record
from triageguard.data_adapter import extract_vitals_features, load_config, load_data
from triageguard.fusion import build_cost_matrix, select_min_cost_level
from triageguard.predict import TriagePipeline
from triageguard.rules import apply_rules

_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _ROOT / "results"


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    p_hat = successes / total
    denom = 1.0 + (z**2) / total
    center = (p_hat + (z**2) / (2.0 * total)) / denom
    half = (z * np.sqrt((p_hat * (1.0 - p_hat) + (z**2) / (4.0 * total)) / total)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def compute_triage_cost(
    pred_level: Optional[int],
    true_level: int,
    cost_matrix: np.ndarray,
    abstain_cost: float = 3.0,
) -> float:
    if pred_level is None or pd.isna(pred_level):
        return abstain_cost
    p_idx = int(pred_level) - 1
    t_idx = int(true_level) - 1
    return float(cost_matrix[t_idx, p_idx])


def evaluate_batch_predictions(
    records: List[Dict],
    pipeline: TriagePipeline,
    cfg: Dict,
    cost_matrix: np.ndarray,
    is_emergency_arr: np.ndarray,
    true_levels: np.ndarray,
    tau_override: Optional[float] = None,
) -> Dict[str, List[Optional[int]]]:
    """
    Run vectorized/per-encounter inference and return predictions for:
    - full
    - no_rules
    - no_screen
    - fusion_only
    - baseline_a (argmax)
    """
    operating_tau = tau_override if tau_override is not None else pipeline.conformal.state.operating_tau
    preds = {
        "full": [],
        "no_rules": [],
        "no_screen": [],
        "fusion_only": [],
        "baseline_a": [],
    }

    for rec in records:
        age = int(rec.get("age", 35))
        cr = clean_record(rec, cfg)
        rr = apply_rules(cr, age=age)

        single_df = pd.DataFrame([{
            "bp_sys_c": cr.bp_sys, "bp_dia_c": cr.bp_dia, "pulse_c": cr.pulse,
            "temp_c_c": cr.temp_c, "spo2_c": cr.spo2, "bp_sys_flag": cr.bp_sys_flag,
            "bp_dia_flag": cr.bp_dia_flag, "pulse_flag": cr.pulse_flag,
            "temp_c_flag": cr.temp_c_flag, "spo2_flag": cr.spo2_flag,
            "age": age, "sex": rec.get("sex", "M"), "n_invalid_vitals": cr.n_invalid_vitals,
            "n_missing_vitals": cr.n_missing_vitals, "data_quality_score": cr.data_quality_score,
            "complaint_clean": cr.complaint_text, "n_red_flags": len(rr.red_flags_detected),
        }])

        X_vit = extract_vitals_features(single_df)
        pv_m = pipeline.model_vitals.predict_proba(X_vit)
        pv_e = pipeline.model_vitals.predict_emergency_proba(X_vit)
        texts = [cr.complaint_text]
        pt_m = pipeline.model_text.predict_proba(texts)
        pt_e = pipeline.model_text.predict_emergency_proba(texts)

        pf_m, calib_p_em = pipeline.fusion.predict_fused(pv_m, pt_m, pv_e, pt_e, single_df)
        fused_probs = pf_m[0]
        p_em = float(calib_p_em[0])
        conformal_set = pipeline.conformal.predict_level_set(fused_probs)
        vit_level = int(np.argmax(pv_m[0]) + 1)
        txt_level = int(np.argmax(pt_m[0]) + 1)
        model_disagree = abs(vit_level - txt_level) >= 2

        # 1. Baseline (a) Argmax
        lvl_argmax = int(np.argmax(fused_probs) + 1)
        preds["baseline_a"].append(lvl_argmax)

        # 2. Fusion only
        lvl_fusion = select_min_cost_level(fused_probs, allowed_levels=[1, 2, 3, 4, 5], cost_matrix=pipeline.cost_matrix)
        preds["fusion_only"].append(lvl_fusion)

        # 3. Full System
        if rr.forced_level is not None:
            lvl_full = int(rr.forced_level)
        elif p_em >= operating_tau:
            lvl_full = select_min_cost_level(fused_probs, allowed_levels=[1, 2], cost_matrix=pipeline.cost_matrix)
        elif cr.needs_remeasure or model_disagree or len(conformal_set) >= 3:
            lvl_full = None
        else:
            c_set = list(conformal_set)
            if rr.no_downgrade_below_L3 and not pipeline._are_vitals_clearly_normal(cr):
                c_set = [l for l in c_set if l <= 3] or [3]
            lvl_full = select_min_cost_level(fused_probs, allowed_levels=c_set, cost_matrix=pipeline.cost_matrix)
        preds["full"].append(lvl_full)

        # 4. Rules Disabled (no forced rule, no soft floor)
        if p_em >= operating_tau:
            lvl_no_rules = select_min_cost_level(fused_probs, allowed_levels=[1, 2], cost_matrix=pipeline.cost_matrix)
        elif cr.needs_remeasure or model_disagree or len(conformal_set) >= 3:
            lvl_no_rules = None
        else:
            lvl_no_rules = select_min_cost_level(fused_probs, allowed_levels=list(conformal_set), cost_matrix=pipeline.cost_matrix)
        preds["no_rules"].append(lvl_no_rules)

        # 5. Screen Disabled (no p_em >= tau)
        if rr.forced_level is not None:
            lvl_no_screen = int(rr.forced_level)
        elif cr.needs_remeasure or model_disagree or len(conformal_set) >= 3:
            lvl_no_screen = None
        else:
            c_set = list(conformal_set)
            if rr.no_downgrade_below_L3 and not pipeline._are_vitals_clearly_normal(cr):
                c_set = [l for l in c_set if l <= 3] or [3]
            lvl_no_screen = select_min_cost_level(fused_probs, allowed_levels=c_set, cost_matrix=pipeline.cost_matrix)
        preds["no_screen"].append(lvl_no_screen)

    return preds


def compute_metrics(
    preds_arr: List[Optional[int]],
    is_em: np.ndarray,
    true_y: np.ndarray,
    cost_matrix: np.ndarray,
) -> Dict[str, Any]:
    n_tot = len(preds_arr)
    n_em = int(np.sum(is_em))
    n_non_em = n_tot - n_em

    strict_hits = sum(1 for i in range(n_tot) if is_em[i] and preds_arr[i] in (1, 2))
    ca_hits = sum(1 for i in range(n_tot) if is_em[i] and (preds_arr[i] in (1, 2) or preds_arr[i] is None))
    silent_misses = sum(1 for i in range(n_tot) if is_em[i] and preds_arr[i] in (3, 4, 5))
    non_em_esc = sum(1 for i in range(n_tot) if not is_em[i] and preds_arr[i] in (1, 2))
    abstain_cnt = sum(1 for i in range(n_tot) if preds_arr[i] is None)

    strict_rec = strict_hits / n_em if n_em > 0 else 0.0
    s_low, s_high = wilson_score_interval(strict_hits, n_em, 0.95)

    ca_rec = ca_hits / n_em if n_em > 0 else 0.0
    ca_low, ca_high = wilson_score_interval(ca_hits, n_em, 0.95)

    tot_cost_3 = sum(compute_triage_cost(preds_arr[i], true_y[i], cost_matrix, 3.0) for i in range(n_tot))
    tot_cost_0 = sum(compute_triage_cost(preds_arr[i], true_y[i], cost_matrix, 0.0) for i in range(n_tot))

    return {
        "strict_hits": strict_hits,
        "n_em": n_em,
        "strict_recall": strict_rec,
        "strict_recall_str": f"{strict_rec:.1%} ({strict_hits}/{n_em})",
        "strict_ci_str": f"[{s_low:.1%}, {s_high:.1%}]",
        "ca_hits": ca_hits,
        "ca_recall": ca_rec,
        "ca_recall_str": f"{ca_rec:.1%} ({ca_hits}/{n_em})",
        "ca_ci_str": f"[{ca_low:.1%}, {ca_high:.1%}]",
        "silent_misses": silent_misses,
        "silent_misses_str": f"{silent_misses/n_em:.1%} ({silent_misses}/{n_em})",
        "non_em_esc": non_em_esc,
        "non_em_esc_str": f"{non_em_esc/n_non_em:.1%} ({non_em_esc}/{n_non_em})",
        "abstain_cnt": abstain_cnt,
        "abstain_rate_str": f"{abstain_cnt/n_tot:.1%} ({abstain_cnt}/{n_tot})",
        "total_cost_abs3": tot_cost_3,
        "mean_cost_abs3": tot_cost_3 / n_tot,
        "total_cost_abs0": tot_cost_0,
        "mean_cost_abs0": tot_cost_0 / n_tot,
    }


def main() -> None:
    t_start = time.time()
    cfg = load_config()
    cost_matrix = build_cost_matrix(cfg)
    pipeline = TriagePipeline.load(_ROOT / "models", cfg)

    print("Loading test and val datasets...")
    df_all = load_data(cfg, include_labels=True)
    df_cleaned = clean_dataframe(df_all, cfg)

    test_df = df_cleaned[df_cleaned["split"] == "test"].copy().reset_index(drop=True)
    val_df = df_cleaned[df_cleaned["split"] == "val"].copy().reset_index(drop=True)

    test_records = test_df.to_dict(orient="records")
    val_records = val_df.to_dict(orient="records")

    test_is_em = test_df["is_emergency"].values.astype(bool)
    test_true_y = test_df["clean_triage_level"].values.astype(int)

    val_is_em = val_df["is_emergency"].values.astype(bool)
    val_true_y = val_df["clean_triage_level"].values.astype(int)

    with open(_ROOT / "lexicon.yaml", "r", encoding="utf-8") as f:
        lexicon = yaml.safe_load(f)

    vague_phrases = lexicon.get("mild_phrases", {}).get("english", [
        "mild discomfort since morning nothing serious",
        "not feeling great just wanted to get checked",
        "slight uneasiness nothing major doctor"
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # 1. TEST Ablation with Costs (4 configs x 3 conditions + Baseline a)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Computing 1. TEST Ablation with Costs ---")
    # Datasets
    # 1a. Clean Test
    clean_test_recs = test_records

    # 1b. Vague Test
    vague_test_recs = []
    v_idx = 0
    for r in test_records:
        r_c = r.copy()
        if r["is_emergency"]:
            r_c["complaint_text"] = vague_phrases[v_idx % len(vague_phrases)]
            v_idx += 1
        vague_test_recs.append(r_c)

    # 1c. 20% Corrupt Test
    rng = np.random.default_rng(42)
    corrupt_test_recs = []
    vital_keys = ["bp_sys", "bp_dia", "pulse", "temp_c", "spo2"]
    for r in test_records:
        r_c = r.copy()
        for vk in vital_keys:
            if rng.uniform(0.0, 1.0) < 0.20:
                mode = rng.choice(["zero", "nan", "extreme"])
                if mode == "zero": r_c[vk] = 0.0
                elif mode == "nan": r_c[vk] = np.nan
                else: r_c[vk] = 999.0
        corrupt_test_recs.append(r_c)

    configs_display = [
        ("Full TriageGuard", "full"),
        ("Rules Disabled", "no_rules"),
        ("Screen Disabled", "no_screen"),
        ("Fusion Only", "fusion_only"),
        ("Baseline (a) Argmax", "baseline_a"),
    ]

    test_ablation_results = []
    test_datasets = [
        ("Clean Test", clean_test_recs),
        ("Vague-Complaint Stress", vague_test_recs),
        ("20% Vital Corruption", corrupt_test_recs),
    ]

    for d_name, d_recs in test_datasets:
        preds_d = evaluate_batch_predictions(d_recs, pipeline, cfg, cost_matrix, test_is_em, test_true_y)
        for c_label, c_key in configs_display:
            p_arr = preds_d[c_key]
            m = compute_metrics(p_arr, test_is_em, test_true_y, cost_matrix)
            test_ablation_results.append({
                "Stress Condition": d_name,
                "Configuration": c_label,
                "Emergency Recall (L1-L2)": m["strict_recall_str"],
                "Silent Misses (L3-L5)": m["silent_misses_str"],
                "Non-Emergency Escalation": m["non_em_esc_str"],
                "Abstain Rate": m["abstain_rate_str"],
                "Total Cost (C_abs=3.0)": f"{m['total_cost_abs3']:,.1f}",
                "Mean Cost (C_abs=3.0)": f"{m['mean_cost_abs3']:.3f}",
                "Total Cost (C_abs=0.0)": f"{m['total_cost_abs0']:,.1f}",
                "Mean Cost (C_abs=0.0)": f"{m['mean_cost_abs0']:.3f}",
            })

    df_test_ablation = pd.DataFrame(test_ablation_results)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. VAL Benchmarks (Clean & Vague-Complaint)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Computing 2. VAL Benchmarks ---")
    val_vague_recs = []
    v_idx = 0
    for r in val_records:
        r_c = r.copy()
        if r["is_emergency"]:
            r_c["complaint_text"] = vague_phrases[v_idx % len(vague_phrases)]
            v_idx += 1
        val_vague_recs.append(r_c)

    val_datasets = [
        ("Clean Validation", val_records),
        ("Vague-Complaint Validation", val_vague_recs),
    ]

    val_benchmark_results = []
    val_preds_cache = {}

    for d_name, d_recs in val_datasets:
        preds_v = evaluate_batch_predictions(d_recs, pipeline, cfg, cost_matrix, val_is_em, val_true_y)
        val_preds_cache[d_name] = preds_v
        for c_label, c_key in configs_display:
            p_arr = preds_v[c_key]
            m = compute_metrics(p_arr, val_is_em, val_true_y, cost_matrix)
            val_benchmark_results.append({
                "Validation Condition": d_name,
                "Configuration": c_label,
                "Strict Recall (L1-L2)": m["strict_recall_str"],
                "Strict 95% CI": m["strict_ci_str"],
                "Caught or Abstained": m["ca_recall_str"],
                "Caught/Abs 95% CI": m["ca_ci_str"],
                "Silent Misses": m["silent_misses_str"],
                "Non-Emergency Escalation": m["non_em_esc_str"],
                "Abstain Rate": m["abstain_rate_str"],
                "Total Cost (C_abs=3.0)": f"{m['total_cost_abs3']:,.1f}",
                "Mean Cost (C_abs=3.0)": f"{m['mean_cost_abs3']:.3f}",
                "Total Cost (C_abs=0.0)": f"{m['total_cost_abs0']:,.1f}",
                "Mean Cost (C_abs=0.0)": f"{m['mean_cost_abs0']:.3f}",
            })

    df_val_benchmarks = pd.DataFrame(val_benchmark_results)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Full Pipeline at Alpha in {0.01, 0.02, 0.05} on VAL and TEST
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Computing 3. Full Pipeline at Alphas on VAL and TEST ---")
    tau_map = pipeline.conformal.state.tau_map
    alpha_candidates = [0.01, 0.02, 0.05]
    alpha_results = []

    splits_eval = [
        ("Validation Split (N=3,000)", val_records, val_is_em, val_true_y),
        ("Test Split (N=4,500)", test_records, test_is_em, test_true_y),
    ]

    for split_label, s_recs, s_is_em, s_true_y in splits_eval:
        for alpha in alpha_candidates:
            tau = tau_map[alpha]
            # evaluate full system with this tau
            p_dict = evaluate_batch_predictions(
                s_recs, pipeline, cfg, cost_matrix, s_is_em, s_true_y, tau_override=tau
            )
            p_full = p_dict["full"]
            m = compute_metrics(p_full, s_is_em, s_true_y, cost_matrix)
            alpha_results.append({
                "Evaluation Split": split_label,
                "Conformal Alpha": alpha,
                "Calibrated Tau": f"{tau:.4f}",
                "Strict Recall (L1-L2)": m["strict_recall_str"],
                "Strict 95% CI": m["strict_ci_str"],
                "Caught or Abstained": m["ca_recall_str"],
                "Caught/Abs 95% CI": m["ca_ci_str"],
                "Silent Misses": m["silent_misses_str"],
                "Non-Emergency Escalation": m["non_em_esc_str"],
                "Abstain Rate": m["abstain_rate_str"],
                "Total Cost (C_abs=3.0)": f"{m['total_cost_abs3']:,.1f}",
                "Mean Cost (C_abs=3.0)": f"{m['mean_cost_abs3']:.3f}",
                "Total Cost (C_abs=0.0)": f"{m['total_cost_abs0']:,.1f}",
                "Mean Cost (C_abs=0.0)": f"{m['mean_cost_abs0']:.3f}",
            })

    df_alphas = pd.DataFrame(alpha_results)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Policy Table on VAL Only (Recall Floors 0.95, 0.98, 0.99)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Computing 4. Policy Table on VAL ---")
    # Candidates on VAL:
    # We consider:
    # - Full pipeline at alpha in {0.005, 0.01, 0.02, 0.05}
    # - Rules Disabled (at operating tau)
    # - Screen Disabled
    # - Fusion Only
    # - Baseline (a) Argmax
    val_candidates = []
    # Test all alphas on Full
    for alpha, tau in tau_map.items():
        p_dict = evaluate_batch_predictions(val_records, pipeline, cfg, cost_matrix, val_is_em, val_true_y, tau_override=tau)
        m = compute_metrics(p_dict["full"], val_is_em, val_true_y, cost_matrix)
        val_candidates.append({
            "candidate_name": f"Full Pipeline (alpha={alpha}, tau={tau:.4f})",
            "config": "full",
            "alpha": alpha,
            "tau": tau,
            "strict_recall": m["strict_recall"],
            "ca_recall": m["ca_recall"],
            "mean_cost_abs3": m["mean_cost_abs3"],
            "total_cost_abs3": m["total_cost_abs3"],
            "mean_cost_abs0": m["mean_cost_abs0"],
            "total_cost_abs0": m["total_cost_abs0"],
            "metrics": m,
        })

    # Add other configurations at default
    p_clean_val = val_preds_cache["Clean Validation"]
    for c_label, c_key in [
        ("Rules Disabled", "no_rules"),
        ("Screen Disabled", "no_screen"),
        ("Fusion Only", "fusion_only"),
        ("Baseline (a) Argmax", "baseline_a"),
    ]:
        m = compute_metrics(p_clean_val[c_key], val_is_em, val_true_y, cost_matrix)
        val_candidates.append({
            "candidate_name": c_label,
            "config": c_key,
            "alpha": None,
            "tau": None,
            "strict_recall": m["strict_recall"],
            "ca_recall": m["ca_recall"],
            "mean_cost_abs3": m["mean_cost_abs3"],
            "total_cost_abs3": m["total_cost_abs3"],
            "mean_cost_abs0": m["mean_cost_abs0"],
            "total_cost_abs0": m["total_cost_abs0"],
            "metrics": m,
        })

    floors = [0.95, 0.98, 0.99]
    policy_rows = []

    for floor in floors:
        # Constraint: strict recall >= floor
        feasible_strict = [c for c in val_candidates if c["strict_recall"] >= floor]
        if feasible_strict:
            best_strict = min(feasible_strict, key=lambda c: c["mean_cost_abs3"])
            best_strict_name = best_strict["candidate_name"]
            best_strict_cost = f"{best_strict['mean_cost_abs3']:.3f} (Total: {best_strict['total_cost_abs3']:,.1f})"
            best_strict_rec = f"{best_strict['strict_recall']:.1%}"
        else:
            best_strict_name = "None feasible"
            best_strict_cost = "N/A"
            best_strict_rec = "N/A"

        # Constraint: caught-or-abstained recall >= floor
        feasible_ca = [c for c in val_candidates if c["ca_recall"] >= floor]
        if feasible_ca:
            best_ca = min(feasible_ca, key=lambda c: c["mean_cost_abs3"])
            best_ca_name = best_ca["candidate_name"]
            best_ca_cost = f"{best_ca['mean_cost_abs3']:.3f} (Total: {best_ca['total_cost_abs3']:,.1f})"
            best_ca_rec = f"{best_ca['ca_recall']:.1%}"
        else:
            best_ca_name = "None feasible"
            best_ca_cost = "N/A"
            best_ca_rec = "N/A"

        policy_rows.append({
            "Recall Floor": f"{floor:.0%}",
            "Min-Cost Config (Strict Recall >= Floor)": best_strict_name,
            "Strict Recall on Val": best_strict_rec,
            "Val Mean Cost (Strict Gate)": best_strict_cost,
            "Min-Cost Config (Caught/Abs Recall >= Floor)": best_ca_name,
            "Caught/Abs Recall on Val": best_ca_rec,
            "Val Mean Cost (Caught/Abs Gate)": best_ca_cost,
        })

    df_policy = pd.DataFrame(policy_rows)

    rule_explanation = (
        "Optimization Rule Used:\n"
        "  arg min_{Config, alpha} Mean_Cost_VAL(Config, alpha | C_abstain=3.0)\n"
        "  subject to: Empirical_Recall_VAL(Config, alpha) >= Target_Floor\n"
        "Where:\n"
        "  - Strict Recall Gate requires Level in {1, 2} for emergency encounters.\n"
        "  - Caught-or-Abstained Gate requires Level in {1, 2} OR safe referral to nurse (abstain) for emergency encounters.\n"
        "  - Evaluated over candidate alphas {0.005, 0.01, 0.02, 0.05} on Full Pipeline plus architectural ablations."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Output JSON & Markdown Generation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Generating Output Artifacts ---")
    output_dict = {
        "metadata": {
            "title": "TriageGuard Operating Points & Layer Ablation Benchmark",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "execution_duration_sec": round(time.time() - t_start, 2),
            "cost_matrix_emergency_penalty": "20x",
            "default_operating_alpha": pipeline.conformal.state.operating_alpha,
            "default_operating_tau": round(pipeline.conformal.state.operating_tau, 4),
        },
        "test_ablation_with_costs": test_ablation_results,
        "val_ablation_benchmarks": val_benchmark_results,
        "full_pipeline_alpha_tradeoff": alpha_results,
        "val_policy_table": policy_rows,
        "policy_rule_used": rule_explanation,
    }

    json_path = _RESULTS_DIR / "operating_points.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2)

    md_lines = [
        "# TriageGuard Operating Points & Layer Ablation Benchmark",
        "",
        "> **Report-Only Deliverable:** Evaluates layer configurations, operational trade-offs, and validation policy selection without modifying underlying models, thresholds, or production defaults.",
        "",
        "---",
        "",
        "## 1. Layer Ablation with Triage Costs on TEST Split (N=4,500)",
        "",
        "Evaluates 4 layer configurations plus Baseline (a) Argmax across Clean, Vague Complaints, and 20% Vital Corruption.",
        "",
        df_test_ablation.to_markdown(index=False),
        "",
        "---",
        "",
        "## 2. Layer Configurations on VALIDATION Split (N=3,000)",
        "",
        "Evaluates performance on Validation split across unperturbed encounters and vague-complaint emergencies ($N=120$ emergencies).",
        "",
        df_val_benchmarks.to_markdown(index=False),
        "",
        "---",
        "",
        "## 3. Full Pipeline Conformal Trade-off at Alpha in {0.01, 0.02, 0.05}",
        "",
        "Evaluates Full TriageGuard operating under pre-calibrated conformal thresholds (without recomputing $\\tau$):",
        "- $\\alpha = 0.01 \\implies \\tau = 0.0018$",
        "- $\\alpha = 0.02 \\implies \\tau = 0.0039$",
        "- $\\alpha = 0.05 \\implies \\tau = 0.0922$ (Current Operating Default)",
        "",
        df_alphas.to_markdown(index=False),
        "",
        "---",
        "",
        "## 4. Policy Table on VALIDATION Split (Recall Floors 95%, 98%, 99%)",
        "",
        df_policy.to_markdown(index=False),
        "",
        "### Optimization Rule Specification",
        "```text",
        rule_explanation,
        "```",
        "",
        "---",
        "",
        f"*Generated programmatically in {round(time.time() - t_start, 2)} seconds. Models, thresholds, and defaults remain untouched.*",
    ]

    md_path = _RESULTS_DIR / "operating_points.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"✓ Saved {json_path}")
    print(f"✓ Saved {md_path}")
    print(f"Total time: {round(time.time() - t_start, 2)}s")


if __name__ == "__main__":
    main()
