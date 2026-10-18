#!/usr/bin/env python3
"""
scripts/rules_report.py
──────────────────────────────────────────────────────────────────────────────
Validation report for Phase 1 (Input Cleaning) and Phase 3 (Rule Layer).

Evaluates the rule layer on the 3,000-case validation split:
1. Emergency recall (vitals rules only), including under-reported emergencies.
2. False escalation split by:
   (a) Vitals rules only (forced_level <= 2)
   (b) None (no vital escalation)
   Across true triage levels L3, L4, L5, and overall.
3. Soft red-flag firing rates on non-emergencies and emergencies.
4. Rule-by-rule firing counts and precision on emergencies.
5. Red-flag firing counts and precision on emergencies.
6. Independence check: verify no phrases in redflags.yaml are copied from
   lexicon.yaml's emergency pools, and print overlap count.

Run:
    .venv/bin/python scripts/rules_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Set

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import yaml

from triageguard.cleaning import clean_record
from triageguard.rules import apply_rules

_DATA_DIR = _ROOT / "data"
_REDFLAGS_PATH = _ROOT / "redflags.yaml"
_RULES_PATH = _ROOT / "rules.yaml"
_LEXICON_PATH = _ROOT / "lexicon.yaml"
_ARCHETYPES_PATH = _ROOT / "archetypes.yaml"


def sep(title: str) -> None:
    bar = "═" * 78
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def check_lexicon_independence() -> int:
    """Verify zero overlap between redflags.yaml and emergency lexicon pools."""
    with open(_REDFLAGS_PATH, encoding="utf-8") as f:
        rf = yaml.safe_load(f)
    with open(_LEXICON_PATH, encoding="utf-8") as f:
        lex = yaml.safe_load(f)
    with open(_ARCHETYPES_PATH, encoding="utf-8") as f:
        arch = yaml.safe_load(f)

    emergency_arch_names = {
        a["name"] for a in arch["archetypes"] if a["triage_level"] in (1, 2)
    }

    emergency_lex_phrases: Set[str] = set()
    for arch_name in emergency_arch_names:
        if arch_name in lex.get("archetypes", {}):
            arch_data = lex["archetypes"][arch_name]
            for lang_key, phrases in arch_data.items():
                if isinstance(phrases, list):
                    for p in phrases:
                        emergency_lex_phrases.add(p.strip().lower())

    for a in arch["archetypes"]:
        if a["triage_level"] in (1, 2):
            for p in a.get("complaint_templates_en", []):
                emergency_lex_phrases.add(p.strip().lower())

    rf_phrases: Set[str] = set()
    for r in rf.get("red_flags", []):
        for k in ["patterns_en", "patterns_tel", "patterns_hin"]:
            for p in r.get(k, []):
                rf_phrases.add(p.strip().lower())

    overlap = rf_phrases.intersection(emergency_lex_phrases)
    overlap_count = len(overlap)

    sep("INDEPENDENCE CHECK: redflags.yaml vs lexicon.yaml Emergency Pools")
    print(f"  Total emergency lexicon & archetype phrases : {len(emergency_lex_phrases):>4}")
    print(f"  Total clinical red-flag phrases (redflags)  : {len(rf_phrases):>4}")
    print(f"  Phrase overlap count                        : {overlap_count:>4}")
    if overlap_count == 0:
        print("  ✓ PASS: Zero phrases copied from generator emergency pools.")
    else:
        print(f"  ✗ FAIL: Overlapping phrases found: {sorted(overlap)}")

    return overlap_count


def evaluate_val_split() -> None:
    sep("EVALUATING PHASE 1 & PHASE 3 ON VALIDATION SPLIT")

    intake_path = _DATA_DIR / "synthetic_intake.csv"
    labels_path = _DATA_DIR / "hidden_labels.csv"
    splits_path = _DATA_DIR / "splits.csv"

    if not (intake_path.exists() and labels_path.exists() and splits_path.exists()):
        print("Data files not found in data/. Run data generation first.")
        sys.exit(1)

    intake_df = pd.read_csv(intake_path)
    labels_df = pd.read_csv(labels_path)
    splits_df = pd.read_csv(splits_path)

    val_df = (
        intake_df.merge(splits_df[splits_df["split"] == "val"], on="patient_id")
        .merge(
            labels_df[
                [
                    "patient_id",
                    "clean_triage_level",
                    "is_emergency",
                    "reporting_bias",
                    "archetype",
                ]
            ],
            on="patient_id",
        )
    )

    n_val = len(val_df)
    print(f"Loaded {n_val:,} validation records.")

    with open(_RULES_PATH, encoding="utf-8") as f:
        rules_cfg = yaml.safe_load(f)
    with open(_REDFLAGS_PATH, encoding="utf-8") as f:
        redflags_cfg = yaml.safe_load(f)

    # Process all rows
    eval_records: List[Dict] = []
    for _, row in val_df.iterrows():
        rec = row.to_dict()
        cr = clean_record(rec)
        rr = apply_rules(cr, age=int(row["age"]), rules_config=rules_cfg, redflags_config=redflags_cfg)

        eval_records.append(
            {
                "patient_id": row["patient_id"],
                "clean_level": int(row["clean_triage_level"]),
                "is_emergency": bool(row["is_emergency"]),
                "reporting_bias": str(row["reporting_bias"]),
                "forced_level": rr.forced_level,
                "triggered_rule_ids": rr.triggered_rule_ids,
                "red_flags_detected": rr.red_flags_detected,
                "no_downgrade_below_L3": rr.no_downgrade_below_L3,
                "needs_remeasure": cr.needs_remeasure,
                "abstain_reason": rr.abstain_reason,
            }
        )

    res = pd.DataFrame(eval_records)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Emergency Recall (Vitals Rules Only)
    # ─────────────────────────────────────────────────────────────────────────
    sep("1. EMERGENCY RECALL (Vitals Rules Only — FORCED)")
    em = res[res["is_emergency"]]
    n_em = len(em)
    em_caught = em[em["forced_level"].notnull()]
    n_em_caught = len(em_caught)
    em_recall = n_em_caught / n_em if n_em > 0 else 0.0

    print(f"{'Emergency Cohort':<35} {'Caught':>8} {'Total':>8} {'Recall':>10}")
    print("-" * 65)

    # Total emergency
    print(f"{'All Emergencies (Level 1 + Level 2)':<35} {n_em_caught:>8} {n_em:>8} {em_recall:>9.1%}")

    # By level
    for lvl in [1, 2]:
        sub = em[em["clean_level"] == lvl]
        sub_caught = sub[sub["forced_level"].notnull()]
        sub_rec = len(sub_caught) / len(sub) if len(sub) > 0 else 0.0
        print(f"  Level {lvl} Emergencies{'':<13} {len(sub_caught):>8} {len(sub):>8} {sub_rec:>9.1%}")

    # By reporting bias
    sub_under = em[em["reporting_bias"] == "under_report"]
    caught_under = sub_under[sub_under["forced_level"].notnull()]
    rec_under = len(caught_under) / len(sub_under) if len(sub_under) > 0 else 0.0
    print(f"  Under-Reported Emergencies{'':<7} {len(caught_under):>8} {len(sub_under):>8} {rec_under:>9.1%}")

    sub_norm = em[em["reporting_bias"] != "under_report"]
    caught_norm = sub_norm[sub_norm["forced_level"].notnull()]
    rec_norm = len(caught_norm) / len(sub_norm) if len(sub_norm) > 0 else 0.0
    print(f"  Standard Emergencies{'':<13} {len(caught_norm):>8} {len(sub_norm):>8} {rec_norm:>9.1%}")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. False Escalation Rate (Vitals Rules vs None, by True Level)
    # ─────────────────────────────────────────────────────────────────────────
    sep("2. FALSE ESCALATION ON NON-EMERGENCIES (Split by Vital Rules vs None)")
    non_em = res[~res["is_emergency"]]
    n_non_em = len(non_em)

    print(
        f"{'True Triage Level':<20} {'Total':>8} {'(a) Vitals Only':>18} {'Vitals %':>10} {'(b) None (Safe)':>18} {'None %':>10}"
    )
    print("-" * 88)

    for lvl in [3, 4, 5]:
        sub = non_em[non_em["clean_level"] == lvl]
        n_sub = len(sub)
        n_vital_esc = len(sub[sub["forced_level"].notnull()])
        n_none = n_sub - n_vital_esc
        pct_vital = n_vital_esc / n_sub if n_sub > 0 else 0.0
        pct_none = n_none / n_sub if n_sub > 0 else 0.0
        print(
            f"Level {lvl} Non-Emergency {n_sub:>8} {n_vital_esc:>18} {pct_vital:>9.1%} {n_none:>18} {pct_none:>9.1%}"
        )

    # Total non-emergencies
    n_tot_vital_esc = len(non_em[non_em["forced_level"].notnull()])
    n_tot_none = n_non_em - n_tot_vital_esc
    pct_tot_vital = n_tot_vital_esc / n_non_em if n_non_em > 0 else 0.0
    pct_tot_none = n_tot_none / n_non_em if n_non_em > 0 else 0.0
    print("-" * 88)
    print(
        f"{'All Non-Emergencies':<20} {n_non_em:>8} {n_tot_vital_esc:>18} {pct_tot_vital:>9.1%} {n_tot_none:>18} {pct_tot_none:>9.1%}"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Red-Flag Firing Rates (Soft Guard: no_downgrade_below_L3)
    # ─────────────────────────────────────────────────────────────────────────
    sep("3. SOFT RED-FLAG TEXT FIRING RATES (no_downgrade_below_L3)")
    print(f"{'Cohort':<25} {'Total':>8} {'Red Flag Triggered':>22} {'Firing Rate':>14}")
    print("-" * 72)

    # On emergencies
    n_em_rf = len(em[em["no_downgrade_below_L3"]])
    pct_em_rf = n_em_rf / n_em if n_em > 0 else 0.0
    print(f"{'True Emergencies (L1-2)':<25} {n_em:>8} {n_em_rf:>22} {pct_em_rf:>13.1%}")

    # On non-emergencies by level
    for lvl in [3, 4, 5]:
        sub = non_em[non_em["clean_level"] == lvl]
        n_sub = len(sub)
        n_rf = len(sub[sub["no_downgrade_below_L3"]])
        pct_rf = n_rf / n_sub if n_sub > 0 else 0.0
        print(f"Level {lvl} Non-Emergency {n_sub:>8} {n_rf:>22} {pct_rf:>13.1%}")

    n_non_em_rf = len(non_em[non_em["no_downgrade_below_L3"]])
    pct_non_em_rf = n_non_em_rf / n_non_em if n_non_em > 0 else 0.0
    print("-" * 72)
    print(f"{'All Non-Emergencies':<25} {n_non_em:>8} {n_non_em_rf:>22} {pct_non_em_rf:>13.1%}")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Rule-by-Rule Firing Counts (Vitals Rules)
    # ─────────────────────────────────────────────────────────────────────────
    sep("4. RULE-BY-RULE FIRING COUNTS (Vitals Rules R001–R013)")
    print(f"{'Rule ID':<8} {'Max Lvl':<8} {'Total Fired':>12} {'Emergencies (TP)':>18} {'Non-Emerg (FP)':>16} {'Precision':>12}")
    print("-" * 78)

    for rule in rules_cfg.get("rules", []):
        r_id = rule["id"]
        lvl = rule["forced_max_level"]
        total_fired = sum(r_id in row_rules for row_rules in res["triggered_rule_ids"])
        tp = sum(
            r_id in row_rules
            for row_rules, is_e in zip(res["triggered_rule_ids"], res["is_emergency"])
            if is_e
        )
        fp = total_fired - tp
        prec = tp / total_fired if total_fired > 0 else 0.0
        print(f"{r_id:<8} L{lvl:<7} {total_fired:>12} {tp:>18} {fp:>16} {prec:>11.1%}")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Red-Flag Firing Counts (RF001–RF010)
    # ─────────────────────────────────────────────────────────────────────────
    sep("5. RED-FLAG TEXT PATTERN FIRING COUNTS (RF001–RF010)")
    print(f"{'Flag ID':<8} {'Flag Name':<28} {'Total Fired':>12} {'Emergencies':>14} {'Non-Emerg':>12}")
    print("-" * 78)

    for rf in redflags_cfg.get("red_flags", []):
        rf_id = rf["id"]
        rf_name = rf["name"]
        total_fired = sum(rf_name in row_flags for row_flags in res["red_flags_detected"])
        tp = sum(
            rf_name in row_flags
            for row_flags, is_e in zip(res["red_flags_detected"], res["is_emergency"])
            if is_e
        )
        fp = total_fired - tp
        print(f"{rf_id:<8} {rf_name:<28} {total_fired:>12} {tp:>14} {fp:>12}")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Abstain Reason Counts
    # ─────────────────────────────────────────────────────────────────────────
    sep("6. ABSTAIN REASON SUMMARY (needs_remeasure and no forced rule)")
    n_remeasure = sum(res["abstain_reason"] == "remeasure")
    print(f"Total validation cases with abstain_reason == 'remeasure': {n_remeasure} ({n_remeasure/n_val:.1%})")


if __name__ == "__main__":
    check_lexicon_independence()
    evaluate_val_split()
