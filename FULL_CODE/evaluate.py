#!/usr/bin/env python3
"""H1 OPD Triage Assistant - official scorer (TalentAP Mega Hackathon 2026).

    python evaluate.py --submission submission/H1_predictions.csv --truth data/train_labels.csv [--out score.json]

Standard library + numpy + pandas only. Deterministic.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- constants (also in README)
COST_MISSED_EMERGENCY = 20.0     # true emergency (level 1/2) predicted level 3, 4 or 5
COST_UNDER_PER_LEVEL = 2.0       # non-emergency predicted too LOW priority (p > t): 2 x (p - t)
COST_OVER_PER_LEVEL = 1.0        # predicted too HIGH priority (p < t): 1 x (t - p), capped below
COST_OVER_CAP = 4.0              # over-triaging a level-5 patient to level 1 costs at most 4
COST_ABSTAIN_ASSESSABLE = 1.5    # ABSTAIN on an assessable non-emergency patient (nurse time)
COST_ABSTAIN_OK = 0.0            # ABSTAIN when the case is not assessable or is a true emergency
REFERENCE_LEVEL = 4              # reference system: predicts level 4 for everyone, never abstains
W_SAFETY, W_RECALL, W_DEPT, W_RF = 0.6, 0.2, 0.1, 0.1
MISSING_PREDICTION_LEVEL = 5     # a truth visit with no prediction is scored as level 5 (worst case)

DEPARTMENTS = ["General Medicine", "Paediatrics", "Obstetrics & Gynaecology", "General Surgery", "Orthopaedics", "ENT",
               "Ophthalmology", "Dermatology", "Psychiatry", "Dental", "Cardiology", "Pulmonology/Chest & TB",
               "Emergency/Casualty"]
RED_FLAG_CODES = ["RF_CHEST_PAIN_CARDIAC", "RF_STROKE_SIGNS", "RF_RESP_DISTRESS", "RF_ALTERED_SENSORIUM", "RF_SEIZURE",
                  "RF_SEVERE_BLEEDING", "RF_PREGNANCY_BLEEDING", "RF_ECLAMPSIA_SIGNS", "RF_REDUCED_FETAL_MOVEMENT",
                  "RF_POISONING", "RF_SNAKEBITE", "RF_SCORPION_STING", "RF_SEVERE_DEHYDRATION", "RF_HIGH_FEVER_INFANT",
                  "RF_CHILD_DANGER_SIGNS", "RF_SEPSIS_SIGNS", "RF_MENINGITIS_SIGNS", "RF_DENGUE_WARNING", "RF_SUICIDAL",
                  "RF_ACUTE_PSYCHOSIS_AGITATION", "RF_TRAUMA_MAJOR", "RF_ANAPHYLAXIS", "RF_DKA_SIGNS", "RF_HYPOGLYCAEMIA",
                  "RF_HEAT_STROKE", "RF_ACUTE_ABDOMEN", "RF_THUNDERCLAP_HEADACHE", "RF_TESTICULAR_TORSION",
                  "RF_EYE_EMERGENCY", "RF_BURNS_MAJOR"]
SUB_COLS = ["visit_id", "triage_level", "department", "red_flags", "confidence"]
TRUTH_COLS = ["visit_id", "triage_level", "department", "true_emergency", "red_flags", "assessable"]
_DEPT_CANON = {d.lower(): d for d in DEPARTMENTS}
_RF_SET = set(RED_FLAG_CODES)


def fail(msg):
    print("\nSUBMISSION FORMAT ERROR\n-----------------------\n" + msg +
          "\n\nFix the file and run evaluate.py again. Expected columns: " + ", ".join(SUB_COLS), file=sys.stderr)
    sys.exit(2)


def _split_flags(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return []
    return [t.strip().upper() for t in s.replace(",", ";").split(";") if t.strip()]


def load_submission(path):
    try:
        sub = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    except FileNotFoundError:
        fail(f"File not found: {path}")
    except Exception as e:  # noqa: BLE001
        fail(f"Could not read {path} as UTF-8 CSV: {e}")
    sub.columns = [c.strip().lower() for c in sub.columns]
    dupc = sorted({c for c in sub.columns if list(sub.columns).count(c) > 1})
    if dupc:
        fail(f"Duplicate column name(s) after ignoring case/spaces: {dupc}. Keep exactly one column of each name.")
    missing = [c for c in SUB_COLS if c not in sub.columns]
    if missing:
        fail(f"Missing column(s): {missing}. Found columns: {list(sub.columns)}")
    sub = sub[SUB_COLS].copy()
    sub["visit_id"] = sub["visit_id"].str.strip()
    if (sub["visit_id"] == "").any():
        fail(f"{int((sub['visit_id'] == '').sum())} row(s) have an empty visit_id.")
    dup = sub["visit_id"][sub["visit_id"].duplicated()]
    if len(dup):
        fail(f"{len(dup)} duplicate visit_id(s), e.g. {list(dup.unique()[:5])}. Each visit must appear once.")
    # triage_level: integer 1-5 or ABSTAIN
    lv = []
    bad = []
    for vid, raw in zip(sub["visit_id"], sub["triage_level"]):
        r = str(raw).strip()
        if r.upper() == "ABSTAIN":
            lv.append(0)
            continue
        try:
            f = float(r)
            if f.is_integer() and 1 <= int(f) <= 5:
                lv.append(int(f))
                continue
        except ValueError:
            pass
        bad.append((vid, raw))
        lv.append(-1)
    if bad:
        fail(f"{len(bad)} invalid triage_level value(s), e.g. {bad[:5]}. Allowed: 1, 2, 3, 4, 5 or ABSTAIN.")
    sub["p_level"] = lv  # 0 = ABSTAIN
    # department
    dep = sub["department"].str.strip()
    canon = dep.str.lower().map(_DEPT_CANON)
    bad_d = sub.loc[canon.isna() & ~((dep == "") & (sub["p_level"] == 0)), ["visit_id", "department"]]
    if len(bad_d):
        fail(f"{len(bad_d)} unknown department value(s), e.g. {bad_d.values[:5].tolist()}.\n"
             f"Allowed (case-insensitive): {DEPARTMENTS}. Department may be empty only when triage_level is ABSTAIN.")
    sub["p_dept"] = canon.fillna("")
    # red flags
    flags, bad_f = [], []
    for vid, raw in zip(sub["visit_id"], sub["red_flags"]):
        fl = _split_flags(raw)
        unk = [f for f in fl if f not in _RF_SET]
        if unk:
            bad_f.append((vid, unk))
        flags.append(sorted(set(fl)))
    if bad_f:
        fail(f"{len(bad_f)} row(s) contain unknown red-flag codes, e.g. {bad_f[:3]}.\n"
             f"Use codes from data/red_flag_codes.csv separated by ';' (or leave empty).")
    sub["p_flags"] = flags
    # confidence: number in [0,1] or empty
    conf = pd.to_numeric(sub["confidence"].replace("", np.nan), errors="coerce")
    bad_c = sub.loc[(conf.isna() & (sub["confidence"].str.strip() != "")) | (conf < 0) | (conf > 1), ["visit_id", "confidence"]]
    if len(bad_c):
        fail(f"{len(bad_c)} invalid confidence value(s), e.g. {bad_c.values[:5].tolist()}. Use a number between 0 and 1 (or leave empty).")
    sub["p_conf"] = conf
    return sub


def load_truth(path):
    t = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    miss = [c for c in TRUTH_COLS if c not in t.columns]
    if miss:
        print(f"Truth file is missing columns {miss}", file=sys.stderr)
        sys.exit(3)
    t["t_level"] = t["triage_level"].astype(int)
    t["t_emerg"] = t["true_emergency"].astype(int)
    t["t_assess"] = t["assessable"].astype(int)
    t["t_flags"] = t["red_flags"].map(lambda x: sorted(set(_split_flags(x))))
    t["t_dept"] = t["department"].str.strip()
    return t


def case_cost(t, p, emerg, assess):
    """t: true level 1-5; p: predicted level 1-5 or 0 (=ABSTAIN)."""
    if p == 0:
        return COST_ABSTAIN_OK if (assess == 0 or emerg == 1) else COST_ABSTAIN_ASSESSABLE
    if emerg == 1:
        return 0.0 if p <= 2 else COST_MISSED_EMERGENCY
    if p > t:
        return COST_UNDER_PER_LEVEL * (p - t)
    if p < t:
        return min(COST_OVER_PER_LEVEL * (t - p), COST_OVER_CAP)
    return 0.0


def score(sub, truth):
    m = truth.merge(sub, on="visit_id", how="left")
    n_missing = int(m["p_level"].isna().sum())
    missing_ids = m.loc[m["p_level"].isna(), "visit_id"].tolist()[:5]
    m["p_level"] = m["p_level"].fillna(MISSING_PREDICTION_LEVEL).astype(int)
    m["p_dept"] = m["p_dept"].fillna("")
    m["p_flags"] = m["p_flags"].map(lambda x: x if isinstance(x, list) else [])
    costs = np.array([case_cost(t, p, e, a) for t, p, e, a in zip(m.t_level, m.p_level, m.t_emerg, m.t_assess)])
    ref = np.array([case_cost(t, REFERENCE_LEVEL, e, a) for t, e, a in zip(m.t_level, m.t_emerg, m.t_assess)])
    team_cost, ref_cost = float(costs.sum()), float(ref.sum())
    safety = float(np.clip(1 - team_cost / ref_cost, -1, 1)) if ref_cost > 0 else 0.0
    em = m[m.t_emerg == 1]
    caught = ((em.p_level >= 1) & (em.p_level <= 2)) | (em.p_level == 0)
    recall = float(caught.mean()) if len(em) else 1.0
    missed_ids = em.loc[~caught, "visit_id"].tolist()
    dm = m[(m.p_level != 0) & (m.t_assess == 1)]
    dept_acc = float((dm.p_dept == dm.t_dept).mean()) if len(dm) else 0.0
    tp = fp = fn = 0
    for tf, pf in zip(m.t_flags, m.p_flags):
        ts, ps = set(tf), set(pf)
        tp += len(ts & ps)
        fp += len(ps - ts)
        fn += len(ts - ps)
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    rf_f1 = 1.0 if (tp + fp + fn) == 0 else (2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    primary = W_SAFETY * safety + W_RECALL * recall + W_DEPT * dept_acc + W_RF * rf_f1
    nonab = m[m.p_level != 0]
    under_nonem = int(((nonab.t_emerg == 0) & (nonab.p_level > nonab.t_level)).sum())
    over = int((nonab.p_level < nonab.t_level).sum())
    conf = pd.crosstab(m.t_level, m.p_level.map(lambda v: "ABSTAIN" if v == 0 else str(v)))
    abst = m[m.p_level == 0]
    res = dict(
        primary_score=round(primary, 6),
        primary_score_range="[-0.6, 1.0], higher is better",
        emergency_recall=round(recall, 6),
        n_true_emergencies=int(len(em)),
        n_missed_emergencies=int((~caught).sum()),
        missed_emergency_examples=missed_ids[:10],
        normalised_safety_score=round(safety, 6),
        team_cost=round(team_cost, 3),
        reference_cost_all_level4=round(ref_cost, 3),
        mean_cost_per_case=round(team_cost / len(m), 6),
        department_accuracy=round(dept_acc, 6),
        red_flag_micro_f1=round(rf_f1, 6),
        red_flag_precision=round(prec, 6),
        red_flag_recall=round(rec, 6),
        exact_level_accuracy_non_abstained=round(float((nonab.p_level == nonab.t_level).mean()) if len(nonab) else 0.0, 6),
        n_scored=int(len(m)),
        n_missing_predictions=n_missing,
        missing_prediction_examples=missing_ids,
        n_abstain=int(len(abst)),
        n_abstain_on_unassessable=int((abst.t_assess == 0).sum()),
        n_unassessable_in_truth=int((m.t_assess == 0).sum()),
        n_under_triage_non_emergency=under_nonem,
        n_over_triage=over,
        cost_breakdown=dict(
            missed_emergencies=round(float(costs[(m.t_emerg == 1).values & (m.p_level.values >= 3)].sum()), 3),
            under_triage_non_emergency=round(float(costs[(m.t_emerg == 0).values & (m.p_level.values > m.t_level.values) & (m.p_level.values != 0)].sum()), 3),
            over_triage=round(float(costs[(m.p_level.values < m.t_level.values) & (m.p_level.values != 0) & (m.t_emerg == 0).values].sum()), 3),
            abstain=round(float(costs[m.p_level.values == 0].sum()), 3),
        ),
        confusion_true_by_pred={str(k): {str(c): int(v) for c, v in row.items()} for k, row in conf.iterrows()},
        weights=dict(safety=W_SAFETY, emergency_recall=W_RECALL, department_accuracy=W_DEPT, red_flag_f1=W_RF),
    )
    return res


def report(res):
    bar = "=" * 64
    print(bar)
    print("  H1 OPD TRIAGE - EVALUATION")
    print(bar)
    print(f"  EMERGENCY RECALL (sensitivity): {res['emergency_recall']:.4f}   "
          f"({res['n_true_emergencies'] - res['n_missed_emergencies']}/{res['n_true_emergencies']} true emergencies caught)")
    if res["n_missed_emergencies"]:
        print(f"  !! {res['n_missed_emergencies']} TRUE EMERGENCIES UNDER-TRIAGED (cost {COST_MISSED_EMERGENCY:g} each), "
              f"e.g. {res['missed_emergency_examples'][:5]}")
    print(bar)
    print(f"  PRIMARY SCORE: {res['primary_score']:.4f}   (range {res['primary_score_range']})")
    print(f"    = {W_SAFETY} x safety {res['normalised_safety_score']:.4f}"
          f" + {W_RECALL} x emergency recall {res['emergency_recall']:.4f}"
          f" + {W_DEPT} x department accuracy {res['department_accuracy']:.4f}"
          f" + {W_RF} x red-flag F1 {res['red_flag_micro_f1']:.4f}")
    print(f"  Safety cost: team {res['team_cost']:.1f} vs reference (all level 4) {res['reference_cost_all_level4']:.1f}")
    cb = res["cost_breakdown"]
    print(f"    missed emergencies {cb['missed_emergencies']:.1f} | under-triage {cb['under_triage_non_emergency']:.1f} | "
          f"over-triage {cb['over_triage']:.1f} | abstain {cb['abstain']:.1f}")
    print(f"  Abstained: {res['n_abstain']} (of which {res['n_abstain_on_unassessable']} were truly unassessable; "
          f"{res['n_unassessable_in_truth']} unassessable cases exist)")
    print(f"  Exact level accuracy (non-abstained): {res['exact_level_accuracy_non_abstained']:.4f}   "
          f"red-flag P/R: {res['red_flag_precision']:.3f}/{res['red_flag_recall']:.3f}")
    if res["n_missing_predictions"]:
        print(bar)
        print(f"  WARNING: {res['n_missing_predictions']} visits in the truth file have NO prediction "
              f"(scored as level {MISSING_PREDICTION_LEVEL}, worst case). e.g. {res['missing_prediction_examples']}")
    print(bar)


def main():
    ap = argparse.ArgumentParser(description="Score an H1 triage submission.")
    ap.add_argument("--submission", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out", default="score.json")
    a = ap.parse_args()
    sub = load_submission(a.submission)
    truth = load_truth(a.truth)
    res = score(sub, truth)
    report(res)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"  Wrote {a.out}")


if __name__ == "__main__":
    main()
