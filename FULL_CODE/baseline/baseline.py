#!/usr/bin/env python3
"""H1 baseline: clean the recorded vitals, apply the primer's vital-sign rules, then simple keyword rules.

    python baseline/baseline.py --input data/phase2/intake_test.csv --out submission/H1_predictions.csv
    python baseline/baseline.py --input data/phase2/intake_train.csv --out baseline_train_predictions.csv

Deliberately simple and honest (no machine learning). It is a starting point, not a good triage system.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

RF_KEYWORDS = {
    "RF_CHEST_PAIN_CARDIAC": ["chest pain", "chest mein dard", "seene mein dard", "chhati mein dard", "chest heaviness",
                              "pressure in chest", "सीने में दर्द", "छाती", "buke byatha", "nenju vali", "chaati noppi", "chhatit dukh"],
    "RF_STROKE_SIGNS": ["face deviated", "mouth tilted", "muh tedha", "munh tedha", "slurred", "zubaan", "paralysis",
                        "one side", "ek side", "ek taraf ka haath", "मुँह टेढ़ा"],
    "RF_RESP_DISTRESS": ["cannot breathe", "gasping", "saans nahi le", "saans bilkul", "saans ukhad", "pasli chal",
                         "साँस नहीं ले"],
    "RF_ALTERED_SENSORIUM": ["unconscious", "not responding", "behosh", "hosh nahi", "confused", "pehchaan nahi", "बेहोश"],
    "RF_SEIZURE": ["fits", "jhatke", "convuls", "seizure", "mirgi", "daura", "झटके"],
    "RF_SEVERE_BLEEDING": ["vomiting blood", "blood in vomit", "khoon ki ulti", "black stool", "kaali tatti", "khoon band nahi",
                           "bleeding a lot", "lot of blood"],
    "RF_PREGNANCY_BLEEDING": ["bleeding in pregnancy", "pregnancy mein bleeding", "garbh mein khoon", "after delivery"],
    "RF_POISONING": ["poison", "pesticide", "insecticide", "zeher", "keede maarne", "keetnashak", "kerosene", "mitti ka tel",
                     "goliyan kha", "tablets", "ज़हर", "कीटनाशक"],
    "RF_SNAKEBITE": ["snake", "saanp", "saap", "paamu", "paambu", "shaap", "साँप"],
    "RF_SCORPION_STING": ["scorpion", "bichhu", "bichhoo", "vinchu", "बिच्छू"],
    "RF_SUICIDAL": ["suicide", "end life", "ending life", "marne ka", "marne ke", "मरने"],
    "RF_ANAPHYLAXIS": ["lips and face swollen", "hont aur", "throat closing", "gala band"],
    "RF_TRAUMA_MAJOR": ["accident", "fell from roof", "fell from tree", "chhat se gir", "ped se gir"],
    "RF_EYE_EMERGENCY": ["chuna", "chemical", "acid", "tezaab"],
    "RF_BURNS_MAJOR": ["burnt", "burns", "jal gay"],
    "RF_THUNDERCLAP_HEADACHE": ["worst headache", "sar phat"],
}
L3_KEYWORDS = ["fracture", "haddi toot", "bone", "appendi", "right side neeche", "stone", "pathri", "jaundice", "peeliya",
               "2 weeks", "weeks", "tb", "khansi jaa hi", "10-12", "deep wound"]
L5_KEYWORDS = ["refill", "dawai lene", "dawai khatam", "medicine", "certificate", "chashma", "glasses", "spectacles",
               "itching", "khujli", "ringworm", "daad", "checkup", "check up", "cold", "zukaam", "acidity", "gas", "tooth",
               "daant", "back pain", "kamar", "knee", "ghutn", "pimple", "report"]
DEPT_KEYWORDS = [
    ("Obstetrics & Gynaecology", ["pregnan", "pet se", "garbh", "period", "mahina", "white discharge", "safed paani", "anc"]),
    ("Dental", ["tooth", "daant", "daadh", "gum", "masood", "pannu", "pal vali", "daat"]),
    ("Ophthalmology", ["eye", "aankh", "vision", "chashma", "glasses", "motiyabind", "dikh"]),
    ("ENT", ["ear", "kaan", "throat", "gala", "gale", "nose", "naak"]),
    ("Dermatology", ["itch", "khujli", "ringworm", "daad", "rash", "pimple", "daag", "skin"]),
    ("Orthopaedics", ["back pain", "kamar", "knee", "ghutn", "sprain", "moch", "fracture", "haddi", "neck", "gardan"]),
    ("General Surgery", ["cut", "kat gaya", "boil", "phoda", "dog", "kutte", "piles", "hernia", "stone", "pathri", "wound"]),
    ("Pulmonology/Chest & TB", ["tb", "weeks", "inhaler", "asthma", "wheez"]),
    ("Psychiatry", ["sleep", "neend", "tension", "udaas", "sad", "alcohol", "daru"]),
]
MEDICAL_DEPTS = {"General Medicine", "Pulmonology/Chest & TB"}
UNINFORMATIVE = {"", "-", "--", ".", "?", "??", "na", "nan", "opd", "same", "check", "xx", "...."}


def clean_vitals(df):
    out = pd.DataFrame(index=df.index)
    s, d = df["bp_systolic"].astype(float), df["bp_diastolic"].astype(float)
    swap = s < d
    s, d = np.where(swap, d, s), np.where(swap, s, d)
    s = pd.Series(s, index=df.index).where((s >= 50) & (s <= 260))
    d = pd.Series(d, index=df.index).where((d >= 20) & (d <= 160))
    out["sbp"], out["dbp"] = s, d
    p = df["pulse_bpm"].astype(float)
    out["hr"] = p.where((p >= 25) & (p <= 250))
    t = df["temp_c"].astype(float)
    t = np.where((t >= 86) & (t <= 113), (t - 32) * 5 / 9, t)
    out["temp"] = pd.Series(t, index=df.index).where(lambda x: (x >= 30) & (x <= 45))
    o = df["spo2_pct"].astype(float)
    out["spo2"] = o.where((o >= 50) & (o <= 100))
    return out


def vitals_level(age, preg, v):
    """Primer vital-sign rules applied to RECORDED (cleaned) vitals. NaN comparisons are False."""
    sbp, dbp, hr, temp, spo2 = v.sbp, v.dbp, v.hr, v.temp, v.spo2
    if spo2 < 85 or (age >= 10 and sbp < 80):
        return 1
    danger = spo2 < 92 or temp <= 35.0 or (age >= 16 and (hr >= 131 or hr <= 40)) or (8 <= age < 16 and (hr > 120 or hr <= 40)) \
        or (age >= 10 and sbp <= 90) \
        or (age >= 16 and sbp >= 220) or (preg == "Y" and (sbp >= 160 or dbp >= 110)) or (age < 0.25 and temp >= 38.0)
    if age < 8 and hr > (180 if age < 0.25 else 160 if age < 3 else 140):
        danger = True
    if danger:
        return 2
    amber = (92 <= spo2 <= 93) or temp >= 39.1 or (age >= 16 and 111 <= hr <= 130) or (preg == "Y" and (sbp >= 140 or dbp >= 90))
    return 3 if amber else 9


def predict(df):
    v = clean_vitals(df)
    rows = []
    for i, r in df.iterrows():
        text = str(r["chief_complaint"]) if pd.notna(r["chief_complaint"]) else ""
        low = text.lower().strip()
        age = float(r["age_months"]) / 12 if pd.notna(r["age_months"]) else float(r["age_years"])
        preg = r["pregnant"] if pd.notna(r["pregnant"]) else ""
        n_usable = int(v.loc[i, ["sbp", "hr", "temp", "spo2"]].notna().sum())
        if low in UNINFORMATIVE and n_usable <= 1:
            rows.append((r["visit_id"], "ABSTAIN", "", "", 0.3))
            continue
        flags = [code for code, kws in RF_KEYWORDS.items() if any(k in low for k in kws)]
        if age < 0.25 and v.loc[i, "temp"] >= 38.0:
            flags.append("RF_HIGH_FEVER_INFANT")
        vl = vitals_level(age, preg, v.loc[i])
        if vl == 1:
            level = 1
        elif flags or vl == 2:
            level = 2
        elif vl == 3 or any(k in low for k in L3_KEYWORDS):
            level = 3
        elif any(k in low for k in L5_KEYWORDS):
            level = 5
        else:
            level = 4
        if level <= 2:
            dept = "Emergency/Casualty"
        else:
            dept = next((d for d, kws in DEPT_KEYWORDS if any(k in low for k in kws)), "General Medicine")
            if age < 12 and dept in MEDICAL_DEPTS:
                dept = "Paediatrics"
        rows.append((r["visit_id"], level, dept, ";".join(sorted(set(flags))), 0.6 if level <= 2 else 0.5))
    return pd.DataFrame(rows, columns=["visit_id", "triage_level", "department", "red_flags", "confidence"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/phase2/intake_test.csv")
    ap.add_argument("--out", default="submission/H1_predictions.csv")
    a = ap.parse_args()
    df = pd.read_csv(a.input, encoding="utf-8")
    pred = predict(df)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pred.to_csv(a.out, index=False, encoding="utf-8")
    print(f"Wrote {len(pred)} predictions to {a.out}")
    print(pred["triage_level"].astype(str).value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
