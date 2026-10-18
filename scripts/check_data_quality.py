#!/usr/bin/env python3
"""
scripts/check_data_quality.py
──────────────────────────────────────────────────────────────────────────────
THROWAWAY data-quality gate for the TriageGuard synthetic dataset.
NOT part of the inference pipeline.  Does NOT import triageguard.data_gen.
Reads only:  data/synthetic_intake.csv
             data/hidden_labels.csv
             data/splits.csv

Sections
─────────
1. 20 sample rows (5 emergencies + 15 non-emergencies) with full label context.
2. Leakage checks: shuffle confirmation, AUC of patient_id / arrival_hour /
   known_conditions / medications versus is_emergency.
3. Difficulty: train TF-IDF+LR (text-only) and LightGBM (vitals-only) on the
   TRAIN split; evaluate AUC and emergency recall@10 %FPR on the VAL split;
   also show a combined (averaged probability) result.
4. Slice report: recall on under-reported emergencies vs other emergencies
   at the threshold corresponding to 10 % FPR on the full val set.
5. Confusable pairs: text-only accuracy/AUC and vitals-only accuracy/AUC for
   each of the five confusable pair groups (70/30 pair-level split).

Run:
    .venv/bin/python scripts/check_data_quality.py
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

DATA = Path("data")
VITAL_COLS = ["bp_sys", "bp_dia", "pulse", "temp_c", "spo2"]
SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def sep(title: str) -> None:
    """Print a section header."""
    bar = "═" * 72
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def recall_at_fpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fpr_target: float = 0.10,
) -> float:
    """Return the maximum TPR achievable at FPR ≤ fpr_target."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    mask = fpr <= fpr_target
    return float(tpr[mask].max()) if mask.any() else 0.0


def threshold_at_fpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fpr_target: float = 0.10,
) -> float:
    """Return the score threshold that achieves FPR ≤ fpr_target."""
    fpr, _, thresholds = roc_curve(y_true, y_prob)
    idx = np.where(fpr <= fpr_target)[0]
    return float(thresholds[idx.max()]) if idx.size else 0.5


def print_table(df: pd.DataFrame, indent: int = 2) -> None:
    pad = " " * indent
    lines = df.to_string().split("\n")
    for line in lines:
        print(pad + line)


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Merge intake, labels, and splits into one DataFrame."""
    intake = pd.read_csv(DATA / "synthetic_intake.csv")
    labels = pd.read_csv(DATA / "hidden_labels.csv")
    splits = pd.read_csv(DATA / "splits.csv")
    df = intake.merge(labels, on="patient_id").merge(splits, on="patient_id")
    df["arrival_hour"] = pd.to_datetime(df["arrival_time"]).dt.hour
    df["pid_int"] = df["patient_id"].str[1:].astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 – 20 sample rows
# ─────────────────────────────────────────────────────────────────────────────

def sec1_sample(df: pd.DataFrame) -> None:
    sep("SECTION 1 – 20 sample rows  (5 emergencies + 15 non-emergencies)")

    em  = df[df.is_emergency].sample(5,  random_state=7)
    non = df[~df.is_emergency].sample(15, random_state=7)
    sample = (
        pd.concat([em, non])
        .sample(frac=1, random_state=7)
        .reset_index(drop=True)
    )

    display_cols = [
        "patient_id", "age", "sex",
        "bp_sys", "bp_dia", "pulse", "temp_c", "spo2",
        "complaint_text",
        "clean_triage_level", "department",
        "is_emergency", "archetype",
        "reporting_bias", "injected_errors",
    ]
    with pd.option_context(
        "display.max_columns",  None,
        "display.width",        220,
        "display.max_colwidth", 48,
    ):
        print(sample[display_cols].to_string(index=True))


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 – Leakage checks
# ─────────────────────────────────────────────────────────────────────────────

def sec2_leakage(df: pd.DataFrame) -> None:
    sep("SECTION 2 – Leakage checks")

    y = df.is_emergency.astype(int).values

    # 2a. Shuffle check: emergency counts in 10-row windows over the first 200 rows
    first200 = df.head(200)["is_emergency"].astype(int).values
    windows  = [int(first200[i : i + 20].sum()) for i in range(0, 200, 20)]
    print(f"\n  2a. Emergency counts per 20-row block (first 200 rows):")
    for i, w in enumerate(windows):
        bar = "█" * w + "░" * (20 - w)
        print(f"       block {i+1:02d}: {bar}  ({w} emergencies out of 20)")
    print(f"       ↳ If shuffled correctly, blocks should vary around"
          f" {df.is_emergency.mean()*20:.1f} with no monotone trend.")

    # 2b. AUC of numeric patient_id and arrival_hour vs is_emergency
    pid_auc  = roc_auc_score(y, df.pid_int.values)
    hour_auc = roc_auc_score(y, df.arrival_hour.values)
    print(f"\n  2b. Proxy AUC (values near 0.50 = no leakage):")
    print(f"       AUC(patient_id_int  → is_emergency) = {pid_auc:.4f}")
    print(f"       AUC(arrival_hour    → is_emergency) = {hour_auc:.4f}")

    # 2c. known_conditions – top-10 values; show emergency fraction
    print(f"\n  2c. known_conditions – top 10 values, emergency fraction:")
    kc = (
        df.groupby("known_conditions")["is_emergency"]
        .agg(n_emergency="sum", n_total="count")
        .assign(em_pct=lambda x: (x.n_emergency / x.n_total * 100).round(2))
        .sort_values("n_total", ascending=False)
        .head(10)
    )
    print_table(kc)
    kc_auc = roc_auc_score(
        y,
        TfidfVectorizer(analyzer="word").fit_transform(
            df.known_conditions.fillna("none")
        ).toarray().dot(
            LogisticRegression(max_iter=500, random_state=SEED)
            .fit(
                TfidfVectorizer(analyzer="word").fit_transform(
                    df.known_conditions.fillna("none")
                ),
                y,
            )
            .coef_[0]
        ),
    )

    # Simpler single-pass version
    vec_kc = TfidfVectorizer(analyzer="word").fit(df.known_conditions.fillna("none"))
    X_kc   = vec_kc.transform(df.known_conditions.fillna("none"))
    lr_kc  = LogisticRegression(max_iter=500, random_state=SEED).fit(X_kc, y)
    p_kc   = lr_kc.predict_proba(X_kc)[:, 1]
    kc_auc = roc_auc_score(y, p_kc)
    print(f"       TF-IDF(known_conditions)+LR  AUC vs is_emergency = {kc_auc:.4f}"
          f"  (train=full set, optimistic upper bound; ~0.50 = no leakage)")

    # 2d. medications
    print(f"\n  2d. medications – top 10 values, emergency fraction:")
    mk = (
        df.groupby("medications")["is_emergency"]
        .agg(n_emergency="sum", n_total="count")
        .assign(em_pct=lambda x: (x.n_emergency / x.n_total * 100).round(2))
        .sort_values("n_total", ascending=False)
        .head(10)
    )
    print_table(mk)
    vec_mk = TfidfVectorizer(analyzer="word").fit(df.medications.fillna("none"))
    X_mk   = vec_mk.transform(df.medications.fillna("none"))
    lr_mk  = LogisticRegression(max_iter=500, random_state=SEED).fit(X_mk, y)
    p_mk   = lr_mk.predict_proba(X_mk)[:, 1]
    mk_auc = roc_auc_score(y, p_mk)
    print(f"       TF-IDF(medications)+LR  AUC vs is_emergency = {mk_auc:.4f}"
          f"  (train=full set, optimistic upper bound; ~0.50 = no leakage)")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 – Model difficulty
# ─────────────────────────────────────────────────────────────────────────────

def _train_text(df_train: pd.DataFrame, df_val: pd.DataFrame, y_train, y_val
                ) -> Tuple[np.ndarray, TfidfVectorizer, LogisticRegression]:
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4),
        max_features=25_000, sublinear_tf=True,
    )
    X_tr = vec.fit_transform(df_train.complaint_text.fillna(""))
    X_vl = vec.transform(df_val.complaint_text.fillna(""))
    lr   = LogisticRegression(
        C=1.0, max_iter=1000, random_state=SEED, class_weight="balanced",
    )
    lr.fit(X_tr, y_train)
    return lr.predict_proba(X_vl)[:, 1], vec, lr


def _train_vitals(df_train: pd.DataFrame, df_val: pd.DataFrame, y_train
                  ) -> Tuple[np.ndarray, LGBMClassifier]:
    lgbm = LGBMClassifier(
        n_estimators=300, learning_rate=0.05,
        random_state=SEED, class_weight="balanced", verbosity=-1,
    )
    lgbm.fit(df_train[VITAL_COLS], y_train)
    return lgbm.predict_proba(df_val[VITAL_COLS])[:, 1], lgbm


def sec3_difficulty(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
) -> Tuple:
    sep("SECTION 3 – Model difficulty  (train = TRAIN split, eval = VAL split)")

    y_train = df_train.is_emergency.astype(int).values
    y_val   = df_val.is_emergency.astype(int).values

    print(f"\n  Train: {len(df_train):,} rows, {y_train.sum():,} emergencies "
          f"({y_train.mean()*100:.1f} %)")
    print(f"  Val:   {len(df_val):,} rows,  {y_val.sum():,} emergencies "
          f"({y_val.mean()*100:.1f} %)")

    p_text,   vec,  lr   = _train_text(df_train, df_val, y_train, y_val)
    p_vitals, lgbm       = _train_vitals(df_train, df_val, y_train)
    p_combined           = (p_text + p_vitals) / 2.0

    rows = []
    for name, p in [
        ("Text-only   (char n-gram TF-IDF + LR)", p_text),
        ("Vitals-only (LightGBM)",                p_vitals),
        ("Combined    (average prob)",             p_combined),
    ]:
        auc = roc_auc_score(y_val, p)
        rec = recall_at_fpr(y_val, p, fpr_target=0.10)
        rows.append({"Model": name, "AUC": f"{auc:.4f}", "Recall@10%FPR": f"{rec:.4f}"})

    print(f"\n  {'Model':<42}  {'AUC':>8}  {'Recall@10%FPR':>14}")
    print(f"  {'-'*68}")
    for r in rows:
        print(f"  {r['Model']:<42}  {r['AUC']:>8}  {r['Recall@10%FPR']:>14}")

    print(f"\n  Guidance:")
    print(f"    Text AUC   < 0.70  → text alone cannot reliably identify emergencies")
    print(f"    Vitals AUC > 0.85  → vitals carry the decisive signal")
    print(f"    Combined   should exceed both individual models")

    return y_val, p_text, p_vitals, p_combined, df_val


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 – Slice report
# ─────────────────────────────────────────────────────────────────────────────

def sec4_slice(
    df_val:     pd.DataFrame,
    y_val:      np.ndarray,
    p_text:     np.ndarray,
    p_vitals:   np.ndarray,
    p_combined: np.ndarray,
) -> None:
    sep("SECTION 4 – Slice report: recall on under-reported vs other emergencies")
    print(f"  Threshold: score at 10 % FPR computed on the full val set.\n")

    # Derive per-model thresholds from the full val set
    threshs = {
        "Text":     threshold_at_fpr(y_val, p_text,     0.10),
        "Vitals":   threshold_at_fpr(y_val, p_vitals,   0.10),
        "Combined": threshold_at_fpr(y_val, p_combined, 0.10),
    }

    # Slice masks (only meaningful for rows that ARE emergencies)
    under_mask  = (df_val.reporting_bias == "under_report").values & y_val.astype(bool)
    other_mask  = y_val.astype(bool) & ~under_mask

    n_under = int(under_mask.sum())
    n_other = int(other_mask.sum())

    print(f"  Under-reported emergencies in val : {n_under}")
    print(f"  Other emergencies in val          : {n_other}")

    hdr = f"  {'Slice':<32}  {'n':>5}  {'Text recall':>12}  {'Vitals recall':>13}  {'Combined recall':>16}"
    print(f"\n{hdr}")
    print(f"  {'-'*82}")

    for slice_label, mask in [
        ("Under-reported emergencies", under_mask),
        ("Other emergencies",          other_mask),
        ("All emergencies (val)",      y_val.astype(bool)),
    ]:
        n = int(mask.sum())
        if n == 0:
            print(f"  {slice_label:<32}  {0:>5}  {'n/a':>12}  {'n/a':>13}  {'n/a':>16}")
            continue

        def slice_recall(probs: np.ndarray, thr: float) -> float:
            """Fraction of slice that exceeds the threshold (= recall since all positive)."""
            return float((probs[mask] >= thr).mean())

        r_t = slice_recall(p_text,     threshs["Text"])
        r_v = slice_recall(p_vitals,   threshs["Vitals"])
        r_c = slice_recall(p_combined, threshs["Combined"])
        print(f"  {slice_label:<32}  {n:>5}  {r_t:>12.4f}  {r_v:>13.4f}  {r_c:>16.4f}")

    print(f"\n  Interpretation:")
    print(f"    Under-reported recall well below 'Other' → text model misses biased cases.")
    print(f"    Vitals model should be more robust to reporting bias.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 – Confusable pairs
# ─────────────────────────────────────────────────────────────────────────────

# (archetype_A, archetype_B, display_label)
# Label = 1 for A, 0 for B in binary classification within the pair.
PAIRS = [
    ("acs_stemi",             "chest_pain_acidity",    "ACS vs Acidity"),
    ("severe_asthma_attack",  "breathlessness_anxiety","Asthma attack vs Anxiety"),
    ("stroke_major",          "migraine",              "Stroke vs Migraine"),
    ("acute_abdomen",         "gastritis_gerd",        "Acute abdomen vs Gastritis"),
    ("sepsis_severe",         "fever_adult_mild",      "Sepsis vs Mild fever"),
]


def sec5_pairs(df: pd.DataFrame) -> None:
    sep("SECTION 5 – Confusable pair analysis  (70 / 30 pair-level split, seed=42)")
    print(
        f"\n  Binary task within each pair: label 1 = first archetype, 0 = second.\n"
        f"  Text acc/AUC ≈ 0.50 means text cannot separate the pair (by design).\n"
        f"  Vitals AUC high means vitals carry the signal (by design).\n"
    )

    hdr = (
        f"  {'Pair':<32}  {'n_tr':>5}  {'n_te':>5}  "
        f"{'base%':>5}  "
        f"{'TextAcc':>8}  {'TextAUC':>8}  "
        f"{'VitAcc':>7}  {'VitAUC':>7}"
    )
    print(hdr)
    print(f"  {'-'*88}")

    for arch_a, arch_b, plabel in PAIRS:
        sub = df[df.archetype.isin([arch_a, arch_b])].copy()
        sub["pair_label"] = (sub.archetype == arch_a).astype(int)

        if sub.pair_label.nunique() < 2 or len(sub) < 20:
            print(f"  {plabel:<32}  too few rows or only one class — skip")
            continue

        try:
            (X_tr_txt, X_te_txt,
             X_tr_vit, X_te_vit,
             y_tr, y_te) = train_test_split(
                sub.complaint_text.fillna(""),
                sub[VITAL_COLS],
                sub.pair_label,
                test_size=0.30,
                random_state=SEED,
                stratify=sub.pair_label,
            )
        except ValueError as e:
            print(f"  {plabel:<32}  split error: {e}")
            continue

        n_tr = len(y_tr)
        n_te = len(y_te)
        base_pct = float(y_te.mean() * 100)  # majority-class baseline

        # --- Text model ---
        vec_p = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4),
            max_features=10_000, sublinear_tf=True,
        )
        X_tr_tfidf = vec_p.fit_transform(X_tr_txt)
        X_te_tfidf = vec_p.transform(X_te_txt)
        lr_p = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
        lr_p.fit(X_tr_tfidf, y_tr)
        p_txt_pair = lr_p.predict_proba(X_te_tfidf)[:, 1]
        acc_txt    = float((lr_p.predict(X_te_tfidf) == y_te).mean())

        # --- Vitals model ---
        lgbm_p = LGBMClassifier(
            n_estimators=200, random_state=SEED, verbosity=-1,
        )
        lgbm_p.fit(X_tr_vit, y_tr)
        p_vit_pair = lgbm_p.predict_proba(X_te_vit)[:, 1]
        acc_vit    = float((lgbm_p.predict(X_te_vit) == y_te).mean())

        try:
            auc_txt = roc_auc_score(y_te, p_txt_pair)
            auc_vit = roc_auc_score(y_te, p_vit_pair)
        except Exception:
            auc_txt = auc_vit = float("nan")

        print(
            f"  {plabel:<32}  {n_tr:>5}  {n_te:>5}  "
            f"{base_pct:>4.0f}%  "
            f"{acc_txt:>8.4f}  {auc_txt:>8.4f}  "
            f"{acc_vit:>7.4f}  {auc_vit:>7.4f}"
        )

    print(f"\n  base% = fraction of test set that is class 1 (majority class accuracy if always predict 1).")
    print(f"  TextACC ≈ base%  → the text model has no edge over a naive baseline.")
    print(f"  VitACC >> base%  → vitals successfully separate the pair.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data ...", end=" ", flush=True)
    df = load_data()
    print(f"OK  ({len(df):,} rows, {df.is_emergency.sum()} emergencies)")

    df_train = df[df.split == "train"].copy()
    df_val   = df[df.split == "val"].copy()

    sec1_sample(df)
    sec2_leakage(df)
    y_val, p_text, p_vitals, p_combined, df_val_ret = sec3_difficulty(df_train, df_val)
    sec4_slice(df_val, y_val, p_text, p_vitals, p_combined)
    sec5_pairs(df)

    print("\n" + "═" * 72)
    print("  Gate complete.  Review numbers above before proceeding to Phase 1.")
    print("═" * 72 + "\n")


if __name__ == "__main__":
    main()
