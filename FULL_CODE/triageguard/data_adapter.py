"""
triageguard/data_adapter.py
─────────────────────────────────────────────────────────────────────────────
Data loading, schema adaptation, and feature extraction for TriageGuard.

ISOLATION RULE
--------------
This module must NOT import triageguard.data_gen.
Predictive features must strictly EXCLUDE known_conditions and medications.
model_vitals features must be TRULY vitals-only (no text-derived n_red_flags).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from triageguard.cleaning import clean_dataframe, clean_record
from triageguard.rules import detect_red_flags

_ROOT = Path(__file__).resolve().parent.parent
_CFG_PATH = _ROOT / "config.yaml"

# Standard internal column names
_COLUMN_ALIASES: Dict[str, str] = {
    "systolic": "bp_sys",
    "sbp": "bp_sys",
    "bp_systolic": "bp_sys",
    "diastolic": "bp_dia",
    "dbp": "bp_dia",
    "bp_diastolic": "bp_dia",
    "heart_rate": "pulse",
    "hr": "pulse",
    "pulse_rate": "pulse",
    "pulse_bpm": "pulse",        # Phase 2 alias
    "temp": "temp_c",
    "temperature": "temp_c",
    "temp_celsius": "temp_c",
    "oxygen": "spo2",
    "saturation": "spo2",
    "o2_sat": "spo2",
    "spo2_pct": "spo2",          # Phase 2 alias
    "complaint": "complaint_text",
    "chief_complaint": "complaint_text",
    "age_years": "age",          # Phase 2 alias
    "visit_id": "patient_id",    # Phase 2 alias
}

# Core vitals features for model_vitals (TRULY VITALS-ONLY)
VITALS_FEATURE_COLS = [
    "bp_sys_c",
    "bp_dia_c",
    "pulse_c",
    "temp_c_c",
    "spo2_c",
    "shock_index",
    "map",
    "age",
    "sex_num",
    "n_invalid_vitals",
    "n_missing_vitals",
    "data_quality_score",
    "bp_sys_is_ok",
    "bp_dia_is_ok",
    "pulse_is_ok",
    "temp_c_is_ok",
    "spo2_is_ok",
]


def load_config(path: Optional[Path] = None) -> Dict:
    p = path or _CFG_PATH
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def adapt_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Rename any recognized column aliases to canonical TriageGuard column names.

    Also handles:
    - Infant age: if age_months is present and age == 0 or NaN, compute age = age_months / 12.0
    - pregnant field: pass through as-is; default to "unknown" if missing.
    """
    rename_map = {
        col: _COLUMN_ALIASES[col.lower()]
        for col in df.columns
        if col.lower() in _COLUMN_ALIASES and _COLUMN_ALIASES[col.lower()] not in df.columns
    }
    if rename_map:
        df = df.rename(columns=rename_map)

    # ── Infant age fix ────────────────────────────────────────────────────────
    # If age_months column exists and age is 0 or NaN, compute age from months.
    if "age_months" in df.columns and "age" in df.columns:
        age_col = pd.to_numeric(df["age"], errors="coerce")
        months_col = pd.to_numeric(df["age_months"], errors="coerce")
        needs_fix = age_col.isna() | (age_col == 0)
        has_months = months_col.notna()
        mask = needs_fix & has_months
        if mask.any():
            df = df.copy()
            # Cast age column to float64 first so fractional values can be stored
            df["age"] = df["age"].astype(float)
            df.loc[mask, "age"] = months_col[mask] / 12.0

    # ── pregnant passthrough ──────────────────────────────────────────────────
    # Keep existing pregnant column as-is; default to "unknown" if missing.
    if "pregnant" not in df.columns:
        df = df.copy()
        df["pregnant"] = "unknown"

    return df


def load_data(
    cfg: Optional[Dict] = None,
    include_labels: bool = True,
) -> pd.DataFrame:
    """
    Load dataset splits using paths from config.yaml.
    """
    if cfg is None:
        cfg = load_config()

    dp = cfg.get("data_paths", {})
    intake_path = _ROOT / dp.get("intake", "data/phase1/synthetic_intake.csv")
    splits_path = _ROOT / dp.get("splits", "data/phase1/splits.csv")

    intake = pd.read_csv(intake_path)
    intake = adapt_schema(intake)
    splits = pd.read_csv(splits_path)

    merged = intake.merge(splits[["patient_id", "split"]], on="patient_id")

    if include_labels:
        labels_path = _ROOT / dp.get("labels", "data/phase1/hidden_labels.csv")
        labels = pd.read_csv(labels_path)
        merged = merged.merge(
            labels[
                [
                    "patient_id",
                    "triage_level",
                    "clean_triage_level",
                    "department",
                    "is_emergency",
                    "reporting_bias",
                    "archetype",
                ]
            ],
            on="patient_id",
        )

    return merged


def extract_vitals_features(df_cleaned: pd.DataFrame) -> pd.DataFrame:
    """
    Extract numerical and flag features for model_vitals.
    Strictly vitals-only: NO text-derived features like n_red_flags.
    LightGBM natively handles NaNs in vitals.
    """
    X = pd.DataFrame(index=df_cleaned.index)

    # Cleaned vitals (NaNs preserved)
    X["bp_sys_c"] = pd.to_numeric(df_cleaned["bp_sys_c"], errors="coerce")
    X["bp_dia_c"] = pd.to_numeric(df_cleaned["bp_dia_c"], errors="coerce")
    X["pulse_c"]  = pd.to_numeric(df_cleaned["pulse_c"], errors="coerce")
    X["temp_c_c"] = pd.to_numeric(df_cleaned["temp_c_c"], errors="coerce")
    X["spo2_c"]   = pd.to_numeric(df_cleaned["spo2_c"], errors="coerce")

    # Derived vitals
    sys_safe = X["bp_sys_c"].where(X["bp_sys_c"] > 0, np.nan)
    X["shock_index"] = X["pulse_c"] / sys_safe
    X["map"] = X["bp_dia_c"] + (X["bp_sys_c"] - X["bp_dia_c"]) / 3.0

    # Demographics
    X["age"] = pd.to_numeric(df_cleaned["age"], errors="coerce").fillna(35.0)
    sex_str = df_cleaned.get("sex", "M").astype(str).str.upper()
    X["sex_num"] = (sex_str == "M").astype(float)

    # Quality & missingness flags
    X["n_invalid_vitals"]   = pd.to_numeric(df_cleaned.get("n_invalid_vitals", 0), errors="coerce").fillna(0)
    X["n_missing_vitals"]   = pd.to_numeric(df_cleaned.get("n_missing_vitals", 0), errors="coerce").fillna(0)
    X["data_quality_score"] = pd.to_numeric(df_cleaned.get("data_quality_score", 1.0), errors="coerce").fillna(1.0)

    # Boolean flags indicating if vital was valid and unmodified
    X["bp_sys_is_ok"] = (df_cleaned.get("bp_sys_flag", "ok") == "ok").astype(float)
    X["bp_dia_is_ok"] = (df_cleaned.get("bp_dia_flag", "ok") == "ok").astype(float)
    X["pulse_is_ok"]  = (df_cleaned.get("pulse_flag", "ok") == "ok").astype(float)
    X["temp_c_is_ok"] = (df_cleaned.get("temp_c_flag", "ok") == "ok").astype(float)
    X["spo2_is_ok"]   = (df_cleaned.get("spo2_flag", "ok") == "ok").astype(float)

    return X[VITALS_FEATURE_COLS]


def extract_red_flag_count(complaint_series: pd.Series) -> pd.Series:
    """Compute number of detected clinical red flags for text/fusion layers."""
    counts = []
    for txt in complaint_series:
        _, names = detect_red_flags(str(txt or ""))
        counts.append(len(names))
    return pd.Series(counts, index=complaint_series.index, dtype=float)
