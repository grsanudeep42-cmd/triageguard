"""Pytest acceptance tests for triageguard.data_gen."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from triageguard.data_gen import generate

# ── Constants matching spec and config.yaml ────────────────────────────────

N_TOTAL = 30_000
SEED    = 42

# Level mix targets and tolerance  # PLACEHOLDER
LEVEL_TARGETS  = {1: 0.005, 2: 0.035, 3: 0.250, 4: 0.400, 5: 0.310}
LEVEL_TOL      = 0.005   # ±0.5 %

# Emergency rate bounds
EM_RATE_LO = 0.035
EM_RATE_HI = 0.045

# Error injection tolerances (±50 % relative of target)  # PLACEHOLDER
ERROR_TARGETS = {
    "bp_zero":         0.015,
    "bp_swapped":      0.007,
    "temp_fahrenheit": 0.020,
    "spo2_invalid":    0.008,
    "pulse_zero":      0.005,
    "decimal_slip":    0.005,
    "missing":         0.040,
}
ERROR_TOL_REL = 0.50   # ±50 % relative

MIN_CALIB_EM = 150

# Hidden-label columns that must NOT appear in synthetic_intake.csv
HIDDEN_COLS = {
    "triage_level", "clean_triage_level", "department",
    "is_emergency", "archetype", "reporting_bias", "injected_errors",
}

# ── Shared fixture ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def generated_data(
    tmp_path_factory: pytest.TempPathFactory,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate the full dataset once; reuse across all tests in this module."""
    out = tmp_path_factory.mktemp("datagen_output")
    generate(n=N_TOTAL, seed=SEED, out=str(out))
    intake = pd.read_csv(out / "synthetic_intake.csv")
    labels = pd.read_csv(out / "hidden_labels.csv")
    splits = pd.read_csv(out / "splits.csv")
    return intake, labels, splits


# ── Test classes ──────────────────────────────────────────────────────────

class TestRowCount:
    """T01 – output must have exactly N_TOTAL rows."""

    def test_intake_row_count(self, generated_data):
        intake, _, _ = generated_data
        assert len(intake) == N_TOTAL, (
            f"synthetic_intake.csv has {len(intake)} rows, expected {N_TOTAL}"
        )

    def test_labels_row_count(self, generated_data):
        _, labels, _ = generated_data
        assert len(labels) == N_TOTAL

    def test_splits_row_count(self, generated_data):
        _, _, splits = generated_data
        assert len(splits) == N_TOTAL

    def test_patient_ids_unique(self, generated_data):
        intake, labels, splits = generated_data
        assert intake["patient_id"].nunique() == N_TOTAL
        assert labels["patient_id"].nunique() == N_TOTAL
        assert splits["patient_id"].nunique() == N_TOTAL


class TestLevelMix:
    """T02 – clean_triage_level distribution must be within ±0.5 % of targets."""

    def test_all_levels_present(self, generated_data):
        _, labels, _ = generated_data
        assert set(labels["clean_triage_level"].unique()) == {1, 2, 3, 4, 5}

    def test_level_fractions_within_tolerance(self, generated_data):
        _, labels, _ = generated_data
        fracs = labels["clean_triage_level"].value_counts() / N_TOTAL
        for lvl, target in LEVEL_TARGETS.items():
            actual = float(fracs.get(lvl, 0.0))
            assert abs(actual - target) <= LEVEL_TOL, (
                f"Level {lvl}: target={target:.4f}, actual={actual:.4f}, "
                f"tol=±{LEVEL_TOL:.4f}"
            )

    def test_level_counts_sum_to_n(self, generated_data):
        _, labels, _ = generated_data
        assert len(labels) == N_TOTAL


class TestEmergencyRate:
    """T03 – emergency rate (is_emergency=True) must be in [3.5 %, 4.5 %]."""

    def test_emergency_rate_bounds(self, generated_data):
        _, labels, _ = generated_data
        em_rate = float(labels["is_emergency"].mean())
        assert EM_RATE_LO <= em_rate <= EM_RATE_HI, (
            f"Emergency rate {em_rate:.4f} not in "
            f"[{EM_RATE_LO}, {EM_RATE_HI}]"
        )

    def test_is_emergency_consistent_with_clean_level(self, generated_data):
        """is_emergency must be True iff clean_triage_level ∈ {1, 2}."""
        _, labels, _ = generated_data
        expected_em = labels["clean_triage_level"].isin([1, 2])
        mismatch = (labels["is_emergency"] != expected_em).sum()
        assert mismatch == 0, (
            f"{mismatch} rows have is_emergency inconsistent with clean_triage_level"
        )


class TestErrorInjectionRates:
    """T04 – each error type's actual rate must be within ±50 % of its target."""

    @pytest.mark.parametrize("code,target", ERROR_TARGETS.items())
    def test_error_rate(self, generated_data, code, target):
        _, labels, _ = generated_data
        actual = labels["injected_errors"].str.contains(code, na=False).sum() / N_TOTAL
        lo = target * (1 - ERROR_TOL_REL)
        hi = target * (1 + ERROR_TOL_REL)
        assert lo <= actual <= hi, (
            f"Error '{code}': target={target:.4f}, actual={actual:.4f}, "
            f"acceptable=[{lo:.4f}, {hi:.4f}]"
        )


class TestCalibrationSplit:
    """T05 – calibration split must contain ≥ 150 emergencies."""

    def test_calib_has_enough_emergencies(self, generated_data):
        _, labels, splits = generated_data
        merged    = splits.merge(labels[["patient_id", "is_emergency"]], on="patient_id")
        calib_em  = int(merged.loc[merged["split"] == "calib", "is_emergency"].sum())
        assert calib_em >= MIN_CALIB_EM, (
            f"Calib split has {calib_em} emergencies, need ≥ {MIN_CALIB_EM}"
        )

    def test_split_labels_valid(self, generated_data):
        _, _, splits = generated_data
        valid = {"train", "calib", "val", "test"}
        bad   = set(splits["split"].unique()) - valid
        assert not bad, f"Unknown split labels: {bad}"

    def test_split_covers_all_rows(self, generated_data):
        intake, _, splits = generated_data
        merged = intake.merge(splits, on="patient_id", how="left")
        assert merged["split"].isna().sum() == 0, "Some rows have no split label"


class TestNoLabelLeakage:
    """T06 – no hidden-label column may appear in synthetic_intake.csv."""

    def test_intake_has_no_hidden_columns(self, generated_data):
        intake, _, _ = generated_data
        leaked = set(intake.columns) & HIDDEN_COLS
        assert not leaked, (
            f"synthetic_intake.csv leaks hidden columns: {leaked}"
        )

    def test_intake_has_expected_columns(self, generated_data):
        intake, _, _ = generated_data
        required = {
            "patient_id", "age", "sex",
            "bp_sys", "bp_dia", "pulse", "temp_c", "spo2",
            "complaint_text", "known_conditions", "medications",
            "arrival_time",
        }
        missing = required - set(intake.columns)
        assert not missing, f"Intake CSV missing columns: {missing}"


class TestReproducibility:
    """T07 – identical seed must produce bit-identical output."""

    def test_same_seed_same_intake(self, tmp_path):
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        generate(n=N_TOTAL, seed=SEED, out=str(out1))
        generate(n=N_TOTAL, seed=SEED, out=str(out2))

        df1 = pd.read_csv(out1 / "synthetic_intake.csv")
        df2 = pd.read_csv(out2 / "synthetic_intake.csv")
        pd.testing.assert_frame_equal(df1, df2, check_like=False)

    def test_same_seed_same_labels(self, tmp_path):
        out1 = tmp_path / "rep1"
        out2 = tmp_path / "rep2"
        generate(n=N_TOTAL, seed=SEED, out=str(out1))
        generate(n=N_TOTAL, seed=SEED, out=str(out2))

        l1 = pd.read_csv(out1 / "hidden_labels.csv")
        l2 = pd.read_csv(out2 / "hidden_labels.csv")
        pd.testing.assert_frame_equal(l1, l2, check_like=False)


# ── Helpers shared by new tests ─────────────────────────────────────────────

def _auc(y_true, x_numeric):
    """Binary AUC of a numeric predictor (handles both orientations)."""
    from sklearn.metrics import roc_auc_score
    raw = roc_auc_score(y_true.astype(int), x_numeric)
    return max(raw, 1.0 - raw)   # normalise so AUC is always ≥ 0.5


class TestDataQuality:
    """T08 – basic sanity checks on column value ranges."""

    def test_age_range(self, generated_data):
        intake, _, _ = generated_data
        assert intake["age"].min() >= 0
        assert intake["age"].max() <= 100

    def test_sex_values(self, generated_data):
        intake, _, _ = generated_data
        assert set(intake["sex"].unique()).issubset({"M", "F"})

    def test_complaint_text_non_empty(self, generated_data):
        intake, _, _ = generated_data
        assert (intake["complaint_text"].str.strip() == "").sum() == 0

    def test_triage_level_in_range(self, generated_data):
        _, labels, _ = generated_data
        assert labels["triage_level"].between(1, 5).all()
        assert labels["clean_triage_level"].between(1, 5).all()

    def test_reporting_bias_values(self, generated_data):
        _, labels, _ = generated_data
        valid = {"none", "under_report", "over_report"}
        bad   = set(labels["reporting_bias"].unique()) - valid
        assert not bad, f"Unknown reporting_bias values: {bad}"

    def test_arrival_time_parseable(self, generated_data):
        intake, _, _ = generated_data
        parsed = pd.to_datetime(intake["arrival_time"], format="%Y-%m-%dT%H:%M:%S",
                                errors="coerce")
        assert parsed.isna().sum() == 0, "Some arrival_time values could not be parsed"


class TestFix1NoIDLeakage:
    """
    T09 – FIX-1: patient IDs are assigned after the global shuffle.
    AUC of patient_id_int and of row_index vs is_emergency must be
    within 0.5 ± 0.05 (i.e. in [0.45, 0.55]).
    """

    AUC_LO = 0.45
    AUC_HI = 0.55

    def test_patient_id_no_label_signal(self, generated_data):
        intake, labels, _ = generated_data
        merged   = intake[["patient_id"]].merge(labels[["patient_id", "is_emergency"]],
                                                on="patient_id")
        pid_int  = merged["patient_id"].str[1:].astype(int).values
        y        = merged["is_emergency"].astype(int).values
        auc      = _auc(y, pid_int)
        assert self.AUC_LO <= auc <= self.AUC_HI, (
            f"patient_id AUC = {auc:.4f}, expected [{self.AUC_LO}, {self.AUC_HI}]  "
            f"(FIX-1: IDs must be assigned after shuffle)"
        )

    def test_row_index_no_label_signal(self, generated_data):
        intake, labels, _ = generated_data
        merged    = intake.reset_index().merge(
            labels[["patient_id", "is_emergency"]], on="patient_id"
        )
        row_index = merged["index"].values
        y         = merged["is_emergency"].astype(int).values
        auc       = _auc(y, row_index)
        assert self.AUC_LO <= auc <= self.AUC_HI, (
            f"row_index AUC = {auc:.4f}, expected [{self.AUC_LO}, {self.AUC_HI}]  "
            f"(FIX-1: row order must not encode label)"
        )


class TestFix2ConditionsMedsNoLeakage:
    """
    T10 – FIX-2: a text-only TF-IDF + LogisticRegression model trained and
    evaluated on the FULL dataset (conditions + medications concatenated) must
    achieve AUC ≤ 0.65 for predicting is_emergency.

    Train=full set (optimistic upper bound).  If even this upper bound is ≤ 0.65,
    the per-split estimate will also be acceptable.
    """

    AUC_MAX = 0.65

    def test_conditions_meds_auc_le_threshold(self, generated_data):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        intake, labels, _ = generated_data
        merged = intake.merge(labels[["patient_id", "is_emergency"]], on="patient_id")
        y      = merged["is_emergency"].astype(int).values

        # Concatenate conditions + medications as a single text field
        text = (
            merged["known_conditions"].fillna("none")
            + " | "
            + merged["medications"].fillna("none")
        )
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
        X   = vec.fit_transform(text)
        lr  = LogisticRegression(max_iter=500, random_state=42, C=1.0)
        lr.fit(X, y)
        probs = lr.predict_proba(X)[:, 1]
        auc   = roc_auc_score(y, probs)
        assert auc <= self.AUC_MAX, (
            f"conditions+medications TF-IDF AUC = {auc:.4f}, must be ≤ {self.AUC_MAX}  "
            f"(FIX-2: conditions/meds must not be a strong proxy for is_emergency)"
        )
