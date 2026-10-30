"""
triageguard/fusion.py
─────────────────────────────────────────────────────────────────────────────
Phase 5 – Stacking fusion and probability calibration for TriageGuard.

FUSION PIPELINE
---------------
1. Generate 5-fold cross-validation out-of-fold (OOF) predictions on train
   from model_vitals and model_text.
2. Train a Stacker (multiclass + binary) on stacked feature vectors:
   [p_vit (5), p_txt (5), p_em_vit (1), p_em_txt (1),
    level_disagreement (1), p_em_disagreement (1),
    data_quality_score (1), n_invalid (1), n_missing (1),
    n_red_flags (1), age (1)]
3. Calibrate fused probabilities on val split using Platt scaling (temperature /
   logistic scaling) suited for sample sizes with ~120 validation emergencies.
4. Calculate and report Expected Calibration Error (ECE) before/after calibration.
5. Cost-aware decision engine minimizing expected clinical triage cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from triageguard.cleaning import clean_dataframe
from triageguard.data_adapter import (
    extract_red_flag_count,
    extract_vitals_features,
    load_config,
)
from triageguard.model_text import TextModel
from triageguard.model_vitals import VitalsModel


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) for binary or multiclass probabilities.
    For binary probs (1D): compares mean predicted probability vs true fraction in each bin.
    For multiclass probs (2D): compares confidence max(p) vs top-1 accuracy in each bin.
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    if probs.ndim == 1:
        # Binary ECE
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_total = len(labels)
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            idx = (probs >= lo) & (probs <= hi if i == n_bins - 1 else probs < hi)
            if np.sum(idx) > 0:
                bin_prob = np.mean(probs[idx])
                bin_true = np.mean(labels[idx])
                weight = np.sum(idx) / n_total
                ece += weight * abs(bin_prob - bin_true)
        return float(ece)
    else:
        # Multiclass ECE
        confs = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)
        # Assuming labels are 0-indexed or 1-indexed:
        if np.min(labels) == 1 and np.max(labels) == 5:
            labels_0 = labels - 1
        else:
            labels_0 = labels
        corrects = (preds == labels_0).astype(float)

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_total = len(labels)
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            idx = (confs >= lo) & (confs <= hi if i == n_bins - 1 else confs < hi)
            if np.sum(idx) > 0:
                bin_conf = np.mean(confs[idx])
                bin_acc = np.mean(corrects[idx])
                weight = np.sum(idx) / n_total
                ece += weight * abs(bin_conf - bin_acc)
        return float(ece)


def build_cost_matrix(cfg: Optional[Dict] = None) -> np.ndarray:
    """
    Build 5x5 cost matrix C[decision-1, true_level-1].
    Over-triage (decision < true): 1.0 * (true - decision)
    Under-triage non-emergency (decision > true, true >= 3): 1.0 * (decision - true)
    Under-triage emergency (decision > true, true <= 2): 20.0 * (decision - true)
    Correct (decision == true): 0.0
    """
    if cfg is None:
        cfg = load_config()

    cw = cfg.get("cost_weights", {})
    over_cost = float(cw.get("over_triage", 1.0))
    under_non_em = float(cw.get("under_triage_non_emergency", 1.0))
    under_mult = float(cw.get("under_triage_multiplier", 20.0))

    matrix = np.zeros((5, 5), dtype=float)
    for d in range(1, 6):
        for y in range(1, 6):
            if d == y:
                matrix[d - 1, y - 1] = 0.0
            elif d < y:
                # Over-triage (e.g. true=4, decision=2)
                matrix[d - 1, y - 1] = over_cost * (y - d)
            else:
                # Under-triage (e.g. true=1, decision=3)
                if y in (1, 2):
                    matrix[d - 1, y - 1] = under_mult * (d - y)
                else:
                    matrix[d - 1, y - 1] = under_non_em * (d - y)
    return matrix


def compute_expected_costs(
    probs_5: np.ndarray,
    cost_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute expected cost for each of the 5 decisions given calibrated 5-class probs.
    Returns: Nx5 array of expected costs for decisions 1..5.
    """
    if cost_matrix is None:
        cost_matrix = build_cost_matrix()
    # E[Cost(d)] = sum_y P(y) * Cost(d, y) -> probs_5 @ cost_matrix.T
    return probs_5 @ cost_matrix.T


def select_min_cost_level(
    probs_5: np.ndarray,
    allowed_levels: Optional[Iterable[int]] = None,
    cost_matrix: Optional[np.ndarray] = None,
) -> int:
    """
    Select the triage level in allowed_levels that minimizes expected cost.
    If allowed_levels is None, searches over all 1..5.
    """
    expected_costs = compute_expected_costs(probs_5.reshape(1, 5), cost_matrix)[0]
    if allowed_levels is None:
        return int(np.argmin(expected_costs) + 1)

    allowed = list(allowed_levels)
    if not allowed:
        return int(np.argmin(expected_costs) + 1)

    best_lvl = allowed[0]
    best_cost = expected_costs[best_lvl - 1]
    for lvl in allowed[1:]:
        c = expected_costs[lvl - 1]
        if c < best_cost:
            best_cost = c
            best_lvl = lvl
    return int(best_lvl)


class StackerFusion:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.stacker_multiclass = LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=random_state
        )
        self.stacker_binary = LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=random_state
        )
        # Platt calibrators fit on validation split
        self.calibrator_binary = LogisticRegression(C=1.0, max_iter=500, random_state=random_state)
        self.is_fitted = False
        self.is_calibrated = False

    def build_stack_features(
        self,
        p_vitals_multi: np.ndarray,
        p_text_multi: np.ndarray,
        p_vitals_em: np.ndarray,
        p_text_em: np.ndarray,
        df_cleaned: pd.DataFrame,
    ) -> np.ndarray:
        """
        Construct feature vectors for stacking:
        • p_vitals (5)
        • p_text (5)
        • p_vitals_em (1)
        • p_text_em (1)
        • level_disagreement (1)
        • p_em_disagreement (1)
        • data_quality_score (1)
        • n_invalid_vitals (1)
        • n_missing_vitals (1)
        • n_red_flags (1)
        • age (1)
        """
        n = len(df_cleaned)
        vit_level = np.argmax(p_vitals_multi, axis=1) + 1
        txt_level = np.argmax(p_text_multi, axis=1) + 1
        level_disagree = np.abs(vit_level - txt_level).reshape(-1, 1)
        em_disagree = np.abs(p_vitals_em - p_text_em).reshape(-1, 1)

        q_score = pd.to_numeric(df_cleaned.get("data_quality_score", 1.0), errors="coerce").fillna(1.0).values.reshape(-1, 1)
        n_invalid = pd.to_numeric(df_cleaned.get("n_invalid_vitals", 0), errors="coerce").fillna(0).values.reshape(-1, 1)
        n_missing = pd.to_numeric(df_cleaned.get("n_missing_vitals", 0), errors="coerce").fillna(0).values.reshape(-1, 1)
        age = pd.to_numeric(df_cleaned.get("age", 35), errors="coerce").fillna(35).values.reshape(-1, 1) / 100.0

        if "n_red_flags" in df_cleaned.columns:
            rf_count = df_cleaned["n_red_flags"].values.reshape(-1, 1)
        else:
            rf_count = extract_red_flag_count(df_cleaned.get("complaint_clean", df_cleaned.get("complaint_text", ""))).values.reshape(-1, 1)

        X_stack = np.hstack([
            p_vitals_multi,
            p_text_multi,
            p_vitals_em.reshape(-1, 1),
            p_text_em.reshape(-1, 1),
            level_disagree,
            em_disagree,
            q_score,
            n_invalid,
            n_missing,
            rf_count,
            age,
        ])
        return X_stack

    def fit_oof(
        self,
        df_train_cleaned: pd.DataFrame,
        y_level: np.ndarray,
        y_is_emergency: np.ndarray,
        model_vitals: VitalsModel,
        model_text: TextModel,
        n_splits: int = 5,
    ) -> "StackerFusion":
        """
        Generate 5-fold cross-validation out-of-fold predictions on train and fit stacker.
        """
        n = len(df_train_cleaned)
        oof_vit_multi = np.zeros((n, 5))
        oof_txt_multi = np.zeros((n, 5))
        oof_vit_em = np.zeros(n)
        oof_txt_em = np.zeros(n)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        X_vit = extract_vitals_features(df_train_cleaned)
        texts = df_train_cleaned.get("complaint_clean", df_train_cleaned.get("complaint_text", ""))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_vit, y_level)):
            # Clone and fit fold models
            fold_vit = VitalsModel(random_state=self.random_state + fold)
            fold_txt = TextModel(random_state=self.random_state + fold)

            fold_vit.fit(X_vit.iloc[train_idx], y_level[train_idx], y_is_emergency[train_idx])
            fold_txt.fit(texts.iloc[train_idx], y_level[train_idx], y_is_emergency[train_idx])

            oof_vit_multi[val_idx] = fold_vit.predict_proba(X_vit.iloc[val_idx])
            oof_txt_multi[val_idx] = fold_txt.predict_proba(texts.iloc[val_idx])
            oof_vit_em[val_idx] = fold_vit.predict_emergency_proba(X_vit.iloc[val_idx])
            oof_txt_em[val_idx] = fold_txt.predict_emergency_proba(texts.iloc[val_idx])

        # Stack OOF features and fit stackers
        X_oof = self.build_stack_features(
            oof_vit_multi, oof_txt_multi, oof_vit_em, oof_txt_em, df_train_cleaned
        )
        self.stacker_multiclass.fit(X_oof, y_level)
        self.stacker_binary.fit(X_oof, y_is_emergency.astype(int))
        self.is_fitted = True
        return self

    def calibrate(
        self,
        raw_val_probs_em: np.ndarray,
        y_val_is_emergency: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Calibrate fused binary emergency probabilities using Platt scaling.
        Returns (ece_before, ece_after).
        """
        ece_before = compute_ece(raw_val_probs_em, y_val_is_emergency)

        # Logit transform of raw probabilities (clamped for stability)
        eps = 1e-6
        p_clipped = np.clip(raw_val_probs_em, eps, 1.0 - eps)
        logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)

        self.calibrator_binary.fit(logits, y_val_is_emergency.astype(int))
        calibrated_probs = self.calibrator_binary.predict_proba(logits)[:, 1]

        ece_after = compute_ece(calibrated_probs, y_val_is_emergency)
        self.is_calibrated = True
        return ece_before, ece_after

    def predict_fused(
        self,
        p_vitals_multi: np.ndarray,
        p_text_multi: np.ndarray,
        p_vitals_em: np.ndarray,
        p_text_em: np.ndarray,
        df_cleaned: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (fused_5_class_probs, calibrated_p_emergency).
        """
        if not self.is_fitted:
            raise RuntimeError("StackerFusion must be fitted before predict_fused.")

        X_stack = self.build_stack_features(
            p_vitals_multi, p_text_multi, p_vitals_em, p_text_em, df_cleaned
        )
        # Multiclass probabilities
        probs_multi = self.stacker_multiclass.predict_proba(X_stack)
        if probs_multi.shape[1] < 5:
            full_p = np.zeros((len(df_cleaned), 5))
            for i, cls in enumerate(self.stacker_multiclass.classes_):
                full_p[:, cls - 1] = probs_multi[:, i]
            probs_multi = full_p

        # Raw binary emergency probability
        raw_em = self.stacker_binary.predict_proba(X_stack)[:, 1]

        if self.is_calibrated:
            eps = 1e-6
            p_clipped = np.clip(raw_em, eps, 1.0 - eps)
            logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
            calib_em = self.calibrator_binary.predict_proba(logits)[:, 1]
        else:
            calib_em = raw_em

        return probs_multi, calib_em

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "StackerFusion":
        return joblib.load(path)
