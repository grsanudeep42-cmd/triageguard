"""
triageguard/model_vitals.py
─────────────────────────────────────────────────────────────────────────────
Phase 4 – Vitals-only model for TriageGuard.

LIGHTGBM ARCHITECTURE
---------------------
• Multiclass classifier: predicts probabilities for levels 1–5.
• Binary classifier: predicts p_emergency with clinical monotone constraints:
    - spo2_c: -1 (lower SpO2 -> higher p_emergency)
    - shock_index: +1 (higher shock index -> higher p_emergency)
    - all other features: 0 (unconstrained)
• Truly vitals-only: strictly NO text-derived features (no n_red_flags).
• Natively handles NaNs in vitals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from triageguard.data_adapter import VITALS_FEATURE_COLS, extract_vitals_features


class VitalsModel:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_names = list(VITALS_FEATURE_COLS)

        # Monotone constraints for binary emergency model
        # -1 = decreasing (lower value -> higher p_emergency)
        # +1 = increasing (higher value -> higher p_emergency)
        #  0 = unconstrained
        self.monotone_constraints: List[int] = [
            -1 if f == "spo2_c" else (1 if f == "shock_index" else 0)
            for f in self.feature_names
        ]

        self.clf_multiclass = LGBMClassifier(
            objective="multiclass",
            num_class=5,
            n_estimators=160,
            learning_rate=0.04,
            max_depth=5,
            num_leaves=24,
            class_weight="balanced",
            random_state=self.random_state,
            verbose=-1,
        )

        self.clf_binary = LGBMClassifier(
            objective="binary",
            n_estimators=160,
            learning_rate=0.04,
            max_depth=5,
            num_leaves=24,
            monotone_constraints=self.monotone_constraints,
            class_weight="balanced",
            random_state=self.random_state,
            verbose=-1,
        )
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y_level: Union[pd.Series, np.ndarray],
        y_is_emergency: Union[pd.Series, np.ndarray],
    ) -> "VitalsModel":
        """
        Fit multiclass and binary emergency models on vitals features.
        y_level: 1 to 5
        y_is_emergency: boolean or 0/1
        """
        X_feat = X[self.feature_names]
        # Shift 1-5 to 0-4 for LightGBM multiclass
        y_multi = np.array(y_level, dtype=int) - 1
        y_bin = np.array(y_is_emergency, dtype=int)

        self.clf_multiclass.fit(X_feat, y_multi)
        self.clf_binary.fit(X_feat, y_bin)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return 5-class probabilities [p_L1, p_L2, p_L3, p_L4, p_L5].
        """
        if not self.is_fitted:
            raise RuntimeError("VitalsModel must be fitted before predict_proba.")
        X_feat = X[self.feature_names]
        return self.clf_multiclass.predict_proba(X_feat)

    def predict_emergency_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return 1D array of p_emergency probabilities.
        """
        if not self.is_fitted:
            raise RuntimeError("VitalsModel must be fitted before predict_emergency_proba.")
        X_feat = X[self.feature_names]
        return self.clf_binary.predict_proba(X_feat)[:, 1]

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "VitalsModel":
        return joblib.load(path)
