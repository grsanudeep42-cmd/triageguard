"""
triageguard/model_dept.py
─────────────────────────────────────────────────────────────────────────────
Phase 4 – Department routing classifier for TriageGuard.

ROUTING ARCHITECTURE
--------------------
• Multiclass classifier predicting one of the 12 clinical departments from:
    - Cleaned complaint text (TF-IDF char_wb)
    - Demographics (age, sex)
    - Cleaned vitals (bp_sys, bp_dia, pulse, temp_c, spo2)
• Dual Department Output (per clinical design):
    - suggested_specialty : Specialty department predicted by this model
    - primary_department  : "Emergency/Casualty" if triage level <= 2,
                            otherwise suggested_specialty
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class DepartmentModel:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 4),
            max_features=8000,
            sublinear_tf=True,
            min_df=2,
        )
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=self.random_state,
        )
        self.classes_: np.ndarray = np.array([])
        self.is_fitted = False

    def _extract_tabular(self, df: pd.DataFrame) -> np.ndarray:
        age = pd.to_numeric(df.get("age", 35), errors="coerce").fillna(35).values
        sex_m = (df.get("sex", "M").astype(str).str.upper() == "M").astype(float).values

        sys = pd.to_numeric(df.get("bp_sys_c", df.get("bp_sys", 120)), errors="coerce").fillna(120).values
        dia = pd.to_numeric(df.get("bp_dia_c", df.get("bp_dia", 80)), errors="coerce").fillna(80).values
        pulse = pd.to_numeric(df.get("pulse_c", df.get("pulse", 75)), errors="coerce").fillna(75).values
        temp = pd.to_numeric(df.get("temp_c_c", df.get("temp_c", 37.0)), errors="coerce").fillna(37.0).values
        spo2 = pd.to_numeric(df.get("spo2_c", df.get("spo2", 98.0)), errors="coerce").fillna(98.0).values

        tab = np.column_stack([age, sex_m, sys, dia, pulse, temp, spo2])
        return tab

    def fit(self, df: pd.DataFrame, y_department: Union[pd.Series, np.ndarray]) -> "DepartmentModel":
        texts = [str(t or "").strip().lower() for t in df.get("complaint_clean", df.get("complaint_text", ""))]
        X_txt = self.vectorizer.fit_transform(texts)

        X_tab = self._extract_tabular(df)
        X_tab_scaled = self.scaler.fit_transform(X_tab)

        X_comb = hstack([X_txt, X_tab_scaled])
        y = np.array(y_department)

        self.clf.fit(X_comb, y)
        self.classes_ = self.clf.classes_
        self.is_fitted = True
        return self

    def predict_specialty(self, df: pd.DataFrame) -> List[str]:
        if not self.is_fitted:
            raise RuntimeError("DepartmentModel must be fitted before predict_specialty.")
        texts = [str(t or "").strip().lower() for t in df.get("complaint_clean", df.get("complaint_text", ""))]
        X_txt = self.vectorizer.transform(texts)

        X_tab = self._extract_tabular(df)
        X_tab_scaled = self.scaler.transform(X_tab)

        X_comb = hstack([X_txt, X_tab_scaled])
        preds = self.clf.predict(X_comb)
        return [str(p) for p in preds]

    def route_departments(
        self,
        df: pd.DataFrame,
        triage_levels: Union[List[Optional[int]], np.ndarray],
    ) -> Tuple[List[str], List[str]]:
        """
        Return (primary_departments, suggested_specialties).
        For level 1 or 2, primary_department is 'Emergency/Casualty'.
        """
        specialties = self.predict_specialty(df)
        primaries = []
        for lvl, spec in zip(triage_levels, specialties):
            if lvl is not None and int(lvl) in (1, 2):
                primaries.append("Emergency/Casualty")
            else:
                primaries.append(spec)
        return primaries, specialties

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DepartmentModel":
        return joblib.load(path)
