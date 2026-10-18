"""
triageguard/model_text.py
─────────────────────────────────────────────────────────────────────────────
Phase 4 – Text-only model for TriageGuard.

TF-IDF + LOGISTIC REGRESSION
----------------------------
• Character n-grams at word boundaries: char_wb (1–5) capture typos, abbreviations,
  and code-mixed words in Telugu-in-English, Hindi-in-English, and English.
• Multiclass classifier: predicts probabilities for levels 1–5.
• Binary classifier: predicts p_emergency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TextModel:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 5),
            max_features=12000,
            sublinear_tf=True,
            min_df=2,
        )
        self.clf_multiclass = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=self.random_state,
        )
        self.clf_binary = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=self.random_state,
        )
        self.is_fitted = False

    def _prepare_texts(self, texts: Union[pd.Series, List[str], np.ndarray]) -> List[str]:
        return [str(t or "").strip().lower() for t in texts]

    def fit(
        self,
        texts: Union[pd.Series, List[str], np.ndarray],
        y_level: Union[pd.Series, np.ndarray],
        y_is_emergency: Union[pd.Series, np.ndarray],
    ) -> "TextModel":
        cleaned_texts = self._prepare_texts(texts)
        X_vec = self.vectorizer.fit_transform(cleaned_texts)

        y_multi = np.array(y_level, dtype=int)
        y_bin = np.array(y_is_emergency, dtype=int)

        self.clf_multiclass.fit(X_vec, y_multi)
        self.clf_binary.fit(X_vec, y_bin)
        self.is_fitted = True
        return self

    def predict_proba(self, texts: Union[pd.Series, List[str], np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TextModel must be fitted before predict_proba.")
        cleaned_texts = self._prepare_texts(texts)
        X_vec = self.vectorizer.transform(cleaned_texts)
        # Ensure 5 columns corresponding to classes 1..5
        probs = self.clf_multiclass.predict_proba(X_vec)
        if probs.shape[1] < 5:
            full_probs = np.zeros((len(texts), 5))
            for i, cls in enumerate(self.clf_multiclass.classes_):
                full_probs[:, cls - 1] = probs[:, i]
            return full_probs
        return probs

    def predict_emergency_proba(self, texts: Union[pd.Series, List[str], np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TextModel must be fitted before predict_emergency_proba.")
        cleaned_texts = self._prepare_texts(texts)
        X_vec = self.vectorizer.transform(cleaned_texts)
        return self.clf_binary.predict_proba(X_vec)[:, 1]

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TextModel":
        return joblib.load(path)
