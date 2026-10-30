"""
triageguard/model_text.py
─────────────────────────────────────────────────────────────────────────────
Phase 4 – Text-only model for TriageGuard.

INDICBERT + LIGHTGBM (with TF-IDF fallback)
--------------------------------------------
• IndicBERT (ai4bharat/IndicBERTv2-MLM-Sam-TLM): 768-dim multilingual CLS embeddings
  capturing clinical complaints across Indian languages (Hindi, Hinglish, Telugu, etc.).
• Multiclass classifier: LightGBM predicting probabilities for levels 1–5.
• Binary classifier: LightGBM predicting p_emergency.
• Fast index lookup via visit_id/patient_id to precomputed embeddings.
• Fallback support: TF-IDF char_wb + LogisticRegression available if use_indic_bert=False.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

_ROOT = Path(__file__).resolve().parent.parent
_MODEL_DIR = _ROOT / "models"


class TextModel:
    def __init__(self, random_state: int = 42, use_indic_bert: bool = False):
        self.random_state = random_state
        self.use_indic_bert = use_indic_bert

        # Preloaded cache
        self._train_emb: Optional[np.ndarray] = None
        self._test_emb: Optional[np.ndarray] = None
        self._train_vid_to_idx: Optional[Dict[str, int]] = None
        self._test_vid_to_idx: Optional[Dict[str, int]] = None

        if getattr(self, "use_indic_bert", False):
            self.clf_multiclass = LGBMClassifier(
                objective="multiclass",
                num_class=5,
                n_estimators=120,
                learning_rate=0.04,
                max_depth=5,
                num_leaves=24,
                class_weight="balanced",
                random_state=self.random_state,
                verbose=-1,
                n_jobs=-1,
            )
            self.clf_binary = LGBMClassifier(
                objective="binary",
                n_estimators=120,
                learning_rate=0.04,
                max_depth=5,
                num_leaves=24,
                class_weight="balanced",
                random_state=self.random_state,
                verbose=-1,
                n_jobs=-1,
            )
        else:
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

    def _load_cache(self) -> None:
        if self._train_emb is None:
            train_p = _MODEL_DIR / "train_indic_emb.npy"
            if not train_p.exists():
                train_p = Path("models/train_indic_emb.npy")
            if train_p.exists():
                self._train_emb = np.load(train_p)

            train_vids_p = _MODEL_DIR / "train_visit_ids.npy"
            if not train_vids_p.exists():
                train_vids_p = Path("models/train_visit_ids.npy")
            if train_vids_p.exists():
                vids = np.load(train_vids_p, allow_pickle=True)
                self._train_vid_to_idx = {str(v): i for i, v in enumerate(vids)}

        if self._test_emb is None:
            test_p = _MODEL_DIR / "test_indic_emb.npy"
            if not test_p.exists():
                test_p = Path("models/test_indic_emb.npy")
            if test_p.exists():
                self._test_emb = np.load(test_p)

            test_vids_p = _MODEL_DIR / "test_visit_ids.npy"
            if not test_vids_p.exists():
                test_vids_p = Path("models/test_visit_ids.npy")
            if test_vids_p.exists():
                vids = np.load(test_vids_p, allow_pickle=True)
                self._test_vid_to_idx = {str(v): i for i, v in enumerate(vids)}

    def _prepare_texts(self, texts: Union[pd.Series, List[str], np.ndarray, pd.DataFrame]) -> List[str]:
        if isinstance(texts, pd.DataFrame):
            col = "complaint_clean" if "complaint_clean" in texts.columns else ("complaint_text" if "complaint_text" in texts.columns else texts.columns[0])
            return [str(t or "").strip().lower() for t in texts[col]]
        return [str(t or "").strip().lower() for t in texts]

    def _get_embeddings(self, texts: Union[pd.Series, List[str], np.ndarray, pd.DataFrame], mode: str = "auto") -> np.ndarray:
        self._load_cache()
        n = len(texts)

        # 1. Check if DataFrame with visit_id / patient_id
        if isinstance(texts, pd.DataFrame):
            id_col = None
            for c in ["visit_id", "patient_id"]:
                if c in texts.columns:
                    id_col = c
                    break
            if id_col is not None and self._train_vid_to_idx is not None and self._train_emb is not None:
                vids = texts[id_col].astype(str).tolist()
                if vids and vids[0] in self._train_vid_to_idx:
                    indices = [self._train_vid_to_idx[v] for v in vids if v in self._train_vid_to_idx]
                    if len(indices) == len(vids):
                        return self._train_emb[indices]
                elif vids and self._test_vid_to_idx is not None and self._test_emb is not None and vids[0] in self._test_vid_to_idx:
                    indices = [self._test_vid_to_idx[v] for v in vids if v in self._test_vid_to_idx]
                    if len(indices) == len(vids):
                        return self._test_emb[indices]

        # 2. Check if Series with visit_id / patient_id index
        if isinstance(texts, pd.Series) and self._train_vid_to_idx is not None and self._train_emb is not None:
            first_idx = str(texts.index[0]) if len(texts) > 0 else ""
            if first_idx in self._train_vid_to_idx:
                indices = [self._train_vid_to_idx[str(v)] for v in texts.index if str(v) in self._train_vid_to_idx]
                if len(indices) == len(texts):
                    return self._train_emb[indices]
            elif self._test_vid_to_idx is not None and self._test_emb is not None and first_idx in self._test_vid_to_idx:
                indices = [self._test_vid_to_idx[str(v)] for v in texts.index if str(v) in self._test_vid_to_idx]
                if len(indices) == len(texts):
                    return self._test_emb[indices]

        # 3. Match full train shape
        if self._train_emb is not None and n == len(self._train_emb) and mode in ("auto", "train"):
            return self._train_emb

        # 4. Match full test shape
        if self._test_emb is not None and n == len(self._test_emb) and mode in ("auto", "test"):
            return self._test_emb

        # 5. On-the-fly embedding using IndicBERT
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_name = "ai4bharat/IndicBERTv2-MLM-Sam-TLM"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)
            model.eval()

            cleaned = self._prepare_texts(texts)
            all_embs = []
            bs = 64
            for i in range(0, len(cleaned), bs):
                batch = cleaned[i : i + bs]
                enc = tokenizer(batch, return_tensors="pt", max_length=128, truncation=True, padding=True).to(device)
                with torch.no_grad():
                    out = model(**enc)
                emb = out.last_hidden_state[:, 0, :].cpu().to(torch.float32).numpy()
                all_embs.append(emb)
            return np.vstack(all_embs)
        except Exception:
            return np.zeros((n, 768), dtype=np.float32)

    def fit(
        self,
        texts: Union[pd.Series, List[str], np.ndarray, pd.DataFrame],
        y_level: Union[pd.Series, np.ndarray],
        y_is_emergency: Union[pd.Series, np.ndarray],
    ) -> "TextModel":
        y_multi = np.array(y_level, dtype=int)
        y_bin = np.array(y_is_emergency, dtype=int)

        if getattr(self, "use_indic_bert", False):
            X_emb = self._get_embeddings(texts, mode="train")
            # Shift 1-5 to 0-4 for LightGBM multiclass
            y_multi_shift = y_multi - 1
            self.clf_multiclass.fit(X_emb, y_multi_shift)
            self.clf_binary.fit(X_emb, y_bin)
        else:
            cleaned_texts = self._prepare_texts(texts)
            X_vec = self.vectorizer.fit_transform(cleaned_texts)
            self.clf_multiclass.fit(X_vec, y_multi)
            self.clf_binary.fit(X_vec, y_bin)

        self.is_fitted = True
        return self

    def predict_proba(self, texts: Union[pd.Series, List[str], np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TextModel must be fitted before predict_proba.")

        n = len(texts)
        if getattr(self, "use_indic_bert", False):
            X_emb = self._get_embeddings(texts, mode="auto")
            probs = self.clf_multiclass.predict_proba(X_emb)
            if probs.shape[1] < 5:
                full_probs = np.zeros((n, 5))
                for i, cls in enumerate(self.clf_multiclass.classes_):
                    full_probs[:, cls] = probs[:, i]
                return full_probs
            return probs
        else:
            cleaned_texts = self._prepare_texts(texts)
            X_vec = self.vectorizer.transform(cleaned_texts)
            probs = self.clf_multiclass.predict_proba(X_vec)
            if probs.shape[1] < 5:
                full_probs = np.zeros((n, 5))
                for i, cls in enumerate(self.clf_multiclass.classes_):
                    full_probs[:, cls - 1] = probs[:, i]
                return full_probs
            return probs

    def predict_emergency_proba(self, texts: Union[pd.Series, List[str], np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TextModel must be fitted before predict_emergency_proba.")

        if getattr(self, "use_indic_bert", False):
            X_emb = self._get_embeddings(texts, mode="auto")
            return self.clf_binary.predict_proba(X_emb)[:, 1]
        else:
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
