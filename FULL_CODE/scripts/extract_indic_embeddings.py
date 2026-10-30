#!/usr/bin/env python3
"""
scripts/extract_indic_embeddings.py
Extract IndicBERT CLS embeddings for Phase 2 train and test intake text.
"""
from pathlib import Path
import sys
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel

_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _ROOT / "data" / "phase2"
_MODEL_DIR = _ROOT / "models"
_MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ai4bharat/IndicBERTv2-MLM-Sam-TLM"

print(f"Loading IndicBERT model: {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def get_cls_embeddings(texts, batch_size=64):
    all_emb = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = list(texts[i:i+batch_size])
        enc = tokenizer(batch, return_tensors="pt",
                        max_length=128, truncation=True,
                        padding=True).to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
        emb = out.last_hidden_state[:, 0, :].cpu().to(torch.float32).numpy()
        all_emb.append(emb)
        if i % 2000 == 0 or i + batch_size >= n:
            print(f"  {min(i + batch_size, n)}/{n} done")
    return np.vstack(all_emb)

train_path = _DATA_DIR / "intake_train.csv"
test_path  = _DATA_DIR / "intake_test.csv"

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

for col in ["chief_complaint", "known_conditions", "current_medications"]:
    train[col] = train[col].fillna("").astype(str)
    test[col]  = test[col].fillna("").astype(str)

# Combine all text fields same way as current TF-IDF does
train_text = (train["chief_complaint"] + " " + 
              train["known_conditions"] + " " + 
              train["current_medications"]).str.strip()
test_text  = (test["chief_complaint"]  + " " + 
              test["known_conditions"] + " " + 
              test["current_medications"]).str.strip()

print(f"Extracting train embeddings for {len(train_text)} texts...")
train_emb = get_cls_embeddings(train_text.tolist(), batch_size=64)
train_out = _MODEL_DIR / "train_indic_emb.npy"
np.save(train_out, train_emb)
print(f"Train shape: {train_emb.shape} saved to {train_out}")  # expect (21455, 768)

# Also save visit_ids for robust index mapping
np.save(_MODEL_DIR / "train_visit_ids.npy", train["visit_id"].values)

print(f"Extracting test embeddings for {len(test_text)} texts...")
test_emb = get_cls_embeddings(test_text.tolist(), batch_size=64)
test_out = _MODEL_DIR / "test_indic_emb.npy"
np.save(test_out, test_emb)
print(f"Test shape: {test_emb.shape} saved to {test_out}")   # expect (8872, 768)
np.save(_MODEL_DIR / "test_visit_ids.npy", test["visit_id"].values)

# Also ensure root models/ has them if root models exists
root_models = Path("models")
if root_models.resolve() != _MODEL_DIR.resolve():
    root_models.mkdir(exist_ok=True)
    np.save(root_models / "train_indic_emb.npy", train_emb)
    np.save(root_models / "test_indic_emb.npy", test_emb)

print("IndicBERT extraction completed successfully!")
