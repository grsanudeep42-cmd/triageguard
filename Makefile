PYTHON := .venv/bin/python
PYTEST  := .venv/bin/pytest

.PHONY: all data train test eval clean venv

all: data train test eval

venv:
	python3 -m venv .venv
	.venv/bin/pip install -q lightgbm pyyaml scikit-learn numpy pandas pytest matplotlib tabulate

## Phase 0 – generate synthetic data
data:
	$(PYTHON) -m triageguard.data_gen --n 30000 --seed 42 --out data

## Phases 4–6 – train models and calibrate pipeline
train:
	$(PYTHON) scripts/train_pipeline.py

## Phases 1, 3, 4–8 – run complete unit & acceptance test suite
test:
	$(PYTEST) tests/ -v

## Phase 8 – evaluate pipeline on test split & generate tables/figures
eval:
	$(PYTHON) scripts/evaluate_pipeline.py

clean:
	rm -rf data/synthetic_intake.csv data/hidden_labels.csv data/splits.csv \
	       models/*.joblib results/*.json results/*.md results/figures/*.png \
	       __pycache__ triageguard/__pycache__ tests/__pycache__ scripts/__pycache__ .pytest_cache
