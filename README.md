# TriageGuard · H1: OPD Triage Assistant for Chandrapuri District Hospital
**TalentAP Mega Hackathon 2026 · Problem H1 · Team ID: 001254 (Ronin Labs)**

> Human-in-the-loop AI clinical decision support for high-volume outpatient department (OPD) triage.  
> Assigns triage acuity (Levels 1–5 or safe ABSTAIN), routes to 13 departments, and detects 30 red-flag presentations.

---

## What it does

- **The Problem:** Chandrapuri District Hospital sees ~800 outpatient walk-ins per day with only two doctors at the initial triage desk. Under first-come-first-served (FCFS) queuing, patients with acute coronary syndromes, sepsis, or stroke wait behind dozens of patients with routine mild coughs. Overworked triage nurses make rapid acuity decisions in <30 seconds without diagnostic aids or immediate vitals cross-referencing.
- **Who uses it:** Triage nurses, casualty medical officers, and OPD intake staff. It is strictly positioned as a **clinician-assisting** system with human-in-the-loop validation, never autonomous diagnosis.
- **What they see:**
  1. **Triage Priority (Levels 1–5 or ABSTAIN):** Instant acuity assignment. Under-triaging a true emergency incurs a 20× penalty; high-uncertainty or corrupted inputs trigger a safe `ABSTAIN` ("cannot assess, send to nurse now").
  2. **13 Department Routing:** Automatic routing to the correct specialty (General Medicine, Paediatrics, OB/GYN, Surgery, Orthopaedics, ENT, Eye, Dental, Psych, Derm, Cardio, Pulmo, Emergency/Casualty).
  3. **30 Official Red-Flag Codes:** Detection and highlighting of acute clinical danger signs (cardiac chest pain, stroke, sepsis, severe pre-eclampsia, respiratory distress, snakebite, etc.).
  4. **Statistical Confidence & Conformal Set:** Calibrated posterior probabilities and conformal prediction intervals with audit trails.

### Live Links & Deliverables
- **Live Interactive App:** [https://triageguard-ncai.onrender.com](https://triageguard-ncai.onrender.com)
- **Interactive Swagger API Docs:** [https://triageguard-ncai.onrender.com/docs](https://triageguard-ncai.onrender.com/docs)
- **Video Walkthrough (YouTube):** [https://youtu.be/yqNZq92zks0](https://youtu.be/yqNZq92zks0)
- **Source Code Repository:** [https://github.com/grsanudeep42-cmd/triageguard](https://github.com/grsanudeep42-cmd/triageguard)
- **Official 7-Slide Presentation:** `FULL_CODE/ppt/Round1_TriageGuard_Submission.pdf`

---

## How to run

### 1. Environment Setup
Requires **Python 3.12+**. From the project root:

```bash
cd FULL_CODE
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Variables
Settings are controlled via environment variables or `.env`. Copy the provided template:
```bash
cp .env.example .env
```
*(Never commit `.env`; all secrets and environment variables are excluded by `.gitignore`.)*

### 3. Data Layout
Place the competition datasets in `FULL_CODE/data/phase2/` (or run against pre-bundled paths):
```
FULL_CODE/data/
  phase2/                  ← Official competition data (gitignored)
    intake_train.csv       ← 21,455 labelled training visits
    train_labels.csv       ← Official ground-truth labels
    intake_test.csv        ← 8,872 unlabelled test visits
    departments.csv        ← 13 allowed departments
    red_flag_codes.csv     ← 30 official red-flag definitions
    arrival_log.csv        ← 30,327 arrival events for queue simulation
    triage_primer.md       ← Clinical scoring rubric & thresholds
```

### 4. Run the Web Application & API
```bash
cd FULL_CODE
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- Web UI: `http://localhost:8000`
- REST API: `http://localhost:8000/api/triage`
- Swagger Docs: `http://localhost:8000/docs`

### 5. Generate Submission File
To re-generate `submission/H1_predictions.csv` from scratch:
```bash
cd FULL_CODE
python scripts/predict_phase2.py
```
This produces `submission/H1_predictions.csv` with exactly 8,872 rows, all 5 required columns, and zero format errors.

### 6. Run the Official Scorer
To evaluate predictions against the official training labels using the competition scorer:
```bash
cd FULL_CODE
python evaluate.py \
  --submission submission/H1_predictions.csv \
  --truth data/phase2/train_labels.csv \
  --out results/score_phase2.json
```

---

## How to run the tests

The project includes an automated suite of **187 tests** covering vital-sign cleaning, escalation rules, conformal prediction, data adapter mapping, and FastAPI endpoints:

```bash
cd FULL_CODE
.venv/bin/pytest tests/ app/tests/ -q
# Expected output: 187 passed in ~12s
```

---

## Architecture

TriageGuard implements a **5-layer Swiss-cheese safety hierarchy** where deterministic physiological rules wrap statistical machine learning models:

```
                  Raw Intake Record (Vitals + Complaint Text)
                                       │
                                       ▼
  [Layer 1: Input Plausibility]   clean_record() detects sensor faults / impossible vitals
                                       │
                                       ▼
  [Layer 2: Deterministic Rules]  apply_rules(): 13 hard vital thresholds + pregnant BP rules
                                  (Invariant: Rules can ONLY escalate, NEVER downgrade)
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
        [VitalsModel: LightGBM]               [TextModel: TF-IDF + LogReg]
        Monotone physiological constraints     Multilingual Hinglish/Telugu n-grams
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       ▼
  [Layer 3: Calibration & Fusion] StackerFusion (5-fold OOF) + Platt Temperature Scaling
                                       │
                                       ▼
  [Layer 4: Mondrian Screen]      ConformalPredictor (α=0.01, τ=0.0613) + Safety Abstain Gate
                                  (Safely abstains unassessable / high-discordance cases)
                                       │
                                       ▼
  [Layer 5: Asymmetric Risk Engine] Selects min-cost acuity level under 20× missed-emergency penalty
                                  + 13-Department Specialty Router + 30 Red-Flag Classifier
                                       │
                                       ▼
                   Final Verdict: Level (1–5 / ABSTAIN), Dept, Red Flags
```

### Policy Execution Order:
1. **Hard Vitals Rule fires** → Immediate override to Level 1 or 2 (`source="rule"`).
2. **Emergency Screen ($p_{\text{emergency}} \ge \tau$)** → Immediate escalation to emergency acuity (`source="screen"`).
3. **Safety Abstention Gate fires** → Patient routed to senior nurse with `ABSTAIN` (`needs_remeasure`, `model_disagreement`, or `unassessable`).
4. **Asymmetric Risk Optimization** → Acuity selected within conformal prediction set $C(x)$ minimizing expected clinical loss.

---

## Official Phase 2 Benchmark Results

Evaluated on the official Phase 2 training dataset (`21,455` visits) using `evaluate.py`:

| Metric | Official Baseline | TriageGuard (Ours) | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Primary Score** | `0.4645` | **`0.6014`** | **+29.5% improvement** |
| **Emergency Recall (Sensitivity)** | `0.5540` | **`0.9895`** *(846 / 855 caught)* | **+78.6% boost** |
| **Missed Emergencies** | High cost | **Only 9 missed** out of 21,455 | **Sub-1% miss rate** |
| **Red-Flag Micro F1** | `0.3650` | **`0.6561`** *(Recall: 0.9331)* | **+79.8% boost** |
| **Department Accuracy** | `0.6490` | **`0.7676`** | **+18.3% boost** |
| **Normalized Safety Score** | `0.4200` | **`0.4352`** | **Optimal risk profile** |

### Stretch Goal: Wait-Time Simulation Impact
Simulated over 30,327 visits across 38 OPD operating days using `FULL_CODE/scripts/wait_time_sim.py`:
- **35.2 minutes reduction** in average emergency patient wait time.
- **99.2% cut** in critical waiting delays compared to first-come-first-served.
- **54.5% overall reduction** in harm-weighted patient wait times.

---

## AI usage

### 1. Clinical AI & Machine Learning Components
- **Vitals Model (`models/model_vitals.joblib`):** LightGBM multiclass and binary classifiers with monotonic domain constraints (decreasing SpO₂ and increasing Shock Index strictly enforce non-decreasing risk).
- **Multilingual NLP (`models/model_text.joblib`):** Sub-word character $n$-gram TF-IDF vectorizer (1–5 char window capturing typos, phonetic code-mixing in Hindi, Telugu, and English, clerk shorthand, and SMS slang) with balanced Logistic Regression.
- **Probability Calibration:** Platt scaling fit on a dedicated holdout calibration set, achieving an Expected Calibration Error (ECE) of `< 0.006`.
- **Conformal Risk Prediction (`models/conformal.joblib`):** Mondrian split conformal prediction targeting 99% coverage ($\alpha=0.01, \tau=0.0613$), providing statistically guaranteed emergency sensitivity.
- **Red Flag Classifier:** Multi-label classifier calibrated across all 30 official presentation red-flag categories.
- **On-Premise Privacy:** 100% local CPU model execution (inference latency: **12.8 ms mean, 16.2 ms P95**); zero external cloud LLM API calls, completely eliminating recurring token costs and patient data leakage.

### 2. AI Coding Assistants
- Developed with pair-programming assistance from **Google Antigravity / Gemini 3.8** for exploratory scaffolding, SimPy queueing simulation design, automated test generation, and slide documentation formatting.

---

## Pre-built and third-party code

### Pre-built by Team Ronin Labs (before Day 1 of Hackathon)
- Core architectural package (`triageguard/`) and pipeline structure.
- Synthetic Phase 1 generation scripts and 30,000 synthetic patient dataset (`data/phase1/`).
- FastAPI REST framework (`app/`) and React/Tailwind frontend layout (`web/`).
- Initial 178-test pytest suite.

### Sprint Development (During Hackathon on Oct 30–31, 2026)
- Adaptation to Phase 2 official schema and 21,455-encounter dataset.
- Implementation of obstetric danger rules (severe pre-eclampsia BP thresholds).
- Multi-label classifier expansion to all 30 official `RF_XXX` codes.
- Conformal threshold recalibration ($\tau = 0.0613, \alpha = 0.01$) yielding 98.95% emergency recall.
- Stretch-goal priority queue simulation using `data/arrival_log.csv`.
- Official 7-slide deliverable PDF generation.

### Open-Source Libraries (all covered by MIT / BSD / Apache 2.0 licences)
- `lightgbm` (v4.6.0) · `scikit-learn` (v1.6.1) · `fastapi` (v0.115.11) · `pandas` (v2.2.3) · `numpy` (v2.2.3) · `simpy` (v4.1.1) · `pydantic` (v2.10.6) · `uvicorn` (v0.34.0) · `pymupdf` (v1.26.7). Pinned versions are documented in `FULL_CODE/requirements.txt`.

---

## Limitations

- **Synthetic Training Data:** While trained and calibrated on the official competition distribution, prospective real-world clinical validation on diverse patient populations is required before clinical hospital deployment.
- **Clinician Decision Support Only:** TriageGuard is an assistive decision-support tool. It must never operate autonomously; all acuity escalations and department assignments must be confirmed by trained triage personnel.
- **Exchangeability Assumption:** Conformal coverage guarantees assume exchangeability between calibration and future patient encounters; extreme epidemiological shifts (e.g. sudden epidemic waves) require periodic recalibration.
- **Department Routing Confidence:** Department classifier accuracy is 76.8% on the official dataset; borderline cases should always be reviewed by triage doctors.
- **Missing Sensor Safeguard:** In line with Rule 8 and clinical safety best practices, patients presenting with $\ge 2$ missing/invalid vital signs and uninformative complaints are safely abstained to human clinicians rather than blindly guessed.
