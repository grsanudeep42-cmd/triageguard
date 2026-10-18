# TriageGuard Engineering & Clinical Insights (Phases 1–8)

This document records the architectural principles, clinical safety invariants, and machine learning methodology implemented in the TriageGuard Emergency Department Triage Decision Support System.

---

## 1. Safety Architecture: Hard Rules vs. Soft Guards vs. Statistical Predictors

### A. The Strict Priority Hierarchy
In clinical triage, safety must supersede statistical optimization. TriageGuard enforces a strict 5-tier evaluation hierarchy:

1. **Forced Vitals Rules (Tier 1 - Override Everything):**
   - Hard-coded clinical escalation thresholds for critical conditions (e.g., $SpO_2 < 85\%$, profound hypotension sys $< 60$ mmHg in adults, infant hyperpyrexia $\ge 40.0^\circ\text{C}$, septic/hypovolemic shock patterns).
   - **Monotonicity guarantee:** Rules can **only escalate** (assign level 1 or 2), never downgrade.
   - Run on cleaned, validated vitals; invalid or missing vitals never fire a rule.

2. **Soft Red Flags & Acuity Floor (Tier 2 - Non-Downgrade Floor):**
   - Natural language complaints are screened for 10 high-risk clinical presentations (e.g., chest pain, respiratory distress, fainted/unresponsive, seizure, unilateral weakness) across English, Telugu, and Hindi transliterations with typo-tolerant Damerau-Levenshtein matching.
   - **Safety invariant:** Text red flags **never set Level 1 or 2 by themselves** (preventing panic/hyperbolic text gaming), but enforce `no_downgrade_below_L3 = True`. If a red flag is present, the pipeline will **never assign Level 4 or 5** unless all objective vitals are explicitly within normal reference ranges.

3. **Mondrian Conformal Emergency Screen (Tier 3 - Guaranteed Recall):**
   - If fused $p_{\text{emergency}} \ge \tau$ (calibrated on the calibration split), the patient is immediately assigned to the emergency stream ($L \in \{1, 2\}$), choosing between Level 1 and Level 2 by minimum expected cost.

4. **Safety Abstention Layer (Tier 4 - "Cannot Assess, Route to Nurse"):**
   - The system explicitly refuses automated triage and abstains under three distinct risk triggers:
     - `needs_remeasure`: $\ge 2$ invalid or missing vital signs.
     - `model_disagreement`: Base vitals and text models disagree by $\ge 2$ triage levels (divergent clinical picture).
     - `conformal_uncertainty`: Conformal level prediction set contains $\ge 3$ levels (high epistemic uncertainty).

5. **Cost-Optimal Bayes Decision (Tier 5 - Cost-Sensitive Argmin):**
   - On stable, non-abstaining patients, selects the triage level within the conformal set $C(x)$ that minimizes expected clinical cost under an asymmetric cost matrix ($20\times$ penalty for emergency under-triage).

---

## 2. Machine Learning Methodology & Anti-Leakage Discipline

### A. Strict Data Split Isolation
- **Train split ($N=18,000$):** Used exclusively for fitting base estimators (`model_vitals`, `model_text`, `model_dept`) and generating 5-fold cross-validation out-of-fold (OOF) stacker features.
- **Validation split ($N=3,000$):** Used exclusively for probability calibration (Platt scaling) and tuning the operating conformal threshold $\alpha^*$.
- **Calibration split ($N=4,500$):** Reserved exclusively for conformal nonconformity score calibration (Mondrian emergency screen and level set coverage).
- **Test split ($N=4,500$):** Evaluated strictly once at the end; never used for tuning or selection.

### B. Feature Independence & The Anti-Leakage Gate
- `model_vitals` is **strictly vitals-only**: it operates on cleaned vitals, validity indicators, shock index, MAP, age, and sex. Text-derived features (`n_red_flags`, complaint text) are **strictly prohibited** from `model_vitals` to avoid cross-modal contamination.
- Patient IDs, synthetic row indices, and historical conditions/medications are strictly isolated from predictive features.
- **Safety Gate:** Both base models must exhibit validation AUC $< 0.99$ before proceeding to fusion, confirming the absence of synthetic label leakage.

### C. Clinical Monotonicity Constraints
- In LightGBM (`model_vitals`), monotone constraints are enforced on features with known unidirectional clinical risk:
  - $SpO_2$: Monotone decreasing ($-1$) $\to$ lower oxygen saturation can only increase $p_{\text{emergency}}$.
  - Shock Index ($\text{pulse} / \text{systolic BP}$): Monotone increasing ($+1$) $\to$ higher shock index can only increase $p_{\text{emergency}}$.

### D. Platt Calibration over Raw Isotonic on Small Val Cohorts
- With $\sim 120$ emergencies in validation cohorts of standard size, non-parametric isotonic regression risks over-fitting step functions to sparse tail bins.
- Logistic (Platt) temperature scaling provides smooth, monotonic calibration that dramatically reduces Expected Calibration Error (ECE from $0.0405$ to $0.0034$) without distorting ranking order.

---

## 3. Conformal Prediction with Finite-Sample Correction

### A. Mondrian Emergency Screen
- To ensure valid conditional coverage specifically on the critical emergency class ($Y \in \{1, 2\}$), conformal calibration is conducted conditionally on calibration emergencies ($n_{\text{calib\_em}}$):
  $$k = \lfloor \alpha (n_{\text{calib\_em}} + 1) \rfloor$$
  $$\tau = s_{(k)} \quad (\text{the } k\text{-th order statistic of emergency scores})$$
- If $k = 0$, the target error rate $\alpha$ is mathematically infeasible at sample size $n$, correctly triggering an "escalate-all" fail-safe rather than a falsely optimistic bound.

### B. Operating $\alpha^*$ Selection on Validation
- Candidate $\alpha \in \{0.005, 0.01, 0.02, 0.05\}$ are calibrated on `calib`, evaluated on `val`, and the operating point $\alpha^*$ is chosen to minimize expected triage cost subject to an empirical recall floor ($\ge 95\%$).

---

## 4. Robustness to Form Gaming & Corrupted Vitals

1. **Text Gaming Resistance:** Patients exaggerating complaints (adding dramatic red-flag keywords) with stable vitals are capped at Level 3. They cannot force Level 1 or 2 unless objective vitals corroborate distress or the ML emergency screen triggers on fused evidence.
2. **Under-Reported Complaints:** Emergency patients with subtle or vague complaints (e.g. elderly myocardial infarction presenting with "feeling tired" or "gas pain") are reliably escalated by vital sign rules ($SpO_2$, shock index) and the vitals-only base model.
3. **Graceful Degradation under Missingness:** Out-of-range sensor readings are flagged as invalid (never silently imputed), and $\ge 2$ missing/invalid vitals trigger immediate nurse remeasurement abstention.
