# TriageGuard Operating Points & Layer Ablation Benchmark

> **Report-Only Deliverable:** Evaluates layer configurations, operational trade-offs, and validation policy selection without modifying underlying models, thresholds, or production defaults.

---

## 1. Layer Ablation with Triage Costs on TEST Split (N=4,500)

Evaluates 4 layer configurations plus Baseline (a) Argmax across Clean, Vague Complaints, and 20% Vital Corruption.

| Stress Condition       | Configuration       | Emergency Recall (L1-L2)   | Silent Misses (L3-L5)   | Non-Emergency Escalation   | Abstain Rate      |   Total Cost (C_abs=3.0) |   Mean Cost (C_abs=3.0) |   Total Cost (C_abs=0.0) |   Mean Cost (C_abs=0.0) |
|:-----------------------|:--------------------|:---------------------------|:------------------------|:---------------------------|:------------------|-------------------------:|------------------------:|-------------------------:|------------------------:|
| Clean Test             | Full TriageGuard    | 92.8% (167/180)            | 2.8% (5/180)            | 4.4% (192/4320)            | 17.0% (763/4500)  |                     8365 |                   1.859 |                     6076 |                   1.35  |
| Clean Test             | Rules Disabled      | 92.8% (167/180)            | 2.8% (5/180)            | 1.1% (49/4320)             | 17.3% (777/4500)  |                     4635 |                   1.03  |                     2304 |                   0.512 |
| Clean Test             | Screen Disabled     | 83.9% (151/180)            | 3.3% (6/180)            | 4.2% (180/4320)            | 17.5% (789/4500)  |                     7864 |                   1.748 |                     5497 |                   1.222 |
| Clean Test             | Fusion Only         | 100.0% (180/180)           | 0.0% (0/180)            | 9.7% (420/4320)            | 0.0% (0/4500)     |                    16843 |                   3.743 |                    16843 |                   3.743 |
| Clean Test             | Baseline (a) Argmax | 93.9% (169/180)            | 6.1% (11/180)           | 1.3% (57/4320)             | 0.0% (0/4500)     |                     2378 |                   0.528 |                     2378 |                   0.528 |
| Vague-Complaint Stress | Full TriageGuard    | 76.7% (138/180)            | 10.6% (19/180)          | 4.4% (192/4320)            | 17.3% (778/4500)  |                     8984 |                   1.996 |                     6650 |                   1.478 |
| Vague-Complaint Stress | Rules Disabled      | 68.9% (124/180)            | 12.2% (22/180)          | 1.1% (49/4320)             | 17.8% (803/4500)  |                     6292 |                   1.398 |                     3883 |                   0.863 |
| Vague-Complaint Stress | Screen Disabled     | 75.0% (135/180)            | 10.6% (19/180)          | 4.2% (180/4320)            | 17.6% (792/4500)  |                     8207 |                   1.824 |                     5831 |                   1.296 |
| Vague-Complaint Stress | Fusion Only         | 92.2% (166/180)            | 7.8% (14/180)           | 9.7% (420/4320)            | 0.0% (0/4500)     |                    18987 |                   4.219 |                    18987 |                   4.219 |
| Vague-Complaint Stress | Baseline (a) Argmax | 68.9% (124/180)            | 31.1% (56/180)          | 1.3% (57/4320)             | 0.0% (0/4500)     |                     3352 |                   0.745 |                     3352 |                   0.745 |
| 20% Vital Corruption   | Full TriageGuard    | 92.2% (166/180)            | 2.2% (4/180)            | 5.5% (236/4320)            | 37.1% (1670/4500) |                    14608 |                   3.246 |                     9598 |                   2.133 |
| 20% Vital Corruption   | Rules Disabled      | 92.2% (166/180)            | 2.2% (4/180)            | 2.8% (121/4320)            | 37.7% (1697/4500) |                    11939 |                   2.653 |                     6848 |                   1.522 |
| 20% Vital Corruption   | Screen Disabled     | 70.6% (127/180)            | 2.2% (4/180)            | 3.2% (139/4320)            | 40.1% (1803/4500) |                     9727 |                   2.162 |                     4318 |                   0.96  |
| 20% Vital Corruption   | Fusion Only         | 99.4% (179/180)            | 0.6% (1/180)            | 15.8% (684/4320)           | 0.0% (0/4500)     |                    32665 |                   7.259 |                    32665 |                   7.259 |
| 20% Vital Corruption   | Baseline (a) Argmax | 93.9% (169/180)            | 6.1% (11/180)           | 3.3% (141/4320)            | 0.0% (0/4500)     |                     7469 |                   1.66  |                     7469 |                   1.66  |

---

## 2. Layer Configurations on VALIDATION Split (N=3,000)

Evaluates performance on Validation split across unperturbed encounters and vague-complaint emergencies ($N=120$ emergencies).

| Validation Condition       | Configuration       | Strict Recall (L1-L2)   | Strict 95% CI   | Caught or Abstained   | Caught/Abs 95% CI   | Silent Misses   | Non-Emergency Escalation   | Abstain Rate     |   Total Cost (C_abs=3.0) |   Mean Cost (C_abs=3.0) |   Total Cost (C_abs=0.0) |   Mean Cost (C_abs=0.0) |
|:---------------------------|:--------------------|:------------------------|:----------------|:----------------------|:--------------------|:----------------|:---------------------------|:-----------------|-------------------------:|------------------------:|-------------------------:|------------------------:|
| Clean Validation           | Full TriageGuard    | 99.2% (119/120)         | [95.4%, 99.9%]  | 100.0% (120/120)      | [96.9%, 100.0%]     | 0.0% (0/120)    | 4.6% (132/2880)            | 15.4% (462/3000) |                     5501 |                   1.834 |                     4115 |                   1.372 |
| Clean Validation           | Rules Disabled      | 99.2% (119/120)         | [95.4%, 99.9%]  | 100.0% (120/120)      | [96.9%, 100.0%]     | 0.0% (0/120)    | 1.1% (31/2880)             | 15.5% (465/3000) |                     2950 |                   0.983 |                     1555 |                   0.518 |
| Clean Validation           | Screen Disabled     | 88.3% (106/120)         | [81.4%, 92.9%]  | 100.0% (120/120)      | [96.9%, 100.0%]     | 0.0% (0/120)    | 4.4% (127/2880)            | 16.0% (480/3000) |                     5175 |                   1.725 |                     3735 |                   1.245 |
| Clean Validation           | Fusion Only         | 100.0% (120/120)        | [96.9%, 100.0%] | 100.0% (120/120)      | [96.9%, 100.0%]     | 0.0% (0/120)    | 8.9% (256/2880)            | 0.0% (0/3000)    |                    10165 |                   3.388 |                    10165 |                   3.388 |
| Clean Validation           | Baseline (a) Argmax | 100.0% (120/120)        | [96.9%, 100.0%] | 100.0% (120/120)      | [96.9%, 100.0%]     | 0.0% (0/120)    | 1.4% (40/2880)             | 0.0% (0/3000)    |                     1615 |                   0.538 |                     1615 |                   0.538 |
| Vague-Complaint Validation | Full TriageGuard    | 81.7% (98/120)          | [73.8%, 87.6%]  | 90.8% (109/120)       | [84.3%, 94.8%]      | 9.2% (11/120)   | 4.6% (132/2880)            | 15.7% (472/3000) |                     6202 |                   2.067 |                     4786 |                   1.595 |
| Vague-Complaint Validation | Rules Disabled      | 75.8% (91/120)          | [67.4%, 82.6%]  | 90.0% (108/120)       | [83.3%, 94.2%]      | 10.0% (12/120)  | 1.1% (31/2880)             | 16.0% (481/3000) |                     4250 |                   1.417 |                     2807 |                   0.936 |
| Vague-Complaint Validation | Screen Disabled     | 73.3% (88/120)          | [64.8%, 80.4%]  | 89.2% (107/120)       | [82.3%, 93.6%]      | 10.8% (13/120)  | 4.4% (127/2880)            | 16.2% (485/3000) |                     5524 |                   1.841 |                     4069 |                   1.356 |
| Vague-Complaint Validation | Fusion Only         | 93.3% (112/120)         | [87.4%, 96.6%]  | 93.3% (112/120)       | [87.4%, 96.6%]      | 6.7% (8/120)    | 8.9% (256/2880)            | 0.0% (0/3000)    |                    11699 |                   3.9   |                    11699 |                   3.9   |
| Vague-Complaint Validation | Baseline (a) Argmax | 67.5% (81/120)          | [58.7%, 75.2%]  | 67.5% (81/120)        | [58.7%, 75.2%]      | 32.5% (39/120)  | 1.4% (40/2880)             | 0.0% (0/3000)    |                     2330 |                   0.777 |                     2330 |                   0.777 |

---

## 3. Full Pipeline Conformal Trade-off at Alpha in {0.01, 0.02, 0.05}

Evaluates Full TriageGuard operating under pre-calibrated conformal thresholds (without recomputing $\tau$):
- $\alpha = 0.01 \implies \tau = 0.0018$
- $\alpha = 0.02 \implies \tau = 0.0039$
- $\alpha = 0.05 \implies \tau = 0.0922$ (Current Operating Default)

| Evaluation Split           |   Conformal Alpha |   Calibrated Tau | Strict Recall (L1-L2)   | Strict 95% CI   | Caught or Abstained   | Caught/Abs 95% CI   | Silent Misses   | Non-Emergency Escalation   | Abstain Rate     |   Total Cost (C_abs=3.0) |   Mean Cost (C_abs=3.0) |   Total Cost (C_abs=0.0) |   Mean Cost (C_abs=0.0) |
|:---------------------------|------------------:|-----------------:|:------------------------|:----------------|:----------------------|:--------------------|:----------------|:---------------------------|:-----------------|-------------------------:|------------------------:|-------------------------:|------------------------:|
| Validation Split (N=3,000) |              0.01 |           0.0018 | 100.0% (120/120)        | [96.9%, 100.0%] | 100.0% (120/120)      | [96.9%, 100.0%]     | 0.0% (0/120)    | 14.5% (417/2880)           | 9.9% (297/3000)  |                    15439 |                   5.146 |                    14548 |                   4.849 |
| Validation Split (N=3,000) |              0.02 |           0.0039 | 100.0% (120/120)        | [96.9%, 100.0%] | 100.0% (120/120)      | [96.9%, 100.0%]     | 0.0% (0/120)    | 10.4% (300/2880)           | 12.0% (361/3000) |                    11314 |                   3.771 |                    10231 |                   3.41  |
| Validation Split (N=3,000) |              0.05 |           0.0922 | 99.2% (119/120)         | [95.4%, 99.9%]  | 100.0% (120/120)      | [96.9%, 100.0%]     | 0.0% (0/120)    | 4.6% (132/2880)            | 15.4% (462/3000) |                     5501 |                   1.834 |                     4115 |                   1.372 |
| Test Split (N=4,500)       |              0.01 |           0.0018 | 100.0% (180/180)        | [97.9%, 100.0%] | 100.0% (180/180)      | [97.9%, 100.0%]     | 0.0% (0/180)    | 14.0% (604/4320)           | 11.2% (505/4500) |                    23164 |                   5.148 |                    21649 |                   4.811 |
| Test Split (N=4,500)       |              0.02 |           0.0039 | 98.3% (177/180)         | [95.2%, 99.4%]  | 99.4% (179/180)       | [96.9%, 99.9%]      | 0.6% (1/180)    | 10.4% (451/4320)           | 13.3% (597/4500) |                    17629 |                   3.918 |                    15838 |                   3.52  |
| Test Split (N=4,500)       |              0.05 |           0.0922 | 92.8% (167/180)         | [88.0%, 95.7%]  | 97.2% (175/180)       | [93.7%, 98.8%]      | 2.8% (5/180)    | 4.4% (192/4320)            | 17.0% (763/4500) |                     8365 |                   1.859 |                     6076 |                   1.35  |

---

## 4. Policy Table on VALIDATION Split (Recall Floors 95%, 98%, 99%)

| Recall Floor   | Min-Cost Config (Strict Recall >= Floor)   | Strict Recall on Val   | Val Mean Cost (Strict Gate)   | Min-Cost Config (Caught/Abs Recall >= Floor)   | Caught/Abs Recall on Val   | Val Mean Cost (Caught/Abs Gate)   |
|:---------------|:-------------------------------------------|:-----------------------|:------------------------------|:-----------------------------------------------|:---------------------------|:----------------------------------|
| 95%            | Baseline (a) Argmax                        | 100.0%                 | 0.538 (Total: 1,615.0)        | Baseline (a) Argmax                            | 100.0%                     | 0.538 (Total: 1,615.0)            |
| 98%            | Baseline (a) Argmax                        | 100.0%                 | 0.538 (Total: 1,615.0)        | Baseline (a) Argmax                            | 100.0%                     | 0.538 (Total: 1,615.0)            |
| 99%            | Baseline (a) Argmax                        | 100.0%                 | 0.538 (Total: 1,615.0)        | Baseline (a) Argmax                            | 100.0%                     | 0.538 (Total: 1,615.0)            |

### Optimization Rule Specification
```text
Optimization Rule Used:
  arg min_{Config, alpha} Mean_Cost_VAL(Config, alpha | C_abstain=3.0)
  subject to: Empirical_Recall_VAL(Config, alpha) >= Target_Floor
Where:
  - Strict Recall Gate requires Level in {1, 2} for emergency encounters.
  - Caught-or-Abstained Gate requires Level in {1, 2} OR safe referral to nurse (abstain) for emergency encounters.
  - Evaluated over candidate alphas {0.005, 0.01, 0.02, 0.05} on Full Pipeline plus architectural ablations.
```

---

*Generated programmatically in 638.53 seconds. Models, thresholds, and defaults remain untouched.*