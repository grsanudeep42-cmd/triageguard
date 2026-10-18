# TriageGuard Executive Evaluation & Safety Benchmark Summary

This document is programmatically generated from `results/*.json` artifacts.
All figures reflect verified on-disk benchmark evaluations without extrapolation.

---

## 1. Headline Emergency Recall (Test Split, N=4,500 Encounters)
| Cohort                     |   Caught |   Total | Recall   | Wilson 95% CI   | Calib Screen Recall   | Calib Full Recall   | Screen In CI   |
|:---------------------------|---------:|--------:|:---------|:----------------|:----------------------|:--------------------|:---------------|
| All Emergencies (Headline) |      167 |     180 | 92.8%    | [88.0%, 95.7%]  | 95.6%                 | 96.7%               | True           |
| Under-Reported Emergencies |       27 |      28 | 96.4%    | [82.3%, 99.4%]  | -                     | -                   | True           |
| Standard Emergencies       |      140 |     152 | 92.1%    | [86.7%, 95.4%]  | -                     | -                   | True           |

## 2. Conformal Coverage & Screening Guarantees
| Mechanism                              | Target Coverage   | Empirical Test Coverage   | Guaranteed Met   |
|:---------------------------------------|:------------------|:--------------------------|:-----------------|
| Mondrian Emergency Screen (alpha=0.05) | 95.0%             | 92.8%                     | False            |
| Non-Emergency Level Sets (alpha=0.1)   | 90.0%             | 95.2%                     | True             |

> **Population Definitions:**
> - **Mondrian Emergency Screen (92.8%):** Covers True Emergencies exclusively ($Y \in \{1, 2\}$, $N=180$). Guaranteed recall $\ge 95.0\%$; empirical test coverage $92.8\%$ (within Wilson 95% CI $[88.0\%, 95.7\%]$).
> - **Non-Emergency Level Sets (95.2%):** Covers True Non-Emergencies ($Y \in \{3, 4, 5\}$, $N=4,320$). Target coverage $\ge 90.0\%$; empirical test coverage $95.2\%$.

## 3. Layer Ablation Benchmark (Clean, Vague, 20% Vital Corruption)
| Stress Condition       | Layer Configuration   | Emergency Recall (L1-L2)   | Silent Misses (L3-L5)   | Non-Emergency Escalation   | Abstain Rate (Nurse Handoff)   |
|:-----------------------|:----------------------|:---------------------------|:------------------------|:---------------------------|:-------------------------------|
| Clean Test             | Full TriageGuard      | 92.8% (167/180)            | 2.8% (5/180)            | 4.4% (192/4320)            | 17.0% (763/4500)               |
| Clean Test             | Rules Disabled        | 92.8% (167/180)            | 2.8% (5/180)            | 1.1% (49/4320)             | 17.3% (777/4500)               |
| Clean Test             | Screen Disabled       | 83.9% (151/180)            | 3.3% (6/180)            | 4.2% (180/4320)            | 17.5% (789/4500)               |
| Clean Test             | Fusion Only           | 100.0% (180/180)           | 0.0% (0/180)            | 9.7% (420/4320)            | 0.0% (0/4500)                  |
| Vague-Complaint Stress | Full TriageGuard      | 76.7% (138/180)            | 10.6% (19/180)          | 4.4% (192/4320)            | 17.3% (778/4500)               |
| Vague-Complaint Stress | Rules Disabled        | 68.9% (124/180)            | 12.2% (22/180)          | 1.1% (49/4320)             | 17.8% (803/4500)               |
| Vague-Complaint Stress | Screen Disabled       | 75.0% (135/180)            | 10.6% (19/180)          | 4.2% (180/4320)            | 17.6% (792/4500)               |
| Vague-Complaint Stress | Fusion Only           | 92.2% (166/180)            | 7.8% (14/180)           | 9.7% (420/4320)            | 0.0% (0/4500)                  |
| 20% Vital Corruption   | Full TriageGuard      | 92.2% (166/180)            | 2.2% (4/180)            | 5.5% (236/4320)            | 37.1% (1670/4500)              |
| 20% Vital Corruption   | Rules Disabled        | 92.2% (166/180)            | 2.2% (4/180)            | 2.8% (121/4320)            | 37.7% (1697/4500)              |
| 20% Vital Corruption   | Screen Disabled       | 70.6% (127/180)            | 2.2% (4/180)            | 3.2% (139/4320)            | 40.1% (1803/4500)              |
| 20% Vital Corruption   | Fusion Only           | 99.4% (179/180)            | 0.6% (1/180)            | 15.8% (684/4320)           | 0.0% (0/4500)                  |

## 4. Text-Gaming Stress Test & Gross Moves (N=500 Gamed Encounters)
| System              |   Total Evaluated | Gross State Changes   | Gross Escalations to L1-L2   | Gross Abstentions to Nurse   | Internal Non-Emergency Shifts   | Unchanged State & Level   |
|:--------------------|------------------:|:----------------------|:-----------------------------|:-----------------------------|:--------------------------------|:--------------------------|
| Full TriageGuard    |               500 | 198 (39.6%)           | 3 (0.6%)                     | 28 (5.6%)                    | 148 (29.6%)                     | 302 (60.4%)               |
| Baseline (a) Argmax |               500 | 62 (12.4%)            | 4 (0.8%)                     | 0 (0.0%)                     | 57 (11.4%)                      | 438 (87.6%)               |
| Baseline (b) Rules  |               500 | 0 (0.0%)              | 0 (0.0%)                     | 0 (0.0%)                     | 0 (0.0%)                        | 500 (100.0%)              |

| Outcome / Triage State              | Pre-Injection (Baseline)   | Post-Injection (Gamed)   | Net Shift   |
|:------------------------------------|:---------------------------|:-------------------------|:------------|
| Escalated to Emergency (L1-L2)      | 7 (1.4%)                   | 10 (2.0%)                | +3 (+0.6%)  |
| Total Abstained (Referred to Nurse) | 84 (16.8%)                 | 92 (18.4%)               | +8 (+1.6%)  |
| └─ Reason: Model Disagreement       | 73 (14.6%)                 | 81 (16.2%)               | +8 (+1.6%)  |
| └─ Reason: Needs Re-measure         | 11 (2.2%)                  | 11 (2.2%)                | 0 (0.0%)    |
| Defended at Non-Emergency (L3-L5)   | 409 (81.8%)                | 398 (79.6%)              | -11 (-2.2%) |

## 5. Clinical Decision Cost Comparison (20x Emergency Under-Triage Penalty)
| System                             |   Mean Cost (Std, C_abstain=3.0) |   Total Cost (Std) |   Mean Cost (Safe-Abstain, Em=0) |   Total Cost (Safe-Abstain) |
|:-----------------------------------|---------------------------------:|-------------------:|---------------------------------:|----------------------------:|
| Baseline (a): Accuracy-Only Argmax |                            0.144 |                649 |                            0.144 |                         649 |
| Baseline (b): Rules Only           |                            2.02  |               9089 |                            2.02  |                        9089 |
| Full TriageGuard System (c)        |                            0.668 |               3007 |                            0.663 |                        2983 |

## 6. Sensor Degradation & Vital Corruption Comparison
| Corruption Rate   | Full System Recall   | Argmax Recall   | Rules Recall   | Full System Abstain Rate   |   Full System Mean Cost |   Argmax Mean Cost |   Rules Mean Cost |
|:------------------|:---------------------|:----------------|:---------------|:---------------------------|------------------------:|-------------------:|------------------:|
| 0%                | 92.8%                | 93.9%           | 54.4%          | 17.0%                      |                   0.668 |              0.144 |             2.02  |
| 5%                | 92.8%                | 92.8%           | 53.3%          | 19.6%                      |                   0.739 |              0.165 |             2.048 |
| 10%               | 93.3%                | 92.2%           | 48.9%          | 23.5%                      |                   0.883 |              0.19  |             2.158 |
| 20%               | 93.9%                | 92.2%           | 48.9%          | 36.7%                      |                   1.283 |              0.221 |             2.178 |

## 7. Department Routing Accuracy
| True Triage Level   |   Total Cases | Primary-Only Accuracy   | Specialty-or-Emergency Accuracy   |
|:--------------------|--------------:|:------------------------|:----------------------------------|
| Level 1             |            25 | 100.0%                  | 100.0%                            |
| Level 2             |           155 | 45.8%                   | 96.8%                             |
| Level 3             |          1114 | 84.1%                   | 95.2%                             |
| Level 4             |          1827 | 84.0%                   | 86.2%                             |
| Level 5             |          1379 | 96.0%                   | 96.2%                             |
| All Answered Cases  |          3737 | 88.5%                   | 95.0%                             |

## 8. Latency Benchmarks (End-to-End Per-Patient Inference)
| Metric                        | Value (ms)   |
|:------------------------------|:-------------|
| Mean Latency                  | 12.77 ms     |
| p50 Latency                   | 12.62 ms     |
| p95 Latency                   | 16.20 ms     |
| p99 Latency                   | 18.59 ms     |
| Latency Target (< 100 ms p95) | PASS         |

## 9. Safety Abstention Distribution
| Metric                                 | Value         |
|:---------------------------------------|:--------------|
| Total Cases                            | 4,500         |
| Answered Cases                         | 3,737 (83.0%) |
| Abstained Cases                        | 763 (17.0%)   |
| Top-1 Level Accuracy on Answered Cases | 91.0%         |
| Mean Cost on Answered Cases            | 0.192         |
| Mean Cost on Abstained Cases           | 3.000         |
