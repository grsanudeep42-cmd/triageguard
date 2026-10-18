# Layer Ablation Stress Testing Across Operating Regimes

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
