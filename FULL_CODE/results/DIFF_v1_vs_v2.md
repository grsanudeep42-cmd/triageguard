# Metric Diff Report: results_v1 vs results

This file records every metric difference between the initial benchmark snapshot (`results_v1/`)
and the reconciled enhanced deliverables (`results/`).

## Summary of Core Metric Invariance
The following core evaluation metrics are **100% numerically identical** between v1 and v2:
- `abstention_metrics.json`: Exactly identical (overall abstain rate, reason breakdowns).
- `conformal_coverage.json`: Exactly identical (95.2% non-emergency level set coverage on N=4,320, 92.8% Mondrian emergency screen coverage on N=180).
- `confusion_matrix.json`: Exactly identical (all cells across all 5 levels + abstain).
- `cost_comparison.json`: Exactly identical (standard costs across all systems).
- `department_accuracy.json`: Exactly identical (primary and specialty-or-emergency accuracy across all levels).
- `emergency_recall.json`: Exactly identical (headline recall 92.8%, under-reported 96.4%, standard 92.1%).
- `vital_corruption_test.json`: Exactly identical (recall across 0%, 5%, 10%, 20% corruption rates).

## Changed Metrics & File-by-File Diffs

### `abstention_metrics.json`
- **Status:** IDENTICAL (0 changed metrics)

### `conformal_coverage.json`
- **Status:** IDENTICAL (0 changed metrics)

### `conformal_tradeoff.json`
- **Status:** CHANGED
  - Array Diff (len v1: 4, len v2: 4):
    - Row 0:
      - **v1 (Old):** `{'alpha': 0.005, 'tau': '0.0000', 'Status': 'Infeasible', 'Emergency Recall': '100.0%', 'Non-Emergency Escalation': '100.0%', 'L3 Escalated': '100.0%', 'L4 Escalated': '100.0%', 'L5 Escalated': '100.0%'}`
      - **v2 (New):** `{'alpha': 0.005, 'tau': '0.0000', 'Status': 'Infeasible', 'Emergency Recall (Screen Only)': '180/180 (100.0%)', 'Non-Emergency Escalation (Screen Only)': '4320/4320 (100.0%)', 'L3 Escalated': '1114/1114 (100.0%)', 'L4 Escalated': '1827/1827 (100.0%)', 'L5 Escalated': '1379/1379 (100.0%)'}`
    - Row 1:
      - **v1 (Old):** `{'alpha': 0.01, 'tau': '0.0018', 'Status': 'Feasible', 'Emergency Recall': '100.0%', 'Non-Emergency Escalation': '11.1%', 'L3 Escalated': '8.2%', 'L4 Escalated': '20.4%', 'L5 Escalated': '1.1%'}`
      - **v2 (New):** `{'alpha': 0.01, 'tau': '0.0018', 'Status': 'Feasible', 'Emergency Recall (Screen Only)': '180/180 (100.0%)', 'Non-Emergency Escalation (Screen Only)': '476/4320 (11.0%)', 'L3 Escalated': '89/1114 (8.0%)', 'L4 Escalated': '372/1827 (20.4%)', 'L5 Escalated': '15/1379 (1.1%)'}`
    - Row 2:
      - **v1 (Old):** `{'alpha': 0.02, 'tau': '0.0039', 'Status': 'Feasible', 'Emergency Recall': '98.3%', 'Non-Emergency Escalation': '7.4%', 'L3 Escalated': '5.3%', 'L4 Escalated': '13.7%', 'L5 Escalated': '0.7%'}`
      - **v2 (New):** `{'alpha': 0.02, 'tau': '0.0039', 'Status': 'Feasible', 'Emergency Recall (Screen Only)': '177/180 (98.3%)', 'Non-Emergency Escalation (Screen Only)': '318/4320 (7.4%)', 'L3 Escalated': '59/1114 (5.3%)', 'L4 Escalated': '250/1827 (13.7%)', 'L5 Escalated': '9/1379 (0.7%)'}`
    - Row 3:
      - **v1 (Old):** `{'alpha': 0.05, 'tau': '0.0922', 'Status': 'Feasible', 'Emergency Recall': '92.8%', 'Non-Emergency Escalation': '0.9%', 'L3 Escalated': '0.7%', 'L4 Escalated': '1.6%', 'L5 Escalated': '0.0%'}`
      - **v2 (New):** `{'alpha': 0.05, 'tau': '0.0922', 'Status': 'Feasible', 'Emergency Recall (Screen Only)': '167/180 (92.8%)', 'Non-Emergency Escalation (Screen Only)': '37/4320 (0.9%)', 'L3 Escalated': '8/1114 (0.7%)', 'L4 Escalated': '29/1827 (1.6%)', 'L5 Escalated': '0/1379 (0.0%)'}`

### `confusion_matrix.json`
- **Status:** IDENTICAL (0 changed metrics)

### `cost_comparison.json`
- **Status:** IDENTICAL (0 changed metrics)

### `demo_cases.json`
- **Status:** CHANGED
  - Key `case1_true_emergency_forced_by_rule`:
    - **v1 (Old):** `None`
    - **v2 (New):** `{'scenario': 'case1_true_emergency_forced_by_rule', 'description': 'True Level 1/2 emergency forced to urgent triage by hard-coded vitals rule.', 'patient_id': 'P000172', 'split': 'test', 'age': 5, 'sex': 'M', 'complaint_text': 'problem ga feel avutundi', 'cleaned_vitals': {'bp_sys': nan, 'bp_dia': 39.0, 'pulse': 156.0, 'temp_c': 36.4, 'spo2': 84.3}, 'true_clean_level': 1, 'true_department': 'Emergency/Casualty', 'triageguard_prediction': {'triage_level': 1, 'primary_department': 'Emergency/Casualty', 'suggested_specialty': 'Emergency/Casualty', 'red_flags': [], 'source_layer': 'rule', 'conformal_set': [1], 'abstain_reason': None, 'p_emergency': 0.1528, 'latency_ms': 10.08}}`
  - Key `case1_vitals_rule_forces_emergency`:
    - **v1 (Old):** `{'scenario': 'case1_vitals_rule_forces_emergency', 'patient_id': 'P000089', 'age': 26, 'sex': 'M', 'complaint_text': 'persistent cough yellow sputum now cannot walk far since 2 days', 'cleaned_vitals': {'bp_sys': 150.0, 'bp_dia': 78.0, 'pulse': 97.0, 'temp_c': 37.2, 'spo2': 83.2}, 'true_clean_level': 3, 'true_department': 'Pulmonology', 'triageguard_prediction': {'triage_level': 1, 'primary_department': 'Emergency/Casualty', 'suggested_specialty': 'Pulmonology', 'red_flags': ['slurred_speech'], 'source_layer': 'rule', 'conformal_set': [3], 'abstain_reason': None, 'p_emergency': 0.0022, 'latency_ms': 15.98}}`
    - **v2 (New):** `None`
  - Key `case1b_rule_over_escalation_honest`:
    - **v1 (Old):** `None`
    - **v2 (New):** `{'scenario': 'case1b_rule_over_escalation_honest', 'description': 'True Level 3 patient whose severe hypoxia (SpO2 83.2%) safely forced Level 1 via Rule R001. Honestly documented as safe over-escalation.', 'patient_id': 'P000089', 'split': 'test', 'age': 26, 'sex': 'M', 'complaint_text': 'persistent cough yellow sputum now cannot walk far since 2 days', 'cleaned_vitals': {'bp_sys': 150.0, 'bp_dia': 78.0, 'pulse': 97.0, 'temp_c': 37.2, 'spo2': 83.2}, 'true_clean_level': 3, 'true_department': 'Pulmonology', 'triageguard_prediction': {'triage_level': 1, 'primary_department': 'Emergency/Casualty', 'suggested_specialty': 'Pulmonology', 'red_flags': [], 'source_layer': 'rule', 'conformal_set': [3], 'abstain_reason': None, 'p_emergency': 0.001, 'latency_ms': 14.08}}`
  - Key `case2_hidden_emergency_conformal_screen`:
    - **v1 (Old):** `{'scenario': 'case2_hidden_emergency_conformal_screen', 'patient_id': 'P001121', 'age': 28, 'sex': 'F', 'complaint_text': '9 months pregnant high BP blurred vision aur peeth mein bhi thoda', 'cleaned_vitals': {'bp_sys': 182.0, 'bp_dia': 93.0, 'pulse': 117.0, 'temp_c': 37.1, 'spo2': 100.0}, 'true_clean_level': 2, 'true_department': 'Obstetrics & Gynecology', 'triageguard_prediction': {'triage_level': 2, 'primary_department': 'Emergency/Casualty', 'suggested_specialty': 'Obstetrics & Gynecology', 'red_flags': [], 'source_layer': 'screen', 'conformal_set': [2], 'abstain_reason': None, 'p_emergency': 0.9953, 'latency_ms': 16.36}}`
    - **v2 (New):** `None`
  - Key `case2_vague_complaint_caught_by_conformal_screen`:
    - **v1 (Old):** `None`
    - **v2 (New):** `{'scenario': 'case2_vague_complaint_caught_by_conformal_screen', 'description': 'True Level 2 emergency with very short or vague complaint successfully rescued by Mondrian conformal screen (p_em >= tau).', 'patient_id': 'P000303', 'split': 'test', 'age': 83, 'sex': 'F', 'complaint_text': 'head pain throbbing', 'cleaned_vitals': {'bp_sys': 195.0, 'bp_dia': 124.0, 'pulse': 69.0, 'temp_c': 36.2, 'spo2': 95.0}, 'true_clean_level': 2, 'true_department': 'Emergency/Casualty', 'triageguard_prediction': {'triage_level': 2, 'primary_department': 'Emergency/Casualty', 'suggested_specialty': 'Emergency/Casualty', 'red_flags': [], 'source_layer': 'screen', 'conformal_set': [2], 'abstain_reason': None, 'p_emergency': 0.9901, 'latency_ms': 19.47}}`
  - Key `case3_gaming_attempt_defended`:
    - **v1 (Old):** `{'scenario': 'case3_gaming_attempt_defended', 'patient_id': 'P000001', 'age': 61, 'sex': 'F', 'complaint_text': 'kannulo noppi discharge vastondi since yesterday from morning also body pan mpld', 'cleaned_vitals': {'bp_sys': 113.0, 'bp_dia': 56.0, 'pulse': 70.0, 'temp_c': 37.6, 'spo2': 98.3}, 'true_clean_level': 4, 'true_department': 'ENT', 'triageguard_prediction': {'triage_level': 4, 'primary_department': 'ENT', 'suggested_specialty': 'ENT', 'red_flags': [], 'source_layer': 'fusion', 'conformal_set': [4], 'abstain_reason': None, 'p_emergency': 0.0001, 'latency_ms': 29.06}}`
    - **v2 (New):** `None`
  - Key `case3_gaming_attempt_defended_at_non_emergency`:
    - **v1 (Old):** `None`
    - **v2 (New):** `{'scenario': 'case3_gaming_attempt_defended_at_non_emergency', 'description': 'Patient from actual 500-patient gaming stress test (first defended patient). Appending dramatic text failed to escalate past objective vitals.', 'patient_id': 'P005556', 'split': 'test', 'selection_method': 'First patient in the 500-patient non-emergency gaming pool that remained defended at non-emergency level.', 'age': 57, 'sex': 'F', 'original_complaint_text': 'mild jwaram 2 days nund cold mariyu daggu also some body ache', 'injected_dramatic_phrase': 'unbearable pain please see me immediately', 'combined_gamed_complaint': 'mild jwaram 2 days nund cold mariyu daggu also some body ache. unbearable pain please see me immediately', 'cleaned_vitals': {'bp_sys': 119.0, 'bp_dia': 61.0, 'pulse': 70.0, 'temp_c': 37.4, 'spo2': 96.7}, 'true_clean_level': 5, 'true_department': 'General Medicine', 'triageguard_prediction': {'triage_level': 5, 'primary_department': 'General Medicine', 'suggested_specialty': 'General Medicine', 'red_flags': ['severe_pain'], 'source_layer': 'fusion', 'conformal_set': [5], 'abstain_reason': None, 'p_emergency': 0.0, 'latency_ms': 17.32}}`
  - Key `case3b_gaming_attempt_abstained_honestly`:
    - **v1 (Old):** `None`
    - **v2 (New):** `{'scenario': 'case3b_gaming_attempt_abstained_honestly', 'description': 'Patient from actual 500-patient gaming stress test where injected dramatic text provoked extreme model discordance, safely abstaining to nurse triage.', 'patient_id': 'P022989', 'split': 'test', 'selection_method': 'First patient in the 500-patient non-emergency gaming pool that triggered safety abstention (model_disagreement).', 'age': 54, 'sex': 'F', 'original_complaint_text': 'feeling off', 'injected_dramatic_phrase': 'emergency please I am extremely sick', 'combined_gamed_complaint': 'feeling off. emergency please I am extremely sick', 'cleaned_vitals': {'bp_sys': 99.0, 'bp_dia': 73.0, 'pulse': 74.0, 'temp_c': 37.0, 'spo2': 99.9}, 'true_clean_level': 4, 'true_department': 'Dermatology', 'triageguard_prediction': {'triage_level': None, 'primary_department': 'Ophthalmology', 'suggested_specialty': 'Ophthalmology', 'red_flags': [], 'source_layer': 'abstain', 'conformal_set': [5], 'abstain_reason': 'model_disagreement', 'p_emergency': 0.0007, 'latency_ms': 13.01}}`
  - Key `case4_corrupted_vitals_abstain`:
    - **v1 (Old):** `{'scenario': 'case4_corrupted_vitals_abstain', 'patient_id': 'P000270', 'age': 49, 'sex': 'M', 'complaint_text': 'medicine marchipoyanu ippudu BP chla you know high nakseer ee roju subah nundi', 'cleaned_vitals': {'bp_sys': 198.0, 'bp_dia': 85.0, 'pulse': 87.0, 'temp_c': nan, 'spo2': nan}, 'true_clean_level': 3, 'true_department': 'General Medicine', 'triageguard_prediction': {'triage_level': None, 'primary_department': 'General Medicine', 'suggested_specialty': 'General Medicine', 'red_flags': [], 'source_layer': 'abstain', 'conformal_set': [3], 'abstain_reason': 'needs_remeasure', 'p_emergency': 0.0001, 'latency_ms': 18.69}}`
    - **v2 (New):** `{'scenario': 'case4_corrupted_vitals_abstain', 'description': 'Encounter with >= 2 invalid or missing vitals safely triggering needs_remeasure abstention to triage nurse.', 'patient_id': 'P000270', 'split': 'test', 'age': 49, 'sex': 'M', 'complaint_text': 'medicine marchipoyanu ippudu BP chla you know high nakseer ee roju subah nundi', 'cleaned_vitals': {'bp_sys': 198.0, 'bp_dia': 85.0, 'pulse': 87.0, 'temp_c': None, 'spo2': None}, 'true_clean_level': 3, 'true_department': 'General Medicine', 'triageguard_prediction': {'triage_level': None, 'primary_department': 'General Medicine', 'suggested_specialty': 'General Medicine', 'red_flags': [], 'source_layer': 'abstain', 'conformal_set': [3], 'abstain_reason': 'needs_remeasure', 'p_emergency': 0.0001, 'latency_ms': 16.25}}`
  - Key `case5_benign_case_routed_correctly`:
    - **v1 (Old):** `{'scenario': 'case5_benign_case_routed_correctly', 'patient_id': 'P000001', 'age': 61, 'sex': 'F', 'complaint_text': 'kannulo noppi discharge vastondi since yesterday from morning also body pan mpld', 'cleaned_vitals': {'bp_sys': 113.0, 'bp_dia': 56.0, 'pulse': 70.0, 'temp_c': 37.6, 'spo2': 98.3}, 'true_clean_level': 4, 'true_department': 'ENT', 'triageguard_prediction': {'triage_level': 4, 'primary_department': 'ENT', 'suggested_specialty': 'ENT', 'red_flags': [], 'source_layer': 'fusion', 'conformal_set': [4], 'abstain_reason': None, 'p_emergency': 0.0001, 'latency_ms': 29.06}}`
    - **v2 (New):** `{'scenario': 'case5_benign_case_routed_correctly', 'description': 'Benign non-emergency encounter (Dermatology skin rash) routed accurately to specialty department without unnecessary emergency escalation.', 'patient_id': 'P000234', 'split': 'test', 'age': 5, 'sex': 'M', 'complaint_text': 'fce meda gajjulu red bumps since yesterday kal se', 'cleaned_vitals': {'bp_sys': 107.0, 'bp_dia': 72.0, 'pulse': 83.0, 'temp_c': 37.4, 'spo2': 97.0}, 'true_clean_level': 4, 'true_department': 'Dermatology', 'triageguard_prediction': {'triage_level': 4, 'primary_department': 'Dermatology', 'suggested_specialty': 'Dermatology', 'red_flags': [], 'source_layer': 'fusion', 'conformal_set': [4], 'abstain_reason': None, 'p_emergency': 0.0, 'latency_ms': 13.09}}`
  - Key `case6_model_disagreement_abstain`:
    - **v1 (Old):** `{'scenario': 'case6_model_disagreement_abstain', 'patient_id': 'P000046', 'age': 16, 'sex': 'M', 'complaint_text': 'check karna tha', 'cleaned_vitals': {'bp_sys': 126.0, 'bp_dia': 69.0, 'pulse': 87.0, 'temp_c': 37.1, 'spo2': 97.8}, 'true_clean_level': 4, 'true_department': 'Orthopedics', 'triageguard_prediction': {'triage_level': None, 'primary_department': 'General Medicine', 'suggested_specialty': 'General Medicine', 'red_flags': [], 'source_layer': 'abstain', 'conformal_set': [4], 'abstain_reason': 'model_disagreement', 'p_emergency': 0.0001, 'latency_ms': 9.21}}`
    - **v2 (New):** `{'scenario': 'case6_model_disagreement_abstain', 'description': 'Encounter where vitals-only model and text-only model disagree by >= 2 levels, triggering safe abstention to human triage nurse.', 'patient_id': 'P000046', 'split': 'test', 'age': 16, 'sex': 'M', 'complaint_text': 'check karna tha', 'cleaned_vitals': {'bp_sys': 126.0, 'bp_dia': 69.0, 'pulse': 87.0, 'temp_c': 37.1, 'spo2': 97.8}, 'true_clean_level': 4, 'true_department': 'Orthopedics', 'triageguard_prediction': {'triage_level': None, 'primary_department': 'General Medicine', 'suggested_specialty': 'General Medicine', 'red_flags': [], 'source_layer': 'abstain', 'conformal_set': [4], 'abstain_reason': 'model_disagreement', 'p_emergency': 0.0001, 'latency_ms': 8.99}}`

### `department_accuracy.json`
- **Status:** IDENTICAL (0 changed metrics)

### `emergency_recall.json`
- **Status:** IDENTICAL (0 changed metrics)

### `latency_benchmarks.json`
- **Status:** CHANGED
  - Array Diff (len v1: 5, len v2: 5):
    - Row 0:
      - **v1 (Old):** `{'Metric': 'Mean Latency', 'Value (ms)': '13.88 ms'}`
      - **v2 (New):** `{'Metric': 'Mean Latency', 'Value (ms)': '12.77 ms'}`
    - Row 1:
      - **v1 (Old):** `{'Metric': 'p50 Latency', 'Value (ms)': '13.72 ms'}`
      - **v2 (New):** `{'Metric': 'p50 Latency', 'Value (ms)': '12.62 ms'}`
    - Row 2:
      - **v1 (Old):** `{'Metric': 'p95 Latency', 'Value (ms)': '18.36 ms'}`
      - **v2 (New):** `{'Metric': 'p95 Latency', 'Value (ms)': '16.20 ms'}`
    - Row 3:
      - **v1 (Old):** `{'Metric': 'p99 Latency', 'Value (ms)': '20.56 ms'}`
      - **v2 (New):** `{'Metric': 'p99 Latency', 'Value (ms)': '18.59 ms'}`

### `vital_corruption_test.json`
- **Status:** IDENTICAL (0 changed metrics)

## Newly Added Enhanced Deliverables in v2
- `abstention_by_split.json`: Added in enhanced reporting pass.
- `baseline_a_confusion_matrix.json`: Added in enhanced reporting pass.
- `baseline_b_confusion_matrix.json`: Added in enhanced reporting pass.
- `cost_decomposition.json`: Added in enhanced reporting pass.
- `cost_sensitivity.json`: Added in enhanced reporting pass.
- `gaming_gross_moves.json`: Added in enhanced reporting pass.
- `gaming_stress_test.json`: Added in enhanced reporting pass.
- `gaming_transition_matrix.json`: Added in enhanced reporting pass.
- `layer_ablation_stress.json`: Added in enhanced reporting pass.
- `layer_escalation_decomposition.json`: Added in enhanced reporting pass.
- `monotonicity_breakdown.json`: Added in enhanced reporting pass.
- `recall_definitions.json`: Added in enhanced reporting pass.
- `stress_test_comparison.json`: Added in enhanced reporting pass.
- `system_escalation.json`: Added in enhanced reporting pass.
- `top_l3_l4_emergency_scores.json`: Added in enhanced reporting pass.
- `val_alpha_selection.json`: Added in enhanced reporting pass.