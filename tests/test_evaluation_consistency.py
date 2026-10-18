"""
tests/test_evaluation_consistency.py
─────────────────────────────────────────────────────────────────────────────
Unit tests asserting mathematical consistency across all generated deliverables,
verifying that row/column totals match cohort sizes, and catching any swapped columns.
"""

import json
from pathlib import Path
import pytest

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def test_confusion_matrix_integrity():
    """Verify confusion matrix row sums equal exact split sizes and total equals 4500."""
    cm_path = _RESULTS_DIR / "confusion_matrix.json"
    assert cm_path.exists(), f"Missing {cm_path}"

    with open(cm_path, encoding="utf-8") as f:
        cm_list = json.load(f)
    cm_data = {r["True Level"]: r for r in cm_list}

    expected_row_totals = {
        "True L1": 25,
        "True L2": 155,
        "True L3": 1114,
        "True L4": 1827,
        "True L5": 1379,
    }

    total_sum = 0
    for row_name, expected_total in expected_row_totals.items():
        assert row_name in cm_data, f"Missing {row_name} in confusion matrix"
        row_dict = cm_data[row_name]
        row_sum = sum(int(v) for k, v in row_dict.items() if k != "True Level")
        assert row_sum == expected_total, f"{row_name} sum {row_sum} != {expected_total}"
        total_sum += row_sum

    assert total_sum == 4500, f"Total confusion matrix sum {total_sum} != 4500"


def test_layer_escalation_decomposition_sums():
    """Verify layer escalation decomposition table sums and matches confusion matrix."""
    decomp_path = _RESULTS_DIR / "layer_escalation_decomposition.json"
    cm_path = _RESULTS_DIR / "confusion_matrix.json"
    assert decomp_path.exists(), f"Missing {decomp_path}"
    assert cm_path.exists(), f"Missing {cm_path}"

    with open(decomp_path, encoding="utf-8") as f:
        decomp_data = json.load(f)

    with open(cm_path, encoding="utf-8") as f:
        cm_list = json.load(f)
    cm_data = {r["True Level"]: r for r in cm_list}

    # Compute expected final escalation from confusion matrix (Pred L1 + Pred L2)
    expected_final_esc = {}
    for lvl in range(1, 6):
        row = cm_data[f"True L{lvl}"]
        expected_final_esc[f"Level {lvl}"] = int(row["Pred L1"]) + int(row["Pred L2"])

    expected_final_esc["All Non-Emergencies (L3-L5)"] = (
        expected_final_esc["Level 3"] + expected_final_esc["Level 4"] + expected_final_esc["Level 5"]
    )

    for row in decomp_data:
        lvl_name = row["True Level"]
        tot = row["Total Cases"]

        # Parse fractions from table
        def parse_num(key):
            val_str = row[key]
            return int(val_str.split("/")[0])

        n_a = parse_num("(a) p_em >= tau")
        n_b = parse_num("(b) source == screen")
        n_c = parse_num("(c) rule fired")
        n_d = parse_num("(d) Final L1/L2 Escalation")

        assert n_a <= tot, f"(a) count {n_a} exceeds total {tot} for {lvl_name}"
        assert n_b <= tot, f"(b) count {n_b} exceeds total {tot} for {lvl_name}"
        assert n_c <= tot, f"(c) count {n_c} exceeds total {tot} for {lvl_name}"
        assert n_d <= tot, f"(d) count {n_d} exceeds total {tot} for {lvl_name}"

        # Assert final escalation matches confusion matrix exactly
        assert n_d == expected_final_esc[lvl_name], (
            f"Escalation {n_d} != expected {expected_final_esc[lvl_name]} for {lvl_name}"
        )


def test_paired_gaming_test_counts():
    """Verify paired gaming stress test row sums to exactly 500 for pre and post."""
    gaming_path = _RESULTS_DIR / "gaming_stress_test.json"
    assert gaming_path.exists(), f"Missing {gaming_path}"

    with open(gaming_path, encoding="utf-8") as f:
        rows = json.load(f)

    row_dict = {r["Outcome / Triage State"]: r for r in rows}

    def parse_count(key, col):
        return int(row_dict[key][col].split()[0])

    for col in ["Pre-Injection (Baseline)", "Post-Injection (Gamed)"]:
        esc = parse_count("Escalated to Emergency (L1-L2)", col)
        abs_tot = parse_count("Total Abstained (Referred to Nurse)", col)
        md = parse_count("  └─ Reason: Model Disagreement", col)
        rem = parse_count("  └─ Reason: Needs Re-measure", col)
        defended = parse_count("Defended at Non-Emergency (L3-L5)", col)

        # Assert reasons sum to total abstained
        assert md + rem == abs_tot, f"Abstain breakdown {md} + {rem} != {abs_tot} in {col}"

        # Assert three mutual outcomes sum to 500
        tot = esc + abs_tot + defended
        assert tot == 500, f"Total {col} sum {tot} != 500"


def test_conformal_tradeoff_row_integrity():
    """Verify conformal tradeoff table preserves all alphas and operating rates."""
    tradeoff_path = _RESULTS_DIR / "conformal_tradeoff.json"
    assert tradeoff_path.exists(), f"Missing {tradeoff_path}"

    with open(tradeoff_path, encoding="utf-8") as f:
        rows = json.load(f)

    alphas = [r["alpha"] for r in rows]
    assert 0.05 in alphas, "Missing operating alpha 0.05 in tradeoff table"

    op_row = next(r for r in rows if r["alpha"] == 0.05)
    assert op_row["Status"] == "Feasible"
    assert "92.8%" in op_row["Emergency Recall (Screen Only)"]
    assert "0.9%" in op_row["Non-Emergency Escalation (Screen Only)"]
