"""
triageguard/conformal.py
─────────────────────────────────────────────────────────────────────────────
Phase 6 – Conformal prediction and Mondrian emergency screening for TriageGuard.

CONFORMAL GUARANTEES (Finite-Sample Corrected)
---------------------------------------------
1. Emergency Screen (Mondrian Conformal):
   Calibrated strictly on calibration emergencies (n_calib_emergencies).
   For a given risk alpha:
     k = floor(alpha * (n + 1))
     tau = k-th smallest emergency score (1-indexed: sorted_scores[k-1])
     If k == 0: marked as "infeasible at this n" (escalate-all threshold = 0.0)
   Guarantee: P(p_em >= tau | emergency) >= 1 - alpha.

2. Non-Emergency Level Sets:
   Split conformal on calib non-emergencies (alpha_level = 0.10).
   Quantile level: q = min(1.0, ceil((n + 1) * (1 - alpha_level)) / n).
   Prediction set: C(x) = { lvl in {1..5} : 1 - p_lvl <= cutoff }.

3. Operating Alpha Selection on VAL:
   Candidate alphas evaluated on the validation split.
   Operating alpha is chosen by minimum expected cost subject to recall floor (>= 95%).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from triageguard.fusion import compute_expected_costs, select_min_cost_level


@dataclass
class ConformalState:
    tau_map: Dict[float, Optional[float]] = field(default_factory=dict)
    infeasible_alphas: List[float] = field(default_factory=list)
    operating_alpha: float = 0.01
    operating_tau: float = 0.5
    level_cutoff: float = 0.85
    level_alpha: float = 0.10


class ConformalPredictor:
    def __init__(
        self,
        candidate_alphas: Optional[List[float]] = None,
        level_alpha: float = 0.10,
        recall_floor_on_val: float = 0.95,
    ):
        self.candidate_alphas = candidate_alphas or [0.005, 0.01, 0.02, 0.05]
        self.level_alpha = level_alpha
        self.recall_floor_on_val = recall_floor_on_val
        self.state = ConformalState(level_alpha=level_alpha)
        self.is_calibrated = False

    def calibrate_on_calib(
        self,
        p_em_calib: np.ndarray,
        p_multi_calib: np.ndarray,
        y_is_emergency_calib: np.ndarray,
        y_level_calib: np.ndarray,
    ) -> ConformalPredictor:
        """
        Calibrate screening thresholds tau_alpha on calib emergencies,
        and level set cutoff on calib non-emergencies.
        """
        em_mask = y_is_emergency_calib.astype(bool)
        calib_em_scores = np.sort(p_em_calib[em_mask])
        n_em = len(calib_em_scores)

        tau_map: Dict[float, Optional[float]] = {}
        infeasible: List[float] = []

        # 1. Mondrian Emergency Screen with finite-sample correction
        for alpha in self.candidate_alphas:
            k = int(math.floor(alpha * (n_em + 1)))
            if k == 0:
                tau_map[alpha] = 0.0  # Infeasible: escalate-all
                infeasible.append(alpha)
            else:
                tau_map[alpha] = float(calib_em_scores[k - 1])

        self.state.tau_map = tau_map
        self.state.infeasible_alphas = infeasible

        # 2. Non-emergency level sets
        non_em_mask = ~em_mask
        n_non_em = int(np.sum(non_em_mask))
        if n_non_em > 0:
            p_non_em = p_multi_calib[non_em_mask]
            y_non_em = y_level_calib[non_em_mask].astype(int) - 1  # 0-indexed
            # Non-conformity score: s_i = 1 - p(y_i)
            true_p = p_non_em[np.arange(n_non_em), y_non_em]
            scores = 1.0 - true_p

            q_level = min(1.0, math.ceil((n_non_em + 1) * (1.0 - self.level_alpha)) / n_non_em)
            self.state.level_cutoff = float(np.quantile(scores, q_level))
        else:
            self.state.level_cutoff = 0.85

        self.is_calibrated = True
        return self

    def select_operating_alpha_on_val(
        self,
        p_em_val: np.ndarray,
        p_multi_val: np.ndarray,
        y_is_emergency_val: np.ndarray,
        y_level_val: np.ndarray,
        cost_matrix: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, pd.DataFrame]:
        """
        Select operating alpha on VAL split by min expected cost subject to recall floor.
        Returns: (best_alpha, best_tau, val_tradeoff_df).
        """
        if not self.is_calibrated:
            raise RuntimeError("ConformalPredictor must be calibrated on calib before selecting on val.")

        val_records = []
        em_mask = y_is_emergency_val.astype(bool)
        n_em_val = int(np.sum(em_mask))
        n_non_em_val = int(np.sum(~em_mask))

        best_alpha = self.candidate_alphas[0]
        best_tau = self.state.tau_map[best_alpha]
        best_cost = float("inf")

        for alpha in self.candidate_alphas:
            tau = self.state.tau_map[alpha]
            is_inf = alpha in self.state.infeasible_alphas

            # Screen decision on val
            screen_fires = p_em_val >= (tau if tau is not None else 0.0)
            caught_em = np.sum(screen_fires & em_mask)
            recall = caught_em / n_em_val if n_em_val > 0 else 0.0

            escalated_non_em = np.sum(screen_fires & (~em_mask))
            esc_rate = escalated_non_em / n_non_em_val if n_non_em_val > 0 else 0.0

            # L3, L4, L5 escalation breakdown
            esc_by_level = {}
            for lvl in [3, 4, 5]:
                sub_mask = y_level_val == lvl
                n_sub = int(np.sum(sub_mask))
                n_esc_sub = int(np.sum(screen_fires & sub_mask))
                esc_by_level[f"esc_L{lvl}_rate"] = n_esc_sub / n_sub if n_sub > 0 else 0.0

            # Compute clinical decision cost under this tau
            # If screen fires -> assign min cost over {1, 2}; else min cost over {1..5}
            costs = []
            for i in range(len(p_em_val)):
                if screen_fires[i]:
                    chosen_lvl = select_min_cost_level(p_multi_val[i], allowed_levels=[1, 2], cost_matrix=cost_matrix)
                else:
                    chosen_lvl = select_min_cost_level(p_multi_val[i], allowed_levels=[1, 2, 3, 4, 5], cost_matrix=cost_matrix)
                true_y = int(y_level_val[i])
                # evaluate cost of chosen_lvl vs true_y
                if chosen_lvl < true_y:
                    c = 1.0 * (true_y - chosen_lvl)
                elif chosen_lvl > true_y:
                    c = 20.0 * (chosen_lvl - true_y) if true_y in (1, 2) else 1.0 * (chosen_lvl - true_y)
                else:
                    c = 0.0
                costs.append(c)

            total_val_cost = float(np.sum(costs))

            val_records.append({
                "alpha": alpha,
                "tau": tau,
                "infeasible": is_inf,
                "val_recall": recall,
                "val_escalation_rate": esc_rate,
                "esc_L3_rate": esc_by_level["esc_L3_rate"],
                "esc_L4_rate": esc_by_level["esc_L4_rate"],
                "esc_L5_rate": esc_by_level["esc_L5_rate"],
                "total_val_cost": total_val_cost,
                "mean_val_cost": total_val_cost / len(p_em_val),
            })

            # Check operating selection criteria
            meets_floor = recall >= self.recall_floor_on_val
            if meets_floor and total_val_cost < best_cost:
                best_cost = total_val_cost
                best_alpha = alpha
                best_tau = tau

        # Fallback if no alpha met the 95% floor: pick highest recall
        if best_cost == float("inf"):
            val_records_sorted = sorted(val_records, key=lambda r: -r["val_recall"])
            best_alpha = val_records_sorted[0]["alpha"]
            best_tau = val_records_sorted[0]["tau"]

        self.state.operating_alpha = float(best_alpha)
        self.state.operating_tau = float(best_tau if best_tau is not None else 0.5)

        tradeoff_df = pd.DataFrame(val_records)
        return best_alpha, self.state.operating_tau, tradeoff_df

    def predict_level_set(self, probs_5: np.ndarray) -> List[int]:
        """
        Return the conformal level prediction set C(x) in {1..5}.
        C(x) = { k in {1..5} : 1 - p_k <= level_cutoff }.
        """
        cutoff = self.state.level_cutoff
        included = [lvl for lvl in range(1, 6) if (1.0 - probs_5[lvl - 1]) <= cutoff]
        if not included:
            included = [int(np.argmax(probs_5) + 1)]
        return included

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ConformalPredictor":
        return joblib.load(path)
