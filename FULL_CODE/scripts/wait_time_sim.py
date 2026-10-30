"""
scripts/wait_time_sim.py
──────────────────────────────────────────────────────────────────────────────
TriageGuard Wait-Time Reduction Simulation
TalentAP Mega Hackathon 2026 — Ronin Labs

Compares two queue strategies for the OPD desk at Chandrapuri District Hospital:

  Strategy A — FCFS (current reality)
    All patients served strictly in token / arrival order.

  Strategy B — TriageGuard Priority Queue
    Patients served by triage level (L1 first, L5 last).
    Within the same level, arrival order is preserved.

Data used:
  FULL_CODE/data/phase2/arrival_log.csv  — 30,327 visits with arrival timestamps
                                    and actual consult_minutes per patient
  FULL_CODE/data/phase2/train_labels.csv — triage_level for the 21,455 labelled visits

For the 8,872 test visits (labels hidden), triage levels are drawn from a
population distribution that matches the training set proportions:
  L1 ~0.4%, L2 ~3.6%, L3 ~18.5%, L4 ~45.1%, L5 ~32.5%

Usage:
  cd FULL_CODE
  python scripts/wait_time_sim.py          # 10 replications (~10s)
  python scripts/wait_time_sim.py --quick  # 1 replication  (~1s)

Output:
  Prints a summary table to stdout
  Saves full results to FULL_CODE/results/wait_time_sim.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── SimPy import with helpful error ──────────────────────────────────────────
try:
    import simpy
except ImportError:
    print("SimPy not found. Installing…")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "simpy", "-q"])
    import simpy  # noqa: F811

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT       = _SCRIPT_DIR.parent          # FULL_CODE/
_DATA_DIR   = _ROOT / "data"
_RESULTS_DIR= _ROOT / "results"

ARRIVAL_LOG_PATH  = _DATA_DIR / "phase2" / "arrival_log.csv"
TRAIN_LABELS_PATH = _DATA_DIR / "phase2" / "train_labels.csv"

# ── Simulation constants ──────────────────────────────────────────────────────
N_DOCTORS          = 2
MEAN_SERVICE_MIN   = 8.0    # exponential mean — overridden by actual consult_minutes

# Level weights for harm-weighted wait
HARM_WEIGHTS: Dict[int, float] = {1: 20.0, 2: 10.0, 3: 4.0, 4: 2.0, 5: 1.0}

# Fallback level distribution for unlabelled visits (matches train proportions)
_LEVEL_PROBS = [0.004, 0.036, 0.185, 0.451, 0.325]   # L1..L5
_LEVELS      = [1, 2, 3, 4, 5]

# Priority mapping: lower number = served first in SimPy PriorityResource
_LEVEL_PRIORITY: Dict[int, int] = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_patient_data() -> pd.DataFrame:
    """
    Load and merge arrival log with triage labels.

    Returns a DataFrame with one row per patient containing:
      visit_id, arrival_minutes (float, minutes from day start),
      service_minutes (float), triage_level (int 1-5), opd_date
    """
    log    = pd.read_csv(ARRIVAL_LOG_PATH, parse_dates=["arrival_ts", "fcfs_consult_start_ts"])
    labels = pd.read_csv(TRAIN_LABELS_PATH)

    # Merge triage levels for the labelled subset
    merged = log.merge(
        labels[["visit_id", "triage_level"]],
        on="visit_id",
        how="left",
    )

    # arrival_minutes: minutes elapsed since midnight of that OPD day
    merged["opd_date_dt"] = pd.to_datetime(merged["opd_date"])
    midnight = merged["opd_date_dt"]
    merged["arrival_minutes"] = (
        merged["arrival_ts"] - midnight.apply(lambda d: d.replace(hour=0, minute=0, second=0))
    ).dt.total_seconds() / 60.0

    # Use actual consult_minutes from the log as the service time
    merged["service_minutes"] = merged["consult_minutes"].clip(lower=0.1)

    # Sort by day then arrival order
    merged = merged.sort_values(["opd_date", "token_no"]).reset_index(drop=True)

    return merged


def assign_missing_levels(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Assign triage levels to unlabelled rows using the empirical distribution
    from the labelled subset of this DataFrame.  Falls back to _LEVEL_PROBS
    (normalised) if no labelled rows exist.

    Labelled rows keep their known level.
    """
    df = df.copy()
    missing_mask = df["triage_level"].isna()
    n_missing    = missing_mask.sum()
    if n_missing > 0:
        labelled = df.loc[~missing_mask, "triage_level"]
        if len(labelled) > 0:
            # Compute exact empirical distribution — guaranteed to sum to 1.0
            counts = np.array([labelled.eq(l).sum() for l in _LEVELS], dtype=float)
            probs  = counts / counts.sum()
        else:
            # Fallback: normalise the hardcoded probs so floating-point is exact
            probs  = np.array(_LEVEL_PROBS, dtype=float)
            probs  = probs / probs.sum()

        df.loc[missing_mask, "triage_level"] = rng.choice(
            _LEVELS, size=n_missing, p=probs
        )
    df["triage_level"] = df["triage_level"].astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SimPy simulation
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_day(
    patients: pd.DataFrame,
    n_doctors: int,
    strategy: str,           # "fcfs" or "priority"
    rng: np.random.Generator,
) -> List[Dict]:
    """
    Run a single OPD day simulation.

    patients: DataFrame for one day, sorted by arrival order.
    Returns list of dicts with wait_time and triage_level per patient.
    """
    env     = simpy.Environment()
    results: List[Dict] = []

    if strategy == "fcfs":
        resource = simpy.Resource(env, capacity=n_doctors)
    else:
        resource = simpy.PriorityResource(env, capacity=n_doctors)

    def patient_process(
        pid: int,
        arrival: float,
        service: float,
        level: int,
    ):
        yield env.timeout(arrival)          # wait until patient arrives
        arrival_actual = env.now

        if strategy == "fcfs":
            req = resource.request()
        else:
            priority = _LEVEL_PRIORITY[level]
            req = resource.request(priority=priority)

        with req:
            yield req
            wait_time = env.now - arrival_actual
            yield env.timeout(service)

        results.append({
            "patient_idx":  pid,
            "triage_level": level,
            "arrival_min":  arrival_actual,
            "wait_min":     wait_time,
            "service_min":  service,
        })

    for idx, row in patients.iterrows():
        env.process(
            patient_process(
                pid     = idx,
                arrival = row["arrival_minutes"],
                service = row["service_minutes"],
                level   = int(row["triage_level"]),
            )
        )

    env.run()
    return results


def run_simulation(
    df: pd.DataFrame,
    strategy: str,
    n_doctors: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Run the simulation day-by-day and return a flat DataFrame of all patient results.
    """
    all_results = []
    for date, day_df in df.groupby("opd_date"):
        # Normalise arrival_minutes so first patient on each day starts at offset
        min_arr  = day_df["arrival_minutes"].min()
        day_copy = day_df.copy()
        day_copy["arrival_minutes"] = day_copy["arrival_minutes"] - min_arr
        day_results = _run_one_day(day_copy, n_doctors, strategy, rng)
        all_results.extend(day_results)

    return pd.DataFrame(all_results)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results_df: pd.DataFrame) -> Dict:
    """Compute all metrics from a simulation results DataFrame."""
    metrics: Dict = {}

    for level in _LEVELS:
        sub = results_df[results_df["triage_level"] == level]["wait_min"]
        if len(sub) > 0:
            metrics[f"avg_wait_L{level}"]  = float(sub.mean())
            metrics[f"p95_wait_L{level}"]  = float(np.percentile(sub, 95))
            metrics[f"n_patients_L{level}"]= int(len(sub))
        else:
            metrics[f"avg_wait_L{level}"]  = float("nan")
            metrics[f"p95_wait_L{level}"]  = float("nan")
            metrics[f"n_patients_L{level}"]= 0

    # Emergency = L1 + L2
    em = results_df[results_df["triage_level"].isin([1, 2])]["wait_min"]
    metrics["avg_wait_emergency"]  = float(em.mean()) if len(em) > 0 else float("nan")
    metrics["p95_wait_emergency"]  = float(np.percentile(em, 95)) if len(em) > 0 else float("nan")
    metrics["n_patients_emergency"]= int(len(em))

    # Overall
    metrics["avg_wait_overall"] = float(results_df["wait_min"].mean())
    metrics["p95_wait_overall"] = float(np.percentile(results_df["wait_min"], 95))
    metrics["n_patients_total"] = int(len(results_df))

    # Harm-weighted wait
    weighted_sum  = 0.0
    weight_total  = 0.0
    for level, w in HARM_WEIGHTS.items():
        sub = results_df[results_df["triage_level"] == level]["wait_min"]
        if len(sub) > 0:
            weighted_sum  += w * float(sub.mean()) * len(sub)
            weight_total  += w * len(sub)
    metrics["harm_weighted_wait"] = (
        weighted_sum / weight_total if weight_total > 0 else float("nan")
    )

    return metrics


def aggregate_replications(rep_metrics: List[Dict]) -> Dict:
    """Average metrics across replications, computing mean ± std."""
    keys = rep_metrics[0].keys()
    agg  = {}
    for k in keys:
        vals = [r[k] for r in rep_metrics if not (isinstance(r[k], float) and np.isnan(r[k]))]
        if vals:
            agg[k]             = float(np.mean(vals))
            agg[f"{k}_std"]    = float(np.std(vals))
        else:
            agg[k]             = float("nan")
            agg[f"{k}_std"]    = float("nan")
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(val: float, decimals: int = 1) -> str:
    if np.isnan(val):
        return "  N/A  "
    return f"{val:6.{decimals}f}"


def print_summary(fcfs: Dict, tg: Dict, n_reps: int) -> None:
    em_fcfs   = fcfs["avg_wait_emergency"]
    em_tg     = tg["avg_wait_emergency"]
    reduction = em_fcfs - em_tg
    pct       = (reduction / em_fcfs * 100) if em_fcfs > 0 else 0.0

    p95_em_fcfs = fcfs["p95_wait_emergency"]
    p95_em_tg   = tg["p95_wait_emergency"]

    W = 66
    sep  = "═" * W
    line = "║"
    mid  = "╠" + sep + "╣"

    def row(label: str, a: str, b: str) -> str:
        content = f"  {label:<30s}│  {a:<10s}│  {b:<16s}"
        return f"║{content}║"

    print()
    print(f"╔{sep}╗")
    print(f"║{'TRIAGEGUARD WAIT-TIME IMPACT ANALYSIS':^{W}}║")
    print(f"║{'(' + str(n_reps) + ' simulation replications  |  ' + str(N_DOCTORS) + ' doctors  |  real OPD arrival data)':^{W}}║")
    print(f"╠{sep}╣")
    print(f"║{'  Metric':<32s}│  {'FCFS':<10s}│  {'TriageGuard':<16s}║")
    print(f"╠{sep}╣")
    print(row("Emergency avg wait  (L1+L2)", f"{_fmt(em_fcfs)} min", f"{_fmt(em_tg)} min"))
    print(row("Emergency 95th pct  (L1+L2)", f"{_fmt(p95_em_fcfs)} min", f"{_fmt(p95_em_tg)} min"))
    print(row("Level 1 avg wait", f"{_fmt(fcfs['avg_wait_L1'])} min", f"{_fmt(tg['avg_wait_L1'])} min"))
    print(row("Level 2 avg wait", f"{_fmt(fcfs['avg_wait_L2'])} min", f"{_fmt(tg['avg_wait_L2'])} min"))
    print(row("Level 3 avg wait", f"{_fmt(fcfs['avg_wait_L3'])} min", f"{_fmt(tg['avg_wait_L3'])} min"))
    print(row("Level 4 avg wait", f"{_fmt(fcfs['avg_wait_L4'])} min", f"{_fmt(tg['avg_wait_L4'])} min"))
    print(row("Level 5 avg wait", f"{_fmt(fcfs['avg_wait_L5'])} min", f"{_fmt(tg['avg_wait_L5'])} min"))
    print(row("Overall avg wait", f"{_fmt(fcfs['avg_wait_overall'])} min", f"{_fmt(tg['avg_wait_overall'])} min"))
    print(row("Harm-weighted wait", f"{_fmt(fcfs['harm_weighted_wait'])}", f"{_fmt(tg['harm_weighted_wait'])}"))
    print(f"╠{sep}╣")
    print(f"║  🚨 HEADLINE: Emergency patients wait {_fmt(reduction).strip()} min less with TriageGuard{'':8s}║")
    print(f"║     That is a {pct:.1f}% reduction in emergency wait time{'':22s}║")
    hw_reduction = fcfs['harm_weighted_wait'] - tg['harm_weighted_wait']
    hw_pct       = (hw_reduction / fcfs['harm_weighted_wait'] * 100) if fcfs['harm_weighted_wait'] > 0 else 0.0
    print(f"║     Harm-weighted wait reduced by {_fmt(hw_reduction).strip()} ({hw_pct:.1f}% improvement){'':12s}║")
    print(f"╚{sep}╝")
    print()

    # Per-level detail
    print("Per-level detail (mean ± std across replications):")
    print(f"  {'Level':<8} {'Strategy':<14} {'Avg wait':>10}  {'±':>4}  {'P95 wait':>10}  {'N patients':>10}")
    print("  " + "-" * 65)
    for lvl in _LEVELS:
        for label, d in [("FCFS", fcfs), ("TriageGuard", tg)]:
            avg = d.get(f"avg_wait_L{lvl}", float("nan"))
            std = d.get(f"avg_wait_L{lvl}_std", float("nan"))
            p95 = d.get(f"p95_wait_L{lvl}", float("nan"))
            n   = d.get(f"n_patients_L{lvl}", 0)
            avg_s = f"{avg:7.2f} min" if not np.isnan(avg) else "    N/A   "
            std_s = f"{std:5.2f}"     if not np.isnan(std) else "  N/A"
            p95_s = f"{p95:7.2f} min" if not np.isnan(p95) else "    N/A   "
            print(f"  L{lvl}      {label:<14} {avg_s}  {std_s}  {p95_s}  {n:>10,}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TriageGuard wait-time reduction simulation"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run 1 replication only (fast testing)",
    )
    args = parser.parse_args()

    n_reps   = 1 if args.quick else 10
    base_seed = 42

    print(f"\nLoading data from {_DATA_DIR} …")
    base_df = load_patient_data()
    print(f"  Loaded {len(base_df):,} patient records across {base_df['opd_date'].nunique()} OPD days")
    print(f"  Labelled visits:   {base_df['triage_level'].notna().sum():,}")
    print(f"  Unlabelled visits: {base_df['triage_level'].isna().sum():,}")
    print(f"\nRunning {n_reps} replication(s) …")

    fcfs_reps = []
    tg_reps   = []

    for rep in range(n_reps):
        seed = base_seed + rep
        rng  = np.random.default_rng(seed)

        # Assign levels for unlabelled visits (different draw each replication)
        df_rep = assign_missing_levels(base_df, rng)

        print(f"  Rep {rep + 1:2d}/{n_reps}  (seed={seed})  "
              f"  emergencies={df_rep['triage_level'].isin([1,2]).sum():,}", end="  ", flush=True)

        # FCFS
        fcfs_results = run_simulation(df_rep, "fcfs",     N_DOCTORS, rng)
        fcfs_m       = compute_metrics(fcfs_results)
        fcfs_reps.append(fcfs_m)

        # Priority queue
        tg_results   = run_simulation(df_rep, "priority", N_DOCTORS, rng)
        tg_m         = compute_metrics(tg_results)
        tg_reps.append(tg_m)

        em_reduction = fcfs_m["avg_wait_emergency"] - tg_m["avg_wait_emergency"]
        print(f"✓  em-reduction={em_reduction:.1f} min")

    # Aggregate
    fcfs_agg = aggregate_replications(fcfs_reps)
    tg_agg   = aggregate_replications(tg_reps)

    # Print summary table
    print_summary(fcfs_agg, tg_agg, n_reps)

    # Save full results
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RESULTS_DIR / "wait_time_sim.json"

    output = {
        "meta": {
            "n_replications":   n_reps,
            "base_seed":        base_seed,
            "n_doctors":        N_DOCTORS,
            "total_patients":   int(len(base_df)),
            "n_labelled":       int(base_df["triage_level"].notna().sum()),
            "n_unlabelled":     int(base_df["triage_level"].isna().sum()),
            "n_opd_days":       int(base_df["opd_date"].nunique()),
            "harm_weights":     HARM_WEIGHTS,
            "level_probs_used_for_unlabelled": dict(zip(_LEVELS, _LEVEL_PROBS)),
        },
        "fcfs":        fcfs_agg,
        "triageguard": tg_agg,
        "headline": {
            "emergency_wait_reduction_min": round(
                fcfs_agg["avg_wait_emergency"] - tg_agg["avg_wait_emergency"], 2
            ),
            "emergency_wait_reduction_pct": round(
                (fcfs_agg["avg_wait_emergency"] - tg_agg["avg_wait_emergency"])
                / fcfs_agg["avg_wait_emergency"] * 100, 1
            ) if fcfs_agg["avg_wait_emergency"] > 0 else 0.0,
            "harm_weighted_reduction_min": round(
                fcfs_agg["harm_weighted_wait"] - tg_agg["harm_weighted_wait"], 2
            ),
            "harm_weighted_reduction_pct": round(
                (fcfs_agg["harm_weighted_wait"] - tg_agg["harm_weighted_wait"])
                / fcfs_agg["harm_weighted_wait"] * 100, 1
            ) if fcfs_agg["harm_weighted_wait"] > 0 else 0.0,
        },
        "per_replication": {
            "fcfs":        fcfs_reps,
            "triageguard": tg_reps,
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Full results saved to {out_path}")


if __name__ == "__main__":
    main()
