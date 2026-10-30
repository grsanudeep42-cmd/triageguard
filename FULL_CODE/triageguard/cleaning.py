"""
triageguard/cleaning.py
─────────────────────────────────────────────────────────────────────────────
Phase 1 – Input cleaning layer for TriageGuard.

ISOLATION RULE
--------------
This module must NOT import triageguard.data_gen or read hidden_labels.csv.

INPUT
-----
A single patient record (dict) or a pandas DataFrame.
Expected vital keys: bp_sys, bp_dia, pulse, temp_c, spo2, complaint_text.

OUTPUT
------
• clean_record()    → CleanResult  (dataclass)
• clean_dataframe() → DataFrame with _c / _flag / summary columns added

Cleaning pipeline (per vital)
──────────────────────────────
1. NaN / None input                 → FLAG_MISSING,   value = None
2. Fahrenheit detection  (temp_c only)
   value ∈ fahrenheit_range         → convert to °C,  FLAG_TEMP_CONVERTED  (uncertain)
3. Decimal slip
   value < lo  AND  lo ≤ value×10 ≤ hi
                                    → FLAG_DECIMAL_SLIP, value = None  (no auto-correct)
4. Out-of-range
   value < lo  OR  value > hi       → FLAG_OUT_OF_RANGE, value = None
5. OK                               → FLAG_OK

Swap check (after per-vital steps)
────────────────────────────────────
If both bp_sys, bp_dia are present AND bp_sys < bp_dia:
  • Swap if swapped values are both plausible → FLAG_SWAPPED_CORRECTED  (uncertain)
  • Otherwise                                → FLAG_OUT_OF_RANGE, both → None

Summary
───────
n_missing_vitals  – originally NaN/None fields
n_invalid_vitals  – failed plausibility (set to None by this module)
data_quality_score – (5 − n_missing − n_invalid) / 5   # PLACEHOLDER
needs_remeasure   – n_missing + n_invalid ≥ 2           # PLACEHOLDER

Text normalisation
──────────────────
Lowercase → strip/collapse whitespace → collapse >N consecutive identical chars.
All thresholds and ranges are  # PLACEHOLDER.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_ROOT     = Path(__file__).resolve().parent.parent
_CFG_PATH = _ROOT / "config.yaml"

# ─────────────────────────────────────────────────────────────────────────────
# Flag constants
# ─────────────────────────────────────────────────────────────────────────────

FLAG_OK           = "ok"
FLAG_MISSING      = "missing"
FLAG_OUT_OF_RANGE = "out_of_range"
FLAG_TEMP_CONV    = "unit_converted"      # Fahrenheit → Celsius (uncertain)
FLAG_SWAPPED      = "swapped_corrected"   # sys/dia transposed then swapped (uncertain)
FLAG_DECIMAL_SLIP = "decimal_slip"        # possible ×10 entry error (not corrected)

# ─────────────────────────────────────────────────────────────────────────────
# CleanResult dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CleanResult:
    """
    Output of clean_record().  All PLACEHOLDER values are noted inline.

    Vitals set to None are definitively unknown — downstream layers must
    never infer acuity from a None vital.
    """

    # ── Cleaned vitals (None = invalid or originally missing) ─────────────
    bp_sys: Optional[float] = None
    bp_dia: Optional[float] = None
    pulse:  Optional[float] = None
    temp_c: Optional[float] = None
    spo2:   Optional[float] = None

    # ── Per-vital flag ────────────────────────────────────────────────────
    bp_sys_flag: str = FLAG_MISSING
    bp_dia_flag: str = FLAG_MISSING
    pulse_flag:  str = FLAG_MISSING
    temp_c_flag: str = FLAG_MISSING
    spo2_flag:   str = FLAG_MISSING

    # ── Uncertainty markers (value is present but may be wrong) ───────────
    bp_sys_uncertain: bool = False   # swapped sys/dia
    bp_dia_uncertain: bool = False   # swapped sys/dia
    temp_c_uncertain: bool = False   # Fahrenheit conversion

    # ── Cleaned text ──────────────────────────────────────────────────────
    complaint_text: str = ""

    # ── Summary ───────────────────────────────────────────────────────────
    n_missing_vitals:   int   = 0      # originally NaN/None
    n_invalid_vitals:   int   = 0      # present but failed plausibility
    data_quality_score: float = 1.0    # (5 − n_missing − n_invalid) / 5  # PLACEHOLDER
    needs_remeasure:    bool  = False  # n_missing + n_invalid ≥ 2         # PLACEHOLDER

    # ── Human-readable warnings ───────────────────────────────────────────
    warnings: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Config loader (lazy, module-level cache)
# ─────────────────────────────────────────────────────────────────────────────

_cfg_cache: Optional[Dict] = None


def _load_cfg() -> Dict:
    global _cfg_cache
    if _cfg_cache is None:
        with open(_CFG_PATH, encoding="utf-8") as fh:
            _cfg_cache = yaml.safe_load(fh)
    return _cfg_cache


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str, max_repeated: int = 2) -> str:
    """
    Normalise free-text complaint:
    1. Lowercase
    2. Collapse all whitespace runs to a single space, strip edges
    3. Collapse runs of identical characters longer than max_repeated
       (e.g. "chesttt" → "chestt" with max_repeated=2)

    max_repeated is taken from config.yaml text_cleaning.max_repeated_chars
    # PLACEHOLDER.
    """
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"\s+", " ", t).strip()
    if max_repeated >= 1:
        pattern = r"(.)\1{" + str(max_repeated) + r",}"
        repl    = r"\1" * max_repeated
        t = re.sub(pattern, repl, t)
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Per-vital plausibility helper
# ─────────────────────────────────────────────────────────────────────────────

def _is_in_range(value: float, lo: float, hi: float) -> bool:
    return lo <= value <= hi


def _is_decimal_slip(value: float, lo: float, hi: float, factor: float) -> bool:
    """True if value < lo AND value*factor is in [lo, hi]."""
    return value < lo and _is_in_range(value * factor, lo, hi)


def _is_fahrenheit(value: float, f_lo: float, f_hi: float) -> bool:
    """True if the value in the Celsius field looks like Fahrenheit."""
    return f_lo <= value <= f_hi


def _f_to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


# ─────────────────────────────────────────────────────────────────────────────
# Single-vital cleaning
# ─────────────────────────────────────────────────────────────────────────────

def _clean_vital(
    name: str,
    raw: object,
    lo: float,
    hi: float,
    slip_factor: float,
    fahrenheit_range: Optional[Tuple[float, float]] = None,
) -> Tuple[Optional[float], str, bool, Optional[str]]:
    """
    Clean one vital reading.

    Returns (value, flag, uncertain, warning_str_or_None).

    'uncertain' is only True when the value has been edited
    (Fahrenheit conversion).  Callers handle swap uncertainty separately.
    """
    # Step 1 – missing
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return (None, FLAG_MISSING, False,
                f"{name.upper()} not recorded — re-measure.")

    value = float(raw)

    # Step 2 – Fahrenheit detection (temp_c only)
    if fahrenheit_range is not None and _is_fahrenheit(value, *fahrenheit_range):
        converted = round(_f_to_c(value), 1)
        if _is_in_range(converted, lo, hi):
            warn = (
                f"temp_c = {value} looks like Fahrenheit "
                f"(→ {converted} °C converted, uncertain): verify unit."
            )
            return (converted, FLAG_TEMP_CONV, True, warn)
        else:
            warn = (
                f"temp_c = {value} looks like Fahrenheit but "
                f"converted value {converted} °C is still out of range "
                f"[{lo}, {hi}]: re-measure."
            )
            return (None, FLAG_OUT_OF_RANGE, False, warn)

    # Step 3 – decimal slip
    if _is_decimal_slip(value, lo, hi, slip_factor):
        corrected_guess = round(value * slip_factor, 1)
        warn = (
            f"{name} = {value} looks like a decimal-slip entry "
            f"(×{int(slip_factor)} = {corrected_guess} would be plausible): "
            f"re-measure."
        )
        return (None, FLAG_DECIMAL_SLIP, False, warn)

    # Step 4 – out of range
    if not _is_in_range(value, lo, hi):
        warn = (
            f"{name} = {value} is outside plausible range "
            f"[{lo}, {hi}]: re-measure."
        )
        return (None, FLAG_OUT_OF_RANGE, False, warn)

    # Step 5 – ok
    return (value, FLAG_OK, False, None)


# ─────────────────────────────────────────────────────────────────────────────
# Swap check
# ─────────────────────────────────────────────────────────────────────────────

def _check_bp_swap(
    bp_sys: Optional[float],
    bp_dia: Optional[float],
    bp_sys_flag: str,
    bp_dia_flag: str,
    lo_sys: float,
    hi_sys: float,
    lo_dia: float,
    hi_dia: float,
) -> Tuple[Optional[float], Optional[float], str, str, bool, bool, Optional[str]]:
    """
    Detect and optionally correct a transposed systolic / diastolic reading.

    Returns:
        bp_sys_val, bp_dia_val,
        bp_sys_flag, bp_dia_flag,
        bp_sys_uncertain, bp_dia_uncertain,
        warning_or_None
    """
    # Both must be present and OK to attempt swap detection
    if bp_sys is None or bp_dia is None:
        return (bp_sys, bp_dia, bp_sys_flag, bp_dia_flag, False, False, None)

    if bp_sys >= bp_dia:
        return (bp_sys, bp_dia, bp_sys_flag, bp_dia_flag, False, False, None)

    # sys < dia — likely transposed
    swapped_sys = bp_dia   # the higher reading
    swapped_dia = bp_sys   # the lower reading

    if _is_in_range(swapped_sys, lo_sys, hi_sys) and _is_in_range(swapped_dia, lo_dia, hi_dia):
        warn = (
            f"BP systolic ({bp_sys}) < diastolic ({bp_dia}): "
            f"values appear transposed. Swapped to sys={swapped_sys}, "
            f"dia={swapped_dia} — UNCERTAIN, please verify."
        )
        return (
            swapped_sys, swapped_dia,
            FLAG_SWAPPED, FLAG_SWAPPED,
            True, True,
            warn,
        )
    else:
        warn = (
            f"BP systolic ({bp_sys}) < diastolic ({bp_dia}): "
            f"values appear transposed but swapped values are also implausible. "
            f"Both set to missing — re-measure."
        )
        return (
            None, None,
            FLAG_OUT_OF_RANGE, FLAG_OUT_OF_RANGE,
            False, False,
            warn,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main: clean_record
# ─────────────────────────────────────────────────────────────────────────────

def clean_record(
    record: Dict,
    cfg: Optional[Dict] = None,
) -> CleanResult:
    """
    Clean and validate a single patient record.

    Parameters
    ----------
    record : dict
        Keys: bp_sys, bp_dia, pulse, temp_c, spo2, complaint_text.
        Extra keys (age, sex, patient_id, etc.) are ignored.
    cfg : dict, optional
        Parsed config.yaml.  Loaded automatically if None.

    Returns
    -------
    CleanResult
        Cleaned vitals, per-vital flags, summary counts, and human-readable
        warnings.  Missing or invalid vitals are NEVER imputed.
    """
    if cfg is None:
        cfg = _load_cfg()

    vr          = cfg["vital_ranges"]
    f_range_cfg = cfg["fahrenheit_range"]
    slip_factor = float(cfg["decimal_slip_detection"]["factor"])
    max_rep     = int(cfg["text_cleaning"]["max_repeated_chars"])
    f_range     = (float(f_range_cfg["lo"]), float(f_range_cfg["hi"]))

    warnings: List[str]       = []
    n_missing_vitals: int      = 0
    n_invalid_vitals: int      = 0

    # ── bp_sys ────────────────────────────────────────────────────────────
    bp_sys, bp_sys_flag, _, bp_sys_warn = _clean_vital(
        "bp_sys", record.get("bp_sys"),
        lo=float(vr["bp_sys"]["lo"]), hi=float(vr["bp_sys"]["hi"]),
        slip_factor=slip_factor,
    )
    if bp_sys_warn:
        warnings.append(bp_sys_warn)
    if bp_sys_flag == FLAG_MISSING:
        n_missing_vitals += 1
    elif bp_sys_flag != FLAG_OK:
        n_invalid_vitals += 1

    # ── bp_dia ────────────────────────────────────────────────────────────
    bp_dia, bp_dia_flag, _, bp_dia_warn = _clean_vital(
        "bp_dia", record.get("bp_dia"),
        lo=float(vr["bp_dia"]["lo"]), hi=float(vr["bp_dia"]["hi"]),
        slip_factor=slip_factor,
    )
    if bp_dia_warn:
        warnings.append(bp_dia_warn)
    if bp_dia_flag == FLAG_MISSING:
        n_missing_vitals += 1
    elif bp_dia_flag != FLAG_OK:
        n_invalid_vitals += 1

    # ── Swap check ────────────────────────────────────────────────────────
    bp_sys_uncertain = bp_dia_uncertain = False
    (
        bp_sys, bp_dia,
        bp_sys_flag, bp_dia_flag,
        bp_sys_uncertain, bp_dia_uncertain,
        swap_warn,
    ) = _check_bp_swap(
        bp_sys, bp_dia, bp_sys_flag, bp_dia_flag,
        lo_sys=float(vr["bp_sys"]["lo"]), hi_sys=float(vr["bp_sys"]["hi"]),
        lo_dia=float(vr["bp_dia"]["lo"]), hi_dia=float(vr["bp_dia"]["hi"]),
    )
    if swap_warn:
        warnings.append(swap_warn)
        # Adjust counts if swap resulted in out_of_range
        if bp_sys_flag == FLAG_OUT_OF_RANGE and bp_dia_flag == FLAG_OUT_OF_RANGE:
            # Both became invalid — remove original count (already added)
            # and add 2 invalid (swap set both to None)
            # We already counted each above; now they became invalid not OK
            # Re-count: both flags changed from OK → OUT_OF_RANGE
            # original both passed → 0 invalid; now both failed → +2 invalid, -0 missing
            n_invalid_vitals += 2
            # (The initial _clean_vital calls set FLAG_OK for both → neither was counted
            #  as invalid above.  So adding 2 here is correct.)

    # ── pulse ─────────────────────────────────────────────────────────────
    pulse, pulse_flag, _, pulse_warn = _clean_vital(
        "pulse", record.get("pulse"),
        lo=float(vr["pulse"]["lo"]), hi=float(vr["pulse"]["hi"]),
        slip_factor=slip_factor,
    )
    if pulse_warn:
        warnings.append(pulse_warn)
    if pulse_flag == FLAG_MISSING:
        n_missing_vitals += 1
    elif pulse_flag != FLAG_OK:
        n_invalid_vitals += 1

    # ── temp_c ────────────────────────────────────────────────────────────
    temp_c, temp_c_flag, temp_c_uncertain, temp_warn = _clean_vital(
        "temp_c", record.get("temp_c"),
        lo=float(vr["temp_c"]["lo"]), hi=float(vr["temp_c"]["hi"]),
        slip_factor=slip_factor,
        fahrenheit_range=f_range,
    )
    if temp_warn:
        warnings.append(temp_warn)
    if temp_c_flag == FLAG_MISSING:
        n_missing_vitals += 1
    elif temp_c_flag not in (FLAG_OK, FLAG_TEMP_CONV):
        # unit_converted is still valid (uncertain but usable)
        n_invalid_vitals += 1

    # ── spo2 ─────────────────────────────────────────────────────────────
    spo2, spo2_flag, _, spo2_warn = _clean_vital(
        "spo2", record.get("spo2"),
        lo=float(vr["spo2"]["lo"]), hi=float(vr["spo2"]["hi"]),
        slip_factor=slip_factor,
    )
    if spo2_warn:
        warnings.append(spo2_warn)
    if spo2_flag == FLAG_MISSING:
        n_missing_vitals += 1
    elif spo2_flag != FLAG_OK:
        n_invalid_vitals += 1

    # ── Summary ───────────────────────────────────────────────────────────
    n_total_issues  = n_missing_vitals + n_invalid_vitals
    score           = max(0.0, (5 - n_total_issues) / 5.0)   # PLACEHOLDER
    needs_remeasure = n_total_issues >= 2                      # PLACEHOLDER

    # ── Text ─────────────────────────────────────────────────────────────
    complaint = clean_text(
        str(record.get("complaint_text", "") or ""),
        max_repeated=max_rep,
    )

    return CleanResult(
        bp_sys=bp_sys,  bp_dia=bp_dia,
        pulse=pulse,    temp_c=temp_c,  spo2=spo2,
        bp_sys_flag=bp_sys_flag,  bp_dia_flag=bp_dia_flag,
        pulse_flag=pulse_flag,    temp_c_flag=temp_c_flag,
        spo2_flag=spo2_flag,
        bp_sys_uncertain=bp_sys_uncertain,
        bp_dia_uncertain=bp_dia_uncertain,
        temp_c_uncertain=temp_c_uncertain,
        complaint_text=complaint,
        n_missing_vitals=n_missing_vitals,
        n_invalid_vitals=n_invalid_vitals,
        data_quality_score=round(score, 2),
        needs_remeasure=needs_remeasure,
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batch: clean_dataframe
# ─────────────────────────────────────────────────────────────────────────────

def clean_dataframe(
    df: pd.DataFrame,
    cfg: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Apply clean_record() to every row of *df*.

    Returns a copy of *df* with new columns appended:
      bp_sys_c, bp_dia_c, pulse_c, temp_c_c, spo2_c   (cleaned vitals)
      bp_sys_flag, bp_dia_flag, pulse_flag,
      temp_c_flag, spo2_flag
      complaint_clean
      n_missing_vitals, n_invalid_vitals,
      data_quality_score, needs_remeasure
      cleaning_warnings  (pipe-separated string)

    Original raw columns are preserved unchanged.
    """
    if cfg is None:
        cfg = _load_cfg()

    results = [
        clean_record(row.to_dict(), cfg)
        for _, row in df.iterrows()
    ]

    out = df.copy()
    out["bp_sys_c"]          = [r.bp_sys   for r in results]
    out["bp_dia_c"]          = [r.bp_dia   for r in results]
    out["pulse_c"]           = [r.pulse    for r in results]
    out["temp_c_c"]          = [r.temp_c   for r in results]
    out["spo2_c"]            = [r.spo2     for r in results]
    out["bp_sys_flag"]       = [r.bp_sys_flag  for r in results]
    out["bp_dia_flag"]       = [r.bp_dia_flag  for r in results]
    out["pulse_flag"]        = [r.pulse_flag   for r in results]
    out["temp_c_flag"]       = [r.temp_c_flag  for r in results]
    out["spo2_flag"]         = [r.spo2_flag    for r in results]
    out["complaint_clean"]   = [r.complaint_text for r in results]
    out["n_missing_vitals"]  = [r.n_missing_vitals  for r in results]
    out["n_invalid_vitals"]  = [r.n_invalid_vitals  for r in results]
    out["data_quality_score"]= [r.data_quality_score for r in results]
    out["needs_remeasure"]   = [r.needs_remeasure   for r in results]
    out["cleaning_warnings"] = [" | ".join(r.warnings) for r in results]

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Unified entrypoint: clean
# ─────────────────────────────────────────────────────────────────────────────

def clean(
    data: Union[Dict, pd.DataFrame],
    cfg: Optional[Dict] = None,
) -> Union[CleanResult, pd.DataFrame]:
    """
    Clean one record (dict) or a batch of records (DataFrame).

    Parameters
    ----------
    data : dict or pd.DataFrame
        Input patient record or batch.
    cfg : dict, optional
        Parsed config.yaml. Loaded automatically if None.

    Returns
    -------
    CleanResult if data is a dict; pd.DataFrame if data is a DataFrame.
    """
    if isinstance(data, pd.DataFrame):
        return clean_dataframe(data, cfg=cfg)
    elif isinstance(data, dict):
        return clean_record(data, cfg=cfg)
    else:
        raise TypeError(f"Expected dict or pd.DataFrame, got {type(data).__name__}")
