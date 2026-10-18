"""
tests/test_cleaning.py
─────────────────────────────────────────────────────────────────────────────
Acceptance tests for triageguard.cleaning  (Phase 1).

Coverage
─────────
• BP = 0 → out_of_range, set to None, warning present
• Temp = 96 (Fahrenheit) → unit_converted, ~35.6 °C, uncertain flag
• SpO2 = 104 → out_of_range, set to None, warning present
• SpO2 = 50.0 → boundary: just valid
• SpO2 = 49.9 → just out of range
• Missing vitals (NaN / None) → n_missing_vitals, FLAG_MISSING
• Swapped sys/dia (plausible) → swapped_corrected, uncertain, warning
• Swapped sys/dia (implausible) → both None, out_of_range
• Decimal slip → FLAG_DECIMAL_SLIP, set to None, no auto-correct
• Two+ issues → needs_remeasure = True
• All clean → data_quality_score = 1.0, needs_remeasure = False
• Text cleaning (lowercase, whitespace, repeated chars)
• Missing or invalid vitals never lower acuity via flag
"""

from __future__ import annotations

import math
import pytest

from triageguard.cleaning import (
    CleanResult,
    clean_record,
    clean_dataframe,
    clean,
    clean_text,
    FLAG_OK,
    FLAG_MISSING,
    FLAG_OUT_OF_RANGE,
    FLAG_TEMP_CONV,
    FLAG_SWAPPED,
    FLAG_DECIMAL_SLIP,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared minimal config (avoids file-system dependency in unit tests)
# ─────────────────────────────────────────────────────────────────────────────

_CFG = {
    "vital_ranges": {
        "bp_sys":  {"lo": 50,   "hi": 240},
        "bp_dia":  {"lo": 20,   "hi": 150},
        "pulse":   {"lo": 25,   "hi": 250},
        "temp_c":  {"lo": 34.0, "hi": 42.5},
        "spo2":    {"lo": 50.0, "hi": 100.0},
    },
    "fahrenheit_range":      {"lo": 88.0,  "hi": 110.0},
    "decimal_slip_detection":{"factor": 10},
    "text_cleaning":         {"max_repeated_chars": 2},
}

def _record(**kwargs):
    base = {
        "bp_sys": 120, "bp_dia": 80, "pulse": 72,
        "temp_c": 37.0, "spo2": 98.0,
        "complaint_text": "chest pain",
    }
    base.update(kwargs)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

class TestTextCleaning:

    def test_lowercase(self):
        assert clean_text("CHEST PAIN") == "chest pain"

    def test_whitespace_normalised(self):
        assert clean_text("  chest   pain  ") == "chest pain"

    def test_repeated_chars_collapsed(self):
        # Default max_repeated=2: "chesttt" → "chestt"
        assert clean_text("chesttt", max_repeated=2) == "chestt"

    def test_repeated_chars_two_allowed(self):
        assert clean_text("aaaa", max_repeated=2) == "aa"

    def test_repeated_chars_two_unchanged(self):
        assert clean_text("aa", max_repeated=2) == "aa"

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_non_string_returns_empty(self):
        assert clean_text(None) == ""   # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# BP = 0
# ─────────────────────────────────────────────────────────────────────────────

class TestBPZero:

    def test_bp_sys_zero_is_flagged(self):
        r = clean_record(_record(bp_sys=0), _CFG)
        assert r.bp_sys is None
        assert r.bp_sys_flag == FLAG_OUT_OF_RANGE

    def test_bp_sys_zero_warning_present(self):
        r = clean_record(_record(bp_sys=0), _CFG)
        assert any("bp_sys" in w for w in r.warnings)

    def test_bp_dia_zero_is_flagged(self):
        r = clean_record(_record(bp_dia=0), _CFG)
        assert r.bp_dia is None
        assert r.bp_dia_flag == FLAG_OUT_OF_RANGE

    def test_n_invalid_incremented_for_bp_zero(self):
        r = clean_record(_record(bp_sys=0), _CFG)
        assert r.n_invalid_vitals >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Fahrenheit detection
# ─────────────────────────────────────────────────────────────────────────────

class TestFahrenheitConversion:

    def test_96f_converted_to_celsius(self):
        r = clean_record(_record(temp_c=96.0), _CFG)
        assert r.temp_c is not None
        assert abs(r.temp_c - 35.6) < 0.2
        assert r.temp_c_flag == FLAG_TEMP_CONV

    def test_unit_converted_flag_set(self):
        r = clean_record(_record(temp_c=96.0), _CFG)
        assert r.temp_c_flag == FLAG_TEMP_CONV

    def test_uncertain_flag_set_after_conversion(self):
        r = clean_record(_record(temp_c=96.0), _CFG)
        assert r.temp_c_uncertain is True

    def test_converted_temp_counted_as_valid_not_invalid(self):
        # unit_converted is still usable — should NOT increment n_invalid
        r = clean_record(_record(temp_c=96.0), _CFG)
        assert r.n_invalid_vitals == 0

    def test_warning_mentions_unit(self):
        r = clean_record(_record(temp_c=96.0), _CFG)
        assert any("fahrenheit" in w.lower() or "unit" in w.lower()
                   for w in r.warnings)

    def test_normal_celsius_not_converted(self):
        r = clean_record(_record(temp_c=37.5), _CFG)
        assert r.temp_c_flag == FLAG_OK
        assert abs(r.temp_c - 37.5) < 0.01

    def test_102f_converted_to_celsius(self):
        r = clean_record(_record(temp_c=102.0), _CFG)
        assert r.temp_c is not None
        expected = round((102 - 32) * 5 / 9, 1)
        assert abs(r.temp_c - expected) < 0.2


# ─────────────────────────────────────────────────────────────────────────────
# SpO2 out of range
# ─────────────────────────────────────────────────────────────────────────────

class TestSpO2OutOfRange:

    def test_spo2_104_flagged(self):
        r = clean_record(_record(spo2=104.0), _CFG)
        assert r.spo2 is None
        assert r.spo2_flag == FLAG_OUT_OF_RANGE

    def test_spo2_104_warning_present(self):
        r = clean_record(_record(spo2=104.0), _CFG)
        assert any("spo2" in w.lower() for w in r.warnings)

    def test_spo2_50_boundary_valid(self):
        r = clean_record(_record(spo2=50.0), _CFG)
        assert r.spo2 == 50.0
        assert r.spo2_flag == FLAG_OK

    def test_spo2_49_just_out_of_range(self):
        r = clean_record(_record(spo2=49.0), _CFG)
        # 49 * 10 = 490 >> 100, so NOT a decimal slip → out_of_range
        assert r.spo2 is None
        assert r.spo2_flag == FLAG_OUT_OF_RANGE

    def test_spo2_100_valid(self):
        r = clean_record(_record(spo2=100.0), _CFG)
        assert r.spo2 == 100.0
        assert r.spo2_flag == FLAG_OK


# ─────────────────────────────────────────────────────────────────────────────
# Missing vitals
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingVitals:

    def test_nan_pulse_flagged_missing(self):
        import math
        r = clean_record(_record(pulse=float("nan")), _CFG)
        assert r.pulse is None
        assert r.pulse_flag == FLAG_MISSING

    def test_none_spo2_flagged_missing(self):
        r = clean_record(_record(spo2=None), _CFG)
        assert r.spo2 is None
        assert r.spo2_flag == FLAG_MISSING

    def test_n_missing_vitals_incremented(self):
        r = clean_record(_record(spo2=None, pulse=None), _CFG)
        assert r.n_missing_vitals == 2

    def test_missing_raises_needs_remeasure(self):
        r = clean_record(_record(spo2=None, pulse=None), _CFG)
        assert r.needs_remeasure is True


# ─────────────────────────────────────────────────────────────────────────────
# Swapped sys/dia
# ─────────────────────────────────────────────────────────────────────────────

class TestSwappedBP:

    def test_plausible_swap_corrected(self):
        # sys=70, dia=120 → swap to sys=120, dia=70 (both plausible)
        r = clean_record(_record(bp_sys=70, bp_dia=120), _CFG)
        assert r.bp_sys == 120.0
        assert r.bp_dia == 70.0
        assert r.bp_sys_flag == FLAG_SWAPPED
        assert r.bp_dia_flag == FLAG_SWAPPED

    def test_plausible_swap_uncertain(self):
        r = clean_record(_record(bp_sys=70, bp_dia=120), _CFG)
        assert r.bp_sys_uncertain is True
        assert r.bp_dia_uncertain is True

    def test_plausible_swap_warning_present(self):
        r = clean_record(_record(bp_sys=70, bp_dia=120), _CFG)
        assert any("transpos" in w.lower() or "swap" in w.lower()
                   for w in r.warnings)

    def test_implausible_swap_both_nulled(self):
        # sys=10, dia=15 → swap to sys=15, dia=10
        # 15 is not in [50, 240] so swap is still implausible → both None
        r = clean_record(_record(bp_sys=10, bp_dia=15), _CFG)
        # Both failed range check; swap attempt also implausible
        assert r.bp_sys is None
        assert r.bp_dia is None

    def test_normal_bp_not_swapped(self):
        r = clean_record(_record(bp_sys=120, bp_dia=80), _CFG)
        assert r.bp_sys == 120.0
        assert r.bp_dia == 80.0
        assert r.bp_sys_flag == FLAG_OK


# ─────────────────────────────────────────────────────────────────────────────
# Decimal slip
# ─────────────────────────────────────────────────────────────────────────────

class TestDecimalSlip:

    def test_pulse_12_flagged_decimal_slip(self):
        # 12 < 25 (lo), 12 × 10 = 120 ∈ [25, 250] → decimal slip
        r = clean_record(_record(pulse=12.0), _CFG)
        assert r.pulse is None
        assert r.pulse_flag == FLAG_DECIMAL_SLIP

    def test_decimal_slip_warning_present(self):
        r = clean_record(_record(pulse=12.0), _CFG)
        assert any("decimal" in w.lower() or "slip" in w.lower()
                   for w in r.warnings)

    def test_decimal_slip_not_auto_corrected(self):
        # Value must be None, not multiplied value
        r = clean_record(_record(pulse=12.0), _CFG)
        assert r.pulse is None

    def test_bp_sys_8_decimal_slip(self):
        # 8 × 10 = 80 ∈ [50, 240]
        r = clean_record(_record(bp_sys=8), _CFG)
        # Note: after decimal slip flagging, swap check cannot fire (sys is None)
        assert r.bp_sys is None
        assert r.bp_sys_flag == FLAG_DECIMAL_SLIP

    def test_bp_sys_45_is_out_of_range_not_slip(self):
        # 45 < 50 (lo), 45 × 10 = 450 > 240 (hi) → NOT decimal slip → out_of_range
        r = clean_record(_record(bp_sys=45), _CFG)
        assert r.bp_sys is None
        assert r.bp_sys_flag == FLAG_OUT_OF_RANGE


# ─────────────────────────────────────────────────────────────────────────────
# Summary stats
# ─────────────────────────────────────────────────────────────────────────────

class TestSummaryStats:

    def test_all_clean_score_is_1(self):
        r = clean_record(_record(), _CFG)
        assert r.data_quality_score == 1.0

    def test_all_clean_needs_remeasure_false(self):
        r = clean_record(_record(), _CFG)
        assert r.needs_remeasure is False

    def test_two_invalids_needs_remeasure(self):
        r = clean_record(_record(bp_sys=0, spo2=104), _CFG)
        assert r.needs_remeasure is True

    def test_one_invalid_no_remeasure(self):
        r = clean_record(_record(bp_sys=0), _CFG)
        assert r.needs_remeasure is False

    def test_score_decreases_with_invalids(self):
        r0 = clean_record(_record(), _CFG)
        r1 = clean_record(_record(bp_sys=0), _CFG)
        r2 = clean_record(_record(bp_sys=0, spo2=104), _CFG)
        assert r0.data_quality_score > r1.data_quality_score
        assert r1.data_quality_score > r2.data_quality_score

    def test_score_between_0_and_1(self):
        r = clean_record(_record(bp_sys=0, spo2=104, pulse=None,
                                 temp_c=None, bp_dia=None), _CFG)
        assert 0.0 <= r.data_quality_score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Safety invariant: invalid vitals must not increase n_invalid unexpectedly
# ─────────────────────────────────────────────────────────────────────────────

class TestSafetyInvariants:

    def test_missing_vital_does_not_set_invalid(self):
        """A None input increments missing, not invalid."""
        r = clean_record(_record(spo2=None), _CFG)
        assert r.n_missing_vitals == 1
        assert r.n_invalid_vitals == 0

    def test_out_of_range_sets_invalid_not_missing(self):
        r = clean_record(_record(spo2=104), _CFG)
        assert r.n_invalid_vitals == 1
        assert r.n_missing_vitals == 0

    def test_unit_converted_not_counted_as_invalid(self):
        # Fahrenheit conversion produces a valid cleaned value
        r = clean_record(_record(temp_c=96.0), _CFG)
        assert r.n_invalid_vitals == 0
        assert r.temp_c_flag == FLAG_TEMP_CONV


# ─────────────────────────────────────────────────────────────────────────────
# Unified clean() interface
# ─────────────────────────────────────────────────────────────────────────────

class TestUnifiedClean:

    def test_clean_dict_returns_clean_result(self):
        rec = _record(bp_sys=120, bp_dia=80)
        res = clean(rec, _CFG)
        assert isinstance(res, CleanResult)
        assert res.bp_sys == 120.0

    def test_clean_dataframe_returns_dataframe(self):
        import pandas as pd
        df = pd.DataFrame([_record(bp_sys=120), _record(bp_sys=0)])
        res = clean(df, _CFG)
        assert isinstance(res, pd.DataFrame)
        assert "bp_sys_c" in res.columns
        assert res.iloc[0]["bp_sys_c"] == 120.0
        assert pd.isna(res.iloc[1]["bp_sys_c"])

    def test_clean_invalid_type_raises(self):
        with pytest.raises(TypeError):
            clean(["not", "a", "record"])  # type: ignore[arg-type]
