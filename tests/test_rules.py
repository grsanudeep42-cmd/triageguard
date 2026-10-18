"""
tests/test_rules.py
─────────────────────────────────────────────────────────────────────────────
Acceptance tests for triageguard.rules  (Phase 3).

Coverage
─────────
• Each vital rule fires on a hand-crafted abnormal case and NOT on a normal one
• Red-flag text patterns trigger on matching text and not on unrelated text
• Red flags only escalate to level 2 (never to level 1)
• Rules with age filters respect age bands
• Missing / invalid vitals do NOT trigger vital rules
• Escalation monotonicity property:
    making any vital worse (↓ SpO2, ↓ sys BP, ↑ pulse, ↑ temp)
    never produces a LESS urgent forced level
• Combined (shock) and multi-condition rules
"""

from __future__ import annotations

import pytest
from triageguard.cleaning import CleanResult, clean_record, FLAG_OK
from triageguard.rules import (
    RuleResult,
    apply_rules,
    detect_red_flags,
    _edit_distance,
    _fuzzy_phrase_match,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared test configs (inline; no file I/O in unit tests)
# ─────────────────────────────────────────────────────────────────────────────

_RULES_CFG = {
    "rules": [
        {   # R001 – critical hypoxia
            "id": "R001", "description": "Critical hypoxia — SpO₂ < 85 %",
            "conditions": {"spo2_lt": 85.0}, "forced_max_level": 1,
        },
        {   # R002 – severe hypoxia
            "id": "R002", "description": "Severe hypoxia — SpO₂ 85–89 %",
            "conditions": {"spo2_lt": 90.0}, "forced_max_level": 2,
        },
        {   # R003 – profound hypotension adult
            "id": "R003", "description": "Profound hypotension — sys < 60",
            "conditions": {"bp_sys_lt": 60.0}, "forced_max_level": 1, "age_min": 13,
        },
        {   # R004 – hypotension adult
            "id": "R004", "description": "Hypotension — sys < 90",
            "conditions": {"bp_sys_lt": 90.0}, "forced_max_level": 2, "age_min": 13,
        },
        {   # R005 – extreme tachycardia adult
            "id": "R005", "description": "Extreme tachycardia > 180 bpm (adult)",
            "conditions": {"pulse_gt": 180.0}, "forced_max_level": 2, "age_min": 13,
        },
        {   # R006 – severe bradycardia
            "id": "R006", "description": "Severe bradycardia < 40 bpm",
            "conditions": {"pulse_lt": 40.0}, "forced_max_level": 2,
        },
        {   # R007 – hyperpyrexia adult
            "id": "R007", "description": "Hyperpyrexia ≥ 41 °C (adult)",
            "conditions": {"temp_c_ge": 41.0}, "forced_max_level": 2, "age_min": 13,
        },
        {   # R008 – infant fever
            "id": "R008", "description": "High fever in infant",
            "conditions": {"temp_c_ge": 39.5}, "forced_max_level": 2, "age_max": 2,
        },
        {   # R009 – infant tachycardia
            "id": "R009", "description": "Tachycardia > 200 bpm in infant (age <= 2)",
            "conditions": {"pulse_gt": 200.0}, "forced_max_level": 2, "age_max": 2,
        },
        {   # R010 – tachycardia child
            "id": "R010", "description": "Extreme tachycardia child",
            "conditions": {"pulse_gt": 160.0}, "forced_max_level": 2,
            "age_min": 2, "age_max": 12,
        },
        {   # R011 – child hypotension
            "id": "R011", "description": "Hypotension — sys < 70 in child (age 2–12)",
            "conditions": {"bp_sys_lt": 70.0}, "forced_max_level": 2,
            "age_min": 2, "age_max": 12,
        },
        {   # R012 – shock pattern
            "id": "R012", "description": "Shock pattern sys < 90 AND pulse > 110",
            "conditions": {"bp_sys_lt": 90.0, "pulse_gt": 110.0},
            "forced_max_level": 2, "age_min": 13,
        },
        {   # R013 – septic shock
            "id": "R013", "description": "Septic shock temp ≥ 38.5 AND sys < 90",
            "conditions": {"temp_c_ge": 38.5, "bp_sys_lt": 90.0},
            "forced_max_level": 2, "age_min": 13,
        },
    ]
}


_RF_CFG = {
    "fuzzy_max_distance": 2,
    "red_flags": [
        {
            "id": "RF001", "name": "chest_pain",
            "escalate_to_max_level": 2,
            "patterns_en": ["chest pain", "chest tightness"],
            "patterns_tel": ["gunde noppi"],
            "patterns_hin": ["seene dard"],
        },
        {
            "id": "RF002", "name": "breathlessness",
            "escalate_to_max_level": 2,
            "patterns_en": ["cannot breathe", "breathlessness"],
            "patterns_tel": ["oopiri ledu"],
            "patterns_hin": ["saans nahi"],
        },
        {
            "id": "RF003", "name": "unconscious_unresponsive",
            "escalate_to_max_level": 2,
            "patterns_en": ["unconscious", "unresponsive"],
            "patterns_tel": ["behosh"],
            "patterns_hin": ["behosh"],
        },
    ],
}

_CLEAN_CFG = {
    "vital_ranges": {
        "bp_sys":  {"lo": 50,   "hi": 240},
        "bp_dia":  {"lo": 20,   "hi": 150},
        "pulse":   {"lo": 25,   "hi": 250},
        "temp_c":  {"lo": 34.0, "hi": 42.5},
        "spo2":    {"lo": 50.0, "hi": 100.0},
    },
    "fahrenheit_range":       {"lo": 88.0, "hi": 110.0},
    "decimal_slip_detection": {"factor": 10},
    "text_cleaning":          {"max_repeated_chars": 2},
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a CleanResult with custom vital values
# ─────────────────────────────────────────────────────────────────────────────

def _cr(**kwargs) -> CleanResult:
    """Build a CleanResult with normal defaults, overrideable per-vital."""
    defaults = dict(
        bp_sys=120.0, bp_dia=80.0, pulse=72.0, temp_c=37.0, spo2=98.0,
        complaint_text="discomfort",
        bp_sys_flag="ok", bp_dia_flag="ok", pulse_flag="ok",
        temp_c_flag="ok", spo2_flag="ok",
    )
    defaults.update(kwargs)
    return CleanResult(**defaults)


def _apply(cleaned: CleanResult, age: int = 35) -> RuleResult:
    return apply_rules(cleaned, age=age,
                       rules_config=_RULES_CFG, redflags_config=_RF_CFG)


# ─────────────────────────────────────────────────────────────────────────────
# Levenshtein helper
# ─────────────────────────────────────────────────────────────────────────────

class TestEditDistance:
    def test_identical(self):
        assert _edit_distance("chest", "chest") == 0

    def test_one_substitution(self):
        assert _edit_distance("chst", "chest") == 1

    def test_empty_strings(self):
        assert _edit_distance("", "") == 0

    def test_one_empty(self):
        assert _edit_distance("abc", "") == 3


class TestFuzzyPhrase:
    def test_exact_match(self):
        assert _fuzzy_phrase_match("chest pain", ["chest", "pain"])

    def test_typo_match(self):
        assert _fuzzy_phrase_match("chest pain", ["chst", "pian"])

    def test_no_match(self):
        assert not _fuzzy_phrase_match("slurred speech", ["stomach", "ache"])

    def test_partial_phrase_no_match(self):
        # Pattern "chest pain" requires both words
        assert not _fuzzy_phrase_match("chest pain", ["chest"])


# ─────────────────────────────────────────────────────────────────────────────
# R001 – critical hypoxia
# ─────────────────────────────────────────────────────────────────────────────

class TestR001CriticalHypoxia:

    def test_fires_on_spo2_84(self):
        r = _apply(_cr(spo2=84.0))
        assert "R001" in r.triggered_rule_ids
        assert r.forced_level == 1

    def test_does_not_fire_on_normal_spo2(self):
        r = _apply(_cr(spo2=98.0))
        assert "R001" not in r.triggered_rule_ids

    def test_does_not_fire_on_spo2_85(self):
        # Boundary: 85 is NOT < 85
        r = _apply(_cr(spo2=85.0))
        assert "R001" not in r.triggered_rule_ids

    def test_spo2_none_does_not_fire(self):
        r = _apply(_cr(spo2=None, spo2_flag="missing"))
        assert "R001" not in r.triggered_rule_ids


# ─────────────────────────────────────────────────────────────────────────────
# R002 – severe hypoxia
# ─────────────────────────────────────────────────────────────────────────────

class TestR002SevereHypoxia:

    def test_fires_on_spo2_88(self):
        r = _apply(_cr(spo2=88.0))
        assert "R002" in r.triggered_rule_ids

    def test_does_not_fire_on_spo2_91(self):
        r = _apply(_cr(spo2=91.0))
        assert "R002" not in r.triggered_rule_ids

    def test_forced_level_not_below_1_when_r001_also_fires(self):
        # R001 fires → forced_level = 1 (min of 1 and 2)
        r = _apply(_cr(spo2=80.0))
        assert r.forced_level == 1


# ─────────────────────────────────────────────────────────────────────────────
# R003 – profound hypotension adult
# ─────────────────────────────────────────────────────────────────────────────

class TestR003ProfoundHypotension:

    def test_fires_on_sys_55_adult(self):
        r = _apply(_cr(bp_sys=55.0), age=40)
        assert "R003" in r.triggered_rule_ids
        assert r.forced_level == 1

    def test_does_not_fire_on_normal_sys(self):
        r = _apply(_cr(bp_sys=120.0), age=40)
        assert "R003" not in r.triggered_rule_ids

    def test_does_not_fire_on_child(self):
        # Age band: age_min=13 → child age=10 should not trigger R003
        r = _apply(_cr(bp_sys=55.0), age=10)
        assert "R003" not in r.triggered_rule_ids


# ─────────────────────────────────────────────────────────────────────────────
# R004 – hypotension adult
# ─────────────────────────────────────────────────────────────────────────────

class TestR004Hypotension:

    def test_fires_on_sys_85_adult(self):
        r = _apply(_cr(bp_sys=85.0), age=40)
        assert "R004" in r.triggered_rule_ids

    def test_does_not_fire_on_sys_90(self):
        # Boundary: 90 is NOT < 90
        r = _apply(_cr(bp_sys=90.0), age=40)
        assert "R004" not in r.triggered_rule_ids

    def test_does_not_fire_on_child_age_12(self):
        r = _apply(_cr(bp_sys=85.0), age=12)
        assert "R004" not in r.triggered_rule_ids


# ─────────────────────────────────────────────────────────────────────────────
# R005 – extreme tachycardia adult
# ─────────────────────────────────────────────────────────────────────────────

class TestR005ExtremeTachycardia:

    def test_fires_on_pulse_200_adult(self):
        r = _apply(_cr(pulse=200.0), age=30)
        assert "R005" in r.triggered_rule_ids

    def test_does_not_fire_on_pulse_180(self):
        # Boundary: 180 is NOT > 180
        r = _apply(_cr(pulse=180.0), age=30)
        assert "R005" not in r.triggered_rule_ids

    def test_pulse_none_does_not_fire(self):
        r = _apply(_cr(pulse=None, pulse_flag="missing"), age=30)
        assert "R005" not in r.triggered_rule_ids


# ─────────────────────────────────────────────────────────────────────────────
# R006 – severe bradycardia (all ages)
# ─────────────────────────────────────────────────────────────────────────────

class TestR006Bradycardia:

    def test_fires_on_pulse_30(self):
        r = _apply(_cr(pulse=30.0), age=40)
        assert "R006" in r.triggered_rule_ids

    def test_does_not_fire_on_pulse_40(self):
        # Boundary: 40 is NOT < 40
        r = _apply(_cr(pulse=40.0), age=40)
        assert "R006" not in r.triggered_rule_ids

    def test_fires_on_infant(self):
        # No age filter on R006
        r = _apply(_cr(pulse=30.0), age=1)
        assert "R006" in r.triggered_rule_ids


# ─────────────────────────────────────────────────────────────────────────────
# R007 – hyperpyrexia adult
# ─────────────────────────────────────────────────────────────────────────────

class TestR007Hyperpyrexia:

    def test_fires_on_temp_41_5_adult(self):
        r = _apply(_cr(temp_c=41.5), age=30)
        assert "R007" in r.triggered_rule_ids

    def test_does_not_fire_on_temp_40_9(self):
        r = _apply(_cr(temp_c=40.9), age=30)
        assert "R007" not in r.triggered_rule_ids

    def test_does_not_fire_on_child_age_12(self):
        r = _apply(_cr(temp_c=41.5), age=12)
        assert "R007" not in r.triggered_rule_ids


# ─────────────────────────────────────────────────────────────────────────────
# R008 – infant fever (age ≤ 2)
# ─────────────────────────────────────────────────────────────────────────────

class TestR008InfantFever:

    def test_fires_on_infant_40c(self):
        r = _apply(_cr(temp_c=40.0), age=1)
        assert "R008" in r.triggered_rule_ids

    def test_does_not_fire_on_adult_40c(self):
        # Adult with 40°C should hit R007 (≥41) not R008 (age_max=2)
        r = _apply(_cr(temp_c=40.0), age=30)
        assert "R008" not in r.triggered_rule_ids

    def test_does_not_fire_on_infant_below_threshold(self):
        r = _apply(_cr(temp_c=39.0), age=1)
        assert "R008" not in r.triggered_rule_ids

    def test_does_not_fire_on_child_age_3(self):
        r = _apply(_cr(temp_c=40.0), age=3)
        assert "R008" not in r.triggered_rule_ids


# ─────────────────────────────────────────────────────────────────────────────
# R009 – infant tachycardia (age ≤ 2)
# ─────────────────────────────────────────────────────────────────────────────

class TestR009InfantTachycardia:

    def test_fires_on_infant_pulse_205(self):
        r = _apply(_cr(pulse=205.0), age=1)
        assert "R009" in r.triggered_rule_ids
        assert r.forced_level == 2

    def test_does_not_fire_on_infant_normal_pulse(self):
        r = _apply(_cr(pulse=140.0), age=1)
        assert "R009" not in r.triggered_rule_ids

    def test_does_not_fire_on_adult(self):
        # Adult age 25 with pulse 205 should hit R005 not R009
        r = _apply(_cr(pulse=205.0), age=25)
        assert "R009" not in r.triggered_rule_ids
        assert "R005" in r.triggered_rule_ids


# ─────────────────────────────────────────────────────────────────────────────
# R010 – tachycardia child (2–12)
# ─────────────────────────────────────────────────────────────────────────────

class TestR010ChildTachycardia:

    def test_fires_on_child_pulse_180(self):
        r = _apply(_cr(pulse=180.0), age=8)
        assert "R010" in r.triggered_rule_ids

    def test_does_not_fire_on_adult(self):
        r = _apply(_cr(pulse=180.0), age=20)
        assert "R010" not in r.triggered_rule_ids

    def test_does_not_fire_on_infant(self):
        r = _apply(_cr(pulse=180.0), age=1)
        assert "R010" not in r.triggered_rule_ids


# ─────────────────────────────────────────────────────────────────────────────
# R011 – child hypotension (2–12)
# ─────────────────────────────────────────────────────────────────────────────

class TestR011ChildHypotension:

    def test_fires_on_child_sys_65(self):
        r = _apply(_cr(bp_sys=65.0), age=7)
        assert "R011" in r.triggered_rule_ids
        assert r.forced_level == 2

    def test_does_not_fire_on_child_normal_sys(self):
        r = _apply(_cr(bp_sys=95.0), age=7)
        assert "R011" not in r.triggered_rule_ids

    def test_does_not_fire_on_adult(self):
        # Adult age 30 with sys 65 should hit R004 not R011
        r = _apply(_cr(bp_sys=65.0), age=30)
        assert "R011" not in r.triggered_rule_ids
        assert "R004" in r.triggered_rule_ids


# ─────────────────────────────────────────────────────────────────────────────
# R012 – shock pattern (combination rule)
# ─────────────────────────────────────────────────────────────────────────────

class TestR012ShockPattern:

    def test_fires_when_both_conditions_met(self):
        r = _apply(_cr(bp_sys=85.0, pulse=120.0), age=40)
        assert "R012" in r.triggered_rule_ids

    def test_does_not_fire_on_low_bp_alone(self):
        # Low BP but pulse 80 — shock rule doesn't fire (pulse ≤ 110)
        r = _apply(_cr(bp_sys=85.0, pulse=80.0), age=40)
        assert "R012" not in r.triggered_rule_ids

    def test_does_not_fire_on_high_pulse_alone(self):
        # High pulse but BP 120 — shock rule doesn't fire
        r = _apply(_cr(bp_sys=120.0, pulse=120.0), age=40)
        assert "R012" not in r.triggered_rule_ids

    def test_does_not_fire_when_bp_none(self):
        r = _apply(_cr(bp_sys=None, bp_sys_flag="missing", pulse=120.0), age=40)
        assert "R012" not in r.triggered_rule_ids


# ─────────────────────────────────────────────────────────────────────────────
# R013 – septic shock pattern
# ─────────────────────────────────────────────────────────────────────────────

class TestR013SepticShock:

    def test_fires_on_fever_plus_hypotension(self):
        r = _apply(_cr(temp_c=39.5, bp_sys=85.0), age=40)
        assert "R013" in r.triggered_rule_ids

    def test_does_not_fire_on_fever_alone(self):
        r = _apply(_cr(temp_c=39.5, bp_sys=120.0), age=40)
        assert "R013" not in r.triggered_rule_ids

    def test_does_not_fire_on_hypotension_alone(self):
        r = _apply(_cr(bp_sys=85.0, temp_c=37.0), age=40)
        assert "R013" not in r.triggered_rule_ids


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Red-flag text detection (SOFT guard)
# ─────────────────────────────────────────────────────────────────────────────

class TestRedFlagDetection:

    def test_chest_pain_fires(self):
        ids, names = detect_red_flags("chest pain since morning", _RF_CFG)
        assert "RF001" in ids
        assert "chest_pain" in names

    def test_typo_chest_pain_fires(self):
        ids, names = detect_red_flags("chst pain severe", _RF_CFG)
        assert "RF001" in ids

    def test_telugu_chest_pain_fires(self):
        ids, names = detect_red_flags("gunde noppi vastondi", _RF_CFG)
        assert "RF001" in ids

    def test_hindi_breathless_fires(self):
        ids, names = detect_red_flags("saans nahi aa rahi", _RF_CFG)
        assert "RF002" in ids

    def test_unrelated_text_does_not_fire(self):
        ids, names = detect_red_flags("knee pain mild since 2 days", _RF_CFG)
        assert ids == []
        assert names == []

    def test_red_flag_sets_no_downgrade_below_L3_and_not_forced_level(self):
        # Red flags are SOFT: they do NOT set forced_level, only no_downgrade_below_L3
        r = _apply(_cr(complaint_text="unconscious patient"))
        assert r.forced_level is None
        assert r.no_downgrade_below_L3 is True
        assert "unconscious_unresponsive" in r.red_flags_detected

    def test_red_flag_alone_never_sets_forced_level_1_or_2(self):
        # Even with prominent complaint, text alone leaves forced_level as None
        cleaned = _cr(complaint_text="cannot breathe shortness of breath")
        r = _apply(cleaned, age=30)
        assert r.forced_level is None
        assert r.no_downgrade_below_L3 is True


class TestAllRedFlagsFromYaml:
    """Verify all 10 red flags from redflags.yaml detect their respective phrases."""

    @pytest.mark.parametrize("flag_id,flag_name,sample_text", [
        ("RF001", "chest_pain", "severe chest pain since morning"),
        ("RF002", "breathlessness", "difficulty breathing and shortness of breath"),
        ("RF003", "unconscious_unresponsive", "patient is unconscious and fainted"),
        ("RF004", "seizure_fit", "had a seizure with shaking uncontrollably"),
        ("RF005", "heavy_bleeding", "heavy bleeding profuse blood loss"),
        ("RF006", "one_sided_weakness", "one sided weakness arm not moving"),
        ("RF007", "slurred_speech", "speech difficulty and slurred speech"),
        ("RF008", "severe_pain", "excruciating pain 10 out of 10 pain"),
        ("RF009", "vomiting_blood", "vomiting blood hematemesis"),
        ("RF010", "pregnancy_bleeding", "bleeding in pregnancy"),
    ])
    def test_each_red_flag_detected(self, flag_id, flag_name, sample_text):
        ids, names = detect_red_flags(sample_text)
        assert flag_id in ids
        assert flag_name in names

    def test_normal_complaint_detects_no_red_flags(self):
        ids, names = detect_red_flags("routine checkup feeling normal")
        assert ids == []
        assert names == []


# ─────────────────────────────────────────────────────────────────────────────
# Abstain reason
# ─────────────────────────────────────────────────────────────────────────────

class TestAbstainReason:

    def test_needs_remeasure_without_forced_rule_sets_abstain_reason(self):
        # 2 invalid vitals -> needs_remeasure = True, vitals otherwise normal
        cleaned = _cr(bp_sys=None, bp_sys_flag="missing",
                      pulse=None, pulse_flag="missing",
                      needs_remeasure=True)
        r = _apply(cleaned, age=35)
        assert r.forced_level is None
        assert r.abstain_reason == "remeasure"
        assert any("remeasure" in reason.lower() for reason in r.reasons)

    def test_needs_remeasure_with_forced_rule_takes_precedence(self):
        # Even if needs_remeasure is True, an emergency vital rule takes precedence
        cleaned = _cr(bp_sys=None, bp_sys_flag="missing",
                      pulse=30.0, pulse_flag="ok",
                      spo2=None, spo2_flag="missing",
                      needs_remeasure=True)
        r = _apply(cleaned, age=35)
        assert r.forced_level == 2  # R006 severe bradycardia
        assert r.abstain_reason is None


# ─────────────────────────────────────────────────────────────────────────────
# Lexicon independence
# ─────────────────────────────────────────────────────────────────────────────

class TestLexiconIndependence:

    def test_zero_overlap_with_emergency_lexicon_pools(self):
        import yaml
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent

        with open(root / "redflags.yaml") as f:
            rf = yaml.safe_load(f)
        with open(root / "lexicon.yaml") as f:
            lex = yaml.safe_load(f)
        with open(root / "archetypes.yaml") as f:
            arch = yaml.safe_load(f)

        em_archs = {a["name"] for a in arch["archetypes"] if a["triage_level"] in (1, 2)}
        em_phrases = set()
        for name in em_archs:
            if name in lex.get("archetypes", {}):
                for lang, phrases in lex["archetypes"][name].items():
                    if isinstance(phrases, list):
                        for p in phrases:
                            em_phrases.add(p.strip().lower())
        for a in arch["archetypes"]:
            if a["triage_level"] in (1, 2):
                for p in a.get("complaint_templates_en", []):
                    em_phrases.add(p.strip().lower())

        rf_phrases = set()
        for r in rf.get("red_flags", []):
            for k in ["patterns_en", "patterns_tel", "patterns_hin"]:
                for p in r.get(k, []):
                    rf_phrases.add(p.strip().lower())

        overlap = rf_phrases.intersection(em_phrases)
        assert len(overlap) == 0, f"Overlapping phrases found: {overlap}"


# ─────────────────────────────────────────────────────────────────────────────
# Never downgrade invariant
# ─────────────────────────────────────────────────────────────────────────────

class TestNeverDowngrade:

    def test_rules_only_produce_levels_1_2_or_none(self):
        """rules.py never outputs level 3/4/5 by itself."""
        cases = [
            _cr(spo2=84.0),
            _cr(bp_sys=55.0),
            _cr(pulse=190.0),
            _cr(temp_c=41.5),
            _cr(),   # all normal → None
        ]
        for cleaned in cases:
            r = _apply(cleaned, age=40)
            if r.forced_level is not None:
                assert r.forced_level in {1, 2}, (
                    f"Rules output level {r.forced_level}, only 1 or 2 allowed"
                )

    def test_all_normal_no_trigger(self):
        r = _apply(_cr(), age=40)
        assert r.forced_level is None
        assert r.triggered_rule_ids == []
        assert r.red_flags_detected == []
        assert r.no_downgrade_below_L3 is False
        assert r.abstain_reason is None


# ─────────────────────────────────────────────────────────────────────────────
# Monotonicity property test
# ─────────────────────────────────────────────────────────────────────────────

class TestMonotonicity:
    """
    Making any vital WORSE must never produce a LESS urgent (higher number)
    forced level.

    Convention: None is treated as "99" (no escalation) when comparing urgency.
    Assert new_level <= old_level when a vital gets worse.
    """

    def _urgency(self, r: RuleResult) -> int:
        return r.forced_level if r.forced_level is not None else 99

    @pytest.mark.parametrize("vital,good_val,bad_val,age", [
        ("spo2",   98.0, 80.0, 35),   # lower SpO2 → more urgent
        ("spo2",   92.0, 86.0, 35),   # within severe range → not worse
        ("bp_sys", 120.0, 55.0, 35),  # lower sys → more urgent
        ("bp_sys",  95.0, 85.0, 35),  # from near-normal to hypotensive
        ("pulse",   72.0, 195.0, 30), # higher pulse (adult) → more urgent
        ("pulse",   72.0,  30.0, 30), # bradycardia
        ("temp_c",  37.0, 41.5, 30),  # hyperpyrexia
        ("temp_c",  38.0, 40.5, 1),   # infant fever
    ])
    def test_worsening_vital_never_less_urgent(
        self, vital, good_val, bad_val, age
    ):
        good_cleaned = _cr(**{vital: good_val})
        bad_cleaned  = _cr(**{vital: bad_val})

        good_r = _apply(good_cleaned, age=age)
        bad_r  = _apply(bad_cleaned,  age=age)

        g_urg  = self._urgency(good_r)
        b_urg  = self._urgency(bad_r)

        assert b_urg <= g_urg, (
            f"Worsening {vital} from {good_val} → {bad_val} (age={age}) "
            f"changed urgency from {g_urg} → {b_urg} (LESS urgent)"
        )

    def test_random_property_worsening_vital_never_less_urgent(self):
        """
        Random property test: 500 randomized cases.
        Represent 'no forced level' as 99, and assert new_level <= old_level
        when any vital gets worse.
        """
        import random
        rng = random.Random(42)

        for trial in range(500):
            age = rng.randint(0, 85)
            # Baseline plausible vitals
            spo2 = rng.uniform(70.0, 100.0)
            bp_sys = rng.uniform(50.0, 160.0)
            bp_dia = rng.uniform(30.0, min(100.0, bp_sys - 5))
            pulse = rng.uniform(35.0, 220.0)
            temp_c = rng.uniform(35.0, 42.0)

            cr_base = _cr(bp_sys=bp_sys, bp_dia=bp_dia, pulse=pulse, temp_c=temp_c, spo2=spo2)
            res_base = _apply(cr_base, age=age)
            base_urg = self._urgency(res_base)

            # Pick a vital to worsen
            choice = rng.choice(["spo2", "bp_sys", "pulse_tach", "pulse_brady", "temp_c"])
            worse_dict = dict(bp_sys=bp_sys, bp_dia=bp_dia, pulse=pulse, temp_c=temp_c, spo2=spo2)

            if choice == "spo2":
                delta = rng.uniform(1.0, 15.0)
                worse_dict["spo2"] = max(50.0, spo2 - delta)
            elif choice == "bp_sys":
                delta = rng.uniform(5.0, 30.0)
                worse_dict["bp_sys"] = max(50.0, bp_sys - delta)
            elif choice == "pulse_tach":
                delta = rng.uniform(10.0, 40.0)
                worse_dict["pulse"] = min(250.0, pulse + delta)
            elif choice == "pulse_brady":
                delta = rng.uniform(5.0, 25.0)
                worse_dict["pulse"] = max(25.0, min(pulse, 45.0) - delta)
            elif choice == "temp_c":
                delta = rng.uniform(0.5, 2.0)
                worse_dict["temp_c"] = min(42.5, temp_c + delta)

            cr_worse = _cr(**worse_dict)
            res_worse = _apply(cr_worse, age=age)
            worse_urg = self._urgency(res_worse)

            assert worse_urg <= base_urg, (
                f"Trial {trial} ({choice}): Worsened vitals {worse_dict} from {cr_base} (age={age}) "
                f"resulted in urgency {worse_urg} > {base_urg} (LESS urgent)"
            )

