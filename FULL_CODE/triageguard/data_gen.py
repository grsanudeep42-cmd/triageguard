"""
TriageGuard – synthetic OPD intake data generator  (v2 – hardened).

ISOLATION RULE
--------------
Label-generation logic lives ONLY here and in archetypes.yaml / lexicon.yaml.
Model code must NEVER import data_gen.  It reads only the output CSVs.

PLACEHOLDER CONVENTION
-----------------------
Every numeric threshold, prior, rate, or distribution parameter is annotated
    # PLACEHOLDER pending official triage primer
so a domain expert can locate and replace each value without reading the full
codebase.

CHANGES vs v1
-------------
FIX-1  Patient IDs are assigned AFTER the global shuffle so the sequential
       index carries no information about the triage level.
FIX-2  known_conditions and medications are sampled from an age/sex-based
       prevalence table, entirely independent of archetype department.
       "not known" is capped at ~8 % across all levels (35 % exception only
       for the unconscious_unresponsive archetype).
FIX-3  Confusable-pair archetypes share 70–80 % of their complaint text.
       Emergency archetypes get a discriminating phrase with only ~25 %
       probability.  Under-reporting suppresses discriminating phrases.
       5 % of all rows get near-empty text; 20 % get an irrelevant extra.

OUTPUTS (written to <output_dir>/)
-----------------------------------
synthetic_intake.csv   – features only (NO hidden-label columns)
hidden_labels.csv      – labels (isolated from model code)
splits.csv             – stratified train / calib / val / test assignments
data/README.md         – column reference, marked SYNTHETIC / PLACEHOLDER
"""

from __future__ import annotations

import argparse
import math
import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_ROOT      = Path(__file__).resolve().parent.parent
_CFG_PATH  = _ROOT / "config.yaml"
_ARCH_PATH = _ROOT / "archetypes.yaml"
_LEX_PATH  = _ROOT / "lexicon.yaml"

# ─────────────────────────────────────────────────────────────────────────────
# FIX-3a  Confusable-pair shared complaint pools
# ─────────────────────────────────────────────────────────────────────────────
# Each entry: "pair_key" → {shared_en, shared_tel, shared_hin,
#                           discrim_en, discrim_tel, discrim_hin}
#
# shared_*   — used by BOTH archetypes in the pair.
# discrim_*  — added only for the FIRST (more-severe) archetype, ~25% of time.
#
# 70–80 % of an archetype's complaints will draw from shared_* pools.
# All phrase selections are  # PLACEHOLDER.

_PAIR_DATA: Dict[str, Dict[str, List[str]]] = {

    # ── ACS vs GERD / acidity ─────────────────────────────────────────────
    "chest_pain": {
        "shared_en": [
            "chest pain since morning",
            "tightness in chest",
            "chest discomfort",
            "pain in chest",
            "heaviness in chest",
            "burning sensation in chest",
            "chest pain after eating",
            "chest pain on and off",
            "chest pain since last night",
            "chest discomfort not sure why",
        ],
        "shared_tel": [
            "gundelo noppi unnadi",
            "chest lo discomfort ga vundi",
            "gundelo heavy ga anipistundi",
            "chest noppi vastondi",
            "gundelo burning ga vundi",
            "gunde lo tight ga anipistundi",
            "chest lo noppi since morning",
        ],
        "shared_hin": [
            "seene mein dard hai",
            "chhati mein daba sa lag raha",
            "seene mein jalan hai",
            "chhati mein dard subah se",
            "seene mein takleef",
            "seene mein heavy feel ho raha",
        ],
        "discrim_en": [     # first member (acs_stemi) only, ~25 %  # PLACEHOLDER
            "radiating to left arm",
            "with sweating and cold sweat",
            "with jaw pain also",
            "radiating to shoulder",
        ],
        "discrim_tel": [
            "cheyyi ki velutundi chema vastondi",
            "jaw ki kuda velutundi noppi",
        ],
        "discrim_hin": [
            "baya haath mein bhi ja raha",
            "pasina aa raha haath mein bhi",
        ],
    },

    # ── Severe asthma vs Anxiety / panic ──────────────────────────────────
    "breathless": {
        "shared_en": [
            "cannot breathe",
            "difficulty breathing",
            "breathlessness",
            "chest tightness and breathing problem",
            "feeling suffocated",
            "shortness of breath",
            "unable to breathe properly",
            "breathing difficulty since some time",
            "breath not coming",
            "feels like air not entering",
        ],
        "shared_tel": [
            "oopiri adatam kastam ga vundi",
            "chest tite ga vundi oopiri ledu",
            "breathlessness ga vundi",
            "oopiri posaledhu",
            "saans lena kastam ga vundi",
            "oopiri adatam ledu",
        ],
        "shared_hin": [
            "saans nahi aa rahi",
            "saans lena mushkil ho raha",
            "chest tight hai saans ki dikkat",
            "dam ghutne jaisa lag raha",
            "saans rukti si lag rahi",
        ],
        "discrim_en": [     # first member (severe_asthma_attack) only, ~25 %
            "with wheezing sound from chest",
            "inhaler used but no relief",
            "oxygen very low",
        ],
        "discrim_tel": [
            "wheeze vastondi inhaler help chesaledhu",
            "oxygen chala tekkuva ga vundi",
        ],
        "discrim_hin": [
            "saans mein khadkhadahat aur inhaler ne kaam nahi kiya",
            "oxygen bahut kam ho gayi",
        ],
    },

    # ── Stroke vs Migraine ────────────────────────────────────────────────
    "headache": {
        "shared_en": [
            "severe headache",
            "very bad headache",
            "terrible headache",
            "head pain",
            "headache with vomiting",
            "one-sided headache",
            "headache since this morning",
            "bad headache and nausea",
            "head pain throbbing",
            "headache light sensitive",
        ],
        "shared_tel": [
            "tala noppi chala theevranga vundi",
            "chala severe headache vachchi",
            "tala noppi tho vamtulu",
            "head pain one side",
            "tala noppi chala kastam ga vundi",
        ],
        "shared_hin": [
            "sir mein bahut dard hai",
            "ek taraf ka sir dard",
            "sir dard aur ulti",
            "sir mein bahut tez dard",
            "sir mein thak thak karna",
        ],
        "discrim_en": [     # first member (stroke_major) only, ~25 %
            "with face drooping on one side",
            "arm not moving properly",
            "speech unclear confused",
            "one-side body weak",
        ],
        "discrim_tel": [
            "muka thippi poyindi matladatam ledu",
            "ooka vaipu kalu cheyyi pani chesaledhu",
        ],
        "discrim_hin": [
            "mooh tedha ho gaya baat nahi kar pa raha",
            "ek taraf haath kaam nahi kar raha",
        ],
    },

    # ── Acute abdomen vs Gastritis / GERD ─────────────────────────────────
    "abdominal": {
        "shared_en": [
            "stomach pain",
            "belly pain",
            "abdominal pain",
            "pain in abdomen since morning",
            "tummy pain and nausea",
            "stomach discomfort",
            "nausea and stomach pain",
            "abdominal ache",
            "stomach hurts",
            "pain in stomach area",
        ],
        "shared_tel": [
            "kadupu noppi unnadi",
            "stomach lo noppi ga vundi",
            "belly lo discomfort",
            "kadupu lo noppi since morning",
            "stomach lo noppi tho nausea",
        ],
        "shared_hin": [
            "pet mein dard hai",
            "pait mein takleef",
            "pait dard aur matli",
            "pet mein dard subah se",
            "pait mein dard aur ulti jaisi",
        ],
        "discrim_en": [     # first member (acute_abdomen) only, ~25 %
            "with fever also",
            "getting much worse since yesterday",
            "abdomen hard on touching",
            "right lower side pain worsening",
        ],
        "discrim_tel": [
            "jwaram tho kadupu noppi unnadi",
            "kadupu chala tight ga vundi kottukuntundi",
        ],
        "discrim_hin": [
            "bukhar bhi aa raha aur dard",
            "dabane par bahut dard right side",
        ],
    },

    # ── Sepsis vs Mild viral fever ────────────────────────────────────────
    "fever_adult": {
        "shared_en": [
            "fever since 2 days",
            "high temperature and body ache",
            "fever and weakness",
            "fever body pain",
            "high fever",
            "temperature since yesterday",
            "fever and fatigue",
            "running temperature and feeling weak",
            "fever body pain headache",
            "fever not going since 2 days",
        ],
        "shared_tel": [
            "jwaram 2 rojulu nundi unnadi",
            "theevranga jwaram mariyu narasam",
            "fever mariyu body ache vundi",
            "temperature unnadi body pain tho",
            "chala jwaram since 2 days",
        ],
        "shared_hin": [
            "2 din se bukhar hai",
            "tez bukhar aur body dard",
            "bukhar aur kamzori",
            "bukhar hai kal se",
            "bukhar aur thakaan bahut",
        ],
        "discrim_en": [     # first member (sepsis_severe) only, ~25 %
            "and unable to stand very weak",
            "feeling confused and dizzy",
            "not passing urine at all",
            "blood pressure very low feeling",
        ],
        "discrim_tel": [
            "chala narasam ga letchaledhu BP tekkuva",
            "mootra raaledhu chala weak ga vundi",
        ],
        "discrim_hin": [
            "khari nahi ho sakti bahut kamzori",
            "BP bahut kam ho gaya aur confusion",
        ],
    },
}

# Maps each archetype name → (pair_key, is_first_member)
# is_first_member=True → gets discriminating phrases at ~25 %
_PAIR_MEMBERSHIP: Dict[str, Tuple[str, bool]] = {
    "acs_stemi":             ("chest_pain",  True),
    "chest_pain_acidity":    ("chest_pain",  False),
    "severe_asthma_attack":  ("breathless",  True),
    "breathlessness_anxiety":("breathless",  False),
    "stroke_major":          ("headache",    True),
    "migraine":              ("headache",    False),
    "acute_abdomen":         ("abdominal",   True),
    "gastritis_gerd":        ("abdominal",   False),
    "sepsis_severe":         ("fever_adult", True),
    "fever_adult_mild":      ("fever_adult", False),
}

# ─────────────────────────────────────────────────────────────────────────────
# FIX-3c  Shared noise pools (all archetypes)
# ─────────────────────────────────────────────────────────────────────────────

_IRRELEVANT_COMPLAINTS: List[str] = [
    "also knee pain",
    "also back pain since few days",
    "also mild cough",
    "also shoulder discomfort",
    "also leg pain",
    "also body pain mild",
    "also slight headache",
    "aur ghutne mein bhi thoda dard",
    "mariyu kadupu lo koddiga discomfort",
    "aur peeth mein bhi thoda",
    "plus mild shoulder pain",
    "also some body ache",
]

_NEAR_EMPTY_TEXTS: List[str] = [
    "not feeling well",
    "unwell since morning",
    "discomfort",
    "thoda problem hai",
    "not good",
    "check karna tha",
    "thoda theek nahi lag raha",
    "koddiga ok ga ledu",
    "problem ga feel avutundi",
    "feeling off",
]

# ─────────────────────────────────────────────────────────────────────────────
# FIX-2  Age/sex-based condition prevalence  (all  # PLACEHOLDER)
# ─────────────────────────────────────────────────────────────────────────────
# Each entry: (condition, age_min, age_max, sex_constraint, probability)
# sex_constraint: "any" | "F" | "M"
# Probability is per-patient independent Bernoulli draw within the eligible
# age/sex window.  These are  # PLACEHOLDER pending validated epidemiology data.

_COND_PREVALENCE: List[Tuple[str, int, int, str, float]] = [
    ("hypertension",             50, 85, "any", 0.28),  # PLACEHOLDER
    ("diabetes mellitus",        40, 85, "any", 0.20),  # PLACEHOLDER
    ("dyslipidaemia",            45, 85, "any", 0.15),  # PLACEHOLDER
    ("hypothyroidism",           35, 85, "F",   0.14),  # PLACEHOLDER
    ("bronchial asthma",          5, 65, "any", 0.05),  # PLACEHOLDER
    ("GERD",                     25, 75, "any", 0.10),  # PLACEHOLDER
    ("osteoporosis",             60, 85, "F",   0.15),  # PLACEHOLDER
    ("rheumatoid arthritis",     30, 75, "F",   0.05),  # PLACEHOLDER
    ("anxiety disorder",         18, 55, "any", 0.05),  # PLACEHOLDER
    ("coronary artery disease",  55, 85, "any", 0.12),  # PLACEHOLDER
    ("chronic sinusitis",         5, 65, "any", 0.04),  # PLACEHOLDER
    ("atopic dermatitis",         5, 35, "any", 0.04),  # PLACEHOLDER
    ("allergic rhinitis",         5, 55, "any", 0.06),  # PLACEHOLDER
    ("glaucoma",                 55, 85, "any", 0.04),  # PLACEHOLDER
]

# Medication pools keyed by condition  # PLACEHOLDER
_COND_MEDICATIONS: Dict[str, List[str]] = {
    "hypertension":            ["amlodipine 5mg", "atenolol 50mg", "telmisartan 40mg"],
    "diabetes mellitus":       ["metformin 500mg", "glipizide 5mg"],
    "dyslipidaemia":           ["atorvastatin 20mg", "rosuvastatin 10mg"],
    "hypothyroidism":          ["levothyroxine 50mcg"],
    "bronchial asthma":        ["salbutamol inhaler", "budesonide inhaler"],
    "GERD":                    ["omeprazole 20mg", "pantoprazole 40mg"],
    "osteoporosis":            ["calcium + vitamin D"],
    "rheumatoid arthritis":    ["methotrexate 7.5mg", "hydroxychloroquine 200mg"],
    "anxiety disorder":        ["alprazolam 0.25mg", "sertraline 50mg"],
    "coronary artery disease": ["aspirin 75mg", "clopidogrel 75mg", "atorvastatin 40mg"],
    "chronic sinusitis":       ["cetirizine 10mg", "nasal drops"],
    "atopic dermatitis":       ["betamethasone cream", "cetirizine 10mg"],
    "allergic rhinitis":       ["cetirizine 10mg", "loratadine 10mg"],
    "glaucoma":                ["timolol eye drops", "latanoprost eye drops"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Style sampling weights (English, Telugu, Hindi, code-mixed)  # PLACEHOLDER
# ─────────────────────────────────────────────────────────────────────────────

_STYLE_WEIGHTS = [0.40, 0.25, 0.20, 0.15]   # PLACEHOLDER
_STYLE_KEYS    = ["english", "telugu", "hindi", "code_mixed"]


# ─────────────────────────────────────────────────────────────────────────────
# YAML loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> Any:
    """Load and return parsed YAML from *path*."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_all() -> Tuple[Dict, List[Dict], Dict]:
    """Return (config_dict, archetypes_list, lexicon_dict)."""
    cfg  = _load_yaml(_CFG_PATH)
    arch = _load_yaml(_ARCH_PATH)["archetypes"]
    lex  = _load_yaml(_LEX_PATH)
    return cfg, arch, lex


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rng_choice(rng: np.random.Generator, pool: list) -> Any:
    """Choose one item from *pool* uniformly."""
    idx = int(rng.integers(0, len(pool)))
    return pool[idx]


def _choose_style(rng: np.random.Generator) -> str:
    """Sample a complaint language style."""
    return _STYLE_KEYS[int(rng.choice(len(_STYLE_KEYS), p=_STYLE_WEIGHTS))]


def _style_lang_key(style: str) -> str:
    """Map style name → key suffix used in _PAIR_DATA."""
    if style == "hindi":
        return "hin"
    elif style in ("telugu", "code_mixed"):
        return "tel"
    return "en"


# ─────────────────────────────────────────────────────────────────────────────
# Step B – vital sampling  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

def _sample_vitals(
    arch: Dict,
    age: int,
    rng: np.random.Generator,
) -> Dict[str, Optional[float]]:
    """
    Draw one set of clean vitals from the archetype's Gaussian distributions.
    All mean/sd values and bounds are  # PLACEHOLDER.
    """
    v = arch["vitals"]

    def g(key: str) -> float:
        return float(rng.normal(v[key]["mean"], v[key]["sd"]))

    bp_sys = g("bp_sys")
    bp_dia = g("bp_dia")
    pulse  = g("pulse")
    temp_c = g("temp_c")
    spo2   = g("spo2")

    if age < 12:                       # children  # PLACEHOLDER
        pulse  *= float(rng.uniform(1.10, 1.30))
        bp_sys -= float(rng.uniform(10, 20))
        bp_dia -= float(rng.uniform(5,  12))
    elif age > 65:                     # elderly   # PLACEHOLDER
        bp_sys += float(rng.uniform(5, 15))
        bp_dia -= float(rng.uniform(0,  8))

    # Hard physiological bounds  # PLACEHOLDER
    bp_sys = min(max(bp_sys,  40.0), 250.0)
    bp_dia = min(max(bp_dia,  20.0), 150.0)
    pulse  = min(max(pulse,   20.0), 220.0)
    temp_c = min(max(temp_c,  34.0),  42.5)
    spo2   = min(max(spo2,    50.0), 100.0)

    return {
        "bp_sys": round(bp_sys),
        "bp_dia": round(bp_dia),
        "pulse":  round(pulse),
        "temp_c": round(temp_c, 1),
        "spo2":   round(spo2,   1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step C (v2) – complaint generation with shared pools
# ─────────────────────────────────────────────────────────────────────────────

def _add_text_noise(text: str, rng: np.random.Generator) -> str:
    """
    Add realistic free-text noise per word (~8 % probability).   # PLACEHOLDER
    Also inserts a filler word with 10 % probability.            # PLACEHOLDER
    """
    words  = text.split()
    result: List[str] = []

    for word in words:
        if len(word) > 3 and rng.random() < 0.08:    # PLACEHOLDER
            ntype = int(rng.integers(0, 3))
            chars = list(word)
            if ntype == 0:                             # typo
                idx = int(rng.integers(1, len(chars)))
                chars[idx] = chr(int(rng.integers(ord("a"), ord("z") + 1)))
                word = "".join(chars)
            elif ntype == 1:                           # drop vowel
                vi = [i for i, c in enumerate(chars) if c in "aeiou" and i > 0]
                if vi:
                    chars.pop(int(_rng_choice(rng, vi)))
                    word = "".join(chars)
            else:                                      # repeat letter
                idx = int(rng.integers(0, len(chars)))
                word = word[:idx] + word[idx] + word[idx:]
        result.append(word)

    fillers = ["actually", "kind of", "like", "you know", "basically", "also"]
    if rng.random() < 0.10:                           # PLACEHOLDER
        pos = int(rng.integers(0, max(1, len(result))))
        result.insert(pos, _rng_choice(rng, fillers))

    return " ".join(result)


def _generate_complaint(
    arch: Dict,
    lex: Dict,
    rng: np.random.Generator,
    bias: str = "none",           # "none" | "under_report" | "over_report"
) -> str:
    """
    Generate one free-text chief complaint.

    FIX-3 changes:
    • 5 % near-empty text across all archetypes.               # PLACEHOLDER
    • Confusable-pair archetypes draw from a SHARED pool
      (70–80 % of their complaints are identical to partner).  # PLACEHOLDER
    • Emergency (first-member) archetype gets a discriminating
      phrase with ~25 % probability (suppressed for under_report). # PLACEHOLDER
    • Over-reporting non-emergency: adds emergency discriminating phrase.
    • 20 % chance of an irrelevant extra complaint appended.   # PLACEHOLDER
    • Duration suffix appended with 40 % probability.         # PLACEHOLDER
    """
    arch_name = arch["name"]

    # ── 5 % near-empty  # PLACEHOLDER ──────────────────────────────────────
    if rng.random() < 0.05 and bias == "none":        # PLACEHOLDER
        text = _rng_choice(rng, _NEAR_EMPTY_TEXTS)
        return _add_text_noise(text, rng)

    # ── Choose language style ───────────────────────────────────────────────
    style    = _choose_style(rng)
    lang_key = _style_lang_key(style)

    # ── Confusable pair path ────────────────────────────────────────────────
    pair_info = _PAIR_MEMBERSHIP.get(arch_name)

    if pair_info is not None:
        pair_key, is_first = pair_info
        pair = _PAIR_DATA[pair_key]

        # Shared pool (same phrases for BOTH archetypes in the pair)
        shared = pair.get(f"shared_{lang_key}", pair["shared_en"])
        text   = _rng_choice(rng, shared)

        # Discriminating phrase logic
        add_discrim = False
        if is_first and bias != "under_report" and rng.random() < 0.25:   # PLACEHOLDER
            add_discrim = True
        elif not is_first and bias == "over_report" and rng.random() < 0.40:  # PLACEHOLDER
            add_discrim = True

        if add_discrim:
            discrim = pair.get(f"discrim_{lang_key}", pair["discrim_en"])
            if discrim:
                text = text + " " + _rng_choice(rng, discrim)

    else:
        # ── Non-confusable archetype path ───────────────────────────────────
        arch_lex = lex.get("archetypes", {}).get(arch_name, {})

        if style == "english":
            pool = arch.get("complaint_templates_en", ["discomfort"])
        else:
            pool = arch_lex.get(style, []) or arch.get("complaint_templates_en", ["discomfort"])

        text = _rng_choice(rng, pool)

    # ── Duration suffix (~40 %)  # PLACEHOLDER ─────────────────────────────
    if rng.random() < 0.40:                            # PLACEHOLDER
        dur_style = _rng_choice(rng, ["telugu", "hindi", "english"])
        dur_pool  = lex.get("duration", {}).get(dur_style, [])
        if dur_pool:
            text = text + " " + _rng_choice(rng, dur_pool)

    # ── Irrelevant extra complaint (~20 %)  # PLACEHOLDER ──────────────────
    if rng.random() < 0.20:                            # PLACEHOLDER
        text = text + " " + _rng_choice(rng, _IRRELEVANT_COMPLAINTS)

    return _add_text_noise(text, rng)


# ─────────────────────────────────────────────────────────────────────────────
# Step D (v2) – reporting bias
# ─────────────────────────────────────────────────────────────────────────────

def _apply_reporting_bias(
    arch: Dict,
    lex: Dict,
    is_emergency: bool,
    clean_level: int,
    cfg: Dict,
    rng: np.random.Generator,
) -> Tuple[str, str]:
    """
    Sample a complaint WITH the appropriate bias baked in.
    Returns (complaint_text, bias_label).

    bias_label ∈ {"none", "under_report", "over_report"}.
    Rates are  # PLACEHOLDER.

    FIX-3b changes:
    • under_report → complaint drawn from SHARED pool, discriminating phrase
      suppressed.  No separate "mild phrases" pool used.
    • over_report  → non-emergency complaint drawn with discriminating phrase
      appended (identical surface form to emergency partner).
    """
    under_rate = float(cfg["bias_rates"]["emergency_under_report"])   # PLACEHOLDER
    over_rate  = float(cfg["bias_rates"]["mild_over_report"])          # PLACEHOLDER

    if is_emergency and rng.random() < under_rate:
        return _generate_complaint(arch, lex, rng, bias="under_report"), "under_report"

    if not is_emergency and clean_level >= 4 and rng.random() < over_rate:
        return _generate_complaint(arch, lex, rng, bias="over_report"), "over_report"

    return _generate_complaint(arch, lex, rng, bias="none"), "none"


# ─────────────────────────────────────────────────────────────────────────────
# Step E – vital error injection  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

def _inject_vital_errors(
    row: Dict[str, Any],
    rates: Dict[str, float],
    rng: np.random.Generator,
) -> Tuple[Dict[str, Any], List[str]]:
    """Inject realistic data-entry errors into the vitals dict.
    All rates are  # PLACEHOLDER."""
    errors: List[str] = []
    r = dict(row)

    if rng.random() < float(rates["bp_zero"]):         # PLACEHOLDER
        field = _rng_choice(rng, ["bp_sys", "bp_dia"])
        r[field] = 0
        errors.append(f"bp_zero:{field}")

    if rng.random() < float(rates["bp_swapped"]):      # PLACEHOLDER
        r["bp_sys"], r["bp_dia"] = r["bp_dia"], r["bp_sys"]
        errors.append("bp_swapped")

    if rng.random() < float(rates["temp_fahrenheit"]): # PLACEHOLDER
        tc = r.get("temp_c") or 37.0
        r["temp_c"] = round(float(tc) * 9.0 / 5.0 + 32.0, 1)
        errors.append("temp_fahrenheit")

    if rng.random() < float(rates["spo2_invalid"]):    # PLACEHOLDER
        r["spo2"] = float(_rng_choice(rng, [0.0, round(float(rng.uniform(101, 110)), 1)]))
        errors.append("spo2_invalid")

    if rng.random() < float(rates["pulse_zero"]):      # PLACEHOLDER
        r["pulse"] = 0
        errors.append("pulse_zero")

    if rng.random() < float(rates["decimal_slip"]):    # PLACEHOLDER
        field = _rng_choice(rng, ["bp_sys", "bp_dia", "pulse"])
        val   = r.get(field)
        if val and float(val) > 20:
            r[field] = round(float(val) / 10.0, 1)
            errors.append(f"decimal_slip:{field}")

    if rng.random() < float(rates["vitals_missing"]):  # PLACEHOLDER
        n_miss = int(rng.integers(1, 3))
        chosen = list(rng.choice(
            ["bp_sys", "bp_dia", "pulse", "temp_c", "spo2"],
            size=n_miss, replace=False,
        ))
        for f in chosen:
            r[f] = None
            errors.append(f"missing:{f}")

    return r, errors


# ─────────────────────────────────────────────────────────────────────────────
# Step F – label noise  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_label_noise(
    clean_level: int,
    is_emergency: bool,
    rate: float,
    rng: np.random.Generator,
) -> int:
    """Apply ±1 label noise on non-emergency rows.  # PLACEHOLDER (rate, clamp)."""
    if is_emergency or rng.random() >= rate:
        return clean_level
    delta = int(rng.choice([-1, 1]))
    noisy = clean_level + delta
    noisy = max(noisy, 3)   # non-emergency floor  # PLACEHOLDER
    noisy = min(noisy, 5)
    return noisy


# ─────────────────────────────────────────────────────────────────────────────
# Step G – arrival time  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_arrival_time(
    day_offset: int,
    cfg: Dict,
    rng: np.random.Generator,
) -> str:
    """
    Generate a realistic arrival datetime (three-component mixture).
    All parameters are  # PLACEHOLDER.
    """
    ac   = cfg["arrival"]
    base = (
        datetime.strptime(ac["start_date"], "%Y-%m-%d")
        + timedelta(days=int(day_offset))
    )
    w = [float(ac["peak1_weight"]),
         float(ac["peak2_weight"]),
         float(ac["uniform_weight"])]
    c = int(rng.choice(3, p=w))
    if c == 0:
        hour_frac = float(rng.normal(ac["peak1_hour"], ac["peak1_sd"]))    # PLACEHOLDER
    elif c == 1:
        hour_frac = float(rng.normal(ac["peak2_hour"], ac["peak2_sd"]))    # PLACEHOLDER
    else:
        hour_frac = float(rng.uniform(float(ac["open_hour"]), float(ac["close_hour"])))
    hour_frac = min(max(hour_frac, float(ac["open_hour"])), float(ac["close_hour"]))
    hour   = int(hour_frac)
    minute = int((hour_frac - hour) * 60)
    second = int(rng.integers(0, 60))
    return base.replace(hour=hour, minute=minute, second=second).strftime("%Y-%m-%dT%H:%M:%S")


# ─────────────────────────────────────────────────────────────────────────────
# FIX-2  Conditions / medications sampling (age/sex-based, archetype-independent)
# ─────────────────────────────────────────────────────────────────────────────

def _sample_conditions_meds(
    age: int,
    sex: str,
    arch_name: str,
    rng: np.random.Generator,
) -> Tuple[str, str]:
    """
    Sample known_conditions and medications from an age/sex prevalence table.

    Changes from v1:
    • NOT keyed by department or archetype (eliminates department proxy leakage).
    • "not known" rate ~8 % across all archetypes, except
      unconscious_unresponsive which gets ~35 % (realistic: patient can't speak).
      # PLACEHOLDER – both rates.
    • Individual conditions sampled independently via Bernoulli draws.
    • Medications derived from sampled conditions (one per condition).
    """
    # "not known" gate  # PLACEHOLDER
    not_known_rate = 0.35 if arch_name == "unconscious_unresponsive" else 0.08
    if rng.random() < not_known_rate:
        return "not known", "not known"

    # Collect eligible conditions
    eligible: List[str] = []
    for cond, age_min, age_max, sex_req, prob in _COND_PREVALENCE:
        if age_min <= age <= age_max:
            if sex_req == "any" or sex_req == sex:
                if rng.random() < prob:            # PLACEHOLDER
                    eligible.append(cond)

    if not eligible:
        return "none", "none"

    # Limit to 1–2 conditions  # PLACEHOLDER
    n = min(len(eligible), int(rng.integers(1, 3)))
    selected = list(rng.choice(eligible, size=n, replace=False))

    # Derive one medication per condition
    meds: List[str] = []
    for cond in selected:
        pool = _COND_MEDICATIONS.get(cond, [])
        if pool:
            meds.append(_rng_choice(rng, pool))

    return "; ".join(selected), "; ".join(meds) if meds else "none"


# ─────────────────────────────────────────────────────────────────────────────
# Stratified split assignment  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

def _make_stratified_splits(
    n_total: int,
    is_emergency: np.ndarray,
    rng: np.random.Generator,
    cfg: Dict,
) -> np.ndarray:
    """
    Assign train/calib/val/test split labels, stratified by is_emergency.
    Calibration split guaranteed ≥ min_calib_emergencies.
    # PLACEHOLDER – split fractions.
    """
    fracs        = cfg["splits"]
    min_calib_em = int(cfg["min_calib_emergencies"])

    em_idx  = np.where(is_emergency)[0]
    non_idx = np.where(~is_emergency)[0]

    def _assign(indices: np.ndarray) -> np.ndarray:
        n      = len(indices)
        keys   = list(fracs.keys())
        counts = []
        remaining = n
        for k in keys[:-1]:
            c = round(float(fracs[k]) * n)
            counts.append(c)
            remaining -= c
        counts.append(remaining)
        labels: np.ndarray = np.empty(n, dtype=object)
        cursor = 0
        for label, count in zip(keys, counts):
            labels[cursor: cursor + count] = label
            cursor += count
        return labels

    rng.shuffle(em_idx)
    rng.shuffle(non_idx)
    em_labels  = _assign(em_idx)
    non_labels = _assign(non_idx)

    # Enforce minimum calib emergencies
    n_calib_em = int((em_labels == "calib").sum())
    if n_calib_em < min_calib_em:
        shortage    = min_calib_em - n_calib_em
        train_pos   = np.where(em_labels == "train")[0]
        rng.shuffle(train_pos)
        em_labels[train_pos[:shortage]] = "calib"

    splits                = np.empty(n_total, dtype=object)
    splits[em_idx]        = em_labels
    splits[non_idx]       = non_labels
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(
    df_intake: pd.DataFrame,
    df_labels: pd.DataFrame,
    df_splits: pd.DataFrame,
) -> None:
    n   = len(df_labels)
    sep = "─" * 60

    print(f"\n{sep}")
    print("  TriageGuard  –  Data Generator Summary  (v2 hardened)")
    print(sep)

    print("\n  CLEAN triage level mix:")
    print(f"  {'Level':<8} {'Count':>6} {'Actual %':>10} {'Target %':>10}")
    targets = {1: 0.5, 2: 3.5, 3: 25.0, 4: 40.0, 5: 31.0}
    for lvl in [1, 2, 3, 4, 5]:
        cnt = int((df_labels["clean_triage_level"] == lvl).sum())
        pct = cnt / n * 100
        print(f"  L{lvl}       {cnt:>6}    {pct:>8.2f} %  {targets[lvl]:>8.1f} %")

    em_rate = df_labels["is_emergency"].mean() * 100
    print(f"\n  Emergency rate: {em_rate:.2f} %  [target 3.5–4.5 %]")

    print("\n  Error injection rates:")
    err_col = df_labels["injected_errors"]
    for code, tgt in [("bp_zero", 1.5), ("bp_swapped", 0.7), ("temp_fahrenheit", 2.0),
                      ("spo2_invalid", 0.8), ("pulse_zero", 0.5),
                      ("decimal_slip", 0.5), ("missing", 4.0)]:
        actual = err_col.str.contains(code, na=False).sum() / n * 100
        print(f"    {code:<20} {actual:>5.2f}%  (target ~{tgt:.1f}%)")

    print("\n  Split sizes:")
    merged = df_splits.merge(df_labels[["patient_id", "is_emergency"]], on="patient_id")
    for spl in ["train", "calib", "val", "test"]:
        sub    = merged[merged["split"] == spl]
        em_cnt = int(sub["is_emergency"].sum())
        print(f"    {spl:<8} {len(sub):>6} rows   {em_cnt:>4} emergencies")

    # 20 sample rows
    em_pids     = df_labels.loc[df_labels["is_emergency"], "patient_id"]
    non_em_pids = df_labels.loc[~df_labels["is_emergency"], "patient_id"]
    sample_pids = (
        list(em_pids.sample(min(5, len(em_pids)), random_state=42).values)
        + list(non_em_pids.sample(min(15, len(non_em_pids)), random_state=42).values)
    )
    sample = (
        df_intake[df_intake["patient_id"].isin(sample_pids)]
        .merge(
            df_labels[["patient_id", "triage_level", "clean_triage_level",
                        "department", "is_emergency", "archetype",
                        "reporting_bias", "injected_errors"]],
            on="patient_id",
        )
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    display_cols = [
        "patient_id", "age", "sex",
        "bp_sys", "bp_dia", "pulse", "temp_c", "spo2",
        "complaint_text",
        "triage_level", "department", "is_emergency",
        "archetype", "reporting_bias", "injected_errors",
    ]
    print(f"\n  20 sample rows:")
    print(sep)
    with pd.option_context(
        "display.max_columns", None, "display.width", 200, "display.max_colwidth", 55
    ):
        print(sample[display_cols].to_string(index=False))
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Main generation function
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    n: int    = 30_000,
    seed: int = 42,
    out: str  = "data",
) -> None:
    """
    Generate the full synthetic dataset and write CSV files to *out/*.

    FIX-1: patient_id is assigned AFTER the global shuffle so the sequential
    index carries no triage-level information.
    """
    rng = np.random.default_rng(seed)

    cfg, archetypes, lex = _load_all()

    level_counts: Dict[int, int] = {int(k): int(v)
                                    for k, v in cfg["level_counts"].items()}
    assert sum(level_counts.values()) == n, (
        f"level_counts sum to {sum(level_counts.values())}, expected {n}"
    )

    level_arch: Dict[int, List[Dict]] = {}
    for arch in archetypes:
        lvl = int(arch["triage_level"])
        level_arch.setdefault(lvl, []).append(arch)

    noise_rate = float(cfg["label_noise"]["rate"])    # PLACEHOLDER
    n_per_day  = int(cfg["arrival"]["n_per_day"])
    total_days = math.ceil(n / n_per_day)

    # ── Build rows WITHOUT patient_id ────────────────────────────────────────
    # FIX-1: IDs are assigned AFTER shuffling, so sequential order carries no
    #         information about the triage level.
    rows_intake: List[Dict] = []
    rows_labels: List[Dict] = []

    for level in sorted(level_counts.keys()):
        target_count = level_counts[level]
        arch_list    = level_arch[level]
        is_emerg     = level in {1, 2}

        raw_priors = np.array([float(a["within_level_prior"]) for a in arch_list])
        priors     = raw_priors / raw_priors.sum()   # PLACEHOLDER

        arch_row_counts = rng.multinomial(target_count, priors)

        for arch, row_count in zip(arch_list, arch_row_counts):
            dept = arch["department"]

            for _ in range(int(row_count)):

                # Demography
                age_min = int(arch.get("age_min", 0))
                age_max = int(arch.get("age_max", 85))
                age     = int(rng.integers(age_min, age_max + 1))

                sex_c = arch.get("sex", "any")
                if sex_c == "F":
                    sex = "F"
                elif sex_c == "M":
                    sex = "M"
                else:
                    sex = "F" if rng.random() < 0.48 else "M"   # PLACEHOLDER

                # Vitals
                vitals = _sample_vitals(arch, age, rng)

                # FIX-2: conditions/meds from age/sex prevalence (not department pool)
                known_cond, meds = _sample_conditions_meds(age, sex, arch["name"], rng)

                # Arrival time
                day_offset   = int(rng.integers(0, total_days))
                arrival_time = _generate_arrival_time(day_offset, cfg, rng)

                # FIX-3: complaint + bias unified in _apply_reporting_bias
                complaint, bias_label = _apply_reporting_bias(
                    arch, lex, is_emerg, level, cfg, rng
                )

                # Vital errors
                vitals_out, error_list = _inject_vital_errors(
                    vitals, cfg["error_rates"], rng
                )

                # Label noise
                noisy_level = _apply_label_noise(level, is_emerg, noise_rate, rng)

                # Intake row (no patient_id yet; placeholder "")
                rows_intake.append({
                    "patient_id":       "",       # filled after shuffle
                    "age":              age,
                    "sex":              sex,
                    "bp_sys":           vitals_out["bp_sys"],
                    "bp_dia":           vitals_out["bp_dia"],
                    "pulse":            vitals_out["pulse"],
                    "temp_c":           vitals_out["temp_c"],
                    "spo2":             vitals_out["spo2"],
                    "complaint_text":   complaint,
                    "known_conditions": known_cond,
                    "medications":      meds,
                    "arrival_time":     arrival_time,
                })

                rows_labels.append({
                    "patient_id":         "",     # filled after shuffle
                    "triage_level":       noisy_level,
                    "clean_triage_level": level,
                    "department":         dept,
                    "is_emergency":       is_emerg,
                    "archetype":          arch["name"],
                    "reporting_bias":     bias_label,
                    "injected_errors":    "|".join(error_list) if error_list else "none",
                })

    # ── FIX-1: SHUFFLE first, THEN assign patient_id ─────────────────────────
    perm        = rng.permutation(n)
    rows_intake = [rows_intake[i] for i in perm]
    rows_labels = [rows_labels[i] for i in perm]

    for idx, (ri, rl) in enumerate(zip(rows_intake, rows_labels)):
        pid = f"P{idx + 1:06d}"
        ri["patient_id"] = pid
        rl["patient_id"] = pid

    df_intake = pd.DataFrame(rows_intake)
    df_labels = pd.DataFrame(rows_labels)

    # ── Stratified splits ─────────────────────────────────────────────────────
    is_emerg_arr = df_labels["is_emergency"].to_numpy(dtype=bool)
    split_labels = _make_stratified_splits(n, is_emerg_arr, rng, cfg)
    df_splits    = pd.DataFrame({
        "patient_id": df_intake["patient_id"].values,
        "split":      split_labels,
    })

    # ── Write outputs ─────────────────────────────────────────────────────────
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_intake.to_csv(out_dir / "synthetic_intake.csv",  index=False)
    df_labels.to_csv(out_dir / "hidden_labels.csv",     index=False)
    df_splits.to_csv(out_dir / "splits.csv",            index=False)

    _print_summary(df_intake, df_labels, df_splits)

    print(f"\n  Files written to {out_dir.resolve()}/")
    print(f"    synthetic_intake.csv  ({len(df_intake)} rows)")
    print(f"    hidden_labels.csv     ({len(df_labels)} rows)")
    print(f"    splits.csv            ({len(df_splits)} rows)\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """python -m triageguard.data_gen [--n N] [--seed S] [--out DIR]"""
    parser = argparse.ArgumentParser(
        description="TriageGuard synthetic data generator (v2 hardened)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n",    type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out",  type=str, default="data")
    args = parser.parse_args()
    generate(n=args.n, seed=args.seed, out=args.out)


if __name__ == "__main__":
    main()
