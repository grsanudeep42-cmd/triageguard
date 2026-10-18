"""
triageguard/rules.py
─────────────────────────────────────────────────────────────────────────────
Phase 3 – Hard-coded escalation layer for TriageGuard.

ESCALATION CONTRACT
--------------------
This module can ONLY escalate (assign a more-urgent / lower numeric level).
It never downgrades.
The fusion layer (Phase 4) takes  min(model_level, rules_result.forced_level).

INPUTS
-------
• CleanResult  from triageguard.cleaning
• age          (integer, years)
• Optional overrides for rules_config and redflags_config (for testing)

OUTPUTS
--------
RuleResult dataclass:
  forced_level       – most urgent level triggered, or None
  triggered_rule_ids – list of all rule IDs that fired
  red_flags_detected – list of red-flag names that matched
  reasons            – human-readable reason strings
  is_uncertain       – True if any triggered rule used an uncertain vital

VITAL RULE EVALUATION
----------------------
Each rule in rules.yaml specifies a 'conditions' dict:
  key format: <vital>_<op>   e.g. spo2_lt, bp_sys_lt, pulse_gt, temp_c_ge
  op: lt (<), le (≤), gt (>), ge (≥)
All conditions in a rule must be True (AND logic) for the rule to fire.
A condition whose vital is None (missing/invalid) causes the rule NOT to fire.
Age filters (age_min, age_max) apply when present.

RED-FLAG TEXT DETECTION
------------------------
Patterns from redflags.yaml are matched against the cleaned complaint text
using fuzzy word-level Levenshtein matching (max_dist from config).
Red flags escalate to AT MOST level 2, never to level 1.

ISOLATION RULE
---------------
This module must NOT import triageguard.data_gen or read hidden_labels.csv.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from triageguard.cleaning import CleanResult

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_ROOT        = Path(__file__).resolve().parent.parent
_RULES_PATH  = _ROOT / "rules.yaml"
_FLAGS_PATH  = _ROOT / "redflags.yaml"

# ─────────────────────────────────────────────────────────────────────────────
# RuleResult dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RuleResult:
    """Output of apply_rules()."""
    forced_level:          Optional[int] = None   # None = no vital rule triggered (vital rules ONLY)
    triggered_rule_ids:    List[str]     = field(default_factory=list) # vital rule IDs
    red_flags_detected:    List[str]     = field(default_factory=list) # red flag names
    no_downgrade_below_L3: bool          = False  # True if any red flag was detected (soft guard)
    abstain_reason:        Optional[str] = None   # "remeasure" if needs_remeasure and no forced level
    reasons:               List[str]     = field(default_factory=list)
    is_uncertain:          bool          = False  # True if uncertain vital used


# ─────────────────────────────────────────────────────────────────────────────
# YAML loaders (module-level lazy cache)
# ─────────────────────────────────────────────────────────────────────────────

_rules_cache:    Optional[Dict] = None
_redflags_cache: Optional[Dict] = None


def _load_rules_cfg() -> Dict:
    global _rules_cache
    if _rules_cache is None:
        with open(_RULES_PATH, encoding="utf-8") as fh:
            _rules_cache = yaml.safe_load(fh)
    return _rules_cache


def _load_redflags_cfg() -> Dict:
    global _redflags_cache
    if _redflags_cache is None:
        with open(_FLAGS_PATH, encoding="utf-8") as fh:
            _redflags_cache = yaml.safe_load(fh)
    return _redflags_cache


# ─────────────────────────────────────────────────────────────────────────────
# Vital accessor
# ─────────────────────────────────────────────────────────────────────────────

_VITAL_ATTRS: Dict[str, str] = {
    "bp_sys":  "bp_sys",
    "bp_dia":  "bp_dia",
    "pulse":   "pulse",
    "temp_c":  "temp_c",
    "spo2":    "spo2",
}

_UNCERTAIN_ATTRS: Dict[str, str] = {
    "bp_sys":  "bp_sys_uncertain",
    "bp_dia":  "bp_dia_uncertain",
    "temp_c":  "temp_c_uncertain",
}


def _get_vital(cleaned: CleanResult, name: str) -> Optional[float]:
    """Return the cleaned vital value (None if missing / invalid)."""
    return getattr(cleaned, _VITAL_ATTRS[name], None)


def _is_uncertain(cleaned: CleanResult, name: str) -> bool:
    """Return True if the vital has the uncertain flag set."""
    attr = _UNCERTAIN_ATTRS.get(name)
    return bool(getattr(cleaned, attr, False)) if attr else False


# ─────────────────────────────────────────────────────────────────────────────
# Condition evaluator
# ─────────────────────────────────────────────────────────────────────────────

_OPS = {
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
}


def _evaluate_condition(
    cond_key: str,
    cond_val: float,
    cleaned: CleanResult,
    age: int,
) -> Tuple[bool, bool]:
    """
    Evaluate one condition clause.

    Returns (fires: bool, used_uncertain_vital: bool).
    Returns (False, False) if the vital is None (missing / invalid).
    """
    # Split on the LAST underscore: e.g. "bp_sys_lt" → ("bp_sys", "lt")
    parts = cond_key.rsplit("_", 1)
    if len(parts) != 2:
        return (False, False)
    vital_name, op = parts

    if vital_name == "age":
        return (_OPS[op](age, cond_val), False)

    if vital_name not in _VITAL_ATTRS:
        return (False, False)

    value = _get_vital(cleaned, vital_name)
    if value is None:
        return (False, False)   # never fire on missing vitals

    op_fn = _OPS.get(op)
    if op_fn is None:
        return (False, False)

    fires     = op_fn(value, float(cond_val))
    uncertain = _is_uncertain(cleaned, vital_name)
    return (fires, uncertain)


def _evaluate_rule(
    rule: Dict,
    cleaned: CleanResult,
    age: int,
) -> Tuple[bool, bool]:
    """
    Evaluate all conditions in a rule (AND logic).

    Returns (fires: bool, any_uncertain: bool).
    """
    age_min = rule.get("age_min")
    age_max = rule.get("age_max")
    if age_min is not None and age < int(age_min):
        return (False, False)
    if age_max is not None and age > int(age_max):
        return (False, False)

    conditions = rule.get("conditions", {})
    any_uncertain = False

    for cond_key, cond_val in conditions.items():
        fires, uncertain = _evaluate_condition(cond_key, float(cond_val), cleaned, age)
        if not fires:
            return (False, False)
        if uncertain:
            any_uncertain = True

    return (True, any_uncertain)


# ─────────────────────────────────────────────────────────────────────────────
# Fuzzy text matching for red flags
# ─────────────────────────────────────────────────────────────────────────────

def _damerau_levenshtein(a: str, b: str) -> int:
    """Damerau-Levenshtein distance (insert, delete, substitute, transposition)."""
    m, n = len(a), len(b)
    if a == b:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,       # deletion
                d[i][j - 1] + 1,       # insertion
                d[i - 1][j - 1] + cost # substitution
            )
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)  # transposition

    return d[m][n]


_edit_distance = _damerau_levenshtein


def _max_allowed_dist(token: str) -> int:
    """
    Max edit distance:
    - 0 for words <= 3 chars (exact match to avoid false positive stopword matches)
    - 1 for words 4-5 chars
    - 2 for words > 5 chars
    """
    if len(token) <= 3:
        return 0
    elif len(token) <= 5:
        return 1
    return 2


def _fuzzy_phrase_match(
    phrase: str,
    text_words: List[str],
) -> bool:
    """
    Return True if *phrase* matches a contiguous sequence of words in *text_words*
    at token boundaries, with:
      - exact match for words <= 3 chars
      - max edit distance 1 for words <= 5 chars
      - max edit distance 2 for words > 5 chars
    """
    pattern_tokens = phrase.lower().split()
    k = len(pattern_tokens)
    if k == 0 or len(text_words) < k:
        return False

    for i in range(len(text_words) - k + 1):
        matched = True
        for j in range(k):
            p_tok = pattern_tokens[j]
            t_tok = text_words[i + j]
            if _damerau_levenshtein(p_tok, t_tok) > _max_allowed_dist(p_tok):
                matched = False
                break
        if matched:
            return True
    return False


def detect_red_flags(
    text: str,
    redflags_config: Optional[Dict] = None,
) -> Tuple[List[str], List[str]]:
    """
    Detect clinical red flags in cleaned complaint text using token-boundary fuzzy matching.

    Returns
    -------
    (flag_ids, flag_names)
      flag_ids   – list of RF ids that matched
      flag_names – list of flag names that matched
    """
    if redflags_config is None:
        redflags_config = _load_redflags_cfg()

    text_words = re.findall(r"\b[a-z0-9]+\b", text.lower())
    matched_ids   : List[str] = []
    matched_names : List[str] = []

    for rf in redflags_config.get("red_flags", []):
        rf_id    = rf["id"]
        rf_name  = rf["name"]

        # Collect all patterns across languages
        patterns: List[str] = []
        for lang_key in ("patterns_en", "patterns_tel", "patterns_hin"):
            patterns.extend(rf.get(lang_key, []))

        for pattern in patterns:
            if _fuzzy_phrase_match(pattern, text_words):
                matched_ids.append(rf_id)
                matched_names.append(rf_name)
                break   # each red flag fires at most once

    return (matched_ids, matched_names)


# ─────────────────────────────────────────────────────────────────────────────
# Main: apply_rules
# ─────────────────────────────────────────────────────────────────────────────

def apply_rules(
    cleaned: CleanResult,
    age: int,
    rules_config: Optional[Dict] = None,
    redflags_config: Optional[Dict] = None,
) -> RuleResult:
    """
    Evaluate all escalation rules (FORCED) and red-flag text patterns (SOFT).

    Parameters
    ----------
    cleaned         : CleanResult from triageguard.cleaning.clean_record()
    age             : patient age in years
    rules_config    : parsed rules.yaml (loaded automatically if None)
    redflags_config : parsed redflags.yaml (loaded automatically if None)

    Returns
    -------
    RuleResult
        forced_level          – most urgent level triggered by vital rules (1 or 2),
                                or None if no vital rule fired.
        triggered_rule_ids    – IDs of all vital rules that fired
        red_flags_detected    – names of red flags matched in text
        no_downgrade_below_L3 – True if any red flag was detected (soft guard)
        abstain_reason        – "remeasure" if needs_remeasure and no forced escalation
        reasons               – human-readable explanations
        is_uncertain          – any fired vital rule used an uncertain vital
    """
    if rules_config is None:
        rules_config = _load_rules_cfg()
    if redflags_config is None:
        redflags_config = _load_redflags_cfg()

    triggered_ids : List[str] = []
    reasons       : List[str] = []
    min_level     : int       = 999
    is_uncertain  : bool      = False

    # ── Vital rules (FORCED) ──────────────────────────────────────────────
    for rule in rules_config.get("rules", []):
        fires, uncertain = _evaluate_rule(rule, cleaned, age)
        if fires:
            rule_id   = rule["id"]
            rule_lvl  = int(rule["forced_max_level"])
            triggered_ids.append(rule_id)
            reasons.append(
                f"[{rule_id}] {rule['description'].strip()} "
                f"→ force level ≤ {rule_lvl}"
            )
            min_level    = min(min_level, rule_lvl)
            is_uncertain = is_uncertain or uncertain

    forced_level = min_level if min_level < 999 else None

    # ── Red-flag text rules (SOFT guard) ──────────────────────────────────
    # Red flags do NOT set forced_level. They populate red_flags_detected
    # and set no_downgrade_below_L3 = True for the downstream fusion layer.
    rf_ids, rf_names = detect_red_flags(
        cleaned.complaint_text, redflags_config
    )
    no_downgrade_below_L3 = bool(rf_names)
    if rf_names:
        for rf_id, rf_name in zip(rf_ids, rf_names):
            reasons.append(
                f"[{rf_id}] Red flag '{rf_name}' detected in complaint text "
                f"→ soft guard: no downgrade below L3"
            )

    # ── Abstain reason ────────────────────────────────────────────────────
    abstain_reason: Optional[str] = None
    if cleaned.needs_remeasure and forced_level is None:
        abstain_reason = "remeasure"
        reasons.append(
            "Abstain reason: remeasure (>= 2 missing/invalid vitals; "
            "no emergency vital rule triggered)"
        )

    return RuleResult(
        forced_level          = forced_level,
        triggered_rule_ids    = triggered_ids,
        red_flags_detected    = rf_names,
        no_downgrade_below_L3 = no_downgrade_below_L3,
        abstain_reason        = abstain_reason,
        reasons               = reasons,
        is_uncertain          = is_uncertain,
    )

