import csv
import re

import pandas as pd

from classification_core.text import parse_decimal_amount


def normalize_rule_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def normalize_match_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().upper())


def normalize_regex_pattern(pattern):
    pattern = str(pattern or "").strip()
    if pattern.startswith("(?i)"):
        return pattern[4:]
    if pattern.startswith("^(?i)"):
        return "^" + pattern[5:]
    return pattern


def parse_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        text = str(value).strip()
        return int(text) if text else default
    except (TypeError, ValueError):
        return default


def split_upper_terms(value, separator=";"):
    terms = []
    for term in str(value or "").split(separator):
        term = normalize_match_text(term)
        if term:
            terms.append(term)
    return terms


def load_rules(rules_file):
    keyword_rules = []
    regex_rules = []
    with open(rules_file, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            counterparty = row.get("counterparty")
            product_type = row.get("product_type", "")
            rule_type = (row.get("rule_type") or "keyword").strip().lower()
            if not counterparty:
                continue
            if rule_type == "regex":
                pattern = str(row.get("keyword", "")).strip()
                if pattern:
                    try:
                        regex_rules.append(
                            (re.compile(pattern, re.IGNORECASE), counterparty, product_type)
                        )
                    except re.error:
                        continue
            else:
                if row.get("keyword"):
                    for keyword in split_upper_terms(row["keyword"]):
                        keyword_rules.append((keyword, counterparty, product_type))
    return keyword_rules, regex_rules


def load_credit_card_rules(rules_file):
    """Load V2 regex-based credit card rules.

    Returns a dict with two tiers of compiled rules:
        "specific" — priority >= 90 (institution-specific)
        "generic"  — priority < 90  (catch-all with repayment signals)
    """
    specific = []
    generic = []

    with open(rules_file, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            pattern = str(row.get("keyword", "")).strip()
            if not pattern:
                continue
            priority = parse_int(row.get("priority"), default=0)

            try:
                compiled = re.compile(normalize_regex_pattern(pattern), re.IGNORECASE)
            except re.error:
                continue

            rule = {
                "priority": priority,
                "bank": normalize_rule_value(row.get("bank", "*")),
                "account_type": normalize_rule_value(row.get("account_type", "*")),
                "dr_cr": normalize_rule_value(row.get("dr_cr", "*")),
                "pattern": compiled,
                "counterparty": str(row.get("counterparty", "")).strip(),
                "product_type": str(row.get("product_type", "")).strip(),
            }

            if priority >= 90:
                specific.append(rule)
            else:
                generic.append(rule)

    specific.sort(key=lambda r: -r["priority"])
    generic.sort(key=lambda r: -r["priority"])

    return {"specific": specific, "generic": generic}


def clean_dataframe_columns(df):
    valid_columns = [
        column
        for column in df.columns
        if (str(column) or "").strip()
        and not str(column).startswith("Unnamed:")
    ]
    return df.loc[:, valid_columns].copy()


def apply_counterparty_rules(df, rules_file):
    """Apply counterparty keyword/regex rules using vectorised operations.

    The original implementation called ``match_text`` once per row via
    ``.map()``, which compiled a new regex inside every call and performed
    O(rows × rules) Python-level iterations.  This version achieves the
    identical result with vectorised ``str.contains()`` passes — one per
    keyword / regex rule — and an early-exit tracker so later rules only
    process still-unmatched rows.
    """
    keyword_rules, regex_rules = load_rules(rules_file)
    output = clean_dataframe_columns(df)
    text_col = output["text"].fillna("").astype(str)
    # Vectorised normalisation — collapse whitespace then uppercase.
    text_normalised = (
        text_col.str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.upper()
    )

    output["counterparty"] = ""
    output["product_type"] = ""
    already = pd.Series(False, index=output.index)

    # Keyword rules — processed in original priority order.
    for keyword, counterparty, product_type in keyword_rules:
        mask = (
            ~already
            & text_normalised.str.contains(
                r"(?<![A-Za-z])" + re.escape(keyword) + r"(?![A-Za-z])",
                na=False,
                regex=True,
            )
        )
        if not mask.any():
            continue
        output.loc[mask, "counterparty"] = counterparty
        output.loc[mask, "product_type"] = product_type
        already |= mask

    # Regex rules — compiled patterns, matched against the normalised text
    # (same as the original ``match_text`` behaviour).
    for pattern, counterparty, product_type in regex_rules:
        mask = (
            ~already
            & text_normalised.str.contains(
                pattern.pattern,
                na=False,
                regex=True,
                flags=re.IGNORECASE,
            )
        )
        if not mask.any():
            continue
        output.loc[mask, "counterparty"] = counterparty
        output.loc[mask, "product_type"] = product_type
        already |= mask

    return output


def apply_credit_card_rules(df, rules_file):
    """Apply V2 regex-based credit card rules to a DataFrame.

    Fully vectorised — no iterrows.  Processing order:
    1. Specific rules (priority >= 90) matched first.
    2. Generic rules (priority < 90) matched on remaining rows.

    All rules run in overwrite mode: when a rule matches, it replaces any
    existing counterparty/product_type.  Duplicate matching within the
    same tier is still prevented via the `already` tracker.
    """
    rules = load_credit_card_rules(rules_file)
    output = df.copy()
    text_col = output["text"].fillna("").astype(str)

    pt_col = output["product_type"].fillna("").astype(str).str.strip()
    already = pt_col.ne("")

    def _col_mask(rule):
        """Build a boolean mask for bank / account_type / dr_cr constraints."""
        mask = pd.Series(True, index=output.index)
        for field_name in ("bank", "account_type", "dr_cr"):
            rv = rule[field_name]
            if rv in ("*", "-"):
                continue
            col = output[field_name].fillna("").astype(str).str.strip().str.lower()
            mask &= col.eq(rv)
        return mask

    def _apply_tier(rule_list):
        nonlocal already
        for rule in rule_list:
            text_hit = text_col.str.contains(rule["pattern"], na=False, regex=True)
            mask = text_hit & _col_mask(rule) & ~already
            if not mask.any():
                continue
            output.loc[mask, "counterparty"] = rule["counterparty"]
            if rule["product_type"]:
                output.loc[mask, "product_type"] = rule["product_type"]
            already |= mask

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")
        _apply_tier(rules["specific"])
        _apply_tier(rules["generic"])

    return output


# ---------------------------------------------------------------------------
# Generic flag-rule matching
# ---------------------------------------------------------------------------


def _load_flag_rules(rules_file):
    """Load generic flag rules from a CSV file.

    Auto-detects two CSV formats:

    Format A (home_loan_car_loan):
        Columns: target_field, match_scope, match_type, pattern,
                 account_type, dr_cr, bank, amount_gt, priority
    Format B (overdrawn / debt_collection / debt_consolidation):
        Columns: keyword (or pattern), match_type, counterparty,
                 product_type (optional)

    Returns a dict keyed by match scope, each value a list of rule dicts.
    """
    with open(rules_file, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {}

    columns = set(rows[0].keys())

    # Format A: has target_field + match_scope columns
    if "target_field" in columns and "match_scope" in columns:
        return _load_flag_rules_format_a(rows)

    # Format B: simpler keyword/regex rules for single target
    return _load_flag_rules_format_b(rows)


def _load_flag_rules_format_a(rows):
    """Load format-A rules with match_scope / account_type / dr_cr / bank indexing."""
    indexed = {
        "text_keyword": [],
        "text_regex": [],
        "text_or_counterparty_keyword": [],
        "text_or_counterparty_regex": [],
        "all_rules": [],
    }

    for row in rows:
        target_field = normalize_rule_value(row.get("target_field", ""))
        match_scope = normalize_rule_value(row.get("match_scope", "")) or "text"
        match_type = normalize_rule_value(row.get("match_type", "")) or "keyword"
        pattern = str(row.get("pattern", "")).strip()
        enabled = str(row.get("enabled", "1")).strip()
        if not target_field or enabled == "0":
            continue
        if not pattern and match_scope != "all":
            continue

        priority = parse_int(row.get("priority"), default=0)
        amount_gt = parse_decimal_amount(row.get("amount_gt"))
        rule = {
            "target_field": target_field,
            "priority": priority,
            "amount_gt": amount_gt,
            "account_type": normalize_rule_value(row.get("account_type", "*")),
            "dr_cr": normalize_rule_value(row.get("dr_cr", "*")),
            "bank": normalize_rule_value(row.get("bank", "*")),
        }

        if match_scope == "all":
            if match_type == "keyword":
                rule["keywords"] = split_upper_terms(pattern)
            else:
                try:
                    rule["pattern"] = re.compile(normalize_regex_pattern(pattern), re.IGNORECASE)
                except re.error:
                    continue
            indexed["all_rules"].append(rule)
            continue

        bucket_key = f"{match_scope}_{match_type}"
        if bucket_key not in indexed:
            continue

        if match_type == "regex":
            try:
                rule["pattern"] = re.compile(normalize_regex_pattern(pattern), re.IGNORECASE)
            except re.error:
                continue
        else:
            keywords = split_upper_terms(pattern)
            if not keywords:
                continue
            rule["keywords"] = keywords

        indexed[bucket_key].append(rule)

    # Merge per-target keywords into a single compiled regex for each bucket key.
    for bucket_key in list(indexed.keys()):
        if "keyword" not in bucket_key:
            indexed[bucket_key].sort(key=lambda r: -r["priority"])
            continue
        by_target = {}
        for rule in indexed[bucket_key]:
            target = rule["target_field"]
            by_target.setdefault(target, []).extend(rule.get("keywords", []))
        merged = []
        for target, kws in sorted(by_target.items()):
            if kws:
                merged.append({
                    "target_field": target,
                    "pattern": re.compile(
                        r"\b(?:" + "|".join(map(re.escape, sorted(set(kws)))) + r")\b",
                        re.IGNORECASE,
                    ),
                })
        indexed[bucket_key] = merged

    indexed["all_rules"].sort(key=lambda r: (-r["priority"], r["target_field"]))
    return indexed


def _load_flag_rules_format_b(rows):
    """Load format-B rules (simple keyword/regex on text field).

    Keywords are merged by (counterparty, product_type) into a single
    compiled regex so matching runs in one str.contains pass.
    """
    indexed = {
        "text_keyword": [],
        "text_regex": [],
    }

    kw_by_key = {}
    for row in rows:
        match_type = normalize_rule_value(row.get("match_type", "")) or "keyword"
        pattern = row.get("keyword") or row.get("pattern") or ""
        counterparty = normalize_rule_value(row.get("counterparty", ""))
        product_type = normalize_rule_value(row.get("product_type", ""))
        if not pattern.strip():
            continue

        if match_type == "regex":
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
            except re.error:
                continue
            indexed["text_regex"].append({
                "pattern": compiled,
                "counterparty": counterparty,
                "product_type": product_type,
            })
        else:
            key = (counterparty, product_type)
            kw_by_key.setdefault(key, set()).update(split_upper_terms(pattern))

    for (counterparty, product_type), kws in kw_by_key.items():
        if kws:
            indexed["text_keyword"].append({
                "pattern": re.compile(
                    r"\b(?:" + "|".join(map(re.escape, sorted(kws))) + r")\b",
                    re.IGNORECASE,
                ),
                "counterparty": counterparty,
                "product_type": product_type,
            })

    return indexed


def _apply_flag_rules(df, rules, output_columns, overwrite=False):
    """Apply loaded flag rules to a dataframe.

    Returns a copy of df with flag columns added (default 0, set to 1 on match).
    """
    if not rules:
        return _ensure_flag_columns(df, output_columns)

    output = df.copy()
    output = _ensure_flag_columns(output, output_columns)

    for col in output_columns:
        if col not in output.columns:
            output[col] = 0

    text_col = output["text"].fillna("").astype(str)
    counterparty_col = output.get("counterparty", pd.Series("", index=output.index))
    counterparty_col = counterparty_col.fillna("").astype(str)

    # Pre-compute the "not yet classified" mask for the non-overwrite path
    # so we can vectorise metadata writes instead of looping row-by-row.
    if not overwrite:
        cp_empty = output["counterparty"].fillna("").astype(str).str.strip().eq("")
        pt_empty = output["product_type"].fillna("").astype(str).str.strip().eq("")
        not_classified = cp_empty & pt_empty
    else:
        not_classified = None

    def _write_metadata(mask, rule, target):
        """Write metadata for matched rows, honouring overwrite mode."""
        if overwrite:
            _bulk_write_metadata(output, mask, rule, target)
        else:
            effective = mask & not_classified
            if effective.any():
                _bulk_write_metadata(output, effective, rule, target)

    # -- text_keyword (vectorised) --
    for rule in rules.get("text_keyword", []):
        target = _resolve_target(rule, output_columns)
        if not target:
            continue
        mask = text_col.str.contains(rule["pattern"].pattern, na=False, regex=True, flags=re.IGNORECASE) & output[target].eq(0)
        if not mask.any():
            continue
        output.loc[mask, target] = 1
        _write_metadata(mask, rule, target)

    # -- text_regex (each rule has its own compiled pattern) --
    for rule in rules.get("text_regex", []):
        target = _resolve_target(rule, output_columns)
        if not target:
            continue
        mask = text_col.str.contains(rule["pattern"].pattern, na=False, regex=True, flags=re.IGNORECASE) & output[target].eq(0)
        if not mask.any():
            continue
        output.loc[mask, target] = 1
        _write_metadata(mask, rule, target)

    # -- text_or_counterparty_keyword --
    for rule in rules.get("text_or_counterparty_keyword", []):
        target = _resolve_target(rule, output_columns)
        if not target:
            continue
        pattern = rule["pattern"]
        hit_text = text_col.str.contains(pattern.pattern, na=False, regex=True, flags=re.IGNORECASE)
        hit_cp = counterparty_col.str.contains(pattern.pattern, na=False, regex=True, flags=re.IGNORECASE)
        mask = (hit_text | hit_cp) & output[target].eq(0)
        if not mask.any():
            continue
        output.loc[mask, target] = 1
        _write_metadata(mask, rule, target)

    # -- text_or_counterparty_regex --
    for rule in rules.get("text_or_counterparty_regex", []):
        target = _resolve_target(rule, output_columns)
        if not target:
            continue
        hit_text = text_col.str.contains(rule["pattern"].pattern, na=False, regex=True, flags=re.IGNORECASE)
        hit_cp = counterparty_col.str.contains(rule["pattern"].pattern, na=False, regex=True, flags=re.IGNORECASE)
        mask = (hit_text | hit_cp) & output[target].eq(0)
        if not mask.any():
            continue
        output.loc[mask, target] = 1
        _write_metadata(mask, rule, target)

    # -- all_rules --
    for rule in rules.get("all_rules", []):
        target = _resolve_target(rule, output_columns)
        if not target:
            continue
        cond_mask = _get_all_rules_mask(output, rule)
        if cond_mask is None or not cond_mask.any():
            continue
        if "pattern" in rule:
            mask = cond_mask & text_col.str.contains(rule["pattern"].pattern, na=False, regex=True, flags=re.IGNORECASE) & output[target].eq(0)
        elif "keywords" in rule:
            mask = cond_mask & text_col.str.contains(rule["pattern"].pattern, na=False, regex=True, flags=re.IGNORECASE) & output[target].eq(0)
        else:
            mask = cond_mask & output[target].eq(0)
        if not mask.any():
            continue
        output.loc[mask, target] = 1
        _write_metadata(mask, rule, target)

    return output


def _get_all_rules_mask(output, rule):
    """Build a boolean mask for all_rules condition checks (vectorised)."""
    mask = pd.Series(True, index=output.index)
    for field_name in ("account_type", "dr_cr", "bank"):
        rule_value = rule.get(field_name, "*")
        if rule_value == "*":
            continue
        col = output.get(field_name, pd.Series(index=output.index))
        col = col.fillna("").astype(str).str.strip().str.lower()
        mask &= col.eq(rule_value)
    amount_gt = rule.get("amount_gt")
    if amount_gt is not None and "amount" in output.columns:
        amount = pd.to_numeric(output["amount"], errors="coerce").abs()
        mask &= amount.gt(amount_gt)
    return mask


def _bulk_write_metadata(output, mask, rule, target):
    """Bulk-assign counterparty/finv_category/product_type for hit rows."""
    if target in _TARGET_METADATA_MAP:
        meta = _TARGET_METADATA_MAP[target]
        if meta.get("counterparty"):
            cp_empty = output["counterparty"].fillna("").astype(str).str.strip().eq("")
            output.loc[mask & cp_empty, "counterparty"] = meta["counterparty"]
        if meta.get("finv_category"):
            output.loc[mask, "finv_category"] = meta["finv_category"]
        if meta.get("product_type"):
            output.loc[mask, "product_type"] = meta["product_type"]
    elif counterparty := rule.get("counterparty", ""):
        output.loc[mask, "counterparty"] = counterparty
        product_type = rule.get("product_type", "")
        if product_type:
            output.loc[mask, "finv_category"] = product_type


def _resolve_target(rule, output_columns):
    """Resolve the target flag column name from a rule dict."""
    target = rule.get("target_field")
    if target:
        return target
    # Format B rules don't have target_field; use the first output column
    return output_columns[0] if output_columns else None


_TARGET_METADATA_MAP = {
    "is_home_loan": {
        "counterparty": "Home Loan",
        "finv_category": "Non SACC Loans",
        "product_type": "home_loan",
    },
    "is_overdrawn": {
        "counterparty": "Overdrawn",
        "finv_category": "Overdrawn",
    },
    "is_debt_collection": {
        "counterparty": "Debt Collection",
        "finv_category": "Debt Collection",
    },
    "is_debt_consolidation": {
        "counterparty": "Debt Consolidation",
        "finv_category": "Debt Consolidation",
    },
    "is_car_loan": {
        "counterparty": "Car Loan",
        "finv_category": "Non SACC Loans",
        "product_type": "car_loan",
    },
}


def _ensure_flag_columns(df, output_columns):
    """Ensure flag columns exist with default value 0."""
    output = df.copy()
    for col in output_columns:
        if col not in output.columns:
            output[col] = 0
    return output


def apply_home_loan_car_loan_flags(df, rules_file):
    """Apply home loan / car loan flag rules."""
    rules = _load_flag_rules(rules_file)
    return _apply_flag_rules(df, rules, ["is_home_loan", "is_car_loan"], overwrite=True)


def apply_overdrawn_flag(df, rules_file):
    """Apply overdrawn flag rules."""
    rules = _load_flag_rules(rules_file)
    return _apply_flag_rules(df, rules, ["is_overdrawn"])


def apply_debt_collection_flag(df, rules_file):
    """Apply debt collection flag rules."""
    rules = _load_flag_rules(rules_file)
    return _apply_flag_rules(df, rules, ["is_debt_collection"])


def apply_debt_consolidation_flag(df, rules_file):
    """Apply debt consolidation flag rules."""
    rules = _load_flag_rules(rules_file)
    return _apply_flag_rules(df, rules, ["is_debt_consolidation"])


def apply_generic_loan_catchall(df):
    """Fallback: classify remaining unclassified transactions with 'Loan' in text.

    Runs after all other rules (counterparty, credit card, home loan, car loan,
    overdrawn, debt collection, debt consolidation, dishonours, stream assignment)
    have been exhausted.  Transactions whose finv_category is still empty and whose
    text contains the word ``LOAN`` are assigned counterparty ``Generic Loans`` and
    finv_category ``Non SACC Loans``.
    """
    output = df.copy()

    fc_col = output.get("finv_category", pd.Series(index=output.index))
    fc_empty = fc_col.isna() | fc_col.astype(str).str.strip().eq("")

    if not fc_empty.any():
        return output

    text_col = output["text"].fillna("").astype(str).str.upper()
    has_loan = text_col.str.contains(r"\bLOAN\b", na=False, regex=True)

    catchall_mask = fc_empty & has_loan

    output.loc[catchall_mask, "counterparty"] = "Generic Loans"
    output.loc[catchall_mask, "product_type"] = "generic_loan"
    output.loc[catchall_mask, "finv_category"] = "Non SACC Loans"

    return output
