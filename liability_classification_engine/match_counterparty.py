import csv
import re

import pandas as pd


def clean_fieldnames(rows, fieldnames):
    valid_fieldnames = [name for name in fieldnames if (name or "").strip()]
    for row in rows:
        for key in list(row):
            if not (key or "").strip():
                row.pop(key, None)
    return valid_fieldnames


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
    with open(rules_file, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            counterparty = row.get("counterparty")
            product_type = row.get("product_type", "")
            if counterparty and row.get("keyword"):
                for keyword in split_upper_terms(row["keyword"]):
                    keyword_rules.append((keyword, counterparty, product_type))
    return keyword_rules


def match_text(text, keyword_rules):
    text = normalize_match_text(text)
    for keyword, counterparty, product_type in keyword_rules:
        if keyword in text:
            return counterparty, product_type
    return "", ""


def build_rule_key(account_type, dr_cr, bank):
    return (
        normalize_rule_value(account_type) or "-",
        normalize_rule_value(dr_cr) or "-",
        normalize_rule_value(bank) or "-",
    )


def candidate_values(value):
    normalized = normalize_rule_value(value) or "-"
    values = [normalized, "*"]
    if normalized != "-":
        values.append("-")
    return list(dict.fromkeys(values))


def candidate_rule_keys(row):
    keys = []
    for account_type in candidate_values(row.get("account_type")):
        for dr_cr in candidate_values(row.get("dr_cr")):
            for bank in candidate_values(row.get("bank")):
                keys.append(build_rule_key(account_type, dr_cr, bank))
    return keys


def load_cc_rules(rules_file):
    indexed_rules = {
        "regex": {},
        "prefix": {},
        "keyword": {},
    }

    sequence = 0
    with open(rules_file, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rule_pattern = row.get("keyword", "")
            if not str(rule_pattern or "").strip():
                continue

            match_type = normalize_rule_value(row.get("match_type")) or "keyword"
            if match_type not in indexed_rules:
                match_type = "keyword"

            key = build_rule_key(
                row.get("account_type", "-"),
                row.get("dr_cr", "-"),
                row.get("bank", "-"),
            )

            rule = {
                "priority": parse_int(row.get("priority"), default=0),
                "sequence": sequence,
                "counterparty": row.get("counterparty", ""),
                "product_type": row.get("product_type", ""),
                "min_prefix_len": parse_int(row.get("min_prefix_len"), default=0),
            }
            sequence += 1

            if match_type == "regex":
                pattern = normalize_regex_pattern(rule_pattern)
                try:
                    rule["patterns"] = [re.compile(pattern, re.IGNORECASE)]
                except re.error:
                    continue
            else:
                keywords = split_upper_terms(rule_pattern)
                if not keywords:
                    continue
                rule["patterns"] = keywords

            indexed_rules[match_type].setdefault(key, []).append(rule)

    for rules_by_key in indexed_rules.values():
        for key, rules in rules_by_key.items():
            rules_by_key[key] = sorted(
                rules,
                key=lambda rule: (-rule["priority"], rule["sequence"]),
            )

    return indexed_rules


def regex_rule_matches(rule, raw_text, match_text):
    return any(pattern.search(raw_text) for pattern in rule["patterns"])


def keyword_rule_matches(rule, raw_text, match_text):
    return any(keyword in match_text for keyword in rule["patterns"])


def prefix_rule_matches(rule, raw_text, match_text):
    for keyword in rule["patterns"]:
        min_prefix_len = rule["min_prefix_len"] or len(keyword)
        min_prefix_len = min(min_prefix_len, len(keyword))

        if keyword in match_text:
            return True

        for length in range(len(keyword) - 1, min_prefix_len - 1, -1):
            if keyword[:length] in match_text:
                return True

    return False


def iter_matching_rules(row, rules):
    matchers = (
        ("keyword", keyword_rule_matches),
        ("prefix", prefix_rule_matches),
        ("regex", regex_rule_matches),
    )
    raw_text = "" if pd.isna(row.get("text")) else str(row.get("text"))
    match_text = normalize_match_text(raw_text)
    for match_type, matcher in matchers:
        for key in candidate_rule_keys(row):
            for rule in rules[match_type].get(key, []):
                if matcher(rule, raw_text, match_text):
                    yield rule


def match_cc_rule(row, rules):
    for rule in iter_matching_rules(row, rules):
        return rule["counterparty"], rule["product_type"]
    return None


def clean_dataframe_columns(df):
    valid_columns = [
        column
        for column in df.columns
        if (str(column) or "").strip()
        and not str(column).startswith("Unnamed:")
    ]
    return df.loc[:, valid_columns].copy()


def apply_counterparty_rules(df, rules_file):
    keyword_rules = load_rules(rules_file)
    output = clean_dataframe_columns(df)
    text_values = output.get("text", pd.Series("", index=output.index))
    matches = text_values.map(lambda text: match_text(text, keyword_rules))
    output["counterparty"] = matches.map(lambda match: match[0])
    output["product_type"] = matches.map(lambda match: match[1])
    return output


def apply_cc_rules(df, rules_file):
    rules = load_cc_rules(rules_file)
    output = df.copy()

    for row_id, row in output.iterrows():
        match = match_cc_rule(row, rules)
        if match is None:
            continue

        counterparty, product_type = match
        if counterparty:
            output.at[row_id, "counterparty"] = counterparty
        if product_type:
            output.at[row_id, "product_type"] = product_type

    return output


def process_file(sample_file, rules_file, output_file):
    keyword_rules = load_rules(rules_file)
    with open(sample_file, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0]) if rows else []
        fieldnames = clean_fieldnames(rows, fieldnames)
        if "counterparty" not in fieldnames:
            fieldnames.append("counterparty")
        if "product_type" not in fieldnames:
            fieldnames.append("product_type")
    for row in rows:
        row["counterparty"], row["product_type"] = match_text(row.get("text"), keyword_rules)
    with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
