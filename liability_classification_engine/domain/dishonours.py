import csv
import re

import pandas as pd

FIELD_NAME = "is_dishonours"


def load_rules(rules_file):
    rules = []
    with open(rules_file, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rule_type = (row.get("rule_type") or "").strip().lower()
            pattern = row.get("pattern") or ""
            required_terms = [x.strip().lower() for x in (row.get("required_terms") or "").split(";") if x.strip()]
            if rule_type and pattern:
                rules.append((rule_type, pattern, required_terms))
    return rules


def is_dishonour(text, rules):
    text = "" if pd.isna(text) else str(text)
    lower_text = text.lower()
    for rule_type, pattern, required_terms in rules:
        if rule_type == "keyword" and pattern.lower() in lower_text:
            return "Yes"
        if rule_type == "regex" and all(term in lower_text for term in required_terms) and re.search(pattern, text):
            return "Yes"
    return "No"


def apply_dishonour_rules(df, rules_file):
    rules = load_rules(rules_file)
    output = df.copy()
    text_values = output.get("text", pd.Series("", index=output.index))
    output[FIELD_NAME] = text_values.map(lambda text: is_dishonour(text, rules))
    return output
