import csv
import os
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


def process_file(input_file, rules_file, output_file):
    rules = load_rules(rules_file)
    with open(input_file, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0]) if rows else []
        if FIELD_NAME not in fieldnames:
            fieldnames.append(FIELD_NAME)
    for row in rows:
        row[FIELD_NAME] = is_dishonour(row.get("text"), rules)
    temp_file = output_file + ".tmp"
    with open(temp_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_file, output_file)
