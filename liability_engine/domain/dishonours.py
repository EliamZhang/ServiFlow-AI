import re

import pandas as pd

from classification_core.rules import load_dishonour_style_rules

FIELD_NAME = "is_dishonours"


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
    rules = load_dishonour_style_rules(rules_file)
    output = df.copy()
    text_values = output.get("text", pd.Series("", index=output.index))
    output[FIELD_NAME] = text_values.map(lambda text: is_dishonour(text, rules))
    return output
