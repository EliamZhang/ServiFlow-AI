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


def dishonour_rows_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows flagged as a dishonour ("Yes", any case).

    Single definition of "is this a dishonour row" for the layers that must
    agree on it: build_summary drops these rows (a dishonour is not a
    stream), and the stream-numbering stage must not spend a stream id on a
    stream made only of them.  Built like add_bscat's dishonour_mask, not
    with ``.map(is_yes)``: on a frame with zero rows ``.map`` yields an
    object-dtype Series, which ``df[...]`` then reads as a column list and
    strips every column.
    """
    return df[FIELD_NAME].astype("string").str.casefold().eq("yes")
