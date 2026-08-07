"""Shared CSV rule loading for engines.

Engines express simple match rules as ``rule_type, pattern, required_terms``
CSV rows (dishonour / all_other_credit).  This loader is the single
implementation of that format; per-engine differences are handled by callers.
"""

from __future__ import annotations

import csv
from pathlib import Path


def load_dishonour_style_rules(
    rules_file: str | Path,
) -> list[tuple[str, str, list[str]]]:
    """Load ``rule_type, pattern, required_terms`` rules from a CSV file.

    ``required_terms`` is semicolon-separated, lower-cased and stripped.
    Rules with no rule_type or no pattern are skipped.
    """
    rules: list[tuple[str, str, list[str]]] = []
    with open(rules_file, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rule_type = (row.get("rule_type") or "").strip().lower()
            pattern = (row.get("pattern") or "").strip()
            required_terms = [
                x.strip().lower()
                for x in (row.get("required_terms") or "").split(";")
                if x.strip()
            ]
            if rule_type and pattern:
                rules.append((rule_type, pattern, required_terms))
    return rules
