"""Category mapping from engine granular categories to illion parent categories.

This is the single source of truth for the illion mapping.
"""

from __future__ import annotations

import pandas as pd

# ── granular (our) → illion parent ────────────────────────────────────────
#
# Exact name matches work automatically (identity mapping) and are NOT listed.

OUR_TO_ILLION: dict[str, str] = {
    # -- singular / plural --
    "fee":                   "Fees",
    "Credit Card Repayment": "Credit Card Repayments",

    # -- case --
    "centrelink":            "Centrelink",

    # -- income: our fine-grained → illion "Wages" --
    "salary_payg":           "Wages",
    "salary_packaging":      "Wages",
    "self_employed_gig":     "Wages",

    # -- liability: our fine-grained → illion coarse-grained --
    "BNPL":                  "Non SACC Loans",
    "Wage Advance":          "Non SACC Loans",
    "Personal Loan Unknown": "Unknown Loans",
    "Contract Loans":        "Non SACC Loans",
    "LOC":                   "Non SACC Loans",
    "Home Loan":             "Non SACC Loans",
    "Car Loan":              "Non SACC Loans",
}


def to_illion_category(our_category: object) -> str:
    """Map a granular engine category to its illion parent.

    Returns the illion category name.  If *our_category* is NA/blank the
    value is returned unchanged; if no mapping is defined the original
    value is returned (identity).
    """
    if pd.isna(our_category):
        return our_category
    cat = str(our_category).strip()
    if not cat:
        return our_category
    return OUR_TO_ILLION.get(cat, cat)
