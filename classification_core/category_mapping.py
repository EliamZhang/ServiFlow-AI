"""Category mapping from engine granular categories to illion parent categories.

This is the single source of truth.  ``compare_labels.py`` imports from here.
"""

from __future__ import annotations

import pandas as pd

# ── granular (our) → illion parent ────────────────────────────────────────
#
# Exact name matches work automatically (identity mapping) and are NOT listed.
# Categories in UNMAPPABLE_CATS have no illion equivalent.

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
    "Personal Loan Unknown": "Non SACC Loans",
    "Contract Loans":        "Non SACC Loans",
    "LOC":                   "Non SACC Loans",
    "Home Loan":             "Non SACC Loans",
    "Car Loan":              "Non SACC Loans",
}

# Our categories that illion has NO equivalent for.
UNMAPPABLE_CATS: frozenset[str] = frozenset({
    "Debt Consolidation",
})

# ── reverse mapping (illion → our labels) for human review ─────────────────

ILLION_TO_OUR: dict[str, list[str]] = {
    # -- exact name match (1:1) --
    "Internal Transfer":            ["[同名] Internal Transfer"],
    "External Transfers":           ["[同名] External Transfers"],
    "Dining Out":                   ["[同名] Dining Out"],
    "Retail":                       ["[同名] Retail"],
    "Groceries":                    ["[同名] Groceries"],
    "Health":                       ["[同名] Health"],
    "Automotive":                   ["[同名] Automotive"],
    "Entertainment":                ["[同名] Entertainment"],
    "Home Improvement":             ["[同名] Home Improvement"],
    "Travel":                       ["[同名] Travel"],
    "Information":                  ["[同名] Information"],
    "Personal Care":                ["[同名] Personal Care"],
    "Transport":                    ["[同名] Transport"],
    "Education":                    ["[同名] Education"],
    "Gambling":                     ["[同名] Gambling"],
    "Gyms and other memberships":   ["[同名] Gyms and other memberships"],
    "Pet Care":                     ["[同名] Pet Care"],
    "Donations":                    ["[同名] Donations"],
    "Utilities":                    ["[同名] Utilities"],
    "Telecommunications":           ["[同名] Telecommunications"],
    "Rent":                         ["[同名] Rent"],
    "Department Stores":            ["[同名] Department Stores"],
    "Insurance":                    ["[同名] Insurance"],
    "Subscription TV":              ["[同名] Subscription TV"],
    "Dishonours":                   ["[同名] Dishonours"],
    "Debt Collection":              ["[同名] Debt Collection"],
    "Overdrawn":                    ["[同名] Overdrawn"],
    "SACC Loans":                   ["[同名] SACC Loans"],

    # -- 1:1, different names --
    "Fees":                         ["fee"],
    "Centrelink":                   ["centrelink"],
    "Credit Card Repayments":       ["Credit Card Repayment"],

    # -- 1:N: illion lumps together our fine-grained categories --
    "Wages":                        ["salary_payg", "salary_packaging", "self_employed_gig"],
    "Non SACC Loans":               ["BNPL", "Wage Advance", "Non SACC Loans",
                                     "Personal Loan Unknown", "Contract Loans", "LOC",
                                     "Home Loan", "Car Loan"],

    # -- illion categories with no direct equivalent in our engine --
    "All Other Credits":            [],
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
