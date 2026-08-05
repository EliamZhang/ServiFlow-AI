"""Helpers for filtering candidates against prior engine claims."""

from __future__ import annotations

import pandas as pd


def exclude_prior_claimed(
    candidates: pd.DataFrame,
    prior_claims: pd.DataFrame,
    keep_categories: set[str] | None = None,
) -> pd.DataFrame:
    """Return *candidates* rows not already claimed by prior engines.

    Rows whose finv_category is in *keep_categories* are always kept (e.g.
    ``{"External Transfers"}`` allows re-matching by a later engine).
    """
    if prior_claims.empty:
        return candidates
    prior_map: dict[tuple[str, str], str] = {
        (str(row["application_id"]), str(row["transaction_id"])): str(
            row["finv_category"]
        )
        for _, row in prior_claims.iterrows()
    }
    keep = pd.Series(True, index=candidates.index)
    for idx, row in candidates.iterrows():
        key = (str(row["application_id"]), str(row["transaction_id"]))
        prior_cat = prior_map.get(key)
        if prior_cat is not None and (
            keep_categories is None or prior_cat not in keep_categories
        ):
            keep.at[idx] = False
    return candidates[keep].copy()
