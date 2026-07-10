from __future__ import annotations

import unittest

from classification_core.reasons import format_classification_reason


class ReasonFormattingTests(unittest.TestCase):
    def test_formats_compact_reason_with_limited_evidence(self) -> None:
        reason = format_classification_reason(
            category="salary_payg",
            rule="salary_payg_wages_rule",
            evidence=[
                "credit",
                "",
                "wages_detector",
                "credit",
                "strong_wage_keyword",
                "repeat_payer",
            ],
        )

        self.assertEqual(
            reason,
            (
                "category=salary_payg; rule=salary_payg_wages_rule; "
                "evidence=credit, wages_detector, strong_wage_keyword"
            ),
        )


if __name__ == "__main__":
    unittest.main()
