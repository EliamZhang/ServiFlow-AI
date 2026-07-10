from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from wages_classification_engine.model_main import parse_args


class IncomeCliTests(unittest.TestCase):
    def test_default_output_is_compact(self) -> None:
        with patch.object(sys, "argv", ["income"]):
            args = parse_args()
        self.assertFalse(args.full)

    def test_full_output_requires_explicit_flag(self) -> None:
        with patch.object(sys, "argv", ["income", "--full"]):
            args = parse_args()
        self.assertTrue(args.full)


if __name__ == "__main__":
    unittest.main()
