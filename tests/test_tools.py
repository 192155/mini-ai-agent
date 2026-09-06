"""
test_tools.py - Unit Tests for Safe Calculator Tool
"""

import unittest
from tools import safe_calculate, detect_and_calculate_math_query


class TestTools(unittest.TestCase):

    def test_basic_arithmetic(self):
        """Test addition, subtraction, multiplication, and division."""
        self.assertEqual(safe_calculate("125 * 48")["result"], 6000)
        self.assertEqual(safe_calculate("10 + 20 - 5")["result"], 25)
        self.assertEqual(safe_calculate("100 / 4")["result"], 25)
        self.assertEqual(safe_calculate("2 ** 4")["result"], 16)

    def test_parentheses_precedence(self):
        """Test operator precedence with parentheses."""
        self.assertEqual(safe_calculate("(10 + 5) * 2")["result"], 30)
        self.assertEqual(safe_calculate("10 + 5 * 2")["result"], 20)

    def test_division_by_zero(self):
        """Test that division by zero returns an error rather than crashing."""
        res = safe_calculate("50 / 0")
        self.assertFalse(res["success"])
        self.assertIn("Division by zero", res["error"])

    def test_security_blocking(self):
        """Verify that arbitrary Python code execution is strictly blocked."""
        unsafe_payloads = [
            "__import__('os').system('dir')",
            "open('config.py').read()",
            "eval('2+2')",
            "exec('print(1)')",
        ]
        for payload in unsafe_payloads:
            res = safe_calculate(payload)
            self.assertFalse(res["success"], f"Failed to block payload: {payload}")

    def test_detect_math_query(self):
        """Test extracting and solving math expressions from natural text."""
        res1 = detect_and_calculate_math_query("calculate 250 + 750")
        self.assertIsNotNone(res1)
        self.assertEqual(res1[1], 1000)

        res2 = detect_and_calculate_math_query("what is (50 * 4) + 20?")
        self.assertIsNotNone(res2)
        self.assertEqual(res2[1], 220)

        res_none = detect_and_calculate_math_query("Who is Alan Turing?")
        self.assertIsNone(res_none)


if __name__ == "__main__":
    unittest.main()
