# test_main.py

import unittest
from main import InputParser, ExpressionEvaluator

class TestInputParser(unittest.TestCase):

    def test_parse_expression_valid(self):
        # Test valid expressions
        self.assertEqual(InputParser.parse_expression("2 + 2"), 4)
        self.assertEqual(InputParser.parse_expression("10 / 2"), 5)
        self.assertEqual(InputParser.parse_expression("3 ** 2"), 9)

    def test_parse_expression_invalid_type(self):
        # Test invalid type (non-string input)
        with self.assertRaises(ValueError):
            InputParser.parse_expression(123)

    def test_parse_expression_invalid_syntax(self):
        # Test invalid syntax
        with self.assertRaises(ValueError):
            InputParser.parse_expression("2 + ")

class TestExpressionEvaluator(unittest.TestCase):

    def test_evaluate_valid_expression(self):
        # Test valid evaluations
        self.assertAlmostEqual(ExpressionEvaluator.evaluate("2 + 2"), 4.0, places=7)
        self.assertAlmostEqual(ExpressionEvaluator.evaluate("10 / 2"), 5.0, places=7)
        self.assertAlmostEqual(ExpressionEvaluator.evaluate("3 ** 2"), 9.0, places=7)

    def test_evaluate_invalid_expression(self):
        # Test invalid expression
        with self.assertRaises(ValueError):
            ExpressionEvaluator.evaluate("2 + ")

    def test_evaluate_invalid_type(self):
        # Test invalid type (non-string input)
        with self.assertRaises(ValueError):
            ExpressionEvaluator.evaluate(123)

if __name__ == '__main__':
    unittest.main()