#!/usr/bin/env python3
import unittest
import math
from unittest.mock import patch, MagicMock
import io
import sys

# Import components from the main module
from main import (
    TokenType, Token, Lexer, Parser, CalculatorEngine, 
    HistoryManager, HelpSystem, ScientificCalculator
)

class TestToken(unittest.TestCase):
    def test_token_initialization(self):
        token = Token(TokenType.NUMBER, "123", 5)
        self.assertEqual(token.type, TokenType.NUMBER)
        self.assertEqual(token.value, "123")
        self.assertEqual(token.position, 5)
        
    def test_token_representation(self):
        token = Token(TokenType.PLUS, "+", 10)
        self.assertEqual(repr(token), "Token(TokenType.PLUS, +, pos=10)")


class TestLexer(unittest.TestCase):
    def setUp(self):
        self.lexer = Lexer()
        
    def test_tokenize_numbers(self):
        tokens = self.lexer.tokenize("123.45")
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, TokenType.NUMBER)
        self.assertEqual(tokens[0].value, "123.45")
        
    def test_tokenize_operators(self):
        tokens = self.lexer.tokenize("1 + 2 - 3 * 4 / 5 % 6 ^ 7")
        self.assertEqual(len(tokens), 13)
        self.assertEqual(tokens[1].type, TokenType.PLUS)
        self.assertEqual(tokens[3].type, TokenType.MINUS)
        self.assertEqual(tokens[5].type, TokenType.MULTIPLY)
        self.assertEqual(tokens[7].type, TokenType.DIVIDE)
        self.assertEqual(tokens[9].type, TokenType.MODULO)
        self.assertEqual(tokens[11].type, TokenType.POWER)
        
    def test_tokenize_parentheses(self):
        tokens = self.lexer.tokenize("(1 + 2) * 3")
        self.assertEqual(len(tokens), 7)
        self.assertEqual(tokens[0].type, TokenType.LPAREN)
        self.assertEqual(tokens[4].type, TokenType.RPAREN)
        
    def test_tokenize_functions(self):
        tokens = self.lexer.tokenize("sin(0.5) + cos(0.5)")
        self.assertEqual(len(tokens), 9)
        self.assertEqual(tokens[0].type, TokenType.FUNCTION)
        self.assertEqual(tokens[0].value, "sin")
        self.assertEqual(tokens[5].type, TokenType.FUNCTION)
        self.assertEqual(tokens[5].value, "cos")
        
    def test_tokenize_constants(self):
        tokens = self.lexer.tokenize("pi + e")
        self.assertEqual(len(tokens), 3)
        self.assertEqual(tokens[0].type, TokenType.CONSTANT)
        self.assertEqual(tokens[0].value, "pi")
        self.assertEqual(tokens[2].type, TokenType.CONSTANT)
        self.assertEqual(tokens[2].value, "e")
        
    def test_tokenize_variables(self):
        tokens = self.lexer.tokenize("x = 5")
        self.assertEqual(len(tokens), 3)
        self.assertEqual(tokens[0].type, TokenType.VARIABLE)
        self.assertEqual(tokens[0].value, "x")
        self.assertEqual(tokens[1].type, TokenType.EQUALS)
        
    def test_tokenize_commands(self):
        tokens = self.lexer.tokenize("help")
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, TokenType.COMMAND)
        self.assertEqual(tokens[0].value, "help")
        
        tokens = self.lexer.tokenize("help functions")
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, TokenType.COMMAND)
        self.assertEqual(tokens[0].value, "help functions")
        
    def test_tokenize_unary_minus(self):
        tokens = self.lexer.tokenize("-5")
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, TokenType.NUMBER)
        self.assertEqual(tokens[0].value, "-5")
        
        tokens = self.lexer.tokenize("5 + -3")
        self.assertEqual(len(tokens), 3)
        self.assertEqual(tokens[2].type, TokenType.NUMBER)
        self.assertEqual(tokens[2].value, "-3")
        
    def test_tokenize_invalid_character(self):
        with self.assertRaises(ValueError):
            self.lexer.tokenize("5 $ 3")
            
    def test_tokenize_invalid_number(self):
        with self.assertRaises(ValueError):
            self.lexer.tokenize("5.5.5")


class TestParser(unittest.TestCase):
    def setUp(self):
        self.parser = Parser()
        self.lexer = Lexer()
        
    def test_parse_simple_expression(self):
        tokens = self.lexer.tokenize("1 + 2")
        rpn = self.parser.parse(tokens)
        self.assertEqual(len(rpn), 3)
        self.assertEqual(rpn[0].value, "1")
        self.assertEqual(rpn[1].value, "2")
        self.assertEqual(rpn[2].value, "+")
        
    def test_parse_expression_with_precedence(self):
        tokens = self.lexer.tokenize("1 + 2 * 3")
        rpn = self.parser.parse(tokens)
        self.assertEqual(len(rpn), 5)
        self.assertEqual(rpn[0].value, "1")
        self.assertEqual(rpn[1].value, "2")
        self.assertEqual(rpn[2].value, "3")
        self.assertEqual(rpn[3].value, "*")
        self.assertEqual(rpn[4].value, "+")
        
    def test_parse_expression_with_parentheses(self):
        tokens = self.lexer.tokenize("(1 + 2) * 3")
        rpn = self.parser.parse(tokens)
        self.assertEqual(len(rpn), 5)
        self.assertEqual(rpn[0].value, "1")
        self.assertEqual(rpn[1].value, "2")
        self.assertEqual(rpn[2].value, "+")
        self.assertEqual(rpn[3].value, "3")
        self.assertEqual(rpn[4].value, "*")
        
    def test_parse_expression_with_function(self):
        tokens = self.lexer.tokenize("sin(0.5)")
        rpn = self.parser.parse(tokens)
        self.assertEqual(len(rpn), 2)
        self.assertEqual(rpn[0].value, "0.5")
        self.assertEqual(rpn[1].value, "sin")
        
    def test_parse_expression_with_unary_minus(self):
        tokens = self.lexer.tokenize("-5")
        rpn = self.parser.parse(tokens)
        self.assertEqual(len(rpn), 1)
        self.assertEqual(rpn[0].value, "-5")
        
    def test_parse_variable_assignment(self):
        tokens = self.lexer.tokenize("x = 5")
        rpn = self.parser.parse(tokens)
        self.assertEqual(len(rpn), 3)
        self.assertEqual(rpn[0].type, TokenType.VARIABLE)
        self.assertEqual(rpn[0].value, "x")
        self.assertEqual(rpn[1].type, TokenType.EQUALS)
        self.assertEqual(rpn[2].type, TokenType.NUMBER)
        
    def test_parse_command(self):
        tokens = self.lexer.tokenize("help")
        rpn = self.parser.parse(tokens)
        self.assertEqual(len(rpn), 1)
        self.assertEqual(rpn[0].type, TokenType.COMMAND)
        
    def test_parse_mismatched_parentheses(self):
        tokens = self.lexer.tokenize("(1 + 2")
        with self.assertRaises(ValueError):
            self.parser.parse(tokens)
            
        tokens = self.lexer.tokenize("1 + 2)")
        with self.assertRaises(ValueError):
            self.parser.parse(tokens)


class TestCalculatorEngine(unittest.TestCase):
    def setUp(self):
        self.calculator = CalculatorEngine()
        
    def test_evaluate_basic_arithmetic(self):
        self.assertAlmostEqual(self.calculator.evaluate_expression("1 + 2"), 3.0)
        self.assertAlmostEqual(self.calculator.evaluate_expression("1 - 2"), -1.0)
        self.assertAlmostEqual(self.calculator.evaluate_expression("2 * 3"), 6.0)
        self.assertAlmostEqual(self.calculator.evaluate_expression("6 / 3"), 2.0)
        self.assertAlmostEqual(self.calculator.evaluate_expression("7 % 3"), 1.0)
        self.assertAlmostEqual(self.calculator.evaluate_expression("2 ^ 3"), 8.0)
        
    def test_evaluate_order_of_operations(self):
        self.assertAlmostEqual(self.calculator.evaluate_expression("1 + 2 * 3"), 7.0)
        self.assertAlmostEqual(self.calculator.evaluate_expression("(1 + 2) * 3"), 9.0)
        self.assertAlmostEqual(self.calculator.evaluate_expression("2 * 3 + 4 * 5"), 26.0)
        self.assertAlmostEqual(self.calculator.evaluate_expression("2 + 3 * 4 ^ 2"), 50.0)
        
    def test_evaluate_functions(self):
        self.assertAlmostEqual(self.calculator.evaluate_expression("sin(0)"), 0.0)
        self.assertAlmostEqual(self.calculator.evaluate_expression("cos(0)"), 1.0)
        self.assertAlmostEqual(self.calculator.evaluate_expression("sqrt(16)"), 4.0)
        self.assertAlmostEqual(self.calculator.evaluate_expression("log(100)"), 2.0)
        self.assertAlmostEqual(self.calculator.evaluate_expression("ln(e)"), 1.0)
        
    def test_evaluate_constants(self):
        self.assertAlmostEqual(self.calculator.evaluate_expression("pi"), math.pi)
        self.assertAlmostEqual(self.calculator.evaluate_expression("e"), math.e)
        self.assertAlmostEqual(self.calculator.evaluate_expression("2 * pi"), 2 * math.pi)
        
    def test_evaluate_variables(self):
        self.calculator.evaluate_expression("x = 5")
        self.assertEqual(self.calculator.variables["x"], 5.0)
        self.assertAlmostEqual(self.calculator.evaluate_expression("x + 3"), 8.0)
        
        self.calculator.evaluate_expression("y = x * 2")
        self.assertEqual(self.calculator.variables["y"], 10.0)
        
    def test_ans_variable(self):
        self.calculator.evaluate_expression("5 + 5")
        self.assertEqual(self.calculator.variables["ans"], 10.0)
        self.assertAlmostEqual(self.calculator.evaluate_expression("ans * 2"), 20.0)
        
    def test_memory_functions(self):
        self.calculator.evaluate_expression("5 + 5")
        self.calculator.store_in_memory(10.0)
        self.assertEqual(self.calculator.recall_from_memory(), 10.0)
        
        self.calculator.clear_memory()
        self.assertIsNone(self.calculator.recall_from_memory())
        
    def test_division_by_zero(self):
        with self.assertRaises(ValueError):
            self.calculator.evaluate_expression("5 / 0")
            
    def test_modulo_by_zero(self):
        with self.assertRaises(ValueError):
            self.calculator.evaluate_expression("5 % 0")
            
    def test_invalid_function_argument(self):
        with self.assertRaises(ValueError):
            self.calculator.evaluate_expression("sqrt(-1)")
            
    def test_undefined_variable(self):
        with self.assertRaises(ValueError):
            self.calculator.evaluate_expression("undefined_var + 5")
            
    def test_invalid_expression(self):
        with self.assertRaises(ValueError):
            self.calculator.evaluate_expression("1 + + 2")


class TestHistoryManager(unittest.TestCase):
    def setUp(self):
        self.history = HistoryManager()
        
    def test_add_entry(self):
        self.history.add_entry("1 + 2", 3.0)
        self.assertEqual(len(self.history.history), 1)
        self.assertEqual(self.history.history[0], ("1 + 2", 3.0))
        
    def test_get_history(self):
        self.history.add_entry("1 + 2", 3.0)
        self.history.add_entry("3 * 4", 12.0)
        history = self.history.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0], ("1 + 2", 3.0))
        self.assertEqual(history[1], ("3 * 4", 12.0))
        
    def test_clear_history(self):
        self.history.add_entry("1 + 2", 3.0)
        self.history.clear_history()
        self.assertEqual(len(self.history.history), 0)
        
    def test_get_last_result(self):
        self.assertIsNone(self.history.get_last_result())
        
        self.history.add_entry("1 + 2", 3.0)
        self.assertEqual(self.history.get_last_result(), 3.0)
        
        self.history.add_entry("3 * 4", 12.0)
        self.assertEqual(self.history.get_last_result(), 12.0)


class TestHelpSystem(unittest.TestCase):
    def setUp(self):
        self.help = HelpSystem()
        
    def test_get_general_help(self):
        help_text = self.help.get_help()
        self.assertIn("Scientific Calculator Help", help_text)
        self.assertIn("Basic usage:", help_text)
        
    def test_get_topic_help(self):
        operators_help = self.help.get_help("operators")
        self.assertIn("Operators", operators_help)
        self.assertIn("Addition", operators_help)
        
        functions_help = self.help.get_help("functions")
        self.assertIn("Mathematical Functions", functions_help)
        self.assertIn("Trigonometric Functions:", functions_help)
        
    def test_get_nonexistent_topic(self):
        help_text = self.help.get_help("nonexistent")
        self.assertIn("Help topic 'nonexistent' not found", help_text)
        
    def test_all_topics(self):
        topics = ["general", "operators", "functions", "constants", 
                 "variables", "memory", "history", "commands"]
        
        for topic in topics:
            help_text = self.help.get_help(topic)
            self.assertIsNotNone(help_text)
            self.assertNotEqual(help_text, "")


class TestScientificCalculator(unittest.TestCase):
    def setUp(self):
        self.calculator = ScientificCalculator()
        
    def test_process_help_command(self):
        result = self.calculator._process_input("help")
        self.assertIn("Scientific Calculator Help", result)
        
        result = self.calculator._process_input("help operators")
        self.assertIn("Operators", result)
        
    def test_process_exit_command(self):
        result = self.calculator._process_input("exit")
        self.assertEqual(result, "Goodbye!")
        self.assertFalse(self.calculator.running)
        
        # Reset for other tests
        self.calculator.running = True
        
        result = self.calculator._process_input("quit")
        self.assertEqual(result, "Goodbye!")
        self.assertFalse(self.calculator.running)
        
    def test_process_history_command(self):
        # Empty history
        result = self.calculator._process_input("history")
        self.assertEqual(result, "History is empty.")
        
        # Add entries to history
        self.calculator._process_input("1 + 2")
        self.calculator._process_input("3 * 4")
        
        result = self.calculator._process_input("history")
        self.assertIn("1: 1 + 2 = 3.0", result)
        self.assertIn("2: 3 * 4 = 12.0", result)
        
    def test_process_clear_command(self):
        self.calculator._process_input("1 + 2")
        result = self.calculator._process_input("clear")
        self.assertEqual(result, "History cleared.")
        
        result = self.calculator._process_input("history")
        self.assertEqual(result, "History is empty.")
        
    def test_process_vars_command(self):
        result = self.calculator._process_input("vars")
        self.assertIn("ans = 0", result)
        
        self.calculator._process_input("x = 5")
        result = self.calculator._process_input("vars")
        self.assertIn("x = 5.0", result)
        
    def test_process_memory_commands(self):
        # No result to store
        result = self.calculator._process_input("store")
        self.assertEqual(result, "No result to store.")
        
        # Store a result
        self.calculator._process_input("5 + 5")
        result = self.calculator._process_input("store")
        self.assertEqual(result, "Stored in memory: 10.0")
        
        # Recall
        result = self.calculator._process_input("recall")
        self.assertEqual(result, "10.0")
        
        # Clear memory
        result = self.calculator._process_input("mclear")
        self.assertEqual(result, "Memory cleared.")
        
        result = self.calculator._process_input("recall")
        self.assertEqual(result, "Memory is empty.")
        
    def test_process_expression(self):
        result = self.calculator._process_input("1 + 2")
        self.assertEqual(result, "3")
        
        result = self.calculator._process_input("2.5 * 3")
        self.assertEqual(result, "7.5")
        
    def test_process_invalid_expression(self):
        with self.assertRaises(ValueError):
            self.calculator._process_input("1 + + 2")
            
    @patch('builtins.input', side_effect=['5 + 5', 'exit'])
    @patch('builtins.print')
    def test_start_method(self, mock_print, mock_input):
        self.calculator.start()
        
        # Check that welcome message was printed
        mock_print.assert_any_call("Scientific Calculator v1.0")
        
        # Check that result was printed
        mock_print.assert_any_call("10")
        
        # Check that exit message was printed
        mock_print.assert_any_call("Goodbye!")
        

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.calculator = ScientificCalculator()
        
    def test_complex_expression(self):
        result = self.calculator._process_input("2 * (3 + 4) - sqrt(16) + sin(pi/2)")
        self.assertAlmostEqual(float(result), 11.0)
        
    def test_variable_expressions(self):
        self.calculator._process_input("x = 5")
        self.calculator._process_input("y = x * 2")
        result = self.calculator._process_input("x + y + 5")
        self.assertEqual(result, "20")
        
    def test_function_composition(self):
        result = self.calculator._process_input("sin(sqrt(9))")
        self.assertAlmostEqual(float(result), math.sin(3))
                
    def test_ans_in_expressions(self):
        self.calculator._process_input("5 * 5")
        result = self.calculator._process_input("ans + 5")
        self.assertEqual(result, "30")
        
    def test_multiple_operations(self):
        # Perform a sequence of calculations
        self.calculator._process_input("x = 5")
        self.calculator._process_input("y = x + 3")
        self.calculator._process_input("z = y * 2")
        result = self.calculator._process_input("z / 4 + x")
        self.assertEqual(result, "9")


if __name__ == '__main__':
    unittest.main()