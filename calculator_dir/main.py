# main.py

"""
Overview of the Implementation:
This implementation includes the core components of a command-line interface (CLI) scientific calculator.
It supports a Read-Eval-Print Loop (REPL) for continuous user input and command-line expression evaluation.
The calculator uses the `sympy` library for parsing and evaluating mathematical expressions, 
including basic arithmetic and scientific calculations. The implementation also includes error handling 
to manage invalid input and provide meaningful error messages to the user.

Modules and Components:
1. Input Parser: Parses and validates user input expressions.
2. Expression Evaluator: Evaluates mathematical expressions using `sympy`.
3. Command-Line Interface: Manages REPL and direct expression evaluation.
4. Error Handling: Centralized error management for user-friendly error messages.

Technology Stack:
- Python
- SymPy for mathematical operations
- Unittest for testing (not included in this file, but suggested for implementation)
"""

import sys
import sympy as sp

class InputParser:
    """
    Class responsible for parsing and validating input expressions.
    """
    
    @staticmethod
    def parse_expression(expression) -> sp.Expr:
        """
        Parse a mathematical expression into a sympy expression.

        :param expression: A string containing the mathematical expression.
        :return: A sympy expression object.
        """
        # Check if the input is a string, raise ValueError if it is not
        if not isinstance(expression, str):
            raise ValueError("Expression must be a string.")
        
        try:
            # SymPy parsing with automatic evaluation of expressions
            return sp.sympify(expression)
        except (sp.SympifyError, TypeError):
            raise ValueError("Invalid mathematical expression.")

class ExpressionEvaluator:
    """
    Class responsible for evaluating parsed mathematical expressions.
    """
    
    @staticmethod
    def evaluate(expression: str) -> float:
        """
        Evaluate a parsed sympy expression.

        :param expression: A string containing the mathematical expression.
        :return: The result of the expression.
        """
        parsed_expr = InputParser.parse_expression(expression)
        
        try:
            # Evaluate the expression to a numerical value
            return float(parsed_expr.evalf())
        except (TypeError, ValueError):
            raise ValueError("Error in evaluating expression.")

class CalculatorCLI:
    """
    Command-Line Interface for the scientific calculator.
    Handles both REPL and direct command-line expression evaluation.
    """
    
    def __init__(self):
        self.exit_commands = {'exit', 'quit'}

    def repl(self):
        """
        Starts the Read-Eval-Print Loop (REPL) for the calculator.
        """
        print("Scientific Calculator REPL (type 'exit' or 'quit' to end)")
        while True:
            try:
                # Accept user input
                user_input = input(">>> ")
                if user_input.lower().strip() in self.exit_commands:
                    print("Exiting REPL.")
                    break
                
                # Evaluate and print the result
                result = ExpressionEvaluator.evaluate(user_input)
                print(result)
            except ValueError as e:
                print(f"Error: {e}")
            except KeyboardInterrupt:
                print("\nExiting REPL.")
                break

    def evaluate_command_line(self, args):
        """
        Evaluate expression provided via command-line arguments.
        
        :param args: List of command-line arguments.
        """
        if len(args) < 2:
            print("Usage: python main.py '<expression>'")
            return
        expression = args[1]
        try:
            result = ExpressionEvaluator.evaluate(expression)
            print(result)
        except ValueError as e:
            print(f"Error: {e}")

def main():
    """
    Main function to run the calculator based on the provided command-line arguments.
    """
    calculator = CalculatorCLI()
    if len(sys.argv) > 1:
        # Handle command-line expression evaluation
        calculator.evaluate_command_line(sys.argv)
    else:
        # Start REPL if no command-line expression is provided
        calculator.repl()

if __name__ == "__main__":
    main()

"""
Notes:
- The implementation uses `sympy` for robust parsing and evaluation of mathematical expressions.
- The REPL loop allows continuous input and can be exited with 'exit' or 'quit'.
- Error handling is implemented to manage invalid expressions and provide user-friendly messages.
- The CLI supports both direct expression evaluation from the command line and interactive REPL.

Fixes Made:
- Added type checking to InputParser.parse_expression to ensure the input is a string.
- Raised a ValueError if the input to parse_expression is not a string, addressing the logical issue reported.

Future Improvements:
- Add unit tests using the `unittest` framework.
- Enhance the parser to support more complex expressions if needed.
- Consider performance optimizations for extremely complex calculations.
"""