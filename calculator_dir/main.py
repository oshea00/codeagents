#!/usr/bin/env python3
"""
Scientific Calculator CLI Application

This application provides a command-line interface for performing scientific calculations
in a REPL (Read-Evaluate-Print Loop) environment.

Features:
- Basic arithmetic operations (+, -, *, /, %)
- Advanced mathematical operations (trigonometric, logarithmic, exponential functions)
- Constants (π, e)
- Memory functions
- Variable assignment and reference
- Calculation history
- Help documentation

The implementation follows a modular architecture with clear separation of concerns between
the lexer, parser, calculator engine, history manager, and help system.
"""

import math
import re
import sys
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Callable
import traceback


class TokenType(Enum):
    """Token types for lexical analysis."""
    NUMBER = 'NUMBER'
    PLUS = 'PLUS'
    MINUS = 'MINUS'
    MULTIPLY = 'MULTIPLY'
    DIVIDE = 'DIVIDE'
    MODULO = 'MODULO'
    POWER = 'POWER'
    LPAREN = 'LPAREN'
    RPAREN = 'RPAREN'
    FUNCTION = 'FUNCTION'
    CONSTANT = 'CONSTANT'
    VARIABLE = 'VARIABLE'
    COMMA = 'COMMA'
    EQUALS = 'EQUALS'
    COMMAND = 'COMMAND'


class Token:
    """Token representation for lexical analysis."""
    def __init__(self, token_type: TokenType, value: str, position: int = 0):
        self.type = token_type
        self.value = value
        self.position = position

    def __repr__(self):
        return f"Token({self.type}, {self.value}, pos={self.position})"


class Lexer:
    """
    Converts input string into tokens for parsing.
    """
    def __init__(self):
        self.functions = {
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'sinh': math.sinh,
            'cosh': math.cosh,
            'tanh': math.tanh,
            'log': math.log10,
            'ln': math.log,
            'exp': math.exp,
            'sqrt': math.sqrt,
            'abs': abs,
            'factorial': math.factorial,
            'rad': math.radians,
            'deg': math.degrees,
        }
        self.constants = {
            'pi': math.pi,
            'e': math.e,
        }
        self.commands = [
            'help', 'exit', 'quit', 'history', 'clear', 'vars',
            'memory', 'store', 'recall', 'mclear'
        ]

    def tokenize(self, text: str) -> List[Token]:
        """
        Convert input string into tokens.
        
        Args:
            text: Input string to tokenize
            
        Returns:
            List of tokens
        """
        tokens = []
        position = 0
        text = text.strip().lower()
        
        # Check if it's a command
        command_match = re.match(r'^([a-z]+)(\s+.*)?$', text)
        if command_match and command_match.group(1) in self.commands:
            return [Token(TokenType.COMMAND, text, 0)]
        
        while position < len(text):
            char = text[position]
            
            # Skip whitespace
            if char.isspace():
                position += 1
                continue
                
            # Numbers
            if char.isdigit() or char == '.':
                num_str = ''
                start_pos = position
                
                # Handle numbers with decimal points
                while position < len(text) and (text[position].isdigit() or text[position] == '.'):
                    num_str += text[position]
                    position += 1
                
                # Ensure it's a valid number
                try:
                    float(num_str)
                    tokens.append(Token(TokenType.NUMBER, num_str, start_pos))
                except ValueError:
                    raise ValueError(f"Invalid number format: {num_str}")
                continue
                
            # Operators
            if char == '+':
                tokens.append(Token(TokenType.PLUS, '+', position))
                position += 1
                continue
                
            if char == '-':
                # Check if it's a unary minus (negative sign)
                if not tokens or tokens[-1].type in (TokenType.PLUS, TokenType.MINUS, TokenType.MULTIPLY, 
                                                  TokenType.DIVIDE, TokenType.MODULO, TokenType.POWER,
                                                  TokenType.LPAREN, TokenType.COMMA, TokenType.EQUALS):
                    # It's a unary minus, treat as part of the number
                    if position + 1 < len(text) and (text[position + 1].isdigit() or text[position + 1] == '.'):
                        start_pos = position
                        num_str = '-'
                        position += 1
                        
                        while position < len(text) and (text[position].isdigit() or text[position] == '.'):
                            num_str += text[position]
                            position += 1
                            
                        try:
                            float(num_str)
                            tokens.append(Token(TokenType.NUMBER, num_str, start_pos))
                        except ValueError:
                            raise ValueError(f"Invalid number format: {num_str}")
                    else:
                        # It's a unary minus before a variable, function, etc.
                        # For test compatibility, tokenize this differently
                        tokens.append(Token(TokenType.MINUS, '-', position))
                        position += 1
                else:
                    # It's a binary minus (subtraction)
                    tokens.append(Token(TokenType.MINUS, '-', position))
                    position += 1
                continue
                
            if char == '*':
                tokens.append(Token(TokenType.MULTIPLY, '*', position))
                position += 1
                continue
                
            if char == '/':
                tokens.append(Token(TokenType.DIVIDE, '/', position))
                position += 1
                continue
                
            if char == '%':
                tokens.append(Token(TokenType.MODULO, '%', position))
                position += 1
                continue
                
            if char == '^':
                tokens.append(Token(TokenType.POWER, '^', position))
                position += 1
                continue
                
            if char == '(':
                tokens.append(Token(TokenType.LPAREN, '(', position))
                position += 1
                continue
                
            if char == ')':
                tokens.append(Token(TokenType.RPAREN, ')', position))
                position += 1
                continue
                
            if char == ',':
                tokens.append(Token(TokenType.COMMA, ',', position))
                position += 1
                continue
                
            if char == '=':
                tokens.append(Token(TokenType.EQUALS, '=', position))
                position += 1
                continue
                
            # Functions, Constants, and Variables
            if char.isalpha():
                identifier = ''
                start_pos = position
                
                while position < len(text) and (text[position].isalnum() or text[position] == '_'):
                    identifier += text[position]
                    position += 1
                
                if identifier in self.functions:
                    tokens.append(Token(TokenType.FUNCTION, identifier, start_pos))
                elif identifier in self.constants:
                    tokens.append(Token(TokenType.CONSTANT, identifier, start_pos))
                else:
                    tokens.append(Token(TokenType.VARIABLE, identifier, start_pos))
                continue
                
            # If we get here, the character is not recognized
            raise ValueError(f"Unrecognized character: '{char}' at position {position}")
            
        return tokens


class Parser:
    """
    Parses tokens into an abstract syntax tree for evaluation.
    Implements the Shunting Yard algorithm for handling operator precedence.
    """
    def __init__(self):
        pass
        
    def parse(self, tokens: List[Token]) -> List[Token]:
        """
        Parse tokens using the Shunting Yard algorithm.
        
        Args:
            tokens: List of tokens to parse
            
        Returns:
            List of tokens in Reverse Polish Notation (RPN)
        """
        # Check if it's a command
        if tokens and tokens[0].type == TokenType.COMMAND:
            return tokens
            
        # Check for variable assignment
        if len(tokens) >= 3 and tokens[0].type == TokenType.VARIABLE and tokens[1].type == TokenType.EQUALS:
            var_name = tokens[0].value
            # Parse the expression after the equals sign
            expression_tokens = self._parse_expression(tokens[2:])
            return [Token(TokenType.VARIABLE, var_name, tokens[0].position), 
                    Token(TokenType.EQUALS, '=', tokens[1].position)] + expression_tokens
        
        return self._parse_expression(tokens)
        
    def _parse_expression(self, tokens: List[Token]) -> List[Token]:
        """
        Parse an expression using the Shunting Yard algorithm.
        
        Args:
            tokens: List of tokens to parse
            
        Returns:
            List of tokens in Reverse Polish Notation (RPN)
        """
        output_queue = []
        operator_stack = []
        
        # Define operator precedence
        precedence = {
            TokenType.PLUS: 1,
            TokenType.MINUS: 1,
            TokenType.MULTIPLY: 2,
            TokenType.DIVIDE: 2,
            TokenType.MODULO: 2,
            TokenType.POWER: 3,
            TokenType.FUNCTION: 4
        }
        
        # Handle unary minus
        i = 0
        processed_tokens = []
        while i < len(tokens):
            if (tokens[i].type == TokenType.MINUS and 
                (i == 0 or tokens[i-1].type in (TokenType.PLUS, TokenType.MINUS, TokenType.MULTIPLY, 
                                             TokenType.DIVIDE, TokenType.MODULO, TokenType.POWER,
                                             TokenType.LPAREN, TokenType.COMMA, TokenType.EQUALS))):
                # It's a unary minus
                if i + 1 < len(tokens) and tokens[i+1].type == TokenType.NUMBER:
                    # Directly negate the number
                    value = -float(tokens[i+1].value)
                    processed_tokens.append(Token(TokenType.NUMBER, str(value), tokens[i].position))
                    i += 2
                else:
                    # Insert a -1 and multiply
                    processed_tokens.append(Token(TokenType.NUMBER, "-1", tokens[i].position))
                    processed_tokens.append(Token(TokenType.MULTIPLY, "*", tokens[i].position))
                    i += 1
            else:
                processed_tokens.append(tokens[i])
                i += 1
        
        tokens = processed_tokens
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            # Handle numbers, constants, and variables
            if token.type in (TokenType.NUMBER, TokenType.CONSTANT, TokenType.VARIABLE):
                output_queue.append(token)
                
            # Handle functions
            elif token.type == TokenType.FUNCTION:
                operator_stack.append(token)
                
            # Handle left parenthesis
            elif token.type == TokenType.LPAREN:
                operator_stack.append(token)
                
            # Handle right parenthesis
            elif token.type == TokenType.RPAREN:
                while operator_stack and operator_stack[-1].type != TokenType.LPAREN:
                    output_queue.append(operator_stack.pop())
                    
                if not operator_stack:
                    raise ValueError("Mismatched parentheses: too many closing parentheses")
                    
                # Pop the left parenthesis
                operator_stack.pop()
                
                # If there's a function at the top of the stack, pop it
                if operator_stack and operator_stack[-1].type == TokenType.FUNCTION:
                    output_queue.append(operator_stack.pop())
                    
            # Handle operators
            elif token.type in precedence:
                while (operator_stack and 
                       operator_stack[-1].type != TokenType.LPAREN and
                       operator_stack[-1].type in precedence and
                       precedence[operator_stack[-1].type] >= precedence[token.type]):
                    output_queue.append(operator_stack.pop())
                    
                operator_stack.append(token)
                
            # Handle comma (function arguments separator)
            elif token.type == TokenType.COMMA:
                while operator_stack and operator_stack[-1].type != TokenType.LPAREN:
                    output_queue.append(operator_stack.pop())
                    
                if not operator_stack:
                    raise ValueError("Misplaced comma or mismatched parentheses")
                    
            else:
                raise ValueError(f"Unexpected token: {token}")
                
            i += 1
            
        # Pop any remaining operators
        while operator_stack:
            if operator_stack[-1].type == TokenType.LPAREN:
                raise ValueError("Mismatched parentheses: too many opening parentheses")
                
            output_queue.append(operator_stack.pop())
            
        return output_queue


class CalculatorEngine:
    """
    Core calculator engine that evaluates expressions and manages variables and memory.
    """
    def __init__(self):
        self.variables = {}
        self.memory = None
        self.functions = {
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'sinh': math.sinh,
            'cosh': math.cosh,
            'tanh': math.tanh,
            'log': math.log10,
            'ln': math.log,
            'exp': math.exp,
            'sqrt': math.sqrt,
            'abs': abs,
            'factorial': math.factorial,
            'rad': math.radians,
            'deg': math.degrees,
        }
        self.constants = {
            'pi': math.pi,
            'e': math.e,
        }
        
        # Add ans variable for last result
        self.variables['ans'] = 0
        
        # Initialize lexer and parser
        self.lexer = Lexer()
        self.parser = Parser()
        
    def evaluate(self, rpn_tokens: List[Token]) -> float:
        """
        Evaluate a list of tokens in Reverse Polish Notation.
        
        Args:
            rpn_tokens: List of tokens in RPN
            
        Returns:
            Result of the evaluation
        """
        stack = []
        
        for token in rpn_tokens:
            if token.type == TokenType.NUMBER:
                stack.append(float(token.value))
                
            elif token.type == TokenType.CONSTANT:
                stack.append(self.constants[token.value])
                
            elif token.type == TokenType.VARIABLE:
                if token.value not in self.variables:
                    raise ValueError(f"Undefined variable: {token.value}")
                stack.append(self.variables[token.value])
                
            elif token.type == TokenType.FUNCTION:
                if token.value not in self.functions:
                    raise ValueError(f"Unknown function: {token.value}")
                    
                # Most functions take one argument
                if len(stack) < 1:
                    raise ValueError(f"Not enough arguments for function {token.value}")
                    
                arg = stack.pop()
                
                try:
                    result = self.functions[token.value](arg)
                    stack.append(result)
                except Exception as e:
                    raise ValueError(f"Error in function {token.value}: {str(e)}")
                    
            elif token.type in (TokenType.PLUS, TokenType.MINUS, TokenType.MULTIPLY, 
                               TokenType.DIVIDE, TokenType.MODULO, TokenType.POWER):
                if len(stack) < 2:
                    raise ValueError(f"Not enough operands for operator {token.value}")
                    
                b = stack.pop()
                a = stack.pop()
                
                if token.type == TokenType.PLUS:
                    stack.append(a + b)
                elif token.type == TokenType.MINUS:
                    stack.append(a - b)
                elif token.type == TokenType.MULTIPLY:
                    stack.append(a * b)
                elif token.type == TokenType.DIVIDE:
                    if b == 0:
                        raise ValueError("Division by zero")
                    stack.append(a / b)
                elif token.type == TokenType.MODULO:
                    if b == 0:
                        raise ValueError("Modulo by zero")
                    stack.append(a % b)
                elif token.type == TokenType.POWER:
                    try:
                        stack.append(a ** b)
                    except Exception as e:
                        raise ValueError(f"Error in power operation: {str(e)}")
                        
        if len(stack) != 1:
            raise ValueError("Invalid expression: too many values")
            
        return stack[0]
        
    def evaluate_expression(self, expression: str) -> float:
        """
        Evaluate a mathematical expression.
        
        Args:
            expression: Mathematical expression as a string
            
        Returns:
            Result of the evaluation
        """
        tokens = self.lexer.tokenize(expression)
        
        # Handle variable assignment
        if len(tokens) >= 3 and tokens[0].type == TokenType.VARIABLE and tokens[1].type == TokenType.EQUALS:
            var_name = tokens[0].value
            # Parse and evaluate the expression after the equals sign
            rpn_tokens = self.parser.parse(tokens[2:])
            result = self.evaluate(rpn_tokens)
            self.variables[var_name] = result
            return result
            
        rpn_tokens = self.parser.parse(tokens)
        result = self.evaluate(rpn_tokens)
        
        # Update ans variable
        self.variables['ans'] = result
        
        return result
        
    def store_in_memory(self, value: float) -> None:
        """Store a value in memory."""
        self.memory = value
        
    def recall_from_memory(self) -> Optional[float]:
        """Recall the value from memory."""
        return self.memory
        
    def clear_memory(self) -> None:
        """Clear the memory."""
        self.memory = None
        
    def get_variables(self) -> Dict[str, float]:
        """Get all defined variables."""
        return self.variables


class HistoryManager:
    """
    Manages the history of calculations and results.
    """
    def __init__(self):
        self.history: List[Tuple[str, float]] = []
        
    def add_entry(self, expression: str, result: float) -> None:
        """
        Add an entry to the history.
        
        Args:
            expression: The expression that was evaluated
            result: The result of the evaluation
        """
        self.history.append((expression, result))
        
    def get_history(self) -> List[Tuple[str, float]]:
        """
        Get the entire history.
        
        Returns:
            List of (expression, result) tuples
        """
        return self.history
        
    def clear_history(self) -> None:
        """Clear the history."""
        self.history = []
        
    def get_last_result(self) -> Optional[float]:
        """
        Get the most recent result.
        
        Returns:
            The most recent result or None if history is empty
        """
        if not self.history:
            return None
        return self.history[-1][1]


class HelpSystem:
    """
    Provides help documentation for the calculator.
    """
    def __init__(self):
        self.topics = {
            'general': self._get_general_help,
            'operators': self._get_operators_help,
            'functions': self._get_functions_help,
            'constants': self._get_constants_help,
            'variables': self._get_variables_help,
            'memory': self._get_memory_help,
            'history': self._get_history_help,
            'commands': self._get_commands_help,
        }
        
    def get_help(self, topic: Optional[str] = None) -> str:
        """
        Get help documentation for a specific topic or general help.
        
        Args:
            topic: The help topic to display
            
        Returns:
            Help text
        """
        if topic is None or topic == '':
            return self._get_general_help()
            
        topic = topic.lower()
        if topic in self.topics:
            return self.topics[topic]()
            
        return f"Help topic '{topic}' not found. Available topics: {', '.join(self.topics.keys())}"
        
    def _get_general_help(self) -> str:
        return """
Scientific Calculator Help
=========================

This calculator supports basic and advanced mathematical operations.
For specific help, try 'help <topic>' where <topic> is one of:
  operators, functions, constants, variables, memory, history, commands

Basic usage:
- Type mathematical expressions and press Enter to evaluate
- Use 'ans' to refer to the previous result
- Define variables with '=' (e.g., 'x = 5')
- Access memory with 'store', 'recall', and 'mclear'
- View history with 'history'
- Exit with 'exit' or 'quit'
"""
    
    def _get_operators_help(self) -> str:
        return """
Operators
=========

The calculator supports the following operators:
+ : Addition
- : Subtraction
* : Multiplication
/ : Division
% : Modulo (remainder)
^ : Exponentiation

Operators follow standard order of precedence:
1. Parentheses
2. Exponentiation
3. Multiplication, Division, Modulo
4. Addition, Subtraction

Examples:
3 + 4 * 2     = 11
(3 + 4) * 2   = 14
5^2 + 3       = 28
10 % 3        = 1
"""
    
    def _get_functions_help(self) -> str:
        return """
Mathematical Functions
=====================

Trigonometric Functions:
- sin(x)  : Sine of x (in radians)
- cos(x)  : Cosine of x (in radians)
- tan(x)  : Tangent of x (in radians)
- asin(x) : Arcsine of x (returns radians)
- acos(x) : Arccosine of x (returns radians)
- atan(x) : Arctangent of x (returns radians)

Hyperbolic Functions:
- sinh(x) : Hyperbolic sine of x
- cosh(x) : Hyperbolic cosine of x
- tanh(x) : Hyperbolic tangent of x

Logarithmic Functions:
- log(x)  : Base-10 logarithm of x
- ln(x)   : Natural logarithm of x

Other Functions:
- sqrt(x)     : Square root of x
- exp(x)      : e raised to the power of x
- abs(x)      : Absolute value of x
- factorial(n): Factorial of n (n must be a non-negative integer)
- rad(x)      : Convert x from degrees to radians
- deg(x)      : Convert x from radians to degrees

Examples:
sin(pi/2)     = 1
log(100)      = 2
sqrt(16)      = 4
factorial(5)  = 120
"""
    
    def _get_constants_help(self) -> str:
        return """
Mathematical Constants
=====================

The calculator supports the following constants:
- pi : The mathematical constant π (3.141592...)
- e  : The mathematical constant e (2.718281...)

Examples:
sin(pi/2)   = 1
e^2         = 7.389...
2*pi        = 6.283...
"""
    
    def _get_variables_help(self) -> str:
        return """
Variables
=========

You can define and use variables in your calculations:
- Assign a value to a variable: x = 5
- Use variables in expressions: x + 10
- The special variable 'ans' always contains the last result

Variables can be used in any expression and can reference other variables.

Examples:
x = 5
y = x * 2       # y becomes 10
z = x + y       # z becomes 15
ans + 5         # adds 5 to the previous result

Type 'vars' to see all defined variables and their values.
"""
    
    def _get_memory_help(self) -> str:
        return """
Memory Functions
===============

The calculator provides memory functions similar to a physical calculator:
- store   : Store the last result in memory
- recall  : Recall the value from memory
- mclear  : Clear the memory

Examples:
5 + 5          # Result: 10
store          # Stores 10 in memory
20 - 5         # Result: 15
recall         # Recalls 10 from memory
mclear         # Clears the memory
"""
    
    def _get_history_help(self) -> str:
        return """
History
=======

The calculator keeps track of your calculations:
- history  : Display the calculation history
- clear    : Clear the calculation history
- ans      : Use the last result in a new calculation

Examples:
5 + 5          # Result: 10
ans * 2        # Result: 20 (10 * 2)
history        # Shows both calculations
clear          # Clears the history
"""
    
    def _get_commands_help(self) -> str:
        return """
Commands
========

The calculator supports the following commands:
- help [topic] : Display help (optionally for a specific topic)
- exit, quit   : Exit the calculator
- history      : Display calculation history
- clear        : Clear the calculation history
- vars         : List all defined variables
- store        : Store the last result in memory
- recall       : Recall the value from memory
- mclear       : Clear the memory

Commands are case-insensitive.
"""


class ScientificCalculator:
    """
    Main calculator application that ties all components together and provides the REPL interface.
    """
    def __init__(self):
        self.calculator_engine = CalculatorEngine()
        self.history_manager = HistoryManager()
        self.help_system = HelpSystem()
        self.running = True
        
    def start(self):
        """Start the calculator REPL."""
        self._print_welcome()
        
        while self.running:
            try:
                # Read
                user_input = input('> ').strip()
                
                if not user_input:
                    continue
                    
                # Evaluate
                result = self._process_input(user_input)
                
                # Print
                if result is not None:
                    print(result)
                    
            except Exception as e:
                print(f"Error: {str(e)}")
                # Uncomment the following line for debugging
                # traceback.print_exc()
                
    def _process_input(self, user_input: str) -> Optional[Union[str, float]]:
        """
        Process user input and return the result.
        
        Args:
            user_input: The user's input string
            
        Returns:
            The result of processing the input, or None if no output is needed
        """
        # Handle commands
        if user_input.lower().startswith('help'):
            parts = user_input.split(maxsplit=1)
            topic = parts[1] if len(parts) > 1 else None
            return self.help_system.get_help(topic)
            
        elif user_input.lower() in ('exit', 'quit'):
            self.running = False
            return "Goodbye!"
            
        elif user_input.lower() == 'history':
            history = self.history_manager.get_history()
            if not history:
                return "History is empty."
                
            result = "Calculation History:\n"
            for i, (expr, res) in enumerate(history, 1):
                result += f"{i}: {expr} = {res}\n"
            return result.strip()
            
        elif user_input.lower() == 'clear':
            self.history_manager.clear_history()
            return "History cleared."
            
        elif user_input.lower() == 'vars':
            variables = self.calculator_engine.get_variables()
            if not variables:
                return "No variables defined."
                
            result = "Defined Variables:\n"
            for name, value in variables.items():
                result += f"{name} = {value}\n"
            return result.strip()
            
        elif user_input.lower() == 'store':
            last_result = self.history_manager.get_last_result()
            if last_result is None:
                return "No result to store."
                
            self.calculator_engine.store_in_memory(last_result)
            return f"Stored in memory: {last_result}"
            
        elif user_input.lower() == 'recall':
            memory_value = self.calculator_engine.recall_from_memory()
            if memory_value is None:
                return "Memory is empty."
            return str(memory_value)
            
        elif user_input.lower() == 'mclear':
            self.calculator_engine.clear_memory()
            return "Memory cleared."
            
        # Handle expressions
        try:
            result = self.calculator_engine.evaluate_expression(user_input)
            self.history_manager.add_entry(user_input, result)
            
            # Format the result to avoid unnecessary decimal places
            if result == int(result):
                return str(int(result))
            else:
                return str(result)
                
        except Exception as e:
            raise ValueError(f"Error evaluating expression: {str(e)}")
            
    def _print_welcome(self):
        """Print the welcome message."""
        print("Scientific Calculator v1.0")
        print("Type 'help' for information or 'exit' to quit.")
        print()


def main():
    """Main entry point for the calculator application."""
    calculator = ScientificCalculator()
    try:
        calculator.start()
    except KeyboardInterrupt:
        print("\nCalculator terminated by user.")
        return 0
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())