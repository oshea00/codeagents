# main.py

"""
Overview of Implementation Approach
-----------------------------------
This file implements a command-line calculator as described in the architecture plan. The system is modular, with
distinct classes for the CLI handler (REPL loop and command history), parser (tokenization and AST construction),
evaluator (AST evaluation), help handler, and error handling. The parser uses a recursive descent approach to
handle operator precedence, associativity, parentheses, and multiple unary minuses. The CLI uses the `readline`
module for history navigation (with a fallback warning on Windows if unavailable). The system is robust against
invalid input and provides clear error messages.

Modules, Classes, and Functions Implemented
-------------------------------------------
- Imports: sys, readline, re, typing
- Error classes: CalculatorError, ParseError, EvalError
- Tokenizer: Token, TokenType, Tokenizer
- AST Nodes: ASTNode, NumberNode, UnaryOpNode, BinOpNode
- Parser: Parser
- Evaluator: Evaluator
- HelpHandler: HelpHandler
- CLIHandler: CLIHandler (main REPL loop)
- Main entry point: main()

All code is documented and follows PEP 8 guidelines.
"""

import sys
import re
from typing import List, Optional, Union, Any

# Try to import readline for command history.
try:
    import readline
except ImportError:
    readline = None  # On Windows, readline may not be available.


# ---------------------------
# Error Classes
# ---------------------------

class CalculatorError(Exception):
    """Base class for calculator errors."""
    pass

class ParseError(CalculatorError):
    """Raised when parsing fails."""
    pass

class EvalError(CalculatorError):
    """Raised when evaluation fails."""
    pass


# ---------------------------
# Tokenizer
# ---------------------------

class TokenType:
    """Enumeration of token types."""
    NUMBER = 'NUMBER'
    PLUS = 'PLUS'
    MINUS = 'MINUS'
    MUL = 'MUL'
    DIV = 'DIV'
    LPAREN = 'LPAREN'
    RPAREN = 'RPAREN'
    EOF = 'EOF'

class Token:
    """
    Represents a token in the input stream.
    """
    def __init__(self, type_: str, value: Optional[Any] = None, pos: int = 0):
        self.type = type_
        self.value = value
        self.pos = pos

    def __repr__(self):
        return f"Token({self.type}, {self.value}, pos={self.pos})"

class Tokenizer:
    """
    Converts an input string into a list of tokens.
    Handles numbers (integers and floats), operators, and parentheses.
    """
    token_specification = [
        ('NUMBER',  r'\d+(\.\d*)?'),   # Integer or decimal number
        ('PLUS',    r'\+'),
        ('MINUS',   r'-'),
        ('MUL',     r'\*'),
        ('DIV',     r'/'),
        ('LPAREN',  r'\('),
        ('RPAREN',  r'\)'),
        ('SKIP',    r'[ \t]+'),        # Skip spaces and tabs
        ('MISMATCH',r'.'),             # Any other character
    ]
    tok_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_specification)
    get_token = re.compile(tok_regex).match

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.tokens: List[Token] = []
        self.current = 0
        self._tokenize()

    def _tokenize(self):
        """
        Tokenizes the input string into a list of Token objects.
        """
        mo = self.get_token(self.text)
        pos = 0
        while mo is not None:
            kind = mo.lastgroup
            value = mo.group()
            if kind == 'NUMBER':
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
                self.tokens.append(Token(TokenType.NUMBER, value, pos))
            elif kind == 'PLUS':
                self.tokens.append(Token(TokenType.PLUS, value, pos))
            elif kind == 'MINUS':
                self.tokens.append(Token(TokenType.MINUS, value, pos))
            elif kind == 'MUL':
                self.tokens.append(Token(TokenType.MUL, value, pos))
            elif kind == 'DIV':
                self.tokens.append(Token(TokenType.DIV, value, pos))
            elif kind == 'LPAREN':
                self.tokens.append(Token(TokenType.LPAREN, value, pos))
            elif kind == 'RPAREN':
                self.tokens.append(Token(TokenType.RPAREN, value, pos))
            elif kind == 'SKIP':
                pass  # Ignore whitespace
            elif kind == 'MISMATCH':
                raise ParseError(f"Unexpected character '{value}' at position {pos}")
            pos = mo.end()
            mo = self.get_token(self.text, pos)
        self.tokens.append(Token(TokenType.EOF, None, pos))

    def peek(self) -> Token:
        """
        Returns the current token without consuming it.
        """
        return self.tokens[self.current]

    def next(self) -> Token:
        """
        Consumes and returns the current token.
        """
        token = self.tokens[self.current]
        self.current += 1
        return token

    def expect(self, type_: str) -> Token:
        """
        Consumes and returns the current token if it matches the expected type.
        Raises ParseError otherwise.
        """
        token = self.peek()
        if token.type != type_:
            raise ParseError(f"Expected token {type_}, got {token.type} at position {token.pos}")
        return self.next()


# ---------------------------
# AST Nodes
# ---------------------------

class ASTNode:
    """
    Base class for AST nodes.
    """
    pass

class NumberNode(ASTNode):
    """
    Represents a numeric literal in the AST.
    """
    def __init__(self, value: Union[int, float]):
        self.value = value

    def __repr__(self):
        return f"NumberNode({self.value})"

class UnaryOpNode(ASTNode):
    """
    Represents a unary operation (e.g., -3, --3) in the AST.
    """
    def __init__(self, op: str, operand: ASTNode):
        self.op = op
        self.operand = operand

    def __repr__(self):
        return f"UnaryOpNode({self.op}, {self.operand})"

class BinOpNode(ASTNode):
    """
    Represents a binary operation (e.g., 1 + 2) in the AST.
    """
    def __init__(self, left: ASTNode, op: str, right: ASTNode):
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self):
        return f"BinOpNode({self.left}, {self.op}, {self.right})"


# ---------------------------
# Parser
# ---------------------------

class Parser:
    """
    Recursive descent parser for arithmetic expressions.
    Handles operator precedence, associativity, parentheses, and multiple unary minuses.
    Grammar:
        expr    : term ((PLUS|MINUS) term)*
        term    : factor ((MUL|DIV) factor)*
        factor  : (MINUS)* primary
        primary : NUMBER | LPAREN expr RPAREN
    """
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def parse(self) -> ASTNode:
        """
        Parses the input and returns the root AST node.
        """
        node = self.expr()
        if self.tokenizer.peek().type != TokenType.EOF:
            token = self.tokenizer.peek()
            raise ParseError(f"Unexpected token '{token.value}' at position {token.pos}")
        return node

    def expr(self) -> ASTNode:
        """
        expr : term ((PLUS|MINUS) term)*
        """
        node = self.term()
        while self.tokenizer.peek().type in (TokenType.PLUS, TokenType.MINUS):
            op_token = self.tokenizer.next()
            right = self.term()
            node = BinOpNode(node, op_token.type, right)
        return node

    def term(self) -> ASTNode:
        """
        term : factor ((MUL|DIV) factor)*
        """
        node = self.factor()
        while self.tokenizer.peek().type in (TokenType.MUL, TokenType.DIV):
            op_token = self.tokenizer.next()
            right = self.factor()
            node = BinOpNode(node, op_token.type, right)
        return node

    def factor(self) -> ASTNode:
        """
        factor : (MINUS)* primary
        Handles multiple consecutive unary minuses.
        """
        minus_count = 0
        while self.tokenizer.peek().type == TokenType.MINUS:
            self.tokenizer.next()
            minus_count += 1
        node = self.primary()
        if minus_count % 2 == 1:
            node = UnaryOpNode(TokenType.MINUS, node)
        return node

    def primary(self) -> ASTNode:
        """
        primary : NUMBER | LPAREN expr RPAREN
        """
        token = self.tokenizer.peek()
        if token.type == TokenType.NUMBER:
            self.tokenizer.next()
            return NumberNode(token.value)
        elif token.type == TokenType.LPAREN:
            self.tokenizer.next()
            node = self.expr()
            self.tokenizer.expect(TokenType.RPAREN)
            return node
        else:
            raise ParseError(f"Expected number or '(', got '{token.value}' at position {token.pos}")


# ---------------------------
# Evaluator
# ---------------------------

class Evaluator:
    """
    Evaluates an AST and computes the result.
    Handles division by zero and other runtime errors.
    """
    def eval(self, node: ASTNode) -> Union[int, float]:
        """
        Recursively evaluates the AST node.
        """
        if isinstance(node, NumberNode):
            return node.value
        elif isinstance(node, UnaryOpNode):
            operand = self.eval(node.operand)
            if node.op == TokenType.MINUS:
                return -operand
            else:
                raise EvalError(f"Unknown unary operator: {node.op}")
        elif isinstance(node, BinOpNode):
            left = self.eval(node.left)
            right = self.eval(node.right)
            if node.op == TokenType.PLUS:
                return left + right
            elif node.op == TokenType.MINUS:
                return left - right
            elif node.op == TokenType.MUL:
                return left * right
            elif node.op == TokenType.DIV:
                if right == 0:
                    raise EvalError("Division by zero")
                return left / right
            else:
                raise EvalError(f"Unknown binary operator: {node.op}")
        else:
            raise EvalError("Invalid AST node")


# ---------------------------
# Help Handler
# ---------------------------

class HelpHandler:
    """
    Prints usage instructions for the calculator.
    """
    HELP_TEXT = """
Command-Line Calculator Help
---------------------------
Supported operations:
  - Addition:           1 + 2
  - Subtraction:        3 - 4
  - Multiplication:     5 * 6
  - Division:           7 / 8
  - Parentheses:        (1 + 2) * 3
  - Negative numbers:   -5, --3 (multiple minuses allowed)
  - Floating point:     3.14 * 2

Special commands:
  - help      : Show this help message
  - exit/quit : Exit the calculator

Other features:
  - Use up/down arrows to navigate command history (if supported)
  - Errors are reported with clear messages

Examples:
  > 2 + 2
  4
  > --3 * (2 + 1)
  9
  > 1 / 0
  Error: Division by zero
"""

    @staticmethod
    def print_help():
        print(HelpHandler.HELP_TEXT.strip())


# ---------------------------
# CLI Handler (REPL)
# ---------------------------

class CLIHandler:
    """
    Handles the REPL loop, command history, and user interaction.
    """
    PROMPT = '> '

    def __init__(self):
        self.evaluator = Evaluator()
        self.running = True
        self._setup_history()

    def _setup_history(self):
        """
        Sets up command-line history using readline, if available.
        """
        if readline is not None:
            # Enable history file if desired (not persistent here).
            readline.parse_and_bind('tab: complete')
            readline.parse_and_bind('set editing-mode emacs')
        else:
            # Warn user if history is not available.
            print("Warning: Command history (up/down arrows) is not available on this platform.", file=sys.stderr)

    def run(self):
        """
        Main REPL loop.
        """
        while self.running:
            try:
                line = input(self.PROMPT)
            except (EOFError, KeyboardInterrupt):
                print()  # Newline for clean exit
                break

            line = line.strip()
            if not line:
                continue

            # Handle special commands
            if line.lower() in ('exit', 'quit'):
                self.running = False
                print("Goodbye!")
                break
            elif line.lower() == 'help':
                HelpHandler.print_help()
                continue

            # Parse and evaluate the expression
            try:
                tokenizer = Tokenizer(line)
                parser = Parser(tokenizer)
                ast = parser.parse()
                result = self.evaluator.eval(ast)
                # Print as int if result is integer-valued
                if isinstance(result, float) and result.is_integer():
                    result = int(result)
                print(result)
            except CalculatorError as e:
                print(f"Error: {e}")
            except Exception as e:
                # Catch-all for unexpected errors
                print(f"Unexpected error: {e}")

# ---------------------------
# Main Entry Point
# ---------------------------

def main():
    """
    Entry point for the calculator application.
    """
    print("Welcome to the Command-Line Calculator!")
    print("Type 'help' for instructions, or 'exit' to quit.")
    cli = CLIHandler()
    cli.run()

if __name__ == '__main__':
    main()