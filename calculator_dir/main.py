# =============================================================================
# Command-Line Scientific Calculator (REPL) with Variable Assignment, History,
# Expression Parsing, Scientific/Logic Functions, and Help Facility
#
# Implementation Approach:
# - Modular design: REPL Engine, Tokenizer, Parser (recursive descent), AST,
#   Evaluator, Symbol Table, and Function Library.
# - Uses Python's `readline` for command history and arrow navigation.
# - Custom tokenizer and parser to handle operator precedence, associativity,
#   unary minus, and function calls.
# - Supports variable assignment and usage.
# - Comprehensive error handling and in-REPL help.
# - Easily extensible for new functions/operators.
# =============================================================================

import math
import operator
import readline  # For command line history and arrow navigation
import sys
import re

# =============================================================================
# Function Library: Scientific and Logic Functions
# =============================================================================

class FunctionLibrary:
    """
    Provides a registry of supported scientific and logic functions.
    """
    def __init__(self):
        # Map function names to (callable, arity)
        self.functions = {
            'sin':   (math.sin, 1),
            'cos':   (math.cos, 1),
            'tan':   (math.tan, 1),
            'asin':  (math.asin, 1),
            'acos':  (math.acos, 1),
            'atan':  (math.atan, 1),
            'exp':   (math.exp, 1),
            'log':   (math.log, 1),
            'log10': (math.log10, 1),
            'sqrt':  (math.sqrt, 1),
            'abs':   (abs, 1),
            'factorial': (math.factorial, 1),
            'floor': (math.floor, 1),
            'ceil':  (math.ceil, 1),
            'round': (round, 1),
            'deg':   (math.degrees, 1),
            'rad':   (math.radians, 1),
            # Logic/bitwise functions
            'bin':   (bin, 1),
            'hex':   (hex, 1),
            'oct':   (oct, 1),
        }

    def is_function(self, name):
        """
        Check if a function is registered.
        """
        return name in self.functions

    def call(self, name, args):
        """
        Call a registered function with arguments.
        """
        if name not in self.functions:
            raise ValueError(f"Unknown function: {name}")
        func, arity = self.functions[name]
        if len(args) != arity:
            raise ValueError(f"Function '{name}' expects {arity} argument(s), got {len(args)}")
        try:
            return func(*args)
        except Exception as e:
            raise ValueError(f"Error in function '{name}': {e}")

    def list_functions(self):
        """
        List all registered function names.
        """
        return sorted(self.functions.keys())

# =============================================================================
# Symbol Table: Variable Assignment and Lookup
# =============================================================================

class SymbolTable:
    """
    Stores variable assignments and provides lookup.
    """
    def __init__(self):
        # Predefine mathematical constants
        self.symbols = {
            'pi': math.pi,
            'e': math.e,
            'tau': math.tau,
        }

    def set(self, name, value):
        """
        Assign a value to a variable.
        """
        self.symbols[name] = value

    def get(self, name):
        """
        Retrieve a variable's value.
        """
        if name not in self.symbols:
            raise NameError(f"Undefined variable: '{name}'")
        return self.symbols[name]

    def list_symbols(self):
        """
        List all user-defined and built-in variables.
        """
        return {k: v for k, v in self.symbols.items() if not k.startswith('_')}

# =============================================================================
# Tokenizer (Lexer)
# =============================================================================

# Token types
NUMBER      = 'NUMBER'
IDENTIFIER  = 'IDENTIFIER'
OPERATOR    = 'OPERATOR'
LPAREN      = 'LPAREN'
RPAREN      = 'RPAREN'
COMMA       = 'COMMA'
ASSIGN      = 'ASSIGN'
EOF         = 'EOF'

# Token specification (regex patterns)
TOKEN_SPEC = [
    (NUMBER,     r'\d+(\.\d*)?([eE][-+]?\d+)?'),  # Integer or decimal number
    (IDENTIFIER, r'[A-Za-z_][A-Za-z0-9_]*'),      # Identifiers (variables/functions)
    (OPERATOR,   r'==|!=|<=|>=|<<|>>|[+\-*/&|^~<>]'),  # Operators
    (ASSIGN,     r'='),                           # Assignment
    (LPAREN,     r'\('),                          # Left parenthesis
    (RPAREN,     r'\)'),                          # Right parenthesis
    (COMMA,      r','),                           # Comma
    ('SKIP',     r'[ \t]+'),                      # Skip spaces/tabs
    ('MISMATCH', r'.'),                           # Any other character
]

TOKEN_REGEX = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in TOKEN_SPEC)

class Token:
    """
    Represents a single token with type and value.
    """
    def __init__(self, type_, value, position):
        self.type = type_
        self.value = value
        self.position = position  # Position in input string

    def __repr__(self):
        return f"Token({self.type}, {self.value}, pos={self.position})"

class Tokenizer:
    """
    Converts input string into a stream of tokens.
    """
    def __init__(self, text):
        self.text = text
        self.tokens = []
        self.pos = 0
        self._tokenize()

    def _tokenize(self):
        """
        Tokenize the input string into a list of Token objects.
        """
        for mo in re.finditer(TOKEN_REGEX, self.text):
            kind = mo.lastgroup
            value = mo.group()
            pos = mo.start()
            if kind == 'NUMBER':
                value = float(value) if '.' in value or 'e' in value or 'E' in value else int(value)
                self.tokens.append(Token(NUMBER, value, pos))
            elif kind == 'IDENTIFIER':
                self.tokens.append(Token(IDENTIFIER, value, pos))
            elif kind == 'OPERATOR':
                self.tokens.append(Token(OPERATOR, value, pos))
            elif kind == 'ASSIGN':
                self.tokens.append(Token(ASSIGN, value, pos))
            elif kind == 'LPAREN':
                self.tokens.append(Token(LPAREN, value, pos))
            elif kind == 'RPAREN':
                self.tokens.append(Token(RPAREN, value, pos))
            elif kind == 'COMMA':
                self.tokens.append(Token(COMMA, value, pos))
            elif kind == 'SKIP':
                continue
            elif kind == 'MISMATCH':
                raise SyntaxError(f"Unexpected character '{value}' at position {pos}")
        self.tokens.append(Token(EOF, None, len(self.text)))

    def peek(self):
        """
        Peek at the current token without consuming it.
        """
        return self.tokens[self.pos]

    def next(self):
        """
        Consume and return the current token.
        """
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def expect(self, type_, value=None):
        """
        Expect the next token to match type (and optionally value).
        """
        token = self.peek()
        if token.type != type_ or (value is not None and token.value != value):
            raise SyntaxError(f"Expected {type_} '{value}' at position {token.position}, got {token.type} '{token.value}'")
        return self.next()

# =============================================================================
# AST Node Definitions
# =============================================================================

class ASTNode:
    """
    Base class for AST nodes.
    """
    pass

class NumberNode(ASTNode):
    """
    AST node for numeric literals.
    """
    def __init__(self, value):
        self.value = value

class VariableNode(ASTNode):
    """
    AST node for variable references.
    """
    def __init__(self, name):
        self.name = name

class UnaryOpNode(ASTNode):
    """
    AST node for unary operations.
    """
    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

class BinaryOpNode(ASTNode):
    """
    AST node for binary operations.
    """
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

class AssignNode(ASTNode):
    """
    AST node for variable assignment.
    """
    def __init__(self, name, expr):
        self.name = name
        self.expr = expr

class FunctionCallNode(ASTNode):
    """
    AST node for function calls.
    """
    def __init__(self, name, args):
        self.name = name
        self.args = args

# =============================================================================
# Parser (Recursive Descent)
# =============================================================================

class Parser:
    """
    Builds an AST from tokens, handling precedence and associativity.
    """
    def __init__(self, tokenizer, function_library):
        self.tokens = tokenizer
        self.function_library = function_library

    def parse(self):
        """
        Entry point: parse an assignment or expression.
        """
        node = self.parse_assignment()
        if self.tokens.peek().type != EOF:
            raise SyntaxError(f"Unexpected token '{self.tokens.peek().value}' at position {self.tokens.peek().position}")
        return node

    def parse_assignment(self):
        """
        assignment ::= IDENTIFIER '=' expression | expression
        """
        token = self.tokens.peek()
        if token.type == IDENTIFIER:
            # Look ahead for assignment
            if self.tokens.pos + 1 < len(self.tokens.tokens):
                next_token = self.tokens.tokens[self.tokens.pos + 1]
                if next_token.type == ASSIGN:
                    var_name = self.tokens.next().value
                    self.tokens.expect(ASSIGN)
                    expr = self.parse_expression()
                    return AssignNode(var_name, expr)
        return self.parse_expression()

    def parse_expression(self):
        """
        Handles lowest precedence: bitwise OR '|'
        """
        node = self.parse_bitwise_xor()
        while self.tokens.peek().type == OPERATOR and self.tokens.peek().value == '|':
            op = self.tokens.next().value
            right = self.parse_bitwise_xor()
            node = BinaryOpNode(node, op, right)
        return node

    def parse_bitwise_xor(self):
        node = self.parse_bitwise_and()
        while self.tokens.peek().type == OPERATOR and self.tokens.peek().value == '^':
            op = self.tokens.next().value
            right = self.parse_bitwise_and()
            node = BinaryOpNode(node, op, right)
        return node

    def parse_bitwise_and(self):
        node = self.parse_shift()
        while self.tokens.peek().type == OPERATOR and self.tokens.peek().value == '&':
            op = self.tokens.next().value
            right = self.parse_shift()
            node = BinaryOpNode(node, op, right)
        return node

    def parse_shift(self):
        node = self.parse_add_sub()
        while self.tokens.peek().type == OPERATOR and self.tokens.peek().value in ('<<', '>>'):
            op = self.tokens.next().value
            right = self.parse_add_sub()
            node = BinaryOpNode(node, op, right)
        return node

    def parse_add_sub(self):
        node = self.parse_mul_div()
        while self.tokens.peek().type == OPERATOR and self.tokens.peek().value in ('+', '-'):
            op = self.tokens.next().value
            right = self.parse_mul_div()
            node = BinaryOpNode(node, op, right)
        return node

    def parse_mul_div(self):
        node = self.parse_unary()
        while self.tokens.peek().type == OPERATOR and self.tokens.peek().value in ('*', '/'):
            op = self.tokens.next().value
            right = self.parse_unary()
            node = BinaryOpNode(node, op, right)
        return node

    def parse_unary(self):
        """
        Handles unary operators: '-', '~', and function calls.
        Multiple consecutive unary minuses are allowed.
        """
        token = self.tokens.peek()
        if token.type == OPERATOR and token.value in ('-', '~'):
            op = self.tokens.next().value
            operand = self.parse_unary()
            return UnaryOpNode(op, operand)
        return self.parse_primary()

    def parse_primary(self):
        """
        Handles numbers, variables, function calls, and parenthesized expressions.
        """
        token = self.tokens.peek()
        if token.type == NUMBER:
            value = self.tokens.next().value
            return NumberNode(value)
        elif token.type == IDENTIFIER:
            name = self.tokens.next().value
            # Function call or variable
            if self.tokens.peek().type == LPAREN:
                self.tokens.next()  # consume '('
                args = []
                if self.tokens.peek().type != RPAREN:
                    while True:
                        args.append(self.parse_expression())
                        if self.tokens.peek().type == COMMA:
                            self.tokens.next()
                        else:
                            break
                self.tokens.expect(RPAREN)
                return FunctionCallNode(name, args)
            else:
                return VariableNode(name)
        elif token.type == LPAREN:
            self.tokens.next()
            node = self.parse_expression()
            self.tokens.expect(RPAREN)
            return node
        else:
            raise SyntaxError(f"Unexpected token '{token.value}' at position {token.position}")

# =============================================================================
# Evaluator: Walks AST, Computes Results, Manages Variables
# =============================================================================

class Evaluator:
    """
    Evaluates the AST, manages variables, and calls functions.
    """
    def __init__(self, symbol_table, function_library):
        self.symbol_table = symbol_table
        self.function_library = function_library

        # Operator mapping: maps operator symbols to functions
        self.binary_ops = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': self._safe_div,
            '&': operator.and_,
            '|': operator.or_,
            '^': operator.xor,
            '<<': operator.lshift,
            '>>': operator.rshift,
        }
        self.unary_ops = {
            '-': operator.neg,
            '~': operator.invert,
        }

    def _safe_div(self, a, b):
        """
        Division with zero check.
        """
        if b == 0:
            raise ZeroDivisionError("Division by zero")
        return a / b

    def eval(self, node):
        """
        Recursively evaluates an AST node.
        """
        if isinstance(node, NumberNode):
            return node.value
        elif isinstance(node, VariableNode):
            return self.symbol_table.get(node.name)
        elif isinstance(node, UnaryOpNode):
            operand = self.eval(node.operand)
            if node.op not in self.unary_ops:
                raise ValueError(f"Unsupported unary operator: {node.op}")
            # Bitwise invert only on integers
            if node.op == '~':
                operand = int(operand)
            return self.unary_ops[node.op](operand)
        elif isinstance(node, BinaryOpNode):
            left = self.eval(node.left)
            right = self.eval(node.right)
            if node.op not in self.binary_ops:
                raise ValueError(f"Unsupported binary operator: {node.op}")
            # For bitwise ops, ensure integer operands
            if node.op in ('&', '|', '^', '<<', '>>'):
                left = int(left)
                right = int(right)
            return self.binary_ops[node.op](left, right)
        elif isinstance(node, AssignNode):
            value = self.eval(node.expr)
            self.symbol_table.set(node.name, value)
            return value
        elif isinstance(node, FunctionCallNode):
            args = [self.eval(arg) for arg in node.args]
            return self.function_library.call(node.name, args)
        else:
            raise TypeError(f"Unknown AST node type: {type(node)}")

# =============================================================================
# REPL Engine: Input Loop, History, Help, and Output
# =============================================================================

HELP_TEXT = """
Scientific Calculator REPL
--------------------------
Type expressions to evaluate them. Supports:

- Arithmetic: +, -, *, /
- Bitwise/Logic: &, |, ^, ~, <<, >>
- Scientific functions: sin(x), cos(x), tan(x), exp(x), log(x), sqrt(x), factorial(x), etc.
- Variable assignment: x = 5
- Use variables in expressions: 2 * x + 1
- Parentheses for grouping: (2 + 3) * 4
- Multiple unary minuses: --3 == 3
- Built-in constants: pi, e, tau

Commands:
  help      Show this help message
  vars      List defined variables
  funcs     List available functions
  exit, quit, Ctrl-D   Exit the calculator

Arrow keys: Navigate command history.
"""

class REPL:
    """
    Read-Eval-Print Loop for the calculator.
    """
    def __init__(self):
        self.function_library = FunctionLibrary()
        self.symbol_table = SymbolTable()
        self.running = True

        # Setup readline for history and arrow navigation
        readline.parse_and_bind('tab: complete')
        readline.set_history_length(100)

    def run(self):
        """
        Main REPL loop.
        """
        print("Scientific Calculator REPL. Type 'help' for instructions.")
        while self.running:
            try:
                line = input('>>> ').strip()
                if not line:
                    continue
                if line.lower() in ('exit', 'quit'):
                    print("Goodbye!")
                    break
                elif line.lower() == 'help':
                    print(HELP_TEXT)
                    continue
                elif line.lower() == 'vars':
                    self._print_vars()
                    continue
                elif line.lower() == 'funcs':
                    self._print_funcs()
                    continue
                # Parse and evaluate expression
                try:
                    tokenizer = Tokenizer(line)
                    parser = Parser(tokenizer, self.function_library)
                    ast = parser.parse()
                    evaluator = Evaluator(self.symbol_table, self.function_library)
                    result = evaluator.eval(ast)
                    print(result)
                except Exception as e:
                    print(f"Error: {e}")
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

    def _print_vars(self):
        """
        Prints all defined variables and their values.
        """
        symbols = self.symbol_table.list_symbols()
        if not symbols:
            print("No variables defined.")
        else:
            for k, v in symbols.items():
                print(f"{k} = {v}")

    def _print_funcs(self):
        """
        Prints all available functions.
        """
        funcs = self.function_library.list_functions()
        print("Available functions:")
        print(', '.join(funcs))

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    # Start the REPL
    repl = REPL()
    repl.run()