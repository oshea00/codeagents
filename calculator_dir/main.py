# Revised implementation of the command-line calculator REPL with lexer, Pratt parser, AST, evaluator, builtins,
# history, and help system.
#
# This revision addresses reported issues:
# - Unifies factorial semantics between postfix '!' operator and builtin factorial(...) via a single helper.
# - Hardens REPL command parsing to avoid shlex.split exceptions and unexpected behavior.
# - Refactors operator precedence tables into explicit prefix/infix/postfix maps to remove ambiguity and ensure
#   correct precedence & associativity handling for complex expressions.
# - Improves parser logic for prefix/infix/postfix handling using the clear precedence tables.
# - Adds robust error messages and safer builtin/function calling.
#
# All imports are at the top, and the implementation is self-contained for testing and running.
# Comments/docstrings explain choices and behaviors. No usage of eval/exec on user input.
#
# Note: This is still a prototype focusing on correctness and safety. It is structured for extension
# and unit testing.

from __future__ import annotations

import math
import os
import json
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Optional UI libs: prompt_toolkit preferred, fallback to readline for history on UNIX.
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.completion import WordCompleter
    PROMPT_TOOLKIT_AVAILABLE = True
except Exception:
    PROMPT_TOOLKIT_AVAILABLE = False
    try:
        import readline  # type: ignore
    except Exception:
        readline = None  # type: ignore

# --------------------------
# Exceptions
# --------------------------

class LexerError(Exception):
    """Raised for errors during tokenization."""
    pass

class ParseError(Exception):
    """Raised for parsing errors with optional position information."""
    pass

class EvalError(Exception):
    """Raised for errors during evaluation, e.g., domain errors, type errors."""
    pass

# --------------------------
# Tokenizer / Lexer
# --------------------------

@dataclass
class Token:
    """Represents a token with type, value, and character position."""
    type: str
    value: Any
    pos: int

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value!r}, pos={self.pos})"

# For lexing, define multi-character operator set to recognize before single-char tokens.
_MULTI_OPS = {'<<', '>>', '**', '&&', '||'}
_SINGLE_OP_CHARS = set('+-*/%^&|~!=<>() ,')

class Lexer:
    """Tokenizer for calculator expressions.

    Produces tokens: NUMBER, IDENT, OP, LPAREN, RPAREN, COMMA, EOF.
    Always tokenizes '-' as operator (no negative literal tokens).
    """
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.len = len(text)

    def _peek(self, n: int = 0) -> str:
        i = self.pos + n
        return self.text[i] if i < self.len else ''

    def _advance(self, n: int = 1) -> None:
        self.pos += n

    def _skip_whitespace(self) -> None:
        while self._peek() and self._peek().isspace():
            self._advance()

    def _read_number(self) -> Token:
        start = self.pos
        has_dot = False
        has_exp = False
        # integer part and fractional/exponent handling
        while True:
            ch = self._peek()
            if ch.isdigit():
                self._advance()
            elif ch == '.' and not has_dot:
                has_dot = True
                self._advance()
            elif (ch == 'e' or ch == 'E') and not has_exp:
                has_exp = True
                self._advance()
                if self._peek() in '+-':
                    self._advance()
                # require at least one digit after e/E
                if not self._peek().isdigit():
                    raise LexerError(f"Invalid numeric literal at pos {self.pos}")
            else:
                break
        raw = self.text[start:self.pos]
        try:
            if not has_dot and not has_exp:
                val: Union[int, float] = int(raw)
            else:
                val = float(raw)
        except ValueError:
            raise LexerError(f"Invalid numeric literal: {raw}")
        return Token('NUMBER', val, start)

    def _read_ident(self) -> Token:
        start = self.pos
        while True:
            ch = self._peek()
            if ch.isalnum() or ch == '_':
                self._advance()
            else:
                break
        raw = self.text[start:self.pos]
        # normalize 'and', 'or', 'not' into OP tokens for parser convenience
        if raw in {'and', 'or', 'not'}:
            return Token('OP', raw, start)
        return Token('IDENT', raw, start)

    def _match_op(self) -> Optional[str]:
        # Try multi-char ops first
        if self.pos + 2 <= self.len:
            two = self.text[self.pos:self.pos + 2]
            if two in _MULTI_OPS:
                return two
        ch = self._peek()
        if ch in _SINGLE_OP_CHARS:
            return ch
        return None

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []
        while True:
            self._skip_whitespace()
            ch = self._peek()
            if ch == '':
                break
            if ch.isdigit() or (ch == '.' and self._peek(1).isdigit()):
                tokens.append(self._read_number())
            elif ch.isalpha() or ch == '_':
                tokens.append(self._read_ident())
            elif ch in '(),':
                if ch == '(':
                    tokens.append(Token('LPAREN', ch, self.pos))
                elif ch == ')':
                    tokens.append(Token('RPAREN', ch, self.pos))
                else:
                    tokens.append(Token('COMMA', ch, self.pos))
                self._advance()
            else:
                op = self._match_op()
                if not op:
                    raise LexerError(f"Unknown character at pos {self.pos}: {repr(ch)}")
                # Normalize C-style logical operators into word forms for parser/evaluator
                if op == '&&':
                    tokens.append(Token('OP', 'and', self.pos))
                elif op == '||':
                    tokens.append(Token('OP', 'or', self.pos))
                else:
                    tokens.append(Token('OP', op, self.pos))
                self._advance(len(op))
        tokens.append(Token('EOF', None, self.pos))
        return tokens

# --------------------------
# AST Nodes
# --------------------------

@dataclass
class ASTNode:
    """Base AST node."""
    pass

@dataclass
class Number(ASTNode):
    value: Union[int, float]

@dataclass
class Variable(ASTNode):
    name: str

@dataclass
class UnaryOp(ASTNode):
    op: str
    operand: ASTNode

@dataclass
class BinaryOp(ASTNode):
    op: str
    left: ASTNode
    right: ASTNode

@dataclass
class PostfixOp(ASTNode):
    op: str
    operand: ASTNode

@dataclass
class FuncCall(ASTNode):
    name: str
    args: List[ASTNode]

@dataclass
class Assignment(ASTNode):
    name: str
    value: ASTNode

# --------------------------
# Parser (Pratt/top-down precedence)
# --------------------------

# Explicit operator precedence maps. Higher number = higher precedence.
# Postfix operators (apply immediately after primary)
POSTFIX_BP: Dict[str, int] = {
    '!': 90,  # factorial
}

# Prefix unary operators: bp indicates the precedence used to parse the operand
PREFIX_BP: Dict[str, int] = {
    '+': 80,
    '-': 80,
    '~': 80,
    'not': 80,
    '!': 80,  # treat '!' prefix as logical not
}

# Infix operators: map to (binding_power, right_assoc)
INFIX_BP: Dict[str, Tuple[int, bool]] = {
    '**': (70, True),   # exponentiation (right-assoc)
    '*': (60, False),
    '/': (60, False),
    '%': (60, False),
    '+': (50, False),
    '-': (50, False),
    '<<': (40, False),
    '>>': (40, False),
    '&': (30, False),
    '^': (25, False),   # bitwise XOR
    '|': (20, False),
    'and': (15, False),
    'or': (10, False),
    '=': (5, True),     # assignment (right-assoc)
}

class Parser:
    """Pratt parser producing an AST for expressions."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def _current(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, typ: str, value: Optional[str] = None) -> Token:
        tok = self._current()
        if tok.type != typ or (value is not None and tok.value != value):
            raise ParseError(f"Expected {typ} {value} at pos {tok.pos}; got {tok.type} {tok.value}")
        return self._advance()

    def parse(self) -> ASTNode:
        node = self.parse_expression(0)
        if self._current().type != 'EOF':
            tok = self._current()
            raise ParseError(f"Unexpected token {tok.value!r} at pos {tok.pos}")
        return node

    def parse_expression(self, rbp: int = 0) -> ASTNode:
        # Parse null denotation
        tok = self._advance()
        left = self.nud(tok)
        # While next token has binding power greater than right-binding precedence, consume led/postfix
        while True:
            cur = self._current()
            # Postfix operators have highest precedence, checked first
            if cur.type == 'OP' and cur.value in POSTFIX_BP and POSTFIX_BP[cur.value] > rbp:
                # consume postfix operator
                self._advance()
                left = PostfixOp(cur.value, left)
                continue
            # Function call: LPAREN immediately after a variable is a function call
            if cur.type == 'LPAREN':
                # Only treat as function call when left is a variable (identifier)
                if isinstance(left, Variable):
                    args = self._parse_argument_list()
                    left = FuncCall(left.name, args)
                    continue
                # else '(' after an expression is a syntax error like (expr)(...)
                raise ParseError(f"Unexpected '(' after expression at pos {cur.pos}")
            # Infix operators
            if cur.type == 'OP' and cur.value in INFIX_BP:
                bp, right_assoc = INFIX_BP[cur.value]
                if bp <= rbp:
                    break
                op_tok = self._advance()
                # For right-assoc operators, parse right-hand side with rbp = bp - 1
                rhs_rbp = bp - 1 if right_assoc else bp
                right = self.parse_expression(rhs_rbp)
                # Special-case assignment validation: left must be Variable
                if op_tok.value == '=':
                    if not isinstance(left, Variable):
                        raise ParseError("Left-hand side of assignment must be a variable")
                    left = Assignment(left.name, right)
                else:
                    left = BinaryOp(op_tok.value, left, right)
                continue
            break
        return left

    def nud(self, tok: Token) -> ASTNode:
        """Null denotation (prefix/primary)."""
        if tok.type == 'NUMBER':
            return Number(tok.value)
        if tok.type == 'IDENT':
            return Variable(tok.value)
        if tok.type == 'LPAREN':
            expr = self.parse_expression(0)
            self._expect('RPAREN')
            return expr
        if tok.type == 'OP':
            val = tok.value
            # Prefix logical not '!' normalized to prefix not
            if val == '!':
                # treat as logical not
                bp = PREFIX_BP['!']
                operand = self.parse_expression(bp)
                return UnaryOp('not', operand)
            if val in PREFIX_BP:
                bp = PREFIX_BP[val]
                operand = self.parse_expression(bp)
                return UnaryOp(val if val != '!' else 'not', operand)
            raise ParseError(f"Unexpected operator {val!r} at pos {tok.pos}")
        raise ParseError(f"Unexpected token {tok.type} {tok.value!r} at pos {tok.pos}")

    def _parse_argument_list(self) -> List[ASTNode]:
        """Parse '(' expr (, expr)* ')' and return list of AST args. Assumes current token is LPAREN."""
        self._expect('LPAREN')
        args: List[ASTNode] = []
        if self._current().type == 'RPAREN':
            # empty arg list
            self._advance()
            return args
        while True:
            expr = self.parse_expression(0)
            args.append(expr)
            cur = self._current()
            if cur.type == 'COMMA':
                self._advance()
                continue
            if cur.type == 'RPAREN':
                self._advance()
                break
            raise ParseError(f"Expected ',' or ')' in argument list at pos {cur.pos}")
        return args

# --------------------------
# Builtins / Standard Library
# --------------------------

def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        raise EvalError(f"Expected numeric value, got {x!r}")

def _ensure_int(x: Any) -> int:
    """Ensure numeric value is integer-like and return int, else raise EvalError."""
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        # Accept floats that represent exact integers (within tolerance)
        if abs(x - round(x)) < 1e-9:
            return int(round(x))
        raise EvalError(f"Expected integer value, got non-integer {x}")
    raise EvalError(f"Expected integer value, got {type(x).__name__}")

# Unified factorial helper used both by builtin 'factorial' and postfix '!'
def _do_factorial(val: Any) -> int:
    """Compute factorial for non-negative integer-like values and raise EvalError for invalid input."""
    n = _ensure_int(val)
    if n < 0:
        raise EvalError("factorial not defined for negative numbers")
    try:
        return math.factorial(n)
    except OverflowError:
        raise EvalError("factorial result too large")

# Builtins registry: map names to callables.
_BUILTINS: Dict[str, Callable[..., Any]] = {}

def _register(name: str, func: Callable[..., Any]) -> None:
    _BUILTINS[name] = func

# Register math functions with safe wrappers.
_register('sin', lambda x: math.sin(_safe_float(x)))
_register('cos', lambda x: math.cos(_safe_float(x)))
_register('tan', lambda x: math.tan(_safe_float(x)))
_register('asin', lambda x: math.asin(_safe_float(x)))
_register('acos', lambda x: math.acos(_safe_float(x)))
_register('atan', lambda x: math.atan(_safe_float(x)))
_register('sqrt', lambda x: math.sqrt(_safe_float(x)))
_register('exp', lambda x: math.exp(_safe_float(x)))
_register('log', lambda *args: math.log(*[_safe_float(a) for a in args]) )
_register('pow', lambda x, y: math.pow(_safe_float(x), _safe_float(y)))
_register('abs', lambda x: abs(_safe_float(x)))
_register('floor', lambda x: math.floor(_safe_float(x)))
_register('ceil', lambda x: math.ceil(_safe_float(x)))
# Register unified factorial function
_register('factorial', _do_factorial)

# Constants are convenient to have as predefined env entries; we also optionally expose names in builtins list
# but in evaluation variables are read from the environment first (so 'pi' returns numeric value).
# We do not register 'pi'/'e' as functions to avoid confusion with pi() call semantics.
_BUILTIN_EXTRA_NAMES = []  # keep builtins dict for functions only

# Provide builtin names list for help/completion (functions only)
_BUILTIN_NAMES = sorted(_BUILTINS.keys())

# --------------------------
# Evaluator
# --------------------------

class Evaluator:
    """Evaluates AST nodes with environment and builtins mapping."""

    def __init__(self):
        self.env: Dict[str, Any] = {}
        # Preload numeric constants into environment
        self.env['pi'] = math.pi
        self.env['e'] = math.e

    def eval(self, node: ASTNode) -> Any:
        """Evaluate given AST node and return the result or raise EvalError."""
        if isinstance(node, Number):
            return node.value
        if isinstance(node, Variable):
            name = node.name
            if name in self.env:
                return self.env[name]
            # If name exists as a builtin zero-arg function, don't auto-call it here to avoid confusion.
            # User can call f() explicitly.
            raise EvalError(f"Undefined variable: {name}")
        if isinstance(node, UnaryOp):
            val = self.eval(node.operand)
            op = node.op
            if op == '+':
                return +_safe_float(val)
            if op == '-':
                return -_safe_float(val)
            if op == '~':
                n = _ensure_int(val)
                return ~n
            if op == 'not':
                return not bool(val)
            raise EvalError(f"Unknown unary operator: {op}")
        if isinstance(node, PostfixOp):
            op = node.op
            if op == '!':
                val = self.eval(node.operand)
                return _do_factorial(val)
            raise EvalError(f"Unknown postfix operator: {op}")
        if isinstance(node, BinaryOp):
            left_val = self.eval(node.left)
            # logical short-circuit
            if node.op == 'and':
                if not bool(left_val):
                    return False
                right_val = self.eval(node.right)
                return bool(right_val)
            if node.op == 'or':
                if bool(left_val):
                    return True
                right_val = self.eval(node.right)
                return bool(right_val)
            right_val = self.eval(node.right)
            op = node.op
            try:
                if op == '+':
                    if isinstance(left_val, int) and isinstance(right_val, int):
                        return left_val + right_val
                    return _safe_float(left_val) + _safe_float(right_val)
                if op == '-':
                    if isinstance(left_val, int) and isinstance(right_val, int):
                        return left_val - right_val
                    return _safe_float(left_val) - _safe_float(right_val)
                if op == '*':
                    if isinstance(left_val, int) and isinstance(right_val, int):
                        return left_val * right_val
                    return _safe_float(left_val) * _safe_float(right_val)
                if op == '/':
                    denom = _safe_float(right_val)
                    if denom == 0:
                        raise EvalError("Division by zero")
                    return _safe_float(left_val) / denom
                if op == '%':
                    if isinstance(left_val, int) and isinstance(right_val, int):
                        if right_val == 0:
                            raise EvalError("Modulo by zero")
                        return left_val % right_val
                    denom = _safe_float(right_val)
                    if denom == 0:
                        raise EvalError("Modulo by zero")
                    return _safe_float(left_val) % denom
                if op == '**':
                    return math.pow(_safe_float(left_val), _safe_float(right_val))
                if op == '^':
                    l = _ensure_int(left_val)
                    r = _ensure_int(right_val)
                    return l ^ r
                if op == '<<':
                    l = _ensure_int(left_val)
                    r = _ensure_int(right_val)
                    if r < 0:
                        raise EvalError("negative shift count")
                    return l << r
                if op == '>>':
                    l = _ensure_int(left_val)
                    r = _ensure_int(right_val)
                    if r < 0:
                        raise EvalError("negative shift count")
                    return l >> r
                if op == '&':
                    l = _ensure_int(left_val)
                    r = _ensure_int(right_val)
                    return l & r
                if op == '|':
                    l = _ensure_int(left_val)
                    r = _ensure_int(right_val)
                    return l | r
            except EvalError:
                raise
            except Exception as e:
                raise EvalError(f"Error evaluating binary op {op}: {e}")
            raise EvalError(f"Unknown binary operator: {op}")
        if isinstance(node, FuncCall):
            name = node.name
            # Look up function in builtins
            if name in _BUILTINS:
                func = _BUILTINS[name]
                # Evaluate arguments before calling
                args = [self.eval(a) for a in node.args]
                try:
                    return func(*args)
                except TypeError as e:
                    raise EvalError(f"Error calling function '{name}': {e}")
                except ValueError as e:
                    raise EvalError(f"Domain error in function '{name}': {e}")
                except EvalError:
                    raise
                except Exception as e:
                    raise EvalError(f"Error in function '{name}': {e}")
            # If user attempts to call a variable, disallow for safety
            raise EvalError(f"Unknown function: {name}")
        if isinstance(node, Assignment):
            val = self.eval(node.value)
            self.env[node.name] = val
            return val
        raise EvalError(f"Unsupported AST node: {type(node).__name__}")

# --------------------------
# REPL, History, Help
# --------------------------

HISTORY_FILE = os.path.expanduser("~/.cli_calc_history")

_HELP_TOPICS: Dict[str, str] = {
    'general': (
        "Calculator REPL help:\n"
        "Supports arithmetic, bitwise, logical operators, functions, variables, and factorial (!).\n"
        "Examples:\n"
        "  x = 3\n"
        "  2 * -3 -> -6\n"
        "  -3! -> -(3!) == -6\n"
        "  --3 -> 3\n"
        "  sin(pi/2) -> 1\n"
        "Commands:\n"
        "  :help, help [topic]    show help\n"
        "  :vars                  list variables\n"
        "  :history               show recent history\n"
        "  :save <file>           save variables to JSON\n"
        "  :load <file>           load variables from JSON\n"
        "  :exit                  exit\n"
    ),
    'operators': (
        "Operators and precedence (high -> low):\n"
        "  postfix: ! (factorial)\n"
        "  prefix: + - ~ (bitwise not) not/! (logical not)\n"
        "  exponent: ** (right-assoc)\n"
        "  * / %\n"
        "  + -\n"
        "  << >>\n"
        "  & ^ |\n"
        "  and, or\n"
        "  = (assignment, right-assoc)\n"
        "Notes:\n"
        "  - Use '**' for exponentiation (2 ** 3 ** 2 == 2 ** (3 ** 2)).\n"
        "  - '^' is bitwise XOR.\n"
        "  - '&&' and '||' are supported as synonyms for 'and' and 'or'.\n"
        "  - '-' always tokenized as operator; negative literals handled via unary minus (so --3 == 3).\n"
    ),
    'functions': (
        "Built-in functions:\n"
        + ", ".join(_BUILTIN_NAMES) +
        "\nExamples: sin(x), cos(pi), sqrt(2), factorial(5)\n"
    ),
}

def show_help(topic: Optional[str] = None) -> str:
    """Return help text for topic or general if None."""
    if not topic:
        return _HELP_TOPICS['general']
    key = topic.lower()
    return _HELP_TOPICS.get(key, f"No help available for topic '{topic}'")

class REPL:
    """Read-Eval-Print Loop for the calculator."""

    def __init__(self):
        self.evaluator = Evaluator()
        self.history_file = HISTORY_FILE
        if PROMPT_TOOLKIT_AVAILABLE:
            self.session = PromptSession(history=FileHistory(self.history_file))
            self.completer = WordCompleter(_BUILTIN_NAMES, ignore_case=True)
        else:
            if 'readline' in globals() and readline is not None:
                try:
                    readline.read_history_file(self.history_file)
                except Exception:
                    # ignore missing history
                    pass

    def _save_history(self) -> None:
        if not PROMPT_TOOLKIT_AVAILABLE and 'readline' in globals() and readline is not None:
            try:
                readline.write_history_file(self.history_file)
            except Exception:
                pass

    def _process_command(self, line: str) -> Optional[str]:
        """Process commands starting with ':' or 'help'. Returns response string if a command, else None.

        Hardened to avoid shlex.split exceptions and to use simple, predictable parsing.
        """
        s = line.strip()
        if not s:
            return None
        # Colon commands like :help, :vars, :save file
        if s.startswith(':'):
            body = s[1:].lstrip()
            if body == '':
                return "No command specified. Use :help for available commands."
            # split into command and rest (max 1 split)
            parts = body.split(None, 1)
            cmd = parts[0]
            args_str = parts[1] if len(parts) > 1 else ''
            args = args_str.split() if args_str else []
            try:
                return self._run_command(cmd, args)
            except EOFError:
                # bubble EOF to caller to exit
                raise
            except Exception as e:
                # Catch unexpected exceptions from commands and return an error string
                return f"Command error: {e}"
        # help keyword at line start: allow 'help' or 'help topic'
        if s.lower().startswith('help'):
            # split into at most two parts to keep topic intact if it contains spaces
            parts = s.split(None, 1)
            if len(parts) == 1:
                return show_help(None)
            else:
                # second part is topic, simple whitespace-trimmed
                topic = parts[1].strip()
                return show_help(topic)
        return None

    def _run_command(self, cmd: str, args: List[str]) -> Optional[str]:
        """Execute a REPL colon command. Raises EOFError for exit/quit to allow outer loop to handle shutdown."""
        cmd_lower = cmd.lower()
        if cmd_lower in {'exit', 'quit'}:
            raise EOFError()
        if cmd_lower == 'help':
            topic = args[0] if args else None
            return show_help(topic)
        if cmd_lower == 'vars':
            items = sorted(self.evaluator.env.items())
            if not items:
                return "(no variables)"
            return "\n".join(f"{k} = {v!r}" for k, v in items)
        if cmd_lower == 'history':
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    all_lines = f.read().splitlines()
                last = all_lines[-50:]
                return "\n".join(last)
            except Exception as e:
                return f"Could not read history: {e}"
        if cmd_lower == 'save':
            if not args:
                return "Usage: :save <file>"
            fname = args[0]
            try:
                with open(fname, 'w', encoding='utf-8') as f:
                    json.dump(self.evaluator.env, f, default=str)
                return f"Saved {len(self.evaluator.env)} variables to {fname}"
            except Exception as e:
                return f"Error saving variables: {e}"
        if cmd_lower == 'load':
            if not args:
                return "Usage: :load <file>"
            fname = args[0]
            try:
                with open(fname, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    return "Invalid file format"
                for k, v in data.items():
                    self.evaluator.env[k] = v
                return f"Loaded {len(data)} variables from {fname}"
            except Exception as e:
                return f"Error loading variables: {e}"
        return f"Unknown command: {cmd}"

    def _normalize_logical_symbols(self, text: str) -> str:
        """Stub for potential normalization; lexer already maps &&/|| to 'and'/'or'."""
        return text

    def evaluate_line(self, line: str) -> Tuple[bool, str]:
        """Evaluate a single line (either command or expression). Returns (ok, output)."""
        # First check for commands
        cmd_out = None
        try:
            cmd_out = self._process_command(line)
        except EOFError:
            # let callers handle EOF by propagating
            raise
        except Exception as e:
            # Catch any command parsing errors
            return False, f"Command processing error: {e}"

        if cmd_out is not None:
            return True, cmd_out

        text = self._normalize_logical_symbols(line)
        try:
            lexer = Lexer(text)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            ast = parser.parse()
            result = self.evaluator.eval(ast)
            return True, repr(result)
        except (LexerError, ParseError, EvalError) as e:
            return False, f"Error: {e}"
        except Exception as e:
            return False, f"Unhandled error: {e}"

    def repl_loop(self) -> None:
        """Interactive REPL loop; integrates with prompt_toolkit if available for history and completion."""
        banner = "Interactive Calculator REPL. Type :help for help. Ctrl-D or :exit to quit."
        print(banner)
        try:
            if PROMPT_TOOLKIT_AVAILABLE:
                while True:
                    try:
                        # update completer with builtins and current variables
                        words = list(_BUILTIN_NAMES) + list(self.evaluator.env.keys())
                        self.completer = WordCompleter(words, ignore_case=True)
                        line = self.session.prompt('> ', completer=self.completer)
                    except KeyboardInterrupt:
                        print("^C")
                        continue
                    except EOFError:
                        print("Exiting.")
                        break
                    if not line.strip():
                        continue
                    ok, out = self.evaluate_line(line)
                    print(out)
            else:
                # readline / input fallback
                while True:
                    try:
                        line = input('> ')
                    except KeyboardInterrupt:
                        print("^C")
                        continue
                    except EOFError:
                        print("Exiting.")
                        break
                    if not line.strip():
                        continue
                    ok, out = self.evaluate_line(line)
                    print(out)
        finally:
            self._save_history()

# --------------------------
# Entry point
# --------------------------

def main(argv: Optional[List[str]] = None) -> int:
    repl = REPL()
    try:
        repl.repl_loop()
    except EOFError:
        # graceful exit
        pass
    return 0

if __name__ == '__main__':
    raise SystemExit(main())