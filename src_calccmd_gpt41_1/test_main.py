# test_main.py

import sys
import math
import types
import pytest

import main
from main import (
    CalculatorError, ParseError, EvalError,
    Tokenizer, TokenType, Token,
    ASTNode, NumberNode, UnaryOpNode, BinOpNode,
    Parser, Evaluator, HelpHandler, CLIHandler
)

# ---------------------------
# Tokenizer Tests
# ---------------------------

def test_tokenizer_simple_expression():
    t = Tokenizer("1 + 2")
    types_ = [token.type for token in t.tokens]
    values = [token.value for token in t.tokens]
    assert types_ == [
        TokenType.NUMBER, TokenType.PLUS, TokenType.NUMBER, TokenType.EOF
    ]
    assert values == [1, '+', 2, None]

def test_tokenizer_float_and_int():
    t = Tokenizer("3.14 + 2")
    assert t.tokens[0].type == TokenType.NUMBER
    assert t.tokens[0].value == 3.14
    assert t.tokens[2].type == TokenType.NUMBER
    assert t.tokens[2].value == 2

def test_tokenizer_parentheses_and_operators():
    t = Tokenizer("(1-2)*3/4")
    types_ = [token.type for token in t.tokens]
    assert types_ == [
        TokenType.LPAREN, TokenType.NUMBER, TokenType.MINUS, TokenType.NUMBER,
        TokenType.RPAREN, TokenType.MUL, TokenType.NUMBER, TokenType.DIV, TokenType.NUMBER, TokenType.EOF
    ]

def test_tokenizer_skip_whitespace():
    t = Tokenizer("  1\t+\t2  ")
    types_ = [token.type for token in t.tokens]
    assert types_ == [
        TokenType.NUMBER, TokenType.PLUS, TokenType.NUMBER, TokenType.EOF
    ]

def test_tokenizer_unexpected_character():
    with pytest.raises(ParseError) as e:
        Tokenizer("1 + $")
    assert "Unexpected character" in str(e.value)

# ---------------------------
# Parser Tests
# ---------------------------

def parse_expr(expr):
    tokenizer = Tokenizer(expr)
    parser = Parser(tokenizer)
    return parser.parse()

def test_parser_number():
    ast = parse_expr("42")
    assert isinstance(ast, NumberNode)
    assert ast.value == 42

def test_parser_simple_addition():
    ast = parse_expr("1 + 2")
    assert isinstance(ast, BinOpNode)
    assert isinstance(ast.left, NumberNode)
    assert ast.left.value == 1
    assert ast.op == TokenType.PLUS
    assert isinstance(ast.right, NumberNode)
    assert ast.right.value == 2

def test_parser_operator_precedence():
    ast = parse_expr("1 + 2 * 3")
    # Should parse as 1 + (2 * 3)
    assert isinstance(ast, BinOpNode)
    assert ast.op == TokenType.PLUS
    assert isinstance(ast.right, BinOpNode)
    assert ast.right.op == TokenType.MUL

def test_parser_parentheses():
    ast = parse_expr("(1 + 2) * 3")
    # Should parse as (1 + 2) * 3
    assert isinstance(ast, BinOpNode)
    assert ast.op == TokenType.MUL
    assert isinstance(ast.left, BinOpNode)
    assert ast.left.op == TokenType.PLUS

def test_parser_multiple_unary_minus():
    ast = parse_expr("--3")
    # Should parse as UnaryOpNode(MINUS, UnaryOpNode(MINUS, NumberNode(3)))
    assert isinstance(ast, NumberNode)
    assert ast.value == 3

    ast = parse_expr("---3")
    assert isinstance(ast, UnaryOpNode)
    assert ast.op == TokenType.MINUS
    assert isinstance(ast.operand, NumberNode)
    assert ast.operand.value == 3

def test_parser_nested_parentheses():
    ast = parse_expr("((2))")
    assert isinstance(ast, NumberNode)
    assert ast.value == 2

def test_parser_missing_parenthesis():
    with pytest.raises(ParseError) as e:
        parse_expr("(1 + 2")
    assert "Expected token RPAREN" in str(e.value)

def test_parser_unexpected_token():
    with pytest.raises(ParseError) as e:
        parse_expr("1 + + 2")
    assert "Expected number or '('" in str(e.value)

def test_parser_trailing_garbage():
    with pytest.raises(ParseError) as e:
        parse_expr("1 + 2 3")
    assert "Unexpected token" in str(e.value)

# ---------------------------
# Evaluator Tests
# ---------------------------

def eval_expr(expr):
    ast = parse_expr(expr)
    return Evaluator().eval(ast)

@pytest.mark.parametrize("expr,expected", [
    ("1 + 2", 3),
    ("2 - 5", -3),
    ("2 * 3", 6),
    ("8 / 2", 4.0),
    ("2 + 3 * 4", 14),
    ("(2 + 3) * 4", 20),
    ("-5", -5),
    ("--5", 5),
    ("---5", -5),
    ("3.5 + 2.5", 6.0),
    ("-3.5 * 2", -7.0),
    ("1 + 2 + 3 + 4", 10),
    ("10 / 4", 2.5),
    ("(1 + 2) * (3 + 4)", 21),
    ("-(-(-2))", -2),
])
def test_evaluator_basic(expr, expected):
    result = eval_expr(expr)
    if isinstance(expected, float):
        assert math.isclose(result, expected)
    else:
        assert result == expected

def test_evaluator_division_by_zero():
    with pytest.raises(EvalError) as e:
        eval_expr("1 / 0")
    assert "Division by zero" in str(e.value)

def test_evaluator_invalid_unary_operator():
    node = UnaryOpNode("INVALID", NumberNode(1))
    with pytest.raises(EvalError) as e:
        Evaluator().eval(node)
    assert "Unknown unary operator" in str(e.value)

def test_evaluator_invalid_binary_operator():
    node = BinOpNode(NumberNode(1), "INVALID", NumberNode(2))
    with pytest.raises(EvalError) as e:
        Evaluator().eval(node)
    assert "Unknown binary operator" in str(e.value)

def test_evaluator_invalid_ast_node():
    class DummyNode(ASTNode):
        pass
    node = DummyNode()
    with pytest.raises(EvalError) as e:
        Evaluator().eval(node)
    assert "Invalid AST node" in str(e.value)

# ---------------------------
# HelpHandler Tests
# ---------------------------

def test_help_handler_prints_help(capsys):
    HelpHandler.print_help()
    out = capsys.readouterr().out
    assert "Command-Line Calculator Help" in out
    assert "Supported operations" in out
    assert "Examples" in out

# ---------------------------
# CLIHandler Tests
# ---------------------------

def test_cli_handler_exit(monkeypatch, capsys):
    # Simulate user typing 'exit'
    inputs = iter(['exit'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    cli = CLIHandler()
    cli.run()
    out = capsys.readouterr().out
    assert "Goodbye!" in out

def test_cli_handler_quit(monkeypatch, capsys):
    # Simulate user typing 'quit'
    inputs = iter(['quit'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    cli = CLIHandler()
    cli.run()
    out = capsys.readouterr().out
    assert "Goodbye!" in out

def test_cli_handler_help(monkeypatch, capsys):
    # Simulate user typing 'help' then 'exit'
    inputs = iter(['help', 'exit'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    cli = CLIHandler()
    cli.run()
    out = capsys.readouterr().out
    assert "Command-Line Calculator Help" in out

def test_cli_handler_empty_line(monkeypatch, capsys):
    # Simulate user typing empty line then 'exit'
    inputs = iter(['', 'exit'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    cli = CLIHandler()
    cli.run()
    out = capsys.readouterr().out
    assert "Goodbye!" in out

def test_cli_handler_valid_expression(monkeypatch, capsys):
    # Simulate user typing '2 + 2' then 'exit'
    inputs = iter(['2 + 2', 'exit'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    cli = CLIHandler()
    cli.run()
    out = capsys.readouterr().out
    assert "4" in out

def test_cli_handler_invalid_expression(monkeypatch, capsys):
    # Simulate user typing '1 / 0' then 'exit'
    inputs = iter(['1 / 0', 'exit'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    cli = CLIHandler()
    cli.run()
    out = capsys.readouterr().out
    assert "Error: Division by zero" in out

def test_cli_handler_parse_error(monkeypatch, capsys):
    # Simulate user typing '1 +' then 'exit'
    inputs = iter(['1 +', 'exit'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    cli = CLIHandler()
    cli.run()
    out = capsys.readouterr().out
    assert "Error:" in out

def test_cli_handler_keyboard_interrupt(monkeypatch, capsys):
    # Simulate KeyboardInterrupt on input
    def raise_keyboard_interrupt(_):
        raise KeyboardInterrupt
    monkeypatch.setattr('builtins.input', raise_keyboard_interrupt)
    cli = CLIHandler()
    cli.run()
    out = capsys.readouterr().out
    # Should print a newline and exit cleanly
    assert out.startswith('\n') or out == '\n'

def test_cli_handler_eof(monkeypatch, capsys):
    # Simulate EOFError on input
    def raise_eof(_):
        raise EOFError
    monkeypatch.setattr('builtins.input', raise_eof)
    cli = CLIHandler()
    cli.run()
    out = capsys.readouterr().out
    assert out.startswith('\n') or out == '\n'

# ---------------------------
# Main Entry Point Test
# ---------------------------

def test_main_entry_point(monkeypatch, capsys):
    # Simulate 'exit' to exit immediately
    inputs = iter(['exit'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    main.main()
    out = capsys.readouterr().out
    assert "Welcome to the Command-Line Calculator!" in out
    assert "Type 'help' for instructions" in out
    assert "Goodbye!" in out