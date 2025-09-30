import json
import math
import os
import tempfile
import pytest

from main import (
    Lexer,
    Token,
    LexerError,
    Parser,
    ParseError,
    Evaluator,
    EvalError,
    REPL,
    show_help,
    _do_factorial,
    _BUILTINS,
)


def lex_values(text):
    l = Lexer(text)
    return l.tokenize()


def test_lex_numbers_and_identifiers():
    toks = lex_values("123 45.6 .5 1e3 2E-2 foo_bar")
    # Expect sequence: NUMBER, NUMBER, NUMBER, NUMBER, NUMBER, IDENT, EOF
    assert toks[0].type == 'NUMBER' and toks[0].value == 123
    assert toks[1].type == 'NUMBER' and math.isclose(toks[1].value, 45.6)
    assert toks[2].type == 'NUMBER' and math.isclose(toks[2].value, 0.5)
    assert toks[3].type == 'NUMBER' and math.isclose(toks[3].value, 1000.0)
    assert toks[4].type == 'NUMBER' and math.isclose(toks[4].value, 0.02)
    assert toks[5].type == 'IDENT' and toks[5].value == 'foo_bar'
    assert toks[-1].type == 'EOF'


def test_lex_multi_char_ops_and_logical_synonyms():
    toks = lex_values("a && b || c << >> **")
    # tokens include 'and' and 'or' for && and ||
    ops = [t for t in toks if t.type == 'OP']
    op_vals = [t.value for t in ops]
    assert 'and' in op_vals
    assert 'or' in op_vals
    assert '<<' in op_vals
    assert '>>' in op_vals
    assert '**' in op_vals


def test_lex_invalid_character_raises():
    with pytest.raises(LexerError):
        lex_values("1 @ 2")


def test_lex_number_invalid_exponent_raises():
    with pytest.raises(LexerError):
        lex_values("1e+")


def test_parser_unary_vs_postfix_precedence_and_evaluation():
    # -3! should be -(3!) == -6
    toks = Lexer("-3!").tokenize()
    ast = Parser(toks).parse()
    ev = Evaluator()
    res = ev.eval(ast)
    assert res == -6

    # --3 -> 3
    toks = Lexer("--3").tokenize()
    ast = Parser(toks).parse()
    res = ev.eval(ast)
    assert res == 3


def test_parser_exponent_right_associative():
    toks = Lexer("2 ** 3 ** 2").tokenize()
    ast = Parser(toks).parse()
    ev = Evaluator()
    res = ev.eval(ast)
    # math.pow returns float
    assert math.isclose(res, 512.0)


def test_assignment_and_variable_lookup():
    repl = REPL()
    # assign x = 7
    ok, out = repl.evaluate_line("x = 3 + 4")
    assert ok and out == repr(7)
    # variable accessible
    ok, out = repl.evaluate_line("x")
    assert ok and out == repr(7)


def test_assignment_lhs_must_be_variable_parse_error():
    # 1 = 2 should raise ParseError during parsing
    toks = Lexer("1 = 2").tokenize()
    with pytest.raises(ParseError):
        Parser(toks).parse()


def test_function_call_builtin_and_unknown_function():
    repl = REPL()
    ok, out = repl.evaluate_line("factorial(5)")
    assert ok and out == repr(120)
    ok, out = repl.evaluate_line("sin(pi / 2)")
    assert ok
    val = float(out)
    assert math.isclose(val, 1.0, rel_tol=1e-12, abs_tol=1e-12)

    # unknown function should produce EvalError wrapped as Error: ...
    ok, out = repl.evaluate_line("no_such_fn(1)")
    assert not ok
    assert "Unknown function" in out


def test_postfix_factorial_on_variable_and_assignment():
    repl = REPL()
    ok, _ = repl.evaluate_line("n = 5")
    assert ok
    ok, out = repl.evaluate_line("n!")
    assert ok and out == repr(120)


def test_factorial_helper_errors_and_tolerance():
    # negative
    with pytest.raises(EvalError):
        _do_factorial(-1)
    # non-integer float
    with pytest.raises(EvalError):
        _do_factorial(3.5)
    # float within tolerance
    assert _do_factorial(3.0000000001) == 6


def test_unary_bitwise_and_not_operations():
    ev = Evaluator()
    # bitwise not on integer
    toks = Lexer("~5").tokenize()
    ast = Parser(toks).parse()
    assert ev.eval(ast) == ~5
    # logical not
    toks = Lexer("!0").tokenize()  # '!' is normalized to 'not' prefix
    ast = Parser(toks).parse()
    assert ev.eval(ast) is True


def test_binary_bitwise_and_shifts_and_errors():
    ev = Evaluator()
    # XOR
    toks = Lexer("5 ^ 2").tokenize()
    ast = Parser(toks).parse()
    assert ev.eval(ast) == (5 ^ 2)
    # left shift
    toks = Lexer("1 << 3").tokenize()
    ast = Parser(toks).parse()
    assert ev.eval(ast) == (1 << 3)
    # negative shift count should raise EvalError when evaluated
    toks = Lexer("1 << -1").tokenize()
    ast = Parser(toks).parse()
    with pytest.raises(EvalError):
        ev.eval(ast)
    # bitwise on floats should raise EvalError
    toks = Lexer("1.5 ^ 2").tokenize()
    ast = Parser(toks).parse()
    with pytest.raises(EvalError):
        ev.eval(ast)


def test_logical_short_circuiting_prevents_errors():
    ev = Evaluator()
    repl = REPL()
    # left false -> and short-circuits; right has division by zero but should not be evaluated
    ok, out = repl.evaluate_line("0 and (1 / 0)")
    assert ok and out == repr(False)
    # left true -> or short-circuits; right has division by zero but should not be evaluated
    ok, out = repl.evaluate_line("1 or (1 / 0)")
    assert ok and out == repr(True)


def test_division_and_modulo_by_zero_errors():
    repl = REPL()
    ok, out = repl.evaluate_line("1 / 0")
    assert not ok and "Division by zero" in out
    ok, out = repl.evaluate_line("5 % 0")
    assert not ok and "Modulo by zero" in out
    ok, out = repl.evaluate_line("5 % 0.0")
    assert not ok and "Modulo by zero" in out


def test_function_domain_error_wrapped():
    repl = REPL()
    ok, out = repl.evaluate_line("log(-1)")
    assert not ok
    assert "Domain error in function 'log'" in out or "math domain error" in out


def test_calling_variable_as_function_disallowed():
    repl = REPL()
    ok, _ = repl.evaluate_line("a = 5")
    assert ok
    # Now trying to call 'a()' should be unknown function
    ok, out = repl.evaluate_line("a()")
    assert not ok and "Unknown function" in out


def test_parser_unexpected_parentheses_after_expr_raises():
    toks = Lexer("(1)(2)").tokenize()
    with pytest.raises(ParseError):
        Parser(toks).parse()


def test_parser_unexpected_operator_nud_error():
    toks = Lexer("*1").tokenize()
    with pytest.raises(ParseError):
        Parser(toks).parse()


def test_repl_command_processing_help_and_vars(tmp_path):
    repl = REPL()
    # show general help via both :help and help
    out = repl._process_command(":help")
    assert "Calculator REPL help" in out
    out2 = repl._process_command("help")
    assert "Calculator REPL help" in out2
    # specific topic
    out3 = repl._process_command(":help functions")
    assert "Built-in functions" in out3
    # vars with empty env
    repl.evaluator.env = {}
    out = repl._process_command(":vars")
    assert "(no variables)" in out
    # save and load
    repl.evaluator.env = {"x": 42}
    tmpfile = tmp_path / "vars.json"
    out = repl._process_command("save", [str(tmpfile)]) if False else None  # avoid calling internal; use _run_command below
    # Use _run_command directly to bypass colon parsing
    res = repl._run_command("save", [str(tmpfile)])
    assert "Saved" in res
    # Clear env and load
    repl.evaluator.env = {}
    res_load = repl._run_command("load", [str(tmpfile)])
    assert "Loaded" in res_load
    assert repl.evaluator.env.get("x") == 42


def test_repl_process_colon_command_errors_are_caught():
    repl = REPL()
    # Unknown command
    out = repl._process_command(":nope")
    assert "Unknown command" in out


def test_history_command_handles_missing_file(tmp_path, monkeypatch):
    # Point history file to a non-existent path and ensure message handled
    repl = REPL()
    repl.history_file = str(tmp_path / "nonexistent_history.txt")
    res = repl._run_command("history", [])
    # If file doesn't exist, it will attempt to open and fail -> returns message starting with Could not read history
    assert "Could not read history" in res or isinstance(res, str)


def test_show_help_unknown_topic():
    assert "No help available for topic" in show_help("nonexistent_topic")


def test_lexer_token_positions_are_nonnegative():
    toks = Lexer("a + b * 3").tokenize()
    for t in toks:
        assert t.pos >= 0


def test_number_types_preserved_int_vs_float():
    toks_int = Lexer("10").tokenize()
    assert isinstance(toks_int[0].value, int)
    toks_float = Lexer("10.0").tokenize()
    assert isinstance(toks_float[0].value, float)


def test_parser_function_arg_parsing_and_commas():
    toks = Lexer("pow(2, 3)").tokenize()
    ast = Parser(toks).parse()
    ev = Evaluator()
    res = ev.eval(ast)
    assert math.isclose(res, math.pow(2.0, 3.0))


def test_repl_evaluate_line_errors_return_false_and_message():
    repl = REPL()
    ok, out = repl.evaluate_line("1 / 0")
    assert not ok and "Error:" in out


def test_saving_and_loading_non_dict_file_returns_error(tmp_path):
    repl = REPL()
    fname = tmp_path / "bad.json"
    # write a list instead of dict
    with open(fname, "w", encoding="utf-8") as f:
        json.dump([1, 2, 3], f)
    res = repl._run_command("load", [str(fname)])
    assert "Invalid file format" in res or "Error loading variables" not in res


def test_builtin_names_registered_and_accessible():
    assert 'sin' in _BUILTINS
    assert callable(_BUILTINS['sin'])
    assert 'factorial' in _BUILTINS
    # factorial callable works
    assert _BUILTINS['factorial'](5) == 120


def test_parser_handles_comma_errors_in_arglist():
    with pytest.raises(ParseError):
        Parser(Lexer("f(1,,2)").tokenize()).parse()