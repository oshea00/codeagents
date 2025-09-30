import math
import operator
import sys
import types
import pytest

from main import (
    FunctionLibrary,
    SymbolTable,
    Tokenizer,
    Parser,
    Evaluator,
    NumberNode,
    VariableNode,
    UnaryOpNode,
    BinaryOpNode,
    AssignNode,
    FunctionCallNode,
    ASTNode,
    EOF,
    IDENTIFIER,
    NUMBER,
    OPERATOR,
    LPAREN,
    RPAREN,
    COMMA,
    ASSIGN,
)

# ---------------------------
# FunctionLibrary Tests
# ---------------------------

def test_function_library_is_function_and_list():
    fl = FunctionLibrary()
    assert fl.is_function('sin')
    assert fl.is_function('log')
    assert not fl.is_function('notafunc')
    funcs = fl.list_functions()
    assert 'sin' in funcs
    assert 'cos' in funcs
    assert 'factorial' in funcs
    assert 'bin' in funcs

def test_function_library_call_valid():
    fl = FunctionLibrary()
    assert fl.call('sin', [math.pi/2]) == pytest.approx(1.0)
    assert fl.call('cos', [0]) == pytest.approx(1.0)
    assert fl.call('sqrt', [9]) == 3.0
    assert fl.call('abs', [-5]) == 5
    assert fl.call('factorial', [5]) == 120
    assert fl.call('bin', [10]) == '0b1010'
    assert fl.call('hex', [255]) == '0xff'
    assert fl.call('oct', [8]) == '0o10'
    assert fl.call('round', [2.7]) == 3

def test_function_library_call_invalid_function():
    fl = FunctionLibrary()
    with pytest.raises(ValueError):
        fl.call('notafunc', [1])

def test_function_library_call_wrong_arity():
    fl = FunctionLibrary()
    with pytest.raises(ValueError):
        fl.call('sin', [1,2])
    with pytest.raises(ValueError):
        fl.call('cos', [])

def test_function_library_call_math_error():
    fl = FunctionLibrary()
    with pytest.raises(ValueError):
        fl.call('sqrt', [-1])
    with pytest.raises(ValueError):
        fl.call('factorial', [-2])

# ---------------------------
# SymbolTable Tests
# ---------------------------

def test_symbol_table_predefined_constants():
    st = SymbolTable()
    assert st.get('pi') == math.pi
    assert st.get('e') == math.e
    assert st.get('tau') == math.tau

def test_symbol_table_set_and_get():
    st = SymbolTable()
    st.set('x', 42)
    assert st.get('x') == 42
    st.set('y', 3.14)
    assert st.get('y') == 3.14

def test_symbol_table_undefined_variable():
    st = SymbolTable()
    with pytest.raises(NameError):
        st.get('not_defined')

def test_symbol_table_list_symbols():
    st = SymbolTable()
    st.set('foo', 1)
    st.set('bar', 2)
    symbols = st.list_symbols()
    assert 'foo' in symbols
    assert 'bar' in symbols
    assert 'pi' in symbols
    assert 'e' in symbols

# ---------------------------
# Tokenizer Tests
# ---------------------------

def test_tokenizer_basic_tokens():
    t = Tokenizer("x = 3 + 4.5 * (y - 2)")
    types = [tok.type for tok in t.tokens]
    values = [tok.value for tok in t.tokens]
    assert types[:7] == [IDENTIFIER, ASSIGN, NUMBER, OPERATOR, NUMBER, OPERATOR, LPAREN]
    assert values[0] == 'x'
    assert values[2] == 3
    assert values[4] == 4.5

def test_tokenizer_operators_and_numbers():
    t = Tokenizer("a<<2 | b&3 ^ ~c")
    ops = [tok.value for tok in t.tokens if tok.type == OPERATOR]
    assert ops == ['<<', '|', '&', '^', '~']
    nums = [tok.value for tok in t.tokens if tok.type == NUMBER]
    assert nums == [2, 3]

def test_tokenizer_function_call():
    t = Tokenizer("sin(3.14)")
    types = [tok.type for tok in t.tokens]
    assert types == [IDENTIFIER, LPAREN, NUMBER, RPAREN, EOF]

def test_tokenizer_invalid_character():
    with pytest.raises(SyntaxError):
        Tokenizer("2 + $")

def test_tokenizer_scientific_notation():
    t = Tokenizer("1e3 + 2.5E-2")
    nums = [tok.value for tok in t.tokens if tok.type == NUMBER]
    assert nums[0] == pytest.approx(1000.0)
    assert nums[1] == pytest.approx(0.025)

# ---------------------------
# Parser Tests
# ---------------------------

def parse_expr(expr):
    fl = FunctionLibrary()
    t = Tokenizer(expr)
    p = Parser(t, fl)
    return p.parse()

def test_parser_number_and_variable():
    node = parse_expr("42")
    assert isinstance(node, NumberNode)
    assert node.value == 42
    node = parse_expr("foo")
    assert isinstance(node, VariableNode)
    assert node.name == "foo"

def test_parser_assignment():
    node = parse_expr("x = 5")
    assert isinstance(node, AssignNode)
    assert node.name == "x"
    assert isinstance(node.expr, NumberNode)
    assert node.expr.value == 5

def test_parser_arithmetic_precedence():
    node = parse_expr("2 + 3 * 4")
    assert isinstance(node, BinaryOpNode)
    assert node.op == '+'
    assert isinstance(node.right, BinaryOpNode)
    assert node.right.op == '*'

def test_parser_parentheses():
    node = parse_expr("(2 + 3) * 4")
    assert isinstance(node, BinaryOpNode)
    assert node.op == '*'
    assert isinstance(node.left, BinaryOpNode)
    assert node.left.op == '+'

def test_parser_unary_minus():
    node = parse_expr("-5")
    assert isinstance(node, UnaryOpNode)
    assert node.op == '-'
    assert isinstance(node.operand, NumberNode)
    assert node.operand.value == 5

def test_parser_multiple_unary_minus():
    node = parse_expr("--3")
    assert isinstance(node, UnaryOpNode)
    assert node.op == '-'
    assert isinstance(node.operand, UnaryOpNode)
    assert node.operand.op == '-'
    assert isinstance(node.operand.operand, NumberNode)
    assert node.operand.operand.value == 3

def test_parser_bitwise_ops():
    node = parse_expr("a & b | c ^ d << 2 >> 1")
    # Just check the top-level node is '|'
    assert isinstance(node, BinaryOpNode)
    assert node.op == '|'

def test_parser_function_call():
    node = parse_expr("sin(0)")
    assert isinstance(node, FunctionCallNode)
    assert node.name == 'sin'
    assert len(node.args) == 1
    assert isinstance(node.args[0], NumberNode)
    assert node.args[0].value == 0

def test_parser_function_call_multiple_args():
    node = parse_expr("round(2.7)")
    assert isinstance(node, FunctionCallNode)
    assert node.name == 'round'
    assert len(node.args) == 1

def test_parser_invalid_syntax():
    with pytest.raises(SyntaxError):
        parse_expr("2 + * 3")
    with pytest.raises(SyntaxError):
        parse_expr("sin(1, 2)")  # Too many args for sin

def test_parser_unexpected_token():
    with pytest.raises(SyntaxError):
        parse_expr(")")

# ---------------------------
# Evaluator Tests
# ---------------------------

def eval_expr(expr, st=None):
    fl = FunctionLibrary()
    if st is None:
        st = SymbolTable()
    t = Tokenizer(expr)
    p = Parser(t, fl)
    ast = p.parse()
    ev = Evaluator(st, fl)
    return ev.eval(ast)

def test_evaluator_basic_arithmetic():
    assert eval_expr("2 + 3 * 4") == 14
    assert eval_expr("(2 + 3) * 4") == 20
    assert eval_expr("10 / 2") == 5.0
    assert eval_expr("7 - 5") == 2

def test_evaluator_unary_minus_and_invert():
    assert eval_expr("-5") == -5
    assert eval_expr("~5") == ~5
    assert eval_expr("--3") == 3

def test_evaluator_bitwise_ops():
    assert eval_expr("5 & 3") == 1
    assert eval_expr("5 | 2") == 7
    assert eval_expr("5 ^ 1") == 4
    assert eval_expr("2 << 3") == 16
    assert eval_expr("8 >> 2") == 2

def test_evaluator_float_bitwise_casts():
    assert eval_expr("5.9 & 3.1") == 1
    assert eval_expr("5.9 | 2.1") == 7

def test_evaluator_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        eval_expr("5 / 0")

def test_evaluator_variable_assignment_and_usage():
    st = SymbolTable()
    assert eval_expr("x = 7", st) == 7
    assert st.get('x') == 7
    assert eval_expr("x + 3", st) == 10

def test_evaluator_undefined_variable():
    with pytest.raises(NameError):
        eval_expr("foo + 1")

def test_evaluator_function_calls():
    assert eval_expr("sin(0)") == pytest.approx(0.0)
    assert eval_expr("cos(0)") == pytest.approx(1.0)
    assert eval_expr("sqrt(9)") == 3.0
    assert eval_expr("factorial(5)") == 120
    assert eval_expr("abs(-7)") == 7
    assert eval_expr("round(2.7)") == 3
    assert eval_expr("bin(10)") == '0b1010'
    assert eval_expr("hex(255)") == '0xff'
    assert eval_expr("oct(8)") == '0o10'

def test_evaluator_function_call_wrong_arity():
    with pytest.raises(ValueError):
        eval_expr("sin()")
    with pytest.raises(ValueError):
        eval_expr("cos(1,2)")

def test_evaluator_function_call_math_error():
    with pytest.raises(ValueError):
        eval_expr("sqrt(-1)")
    with pytest.raises(ValueError):
        eval_expr("factorial(-2)")

def test_evaluator_constants():
    assert eval_expr("pi") == math.pi
    assert eval_expr("e") == math.e
    assert eval_expr("tau") == math.tau

def test_evaluator_chained_assignments():
    st = SymbolTable()
    assert eval_expr("x = 2", st) == 2
    assert eval_expr("y = x + 3", st) == 5
    assert st.get('y') == 5

def test_evaluator_multiple_unary_minus():
    assert eval_expr("--3") == 3
    assert eval_expr("---3") == -3

def test_evaluator_parentheses_and_precedence():
    assert eval_expr("2 + 3 * 4") == 14
    assert eval_expr("(2 + 3) * 4") == 20
    assert eval_expr("2 + (3 * 4)") == 14

def test_evaluator_logic_and_scientific_mix():
    st = SymbolTable()
    eval_expr("x = 5", st)
    assert eval_expr("sin(x)", st) == pytest.approx(math.sin(5))
    assert eval_expr("x & 3", st) == 1

def test_evaluator_invalid_operator():
    class DummyNode(ASTNode):
        pass
    ev = Evaluator(SymbolTable(), FunctionLibrary())
    with pytest.raises(TypeError):
        ev.eval(DummyNode())

def test_evaluator_unsupported_unary_operator():
    node = UnaryOpNode('!', NumberNode(5))
    ev = Evaluator(SymbolTable(), FunctionLibrary())
    with pytest.raises(ValueError):
        ev.eval(node)

def test_evaluator_unsupported_binary_operator():
    node = BinaryOpNode(NumberNode(1), '**', NumberNode(2))
    ev = Evaluator(SymbolTable(), FunctionLibrary())
    with pytest.raises(ValueError):
        ev.eval(node)

def test_evaluator_bitwise_float_cast():
    assert eval_expr("3.9 & 2.1") == 2

def test_evaluator_function_call_with_variable():
    st = SymbolTable()
    eval_expr("x = 0", st)
    assert eval_expr("cos(x)", st) == pytest.approx(1.0)

def test_evaluator_function_call_with_expression():
    assert eval_expr("sin(3.141592653589793/2)") == pytest.approx(1.0)

def test_evaluator_large_expression():
    expr = "((2 + 3) * (4 + 5) - 6) / 3"
    assert eval_expr(expr) == pytest.approx(13.0)

def test_evaluator_logic_bitwise_chain():
    assert eval_expr("1 | 2 & 3 ^ 4") == 1 | (2 & 3) ^ 4

def test_evaluator_shift_ops():
    assert eval_expr("1 << 4") == 16
    assert eval_expr("16 >> 2") == 4

def test_evaluator_rounding():
    assert eval_expr("round(2.5)") == 2  # Python's round uses "banker's rounding"
    assert eval_expr("round(3.5)") == 4

def test_evaluator_deg_rad():
    assert eval_expr("deg(pi)") == pytest.approx(180.0)
    assert eval_expr("rad(180)") == pytest.approx(math.pi)

def test_evaluator_log_log10():
    assert eval_expr("log(e)") == pytest.approx(1.0)
    assert eval_expr("log10(100)") == pytest.approx(2.0)

def test_evaluator_floor_ceil():
    assert eval_expr("floor(2.7)") == 2
    assert eval_expr("ceil(2.1)") == 3

def test_evaluator_assign_and_use_in_expression():
    st = SymbolTable()
    eval_expr("x = 2", st)
    eval_expr("y = 3", st)
    assert eval_expr("x * y + 1", st) == 7

def test_evaluator_assign_overwrite():
    st = SymbolTable()
    eval_expr("x = 2", st)
    eval_expr("x = 5", st)
    assert st.get('x') == 5

def test_evaluator_assign_to_constant():
    st = SymbolTable()
    eval_expr("pi = 3", st)
    assert st.get('pi') == 3

def test_evaluator_assign_and_use_constant():
    st = SymbolTable()
    eval_expr("x = pi", st)
    assert st.get('x') == math.pi

def test_evaluator_assign_and_use_function_result():
    st = SymbolTable()
    eval_expr("x = sin(pi/2)", st)
    assert st.get('x') == pytest.approx(1.0)

def test_evaluator_assign_and_use_bitwise():
    st = SymbolTable()
    eval_expr("x = 5 & 3", st)
    assert st.get('x') == 1

def test_evaluator_assign_and_use_shift():
    st = SymbolTable()
    eval_expr("x = 1 << 3", st)
    assert st.get('x') == 8

def test_evaluator_assign_and_use_unary():
    st = SymbolTable()
    eval_expr("x = -5", st)
    assert st.get('x') == -5

def test_evaluator_assign_and_use_multiple():
    st = SymbolTable()
    eval_expr("x = 2", st)
    eval_expr("y = 3", st)
    eval_expr("z = x + y", st)
    assert st.get('z') == 5

def test_evaluator_assign_and_use_in_function():
    st = SymbolTable()
    eval_expr("x = 0", st)
    assert eval_expr("sin(x)", st) == pytest.approx(0.0)

def test_evaluator_assign_and_use_in_complex_expr():
    st = SymbolTable()
    eval_expr("x = 2", st)
    eval_expr("y = 3", st)
    assert eval_expr("x * (y + 1)", st) == 8

def test_evaluator_assign_and_use_in_nested_expr():
    st = SymbolTable()
    eval_expr("x = 2", st)
    eval_expr("y = x + 3", st)
    assert eval_expr("y * 2", st) == 10

def test_evaluator_assign_and_use_in_bitwise_expr():
    st = SymbolTable()
    eval_expr("x = 5", st)
    assert eval_expr("x & 3", st) == 1

def test_evaluator_assign_and_use_in_shift_expr():
    st = SymbolTable()
    eval_expr("x = 1", st)
    assert eval_expr("x << 3", st) == 8

def test_evaluator_assign_and_use_in_unary_expr():
    st = SymbolTable()
    eval_expr("x = 5", st)
    assert eval_expr("-x", st) == -5

def test_evaluator_assign_and_use_in_function_expr():
    st = SymbolTable()
    eval_expr("x = 0", st)
    assert eval_expr("cos(x)", st) == pytest.approx(1.0)

def test_evaluator_assign_and_use_in_complex_function_expr():
    st = SymbolTable()
    eval_expr("x = pi/2", st)
    assert eval_expr("sin(x)", st) == pytest.approx(1.0)

def test_evaluator_assign_and_use_in_nested_function_expr():
    st = SymbolTable()
    eval_expr("x = 2", st)
    eval_expr("y = x + 3", st)
    assert eval_expr("sin(y)", st) == pytest.approx(math.sin(5))

def test_evaluator_assign_and_use_in_bitwise_function_expr():
    st = SymbolTable()
    eval_expr("x = 5", st)
    assert eval_expr("bin(x)", st) == '0b101'

def test_evaluator_assign_and_use_in_shift_function_expr():
    st = SymbolTable()
    eval_expr("x = 1", st)
    assert eval_expr("hex(x << 3)", st) == '0x8'

def test_evaluator_assign_and_use_in_unary_function_expr():
    st = SymbolTable()
    eval_expr("x = 5", st)
    assert eval_expr("abs(-x)", st) == 5

def test_evaluator_assign_and_use_in_complex_nested_expr():
    st = SymbolTable()
    eval_expr("x = 2", st)
    eval_expr("y = x + 3", st)
    eval_expr("z = y * 2", st)
    assert eval_expr("z + 1", st) == 11

def test_evaluator_assign_and_use_in_complex_nested_function_expr():
    st = SymbolTable()
    eval_expr("x = 2", st)
    eval_expr("y = x + 3", st)
    eval_expr("z = y * 2", st)
    assert eval_expr("sin(z)", st) == pytest.approx(math.sin(10))

def test_evaluator_assign_and_use_in_complex_nested_bitwise_expr():
    st = SymbolTable()
    eval_expr("x = 5", st)
    eval_expr("y = x & 3", st)
    assert eval_expr("y | 2", st) == 3

def test_evaluator_assign_and_use_in_complex_nested_shift_expr():
    st = SymbolTable()
    eval_expr("x = 1", st)
    eval_expr("y = x << 3", st)
    assert eval_expr("y >> 2", st) == 2

def test_evaluator_assign_and_use_in_complex_nested_unary_expr():
    st = SymbolTable()
    eval_expr("x = 5", st)
    eval_expr("y = -x", st)
    assert eval_expr("abs(y)", st) == 5

def test_evaluator_assign_and_use_in_complex_nested_function_expr_2():
    st = SymbolTable()
    eval_expr("x = 2", st)
    eval_expr("y = x + 3", st)
    eval_expr("z = y * 2", st)
    assert eval_expr("cos(z)", st) == pytest.approx(math.cos(10))

def test_evaluator_assign_and_use_in_complex_nested_bitwise_function_expr():
    st = SymbolTable()
    eval_expr("x = 5", st)
    eval_expr("y = x & 3", st)
    assert eval_expr("bin(y)", st) == '0b1'

def test_evaluator_assign_and_use_in_complex_nested_shift_function_expr():
    st = SymbolTable()
    eval_expr("x = 1", st)
    eval_expr("y = x << 3", st)
    assert eval_expr("hex(y)", st) == '0x8'

def test_evaluator_assign_and_use_in_complex_nested_unary_function_expr():
    st = SymbolTable()
    eval_expr("x = 5", st)
    eval_expr("y = -x", st)
    assert eval_expr("abs(y)", st) == 5

def test_evaluator_assign_and_use_in_complex_nested_function_expr_3():
    st = SymbolTable()
    eval_expr("x = 2", st)
    eval_expr("y = x + 3", st)
    eval_expr("z = y * 2", st)
    assert eval_expr("sin(z)", st) == pytest.approx(math.sin(10))

def test_evaluator_assign_and_use_in_complex_nested_function_expr_4():
    st = SymbolTable()
    eval_expr("x = 2", st)
    eval_expr("y = x + 3", st)
    eval_expr("z = y * 2", st)
    assert eval_expr("cos(z)", st) == pytest.approx(math.cos(10))

def test_evaluator_assign_and_use_in_complex_nested_function_expr_5():
    st = SymbolTable()
    eval_expr("x = 2", st)
    eval_expr("y = x + 3", st)
    eval_expr("z = y * 2", st)
    assert eval_expr("tan(z)", st) == pytest.approx(math.tan(10))

def test_evaluator_assign_and_use_in_complex_nested_function_expr_6():
    st = SymbolTable()
    eval_expr("x = 2", st)
    eval_expr("y = x + 3", st)
    eval_expr("z = y * 2", st)
    assert eval_expr("exp(z)", st) == pytest.approx(math.exp(10))

def test_evaluator_assign_and_use_in_complex_nested_function_expr_7():
    st = SymbolTable()
    eval_expr("x = 2", st)
    eval_expr("y = x + 3", st)
    eval_expr("z = y * 2", st)
    assert eval_expr("log(z)", st) == pytest.approx(math.log(10))

def test_evaluator_assign_and_use_in_complex_nested_function_expr_8():
    st = SymbolTable()
    eval_expr("x = 2", st)
    eval_expr("y = x + 3", st)
    eval_expr("z = y * 2", st)
    assert eval_expr("log10(z)", st) == pytest.approx(math.log10(10))

def test_evaluator_assign_and_use_in_complex_nested_function_expr_9():
    st = SymbolTable()
    eval_expr("x = 2", st)
    eval_expr("y = x + 3", st)
    eval_expr("z = y * 2", st)
    assert eval_expr("sqrt(z)", st) == pytest.approx(math.sqrt(10))

def test_evaluator_assign_and_use_in_complex_nested_function_expr_10():
    st = SymbolTable()
    eval_expr("x = 2", st)
    eval_expr("y = x + 3", st)
    eval_expr("z = y * 2", st)
    assert eval_expr("abs(-z)", st) == 10

def test_evaluator_assign_and_use_in_complex_nested_function_expr_11():
    st = SymbolTable()
    eval_expr("x = 2", st)
    eval_expr("y = x + 3", st)
    eval_expr("z = y * 2", st)
    assert eval_expr("factorial(5)") == 120

def test_evaluator_assign_and_use_in_complex_nested_function_expr_12():
    st = SymbolTable()
    eval_expr("x = 2", st)
    eval_expr("y = x + 3", st)
    eval_expr("z = y * 2", st)
    assert eval_expr("floor(2.7)") == 2

def test_evaluator_assign_and_use_in_complex_nested_function_expr_13():
    st = SymbolTable()
    eval_expr("x = 2", st)
    eval_expr("y = x + 3", st)
    eval_expr("z = y * 2", st)
    assert eval_expr("ceil(2.1)") == 3

def test_evaluator_assign_and_use_in_complex_nested_function_expr_14():
    st = SymbolTable()
    eval_expr("x = 2", st)
    eval_expr("y = x + 3", st)
    eval_expr("z = y * 2", st)
    assert eval_expr("round(2.7)") == 3

def test_evaluator_assign_and_use_in_complex_nested_function_expr_15():
    st = SymbolTable()
    eval_expr("x = 2", st)
    eval_expr("y = x + 3", st)
    eval_expr("z = y * 2", st)
    assert eval_expr("deg(pi)") == pytest.approx(180.0)

def test_evaluator_assign_and_use_in_complex_nested_function_expr_16():
    st = SymbolTable()
    eval_expr("x = 2", st)
    eval_expr("y = x + 3", st)
    eval_expr("z = y * 2", st)
    assert eval_expr("rad(180)") == pytest.approx(math.pi)

def test_evaluator_assign_and_use_in_complex_nested_function_expr_17():
    st = SymbolTable()
    eval_expr("x = 2", st)
    eval_expr("y = x + 3", st)
    eval_expr("z = y * 2", st)
    assert eval_expr("bin(z)", st) == '0b1010'

def test_evaluator_assign_and_use_in_complex_nested_function_expr_18():
    st = SymbolTable()
    eval_expr("x = 2", st)
    eval_expr("y = x + 3", st)
    eval_expr("z = y * 2", st)
    assert eval_expr("hex(z)", st) == '0xa'

def test_evaluator_assign_and_use_in_complex_nested_function_expr_19():
    st = SymbolTable()
    eval_expr("x = 2", st)
    eval_expr("y = x + 3", st)
    eval_expr("z = y * 2", st)
    assert eval_expr("oct(z)", st) == '0o12'

def test_evaluator_assign_and_use_in_complex_nested_function_expr_20():
    st = SymbolTable()
    eval_expr("x = 2", st)
    eval_expr("y = x + 3", st)
    eval_expr("z = y * 2", st)
    assert eval_expr("round(z/3)", st) == 3

# ---------------------------
# REPL Command/Help/Vars/Funcs (Non-interactive)
# ---------------------------

def test_help_text_present():
    import main
    assert "Scientific Calculator REPL" in main.HELP_TEXT
    assert "Arithmetic" in main.HELP_TEXT
    assert "Bitwise/Logic" in main.HELP_TEXT
    assert "Scientific functions" in main.HELP_TEXT
    assert "Variable assignment" in main.HELP_TEXT
    assert "Parentheses for grouping" in main.HELP_TEXT
    assert "help" in main.HELP_TEXT
    assert "vars" in main.HELP_TEXT
    assert "funcs" in main.HELP_TEXT

def test_repl_print_vars_and_funcs(capsys):
    from main import REPL
    repl = REPL()
    # Add some variables
    repl.symbol_table.set('foo', 123)
    repl.symbol_table.set('bar', 456)
    repl._print_vars()
    out = capsys.readouterr().out
    assert "foo = 123" in out
    assert "bar = 456" in out
    repl._print_funcs()
    out = capsys.readouterr().out
    assert "Available functions:" in out
    assert "sin" in out
    assert "cos" in out
    assert "factorial" in out

def test_repl_print_vars_empty(capsys):
    from main import REPL
    repl = REPL()
    # Remove all user variables
    repl.symbol_table.symbols = {k: v for k, v in repl.symbol_table.symbols.items() if k in ('pi', 'e', 'tau')}
    repl._print_vars()
    out = capsys.readouterr().out
    assert "pi" in out
    assert "e" in out
    assert "tau" in out

# ---------------------------
# Edge Cases
# ---------------------------

def test_tokenizer_empty_input():
    t = Tokenizer("")
    assert t.tokens[0].type == EOF

def test_parser_empty_input():
    fl = FunctionLibrary()
    t = Tokenizer("")
    p = Parser(t, fl)
    with pytest.raises(SyntaxError):
        p.parse()

def test_parser_only_whitespace():
    fl = FunctionLibrary()
    t = Tokenizer("   ")
    p = Parser(t, fl)
    with pytest.raises(SyntaxError):
        p.parse()

def test_tokenizer_only_whitespace():
    t = Tokenizer("   ")
    assert t.tokens[0].type == EOF

def test_parser_invalid_function_call():
    with pytest.raises(SyntaxError):
        parse_expr("sin(")

def test_parser_missing_rparen():
    with pytest.raises(SyntaxError):
        parse_expr("(2 + 3")

def test_parser_extra_token():
    with pytest.raises(SyntaxError):
        parse_expr("2 + 3 4")

def test_parser_invalid_assignment():
    with pytest.raises(SyntaxError):
        parse_expr("= 5")

def test_parser_invalid_identifier():
    with pytest.raises(SyntaxError):
        parse_expr("3x = 5")

def test_parser_invalid_operator_sequence():
    with pytest.raises(SyntaxError):
        parse_expr("2 + + 3")

def test_parser_invalid_comma():
    with pytest.raises(SyntaxError):
        parse_expr("sin(1, )")

def test_parser_invalid_function_name():
    with pytest.raises(ValueError):
        eval_expr("notafunc(1)")

def test_parser_invalid_function_args():
    with pytest.raises(ValueError):
        eval_expr("sin(1,2)")

def test_evaluator_invalid_node_type():
    class Dummy(ASTNode):
        pass
    ev = Evaluator(SymbolTable(), FunctionLibrary())
    with pytest.raises(TypeError):
        ev.eval(Dummy())

def test_evaluator_invalid_unary_operator():
    node = UnaryOpNode('!', NumberNode(1))
    ev = Evaluator(SymbolTable(), FunctionLibrary())
    with pytest.raises(ValueError):
        ev.eval(node)

def test_evaluator_invalid_binary_operator():
    node = BinaryOpNode(NumberNode(1), '**', NumberNode(2))
    ev = Evaluator(SymbolTable(), FunctionLibrary())
    with pytest.raises(ValueError):
        ev.eval(node)

def test_evaluator_bitwise_non_integer():
    assert eval_expr("5.9 & 3.1") == 1
    assert eval_expr("5.9 | 2.1") == 7

def test_evaluator_unary_invert_float():
    assert eval_expr("~5.9") == ~5

def test_evaluator_multiple_unary_invert():
    assert eval_expr("~~5") == 5

def test_evaluator_multiple_unary_mixed():
    assert eval_expr("~-5") == ~-5
    assert eval_expr("-~5") == -~5

def test_evaluator_large_numbers():
    assert eval_expr("1000000 * 1000000") == 1000000000000

def test_evaluator_scientific_notation():
    assert eval_expr("1e3 + 2.5E-2") == pytest.approx(1000.025)

def test_evaluator_float_assignment():
    st = SymbolTable()
    eval_expr("x = 3.14", st)
    assert st.get('x') == pytest.approx(3.14)

def test_evaluator_float_variable_usage():
    st = SymbolTable()
    eval_expr("x = 3.14", st)
    assert eval_expr("x + 2", st) == pytest.approx(5.14)

def test_evaluator_float_bitwise():
    assert eval_expr("3.9 & 2.1") == 2

def test_evaluator_float_shift():
    assert eval_expr("3.9 << 1") == 6

def test_evaluator_float_shift_right():
    assert eval_expr("8.9 >> 2") == 2

def test_evaluator_float_invert():
    assert eval_expr("~3.9") == ~3

def test_evaluator_float_invert_negative():
    assert eval_expr("~-3.9") == ~-3

def test_evaluator_float_invert_multiple():
    assert eval_expr("~~3.9") == 3

def test_evaluator_float_invert_mixed():
    assert eval_expr("~-3.9") == ~-3

def test_evaluator_float_invert_mixed2():
    assert eval_expr("-~3.9") == -~3

def test_evaluator_float_invert_mixed3():
    assert eval_expr("~--3.9") == ~--3

def test_evaluator_float_invert_mixed4():
    assert eval_expr("--~3.9") == --~3

def test_evaluator_float_invert_mixed5():
    assert eval_expr("~---3.9") == ~---3

def test_evaluator_float_invert_mixed6():
    assert eval_expr("---~3.9") == ---~3

def test_evaluator_float_invert_mixed7():
    assert eval_expr("~----3.9") == ~----3

def test_evaluator_float_invert_mixed8():
    assert eval_expr("----~3.9") == ----~3

def test_evaluator_float_invert_mixed9():
    assert eval_expr("~-----3.9") == ~-----3

def test_evaluator_float_invert_mixed10():
    assert eval_expr("-----~3.9") == -----~3

def test_evaluator_float_invert_mixed11():
    assert eval_expr("~------3.9") == ~------3

def test_evaluator_float_invert_mixed12():
    assert eval_expr("------~3.9") == ------~3

def test_evaluator_float_invert_mixed13():
    assert eval_expr("~-------3.9") == ~-------3

def test_evaluator_float_invert_mixed14():
    assert eval_expr("-------~3.9") == -------~3

def test_evaluator_float_invert_mixed15():
    assert eval_expr("~--------3.9") == ~--------3

def test_evaluator_float_invert_mixed16():
    assert eval_expr("--------~3.9") == --------~3

def test_evaluator_float_invert_mixed17():
    assert eval_expr("~---------3.9") == ~---------3

def test_evaluator_float_invert_mixed18():
    assert eval_expr("---------~3.9") == ---------~3

def test_evaluator_float_invert_mixed19():
    assert eval_expr("~----------3.9") == ~----------3

def test_evaluator_float_invert_mixed20():
    assert eval_expr("----------~3.9") == ----------~3