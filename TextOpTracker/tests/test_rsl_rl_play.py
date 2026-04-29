from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "rsl_rl" / "play.py"


def load_tree() -> ast.AST:
    return ast.parse(SCRIPT_PATH.read_text(), filename=str(SCRIPT_PATH))


def load_function(name: str):
    tree = load_tree()
    function_node = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name),
        None,
    )
    if function_node is None:
        raise AssertionError(f"Expected play.py to define {name}()")
    module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"time": time}
    exec(compile(module, str(SCRIPT_PATH), "exec"), namespace)
    return namespace[name]


def test_play_parser_supports_real_time_flag():
    tree = load_tree()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant) or node.args[0].value != "--real_time":
            continue

        keyword_values = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in node.keywords
            if keyword.arg in {"action", "default"}
        }
        assert keyword_values["action"] == "store_true"
        assert keyword_values["default"] is False
        return

    raise AssertionError("Expected play.py to register a --real_time store_true flag")


def test_play_parser_supports_export_onnx_flag():
    tree = load_tree()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant) or node.args[0].value != "--export_onnx":
            continue

        keyword_values = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in node.keywords
            if keyword.arg in {"action", "default"}
        }
        assert keyword_values["action"] == "store_true"
        assert keyword_values["default"] is False
        return

    raise AssertionError("Expected play.py to register a --export_onnx store_true flag")


def test_play_parser_supports_playback_speed_flag():
    tree = load_tree()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant) or node.args[0].value != "--playback_speed":
            continue

        keyword_values = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in node.keywords
            if keyword.arg in {"default"}
        }
        assert keyword_values["default"] == 1.0
        return

    raise AssertionError("Expected play.py to register a --playback_speed flag")


def test_sleep_to_real_time_sleeps_for_remaining_duration_at_selected_playback_speed():
    sleep_to_real_time = load_function("_sleep_to_real_time")
    sleep_calls = []

    sleep_to_real_time(True, 0.5, 0.02, 100.0, now_fn=lambda: 100.005, sleep_fn=sleep_calls.append)

    assert sleep_calls == [pytest.approx(0.035)]


def test_sleep_to_real_time_skips_sleep_when_disabled_or_late():
    sleep_to_real_time = load_function("_sleep_to_real_time")
    sleep_calls = []

    sleep_to_real_time(False, 1.0, 0.02, 100.0, now_fn=lambda: 100.005, sleep_fn=sleep_calls.append)
    sleep_to_real_time(True, 2.0, 0.02, 100.0, now_fn=lambda: 100.025, sleep_fn=sleep_calls.append)

    assert sleep_calls == []


def test_sleep_to_real_time_rejects_non_positive_playback_speed():
    sleep_to_real_time = load_function("_sleep_to_real_time")

    with pytest.raises(ValueError, match="playback_speed"):
        sleep_to_real_time(True, 0.0, 0.02, 100.0)


def test_play_loop_invokes_real_time_helper_with_cli_flag():
    tree = load_tree()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "_sleep_to_real_time":
            continue
        assert node.args, "Expected _sleep_to_real_time to be called with positional arguments"
        first_arg = node.args[0]
        assert isinstance(first_arg, ast.Attribute)
        assert isinstance(first_arg.value, ast.Name)
        assert first_arg.value.id == "args_cli"
        assert first_arg.attr == "real_time"
        second_arg = node.args[1]
        assert isinstance(second_arg, ast.Attribute)
        assert isinstance(second_arg.value, ast.Name)
        assert second_arg.value.id == "args_cli"
        assert second_arg.attr == "playback_speed"
        return

    raise AssertionError("Expected play.py to call _sleep_to_real_time(args_cli.real_time, args_cli.playback_speed, ...)")


def test_onnx_export_is_guarded_by_cli_flag():
    tree = load_tree()

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Attribute):
            continue
        if not isinstance(test.value, ast.Name) or test.value.id != "args_cli":
            continue
        if test.attr != "export_onnx":
            continue

        called_functions = {
            child.func.id
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        }
        assert "export_motion_policy_as_onnx" in called_functions
        assert "attach_onnx_metadata" in called_functions
        return

    raise AssertionError("Expected ONNX export calls to be guarded by if args_cli.export_onnx")
