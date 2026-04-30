

from __future__ import annotations

import importlib.util
import os
import py_compile
import shutil
import subprocess
import tempfile
import traceback
from typing import Dict, Optional

def _run_tool(cmd: list[str], timeout: int = 30) -> str | None:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode == 0:
            return None
        output = f"{proc.stdout}{proc.stderr}".strip()
        return output if output else f"Tool failed with exit code {proc.returncode}"
    except subprocess.TimeoutExpired:
        return f"Tool timed out after {timeout}s."
    except FileNotFoundError as e:
        return f"Tool not found: {e}"
    except Exception as e:
        return f"Tool execution failed: {e}"


def _load_solution_module(code: str, tmpdir: str):
    code_path = os.path.join(tmpdir, "solution.py")
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(code)

    spec = importlib.util.spec_from_file_location("solution", code_path)
    if spec is None or spec.loader is None:
        return None, "Internal error: could not create module spec for generated code."

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SyntaxError as exc:
        return None, f"SyntaxError while loading generated code: {exc}"
    except Exception as exc:
        return None, f"Runtime error while loading generated code: {type(exc).__name__}: {exc}"

    return module, None


def _run_test_functions(namespace: dict, new_names: set) -> tuple[int, int, list[str]]:
    test_funcs = [
        (name, namespace[name])
        for name in sorted(new_names)
        if name.startswith("test_") and callable(namespace.get(name))
    ]

    passed = 0
    failures: list[str] = []

    for name, func in test_funcs:
        try:
            func()
            passed += 1
        except AssertionError as exc:
            msg = str(exc).strip()
            if not msg:
                msg = (
                    "(no assertion message — add a message to your assert for "
                    "clearer output, e.g. assert x == y, f'got {x}')"
                )
            failures.append(f"  [FAIL] {name}: AssertionError: {msg}")
        except Exception as exc:
            failures.append(f"  [ERROR] {name}: {type(exc).__name__}: {exc}")

    return passed, len(test_funcs), failures


def _gate_result(tool: str, error: str) -> Dict:
    return {"tool": tool, "error_output": error, "passed": False}


def _gate_pass() -> Dict:
    return {"tool": None, "error_output": None, "passed": True}



def run_gates(code: str, use_static_gates: bool, problem: Optional[Dict] = None) -> Dict:
    
    print(
        f"Running gates (static={'on' if use_static_gates else 'off'}, "
        f"tests={'on' if problem else 'off'})...", flush=True
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        temp_path = f.name

    try:
        if use_static_gates:
            try:
                py_compile.compile(temp_path, doraise=True)
            except py_compile.PyCompileError as exc:
                return _gate_result("syntax", str(exc))
            lint_error = _run_tool(["ruff", "check", temp_path])
            if lint_error is not None:
                return _gate_result("lint", lint_error)

            type_error = _run_tool(
                ["mypy", temp_path, "--ignore-missing-imports"]
            )
            if type_error is not None:
                return _gate_result("types", type_error)

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

  
    tmpdir = tempfile.mkdtemp()
    try:
        module, load_error = _load_solution_module(code, tmpdir)
        if load_error:
            return _gate_result("execution", load_error)

        if problem is None:
            return _gate_pass()

        test_code = problem.get("test")
        entry_point = problem.get("entry_point")

        if not test_code:
            return _gate_pass()

        if entry_point:
            available_callables = [
                k for k, v in vars(module).items()
                if callable(v) and not k.startswith("_")
            ]
            if entry_point not in vars(module):
                return _gate_result(
                    "tests",
                    f"Entry point '{entry_point}' not found in your code.\n"
                    f"Your top-level function MUST be named exactly '{entry_point}'.\n"
                    f"Callable names found in your code: {available_callables}"
                )

        namespace = {**vars(module), "__builtins__": __builtins__}
        pre_exec_names = set(namespace.keys())

        try:
            exec(test_code, namespace)
        except AssertionError as exc:
            msg = str(exc).strip()
            if not msg:
                msg = (
                    "(no assertion message — the check() function does not "
                    "report which specific test case failed; compare your "
                    "function's output against the examples in the docstring)"
                )
            return _gate_result("tests", f"Test failed: AssertionError: {msg}")
        except Exception as exc:
            return _gate_result(
                "tests",
                f"Test execution error: {type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc()}"
            )

        new_names = set(namespace.keys()) - pre_exec_names
        passed, total, failures = _run_test_functions(namespace, new_names)

        if total > 0 and failures:
            summary = f"Tests: {passed}/{total} passed.\n" + "\n".join(failures)
            return _gate_result("tests", summary)

        return _gate_pass()

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)



def sanitize_code(code: str) -> str:
    if not code:
        return ""

    code = code.strip()
    lines = code.splitlines()

    PYTHON_STARTERS = ("def ", "class ", "import ", "from ", "async def ")
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith(PYTHON_STARTERS):
            start_idx = i
            break

    lines = lines[start_idx:]

    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]

    end_idx = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "```":
            end_idx = i
            break

    lines = lines[:end_idx]

    return "\n".join(lines).strip()