"""Sequential tool gates for syntax, lint, type checking, and tests."""

from __future__ import annotations
import os
import py_compile
import tempfile
import subprocess
import traceback
import re
from typing import Dict

def _run_tool(cmd: list[str], timeout: int = 10) -> str | None:
    """Run a subprocess tool and normalize failures to strings."""
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
    except Exception as e:
        return f"Tool execution failed: {e}"


def run_gates(code: str) -> Dict:
    """
    Validates the generated code sequentially through syntax, linting, 
    type checking, and execution gates.
    """
    print("Running gates...")

    # Create one temporary file for all three static analysis checks
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(code)
        temp_path = f.name

    try:
        # 1. Syntax Check
        try:
            py_compile.compile(temp_path, doraise=True)
        except Exception as e:
            return {
                "tool": "syntax",
                "error_output": str(e),
                "passed": False
            }

        # 2. Lint Check (ruff)
        lint_error = _run_tool(["ruff", "check", temp_path])
        if lint_error is not None:
            return {
                "tool": "lint",
                "error_output": lint_error,
                "passed": False
            }

        # 3. Type Checking (mypy)
        type_error = _run_tool(["mypy", temp_path, "--ignore-missing-imports"])
        if type_error is not None:
            return {
                "tool": "types",
                "error_output": type_error,
                "passed": False
            }

        # 4. Module Execution Gate
        try:
            exec_scope = {}
            exec(code, {}, exec_scope)
        except Exception:
            return {
                "tool": "execution",
                "error_output": traceback.format_exc(),
                "passed": False
            }

    finally:
        # Ensure we always clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    return {
        "tool": None,
        "error_output": None,
        "passed": True
    }


import re
from typing import Optional

def sanitize_code(code: Optional[str]) -> str:
    if code is None:
        return ""
        
    code = code.strip()
    lines = code.splitlines()
    
    # 1. Find the starting index of the actual Python code
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Stop when we find a declaration, import, or comment
        if stripped.startswith(('def ', 'class ', 'import ', 'from ', '#', 'async def ')):
            start_idx = i
            break
            
    lines = lines[start_idx:]
    
    # If the first remaining line is a markdown fence like ```python, skip it
    if lines and lines[0].strip().startswith('```'):
        lines = lines[1:]
        
    # 2. Find the ending index of the code
    end_idx = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        # Stop at the closing markdown fence if present
        if stripped == '```':
            end_idx = i
            break
            
    lines = lines[:end_idx]
    
    return '\n'.join(lines).strip()