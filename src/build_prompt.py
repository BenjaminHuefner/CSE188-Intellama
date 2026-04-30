DRAFT_SYSTEM = """You are an expert Python 3.11 programmer.

TASK: Write a COMPLETE, working Python function implementation.

STRICT RULES:
1. Write a real implementation that solves the problem as described in the prompt. Do not write pseudocode or placeholders.
2. The function body must contain actual executable logic that solves the problem.
3. Output ONLY valid Python source code — no explanations, no markdown fences.
4. Only write the solution function."""


CODE_REVIEW_SYSTEM = """You are an expert Python 3.11 debugger.

TASK: The Python function implementation you are given is not working correctly. 
Use the error output and the original problem statement to write a complete to do list of what needs to be fixed in the code.

STRICT_RULES:
1. Do not write any code. Only write a to do list of what needs to be fixed in the code.
2. The to do list should be as detailed as possible, breaking down the problem into small steps.
3. Use the error output to identify specific issues in the code and include them in the to do list.
"""

CODE_REPAIR_SYSTEM = """You are an expert Python 3.11 debugger.

TASK: The python function implementation you are given is not working correctly.
Use the error output and the original problem statement to fix the code and return a complete, working implementation.

STRICT RULES:
1. Edit the code to fix the issues identified in the error output. Do not write any new code that is not directly related to fixing the issues.
2. The function body must contain actual executable logic that solves the problem.
3. Output ONLY valid Python source code — no explanations, no markdown fences.
4. Only write the solution function."""

def build_draft_prompt(problem_prompt: str) -> str:
    prompt = f"{DRAFT_SYSTEM}\n\nProblem:\n{problem_prompt}"
    return prompt


def build_code_review_prompt(
    problem_prompt: str,
    current_code: str,
    tool_name: str,
    error_output: str,
) -> str:
    error_output = error_output[-1000:]
    prompt = (
        f"{CODE_REVIEW_SYSTEM}\n\n"
        f"--- ORIGINAL PROBLEM ---\n{problem_prompt}\n\n"
        f"--- CURRENT CODE ---\n{current_code}\n\n"
        f"--- TOOL ERROR ({tool_name}) ---\n{error_output}\n\n"
        "Write the complete list of what needs to be fixed in the code:"
    )
    return prompt

def build_code_repair_prompt(
    problem_prompt: str,
    current_code: str,
    current_feedback: str,
    tool_name: str,
    error_output: str,
) -> str:
    error_output = error_output[-1000:]
    tool_error = ""
    if(current_feedback):
        current_feedback = f"\n\n Another debugger has provided feedback on the broken code:\n{current_feedback}\n\n"
    if(tool_name and error_output):
        tool_error = f"--- TOOL ERROR ({tool_name}) ---\n{error_output}\n\n"
    prompt = (
        f"{CODE_REPAIR_SYSTEM}\n\n"
        f"--- ORIGINAL PROBLEM ---\n{problem_prompt}\n\n"
        f"--- CURRENT CODE ---\n{current_code}\n\n"
        f"{tool_error if tool_name and error_output else ''}"
        f"{current_feedback if current_feedback else ''}"
        "Now fix the code:"
    )
    return prompt


