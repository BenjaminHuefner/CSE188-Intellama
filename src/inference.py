from build_prompt import build_draft_prompt, build_code_review_prompt, build_code_repair_prompt
from typing import Dict
from src.llm import generate

_MODEL = None
_TOKENIZER = None

def get_model():
    """Lazily loads the model into the GPU process space."""
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        from src.llm import load_model
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        _MODEL, _TOKENIZER = load_model(model_id)
    return _MODEL, _TOKENIZER

def first_draft(problem: Dict, code_temp: float) -> str:
    """Generates the first draft of code for a given problem."""
    model, tokenizer = get_model()
    prompt = build_draft_prompt(problem["prompt"])
    return generate(model, tokenizer, prompt, temperature=code_temp)

def code_review(problem: Dict, previous_result: Dict, tool_error: Dict, review_temp: float) -> str:
    """Generates a code review for the previous code."""
    model, tokenizer = get_model()
    prompt = build_code_review_prompt(
        problem["prompt"],
        previous_result["code"],
        tool_error["tool"] if tool_error else "None",
        tool_error["error_output"] if tool_error else "None"
    )
    return generate(model, tokenizer, prompt, temperature=review_temp)

def code_repair(problem: Dict, previous_result: Dict, current_feedback: str, tool_error: Dict, code_temp: float) -> str:
    """Generates a code repair based on the previous code, feedback, and tool errors."""
    model, tokenizer = get_model()
    prompt = build_code_repair_prompt(
        problem["prompt"],
        previous_result["code"],
        current_feedback,
        tool_error["tool"] if tool_error else "None",
        tool_error["error_output"] if tool_error else "None"
    )
    return generate(model, tokenizer, prompt, temperature=code_temp)