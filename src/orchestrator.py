import json
import os
from src.logs import save_turn, save_result, problem_dir as get_problem_dir
from src.inference import first_draft, code_review, code_repair
from src.gates import run_gates, sanitize_code
from typing import Dict, Any, Optional



_EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "full_experiment": {
        "gates": True,
        "review": True,
        "max_turns": 5,
        "review_temperature": 0.7,
        "code_temperature": 0.2,
    },
    "no_gates": {
        "gates": False,
        "review": True,
        "max_turns": 5,
        "review_temperature": 0.7,
        "code_temperature": 0.2,
    },
    "no_review": {
        "gates": True,
        "review": False,
        "max_turns": 5,
        "review_temperature": 0.7,
        "code_temperature": 0.2,
    },
    "no_review_no_gates": {
        "gates": False,
        "review": False,
        "max_turns": 5,
        "review_temperature": 0.7,
        "code_temperature": 0.2,
    },
    "zero_shot": {
        "gates": False,
        "review": False,
        "max_turns": 1,
        "review_temperature": 0.7,
        "code_temperature": 0.2,
    },
}

RUN_DIR = "data/outputs"


def process_turn(
    turn: int,
    code_temp: float,
    review_temp: float,
    use_gates: bool,
    use_review: bool,
    problem: Dict,
    previous_result: Optional[Dict] = None,
    tool_error: Optional[Dict] = None,
) -> tuple[Dict, Dict]:
    turn_result = {
        "turn": turn,
        "previous_code": previous_result.get("code") if previous_result else None,
        "status": "in_progress",
        "tool_error": tool_error,
        "review_feedback": None,
        "code": None,
    }

    if previous_result is None:
        turn_result["code"] = first_draft(problem, code_temp)
    else:
        if use_review:
            turn_result["review_feedback"] = code_review(
                problem, previous_result, tool_error, review_temp
            )
        turn_result["code"] = code_repair(
            problem,
            previous_result,
            turn_result["review_feedback"],
            tool_error,
            code_temp,
        )

    turn_result["code"] = sanitize_code(turn_result["code"])

    gate_result = run_gates(turn_result["code"], use_gates, problem=problem)

    if gate_result.get("passed", False):
        turn_result["status"] = "completed"

    return turn_result, gate_result

def run_experiment_problem(problem: Dict, experiment_name: str) -> Dict:

    if experiment_name not in _EXPERIMENTS:
        raise ValueError(
            f"Experiment '{experiment_name}' is not defined. "
            f"Valid options: {list(_EXPERIMENTS.keys())}"
        )

    cfg = _EXPERIMENTS[experiment_name]
    code_temp = cfg["code_temperature"]
    review_temp = cfg["review_temperature"]
    max_turns = cfg["max_turns"]
    use_gates = cfg["gates"]
    use_review = cfg["review"]
    task_id = problem["task_id"]

    this_problem_dir = get_problem_dir(RUN_DIR, task_id)
    result_path = os.path.join(this_problem_dir, "result.json")

    if os.path.exists(result_path):
        print(f"Skipping {task_id} (result already exists).", flush=True)
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f)

    final_result: Dict = {
        "status": "failed",
        "task_id": task_id,
        "experiment": experiment_name,
        "num_turns": 0,
        "turns": [],
    }

    previous_result: Optional[Dict] = None
    tool_error: Dict = {}

    for turn in range(max_turns):
        turn_path = os.path.join(this_problem_dir, f"turn_{turn}.json")

        if not os.path.exists(turn_path):
            turn_result, tool_error = process_turn(
                turn, code_temp, review_temp,
                use_gates, use_review,
                problem, previous_result, tool_error,
            )
            save_turn(RUN_DIR, task_id, turn, turn_result)
        else:
            with open(turn_path, "r", encoding="utf-8") as f:
                turn_result = json.load(f)
            tool_error = run_gates(
                turn_result.get("code", ""), use_gates, problem=problem
            )

        final_result["turns"].append(turn_result)
        final_result["num_turns"] = turn_result["turn"]

        if turn_result.get("status") == "completed":
            final_result["status"] = "completed"
            break

        previous_result = turn_result

    save_result(RUN_DIR, task_id, final_result)
    return final_result