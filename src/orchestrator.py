import json
import os
from src.logs import save_turn, save_result, _problem_dir
from src.inference import first_draft, code_review, code_repair
from src.gates import run_gates, sanitize_code
from pathlib import Path
from typing import Dict, Any


# Experiment parameters dictionary
_EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "full_experiment": {
        "gates": True,
        "review": True,
        "max_turns": 5,
        "review_temperature": 0.7,
        "code_temperature": 0.2
    },
    "no_gates": {
        "gates": False,
        "review": True,
        "max_turns": 5,
        "review_temperature": 0.7,
        "code_temperature": 0.2
    },
    "no_review": {
        "gates": True,
        "review": False,
        "max_turns": 5,
        "review_temperature": 0.7,
        "code_temperature": 0.2
    },
    "no_review_no_gates": {
        "gates": False,
        "review": False,
        "max_turns": 5,
        "review_temperature": 0.7,
        "code_temperature": 0.2
    },
    "zero_shot": {
        "gates": False,
        "review": False,
        "max_turns": 1,
        "review_temperature": 0.7,
        "code_temperature": 0.2
    }
}


def process_turn(turn: int, code_temp: float, review_temp: float, gates: bool, review: bool, problem: Dict, previous_result: Dict = None, tool_error: Dict = None) -> tuple[Dict, Dict]: 
    turn_result= {
                    "turn": turn,
                    "previous_code": previous_result.get("code") if previous_result else None,
                    "status": previous_result.get("status") if previous_result else "in_progress",
                    "tool_error": tool_error,
                    "review_feedback": None,
                    "code": None
                }
    if(previous_result is None):
        turn_result["code"] = first_draft(problem["prompt"], code_temp)
    else:
        if review:
            # Implement code review logic here (e.g., call a function that generates a review based on previous_result)
            turn_result["review_feedback"] = code_review(problem, previous_result, tool_error, review_temp)
        turn_result["code"] = code_repair(problem, previous_result, turn_result["review_feedback"], tool_error, code_temp)

    turn_result["code"] = sanitize_code(turn_result["code"])

    tool_error = run_gates(turn_result["code"])
    if tool_error.get("passed", False):
        turn_result["status"] = "completed"
    
    return turn_result, tool_error

def run_experiment_problem(problem: Dict, experiment_name: str)->Dict:
        
    # 2. Retrieve configuration parameters safely without shared state
    if experiment_name not in _EXPERIMENTS:
        raise ValueError(f"Experiment '{experiment_name}' is not defined in _EXPERIMENTS")
        
    experiment_config = _EXPERIMENTS[experiment_name]
    
    # 3. Read parameters locally for use during generation/processing
    code_temp = experiment_config["code_temperature"]
    review_temp = experiment_config["review_temperature"]
    max_turns = experiment_config["max_turns"]
    gates = experiment_config["gates"]
    review = experiment_config["review"]
    task_id = problem['task_id']
    run_dir = "data/outputs"
    
    problem_dir = _problem_dir(run_dir, task_id)
    result_path = os.path.join(problem_dir, "result.json")

    if os.path.exists(result_path):
        print(f"Skipping problem {task_id} (already completed).")
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f)

    final_result= {
                    "status": "failed",
                    "task_id": task_id,
                    "num_turns": 0,
                    "turns":[]
                }
    
    previous_result=None
    tool_error = {}
    for turn in range(max_turns):
        turn_path = os.path.join(problem_dir, f"turn_{turn}.json")
        if not os.path.exists(turn_path):
            turn_result, tool_error=process_turn(turn, code_temp, review_temp, gates, review, problem, previous_result, tool_error)
            save_turn("data/outputs/", problem['task_id'], turn, turn_result)
            final_result["turns"].append(turn_result)
            final_result["num_turns"] = turn_result["turn"]
            if turn_result["status"] == "completed":
                final_result["status"] = "completed"
                break
            else:
                previous_result = turn_result
        else:
            with open(turn_path, "r", encoding="utf-8") as f:
                saved_turn = json.load(f)
                final_result["turns"].append(saved_turn)
                final_result["num_turns"] = saved_turn["turn"]
                if saved_turn.get("status") == "completed":
                    final_result["status"] = "completed"
                    break
                else:
                    tool_error = run_gates(saved_turn["code"])
                    previous_result = saved_turn
    
    save_result("data/outputs/", problem['task_id'], final_result)

    return final_result