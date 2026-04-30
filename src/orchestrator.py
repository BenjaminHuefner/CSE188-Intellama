import json
import os
from src.logs import save_turn, save_result, _problem_dir
from pathlib import Path
from typing import Dict, Any

_MODEL = None
_TOKENIZER = None

# Experiment parameters dictionary
_EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "full_experiment": {
        "repair": True,
        "review": True,
        "max_turns": 5,
        "review_temperature": 0.7,
        "code_temperature": 0.2
    },
    "no_repair": {
        "repair": False,
        "review": True,
        "max_turns": 5,
        "review_temperature": 0.7,
        "code_temperature": 0.2
    },
    "no_review": {
        "repair": True,
        "review": False,
        "max_turns": 5,
        "review_temperature": 0.7,
        "code_temperature": 0.2
    },
    "zero_shot": {
        "repair": False,
        "review": False,
        "max_turns": 5,
        "review_temperature": 0.7,
        "code_temperature": 0.2
    }
}

def get_model():
    """Lazily loads the model into the GPU process space."""
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        from src.llm import load_model
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        _MODEL, _TOKENIZER = load_model(model_id)
    return _MODEL, _TOKENIZER

def process_turn(turn: int, code_temp: float, review_temp: float, repair: bool, review: bool, previous_result: Dict = None):
    # actual logic 
    results= {
                    "status": "failed"
                }
    return results

def run_experiment_problem(problem: Dict, experiment_name: str)->Dict:
    if(_MODEL is None):
        get_model()
        
    # 2. Retrieve configuration parameters safely without shared state
    if experiment_name not in _EXPERIMENTS:
        raise ValueError(f"Experiment '{experiment_name}' is not defined in _EXPERIMENTS")
        
    experiment_config = _EXPERIMENTS[experiment_name]
    
    # 3. Read parameters locally for use during generation/processing
    code_temp = experiment_config["code_temperature"]
    review_temp = experiment_config["review_temperature"]
    max_turns = experiment_config["max_turns"]
    repair = experiment_config["repair"]
    review = experiment_config["review"]
    
    task_id = problem['task_id']
    run_dir = "data/outputs"
    
    problem_dir = _problem_dir(run_dir, task_id)
    result_path = os.path.join(problem_dir, "result.json")

    if os.path.exists(result_path):
        print(f"Skipping problem {task_id} (already completed).")
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f)

    completed = False
    final_result= {
                    "status": "failed",
                    "task_id": task_id
                }
    
    previous_result=None
    for turn in range(max_turns):
        turn_path = os.path.join(problem_dir, f"turn_{turn}.json")
        if not os.path.exists(turn_path):
            turn_result=process_turn(turn, code_temp, review_temp, repair, review, previous_result)
            save_turn("data/outputs/", problem['task_id'], turn, turn_result)
            if turn_result["status"] == "completed":
                completed = True
                final_result["status"] = "completed"
                break
        else:
            with open(turn_path, "r", encoding="utf-8") as f:
                saved_turn = json.load(f)
                if saved_turn.get("status") == "completed":
                    completed = True
                    final_result["status"] = "completed"
                    break
                else:
                    previous_result = saved_turn
    
    save_result("data/outputs/", problem['task_id'], final_result)

    return final_result