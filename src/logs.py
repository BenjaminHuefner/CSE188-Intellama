import json
import os


def _problem_dir(run_dir: str, task_id: str) -> str:
    safe_id = task_id.replace("/", "_")
    path = os.path.join(run_dir, safe_id)
    os.makedirs(path, exist_ok=True)
    return path


def save_turn(run_dir: str, task_id: str, turn: int, turn_data: dict) -> None:
    """Persist one turn log to disk as JSON."""
    path = os.path.join(_problem_dir(run_dir, task_id), f"turn_{turn}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(turn_data, f, indent=2)


def save_result(run_dir: str, task_id: str, result: dict) -> None:
    """Persist the final result JSON and final code artifact."""
    pdir = _problem_dir(run_dir, task_id)
    with open(os.path.join(pdir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
