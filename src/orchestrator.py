import time
from typing import Dict


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


def run_experiment_problem(problem: Dict, experiment_name: str)->Dict:
    if(_MODEL is None):
        get_model()
    time.sleep(1)  # Simulate time-consuming processing
    results = {
                    "task_id": problem['task_id'],
                    "status": "completed"
                }
    return results