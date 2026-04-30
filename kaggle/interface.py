import os
import json
import time
import torch
import torch.multiprocessing as mp
from pathlib import Path
from typing import List, Dict

def worker_process(gpu_id: int, problems: List[Dict], results_dict: Dict):
    """
    Logic for a single GPU worker.
    """
    # 1. Hardware Isolation: Must happen before any torch.cuda calls
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 2. Imports inside the worker to ensure clean process state
    import torch
    from src.llm import load_model#, generate

    print(f"Worker {gpu_id}: Initializing GPU {gpu_id} with {len(problems)} problems.")
    
    try:
        # 3. Load model ONCE per worker
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        model, tokenizer = load_model(model_id)
        
        worker_results = []
        for problem in problems:
            task_id = problem['task_id']
            print(f"Worker {gpu_id}: Processing {task_id}",flush=True)
            
            try:
                # --- ACTUAL EXPERIMENT LOGIC ---
                time.sleep(1)  # Simulate time-consuming processing
                # Placeholder result
                results = {
                    "task_id": task_id,
                    "status": "completed",
                    "code": "print('Hello World')" 
                }
                worker_results.append(results)
                
            except Exception as e:
                print(f"Worker {gpu_id} failed on {task_id}: {str(e)}")
                worker_results.append({"task_id": task_id, "status": "failed", "error": str(e)})
        
        # 4. Push results back to the manager
        results_dict[gpu_id] = worker_results
        
    except Exception as e:
        print(f"Worker {gpu_id} CRITICAL FAILURE: {str(e)}")

def run_experiment(experiment_name: str):
    """
    Main entry point for the Kaggle Notebook.
    """
    print(f"Starting experiment: {experiment_name}")
    
    # 1. Load Data
    data_path = Path('data/humaneval/problems.json')
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Ensure your setup script ran.")
        
    with open(data_path, 'r') as f:
        problems = json.load(f)
        
    # 2. Determine Workers
    num_gpus = torch.cuda.device_count()
    num_workers = 2 if num_gpus >= 2 else 1
    if num_workers < 2:
        print("Running in single-GPU mode.")

    # 3. Robust Chunking: Use slicing with strides for even distribution
    # This handles remainders (e.g., 163 problems) automatically
    chunks = [problems[i::num_workers] for i in range(num_workers)]

    # 4. Launch Processes using 'spawn'
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # Already set
        
    manager = mp.Manager()
    results_dict = manager.dict()
    processes = []

    for i in range(num_workers):
        p = mp.Process(target=worker_process, args=(i, chunks[i], results_dict))
        p.start()
        processes.append(p)

    # 5. Wait for completion
    for p in processes:
        p.join()

    # 6. Consolidate and Save
    final_results = []
    for i in range(num_workers):
        if i in results_dict:
            final_results.extend(results_dict[i])
        
    output_file = Path(f"{experiment_name}_results.json")
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
        
    print(f"Success! {len(final_results)} results saved to {output_file}")