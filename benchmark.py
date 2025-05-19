#!/usr/bin/env python3
"""
Ollama GPU Inference Benchmark Tool

This script benchmarks inference performance of various LLM models using Ollama
across available GPUs on the system.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import requests
import numpy as np

# Default configuration
DEFAULT_PROMPT = "Write an essay about the USA revolution."
DEFAULT_MODELS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.json")
DEFAULT_MODELS = [
    "llama3:8b",
    "llama3:70b",
    "mistral:7b",
    "gemma:7b",
    "phi3:14b"
]
# Default API host
DEFAULT_API_HOST = "http://localhost:11434"
# Set API host with environment variable override
OLLAMA_API_HOST = os.environ.get("OLLAMA_API_HOST", DEFAULT_API_HOST)


def get_available_gpus() -> List[int]:
    """
    Detect available GPUs on the system.
    Returns a list of GPU indices.
    """
    try:
        # Try using nvidia-smi for NVIDIA GPUs
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        gpu_indices = [int(idx.strip()) for idx in result.stdout.splitlines() if idx.strip()]
        if gpu_indices:
            print(f"Detected {len(gpu_indices)} NVIDIA GPUs: {gpu_indices}")
            return gpu_indices
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    try:
        # Try using rocm-smi for AMD GPUs
        result = subprocess.run(
            ["rocm-smi", "--showid"],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse rocm-smi output to extract GPU indices
        lines = result.stdout.splitlines()
        gpu_indices = []
        for line in lines:
            if "GPU[" in line:
                try:
                    idx = int(line.split("GPU[")[1].split("]")[0])
                    gpu_indices.append(idx)
                except (IndexError, ValueError):
                    continue
        if gpu_indices:
            print(f"Detected {len(gpu_indices)} AMD GPUs: {gpu_indices}")
            return gpu_indices
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    # Check for Apple Silicon GPUs
    if platform.system() == "Darwin" and platform.processor() == "arm":
        print("Detected Apple Silicon GPU")
        return [0]  # Apple Silicon is treated as a single GPU
    
    print("No GPUs detected. Will run on CPU.")
    return []


def get_available_models() -> List[str]:
    """
    Get list of available models from Ollama API.
    """
    try:
        response = requests.get(f"{OLLAMA_API_HOST}/api/tags")
        if response.status_code == 200:
            models = [model["name"] for model in response.json().get("models", [])]
            return models
        else:
            print(f"Failed to get models from Ollama API: {response.status_code}")
            return []
    except requests.RequestException as e:
        print(f"Error connecting to Ollama API: {e}")
        return []


def pull_model(model_name: str) -> bool:
    """
    Pull a model from Ollama if it's not already available.
    """
    available_models = get_available_models()
    
    if model_name in available_models:
        print(f"Model {model_name} is already available.")
        return True
    
    print(f"Pulling model {model_name}...")
    try:
        response = requests.post(
            f"{OLLAMA_API_HOST}/api/pull",
            json={"name": model_name},
            stream=True
        )
        
        for line in response.iter_lines():
            if line:
                progress = json.loads(line)
                status = progress.get("status")
                if status:
                    # Print download progress
                    if "completed" in progress:
                        print(f"\r{status}: {progress.get('completed')}/{progress.get('total')} layers", end="")
                    else:
                        print(f"\r{status}", end="")
        
        print("\nModel pull completed.")
        return True
    except requests.RequestException as e:
        print(f"Error pulling model {model_name}: {e}")
        return False


def get_gpu_metrics_nvidia(gpu_idx: int) -> dict:
    """
    Collect GPU metrics using nvidia-smi for a specific GPU index.
    Returns a dict with temperature, utilization, memory, and power.
    Returns None if nvidia-smi is not available or fails.
    """
    try:
        result = subprocess.run([
            "nvidia-smi",
            f"--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
            f"-i", str(gpu_idx)
        ], capture_output=True, text=True, check=True)
        line = result.stdout.strip().split(',')
        if len(line) != 5:
            return None
        temp_c = float(line[0])
        util_percent = float(line[1])
        mem_used = float(line[2])
        mem_total = float(line[3])
        mem_util_percent = (mem_used / mem_total) * 100 if mem_total > 0 else 0
        power_w = float(line[4])
        return {
            "temperature_c": temp_c,
            "utilization_percent": util_percent,
            "memory_utilization_percent": mem_util_percent,
            "power_draw_w": power_w
        }
    except Exception:
        return None


def run_inference(
    model_name: str,
    prompt: str,
    gpu_idx: Optional[int] = None
) -> Dict[str, Union[float, str, int]]:
    """
    Run inference on a model and measure performance, including GPU metrics if available.
    """
    # Set environment variables for GPU selection
    env = os.environ.copy()
    if gpu_idx is not None:
        if platform.system() == "Linux":
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        elif platform.system() == "Darwin":
            # Apple Silicon doesn't need explicit GPU selection
            pass

    print(f"Running inference on model {model_name} {'on GPU ' + str(gpu_idx) if gpu_idx is not None else 'on CPU'}")

    # Prepare the request payload
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 100  # Limit token generation for benchmarking
        }
    }

    # GPU metrics sampling (NVIDIA only)
    gpu_metrics_samples = []
    sample_gpu_metrics = (gpu_idx is not None and platform.system() == "Linux")
    try:
        if sample_gpu_metrics:
            m = get_gpu_metrics_nvidia(gpu_idx)
            if m:
                gpu_metrics_samples.append(m)
    except Exception:
        pass

    # Measure inference time
    start_time = time.time()
    try:
        response = requests.post(f"{OLLAMA_API_HOST}/api/generate", json=payload)
        response_json = response.json()
        end_time = time.time()

        # Sample GPU metrics again after inference
        try:
            if sample_gpu_metrics:
                m = get_gpu_metrics_nvidia(gpu_idx)
                if m:
                    gpu_metrics_samples.append(m)
        except Exception:
            pass

        # Extract metrics
        total_duration = end_time - start_time
        eval_count = response_json.get("eval_count", 0)
        eval_duration = response_json.get("eval_duration", 0)

        if eval_count > 0 and eval_duration > 0:
            tokens_per_second = eval_count / (eval_duration / 1e9)  # eval_duration is in nanoseconds
        else:
            tokens_per_second = 0

        # Average GPU metrics if collected
        gpu_avg_metrics = None
        if gpu_metrics_samples:
            arr = lambda key: [s[key] for s in gpu_metrics_samples if key in s]
            gpu_avg_metrics = {
                "avg_temperature_c": float(np.mean(arr("temperature_c"))) if arr("temperature_c") else None,
                "avg_utilization_percent": float(np.mean(arr("utilization_percent"))) if arr("utilization_percent") else None,
                "avg_memory_utilization_percent": float(np.mean(arr("memory_utilization_percent"))) if arr("memory_utilization_percent") else None,
                "avg_power_draw_w": float(np.mean(arr("power_draw_w"))) if arr("power_draw_w") else None,
            }

        # Create result dictionary
        result = {
            "model": model_name,
            "gpu_idx": gpu_idx,
            "total_duration_seconds": total_duration,
            "eval_count": eval_count,
            "eval_duration_ns": eval_duration,
            "tokens_per_second": tokens_per_second,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "system_info": {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "processor": platform.processor()
            }
        }
        if gpu_avg_metrics:
            result["gpu_avg_metrics"] = gpu_avg_metrics

        print(f"Inference completed in {total_duration:.2f} seconds, {tokens_per_second:.2f} tokens/sec")
        return result

    except requests.RequestException as e:
        print(f"Error during inference: {e}")
        return {
            "model": model_name,
            "gpu_idx": gpu_idx,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def save_results(results: List[Dict], output_file: str):
    """
    Save benchmark results to a JSON file.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")


def print_summary(results: List[Dict]):
    """
    Print a summary of benchmark results.
    """
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    
    # Group results by model
    model_results = {}
    for result in results:
        model = result.get("model")
        if model not in model_results:
            model_results[model] = []
        model_results[model].append(result)
    
    # Print summary for each model
    for model, model_data in model_results.items():
        print(f"\nModel: {model}")
        
        # Group by GPU
        gpu_results = {}
        for data in model_data:
            gpu_idx = data.get("gpu_idx")
            gpu_key = f"GPU {gpu_idx}" if gpu_idx is not None else "CPU"
            if gpu_key not in gpu_results:
                gpu_results[gpu_key] = []
            gpu_results[gpu_key].append(data)
        
        # Print results for each GPU
        for gpu, gpu_data in gpu_results.items():
            tps_values = [d.get("tokens_per_second", 0) for d in gpu_data if "error" not in d]
            if tps_values:
                avg_tps = np.mean(tps_values)
                print(f"  {gpu}: {avg_tps:.2f} tokens/sec (avg of {len(tps_values)} runs)")
            else:
                print(f"  {gpu}: No successful runs")


def load_models_from_json(json_file: str) -> List[str]:
    """
    Load models from a JSON file.
    
    Expected format:
    {
        "models": [
            {"name": "model1", "description": "..."}, 
            {"name": "model2", "description": "..."}
        ]
    }
    
    Returns a list of model names.
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, dict) or "models" not in data:
            print(f"Error: Invalid JSON format in {json_file}. Expected 'models' key.")
            return []
        
        models = []
        for model_entry in data.get("models", []):
            if isinstance(model_entry, dict) and "name" in model_entry:
                models.append(model_entry["name"])
            elif isinstance(model_entry, str):
                models.append(model_entry)
        
        if not models:
            print(f"Warning: No valid models found in {json_file}")
        else:
            print(f"Loaded {len(models)} models from {json_file}")
        
        return models
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading models from {json_file}: {e}")
        return []


def main():
    global OLLAMA_API_HOST
    parser = argparse.ArgumentParser(description="Ollama GPU Inference Benchmark Tool")
    parser.add_argument(
        "--models", 
        nargs="+", 
        default=None,
        help="List of models to benchmark (space-separated)"
    )
    parser.add_argument(
        "--models-file",
        default=None,
        help=f"JSON file containing models to benchmark"
    )
    parser.add_argument(
        "--prompt", 
        default=DEFAULT_PROMPT,
        help=f"Prompt to use for inference (default: '{DEFAULT_PROMPT}')"
    )
    parser.add_argument(
        "--gpus", 
        nargs="+", 
        type=int,
        help="Specific GPU indices to use (default: all available GPUs)"
    )
    parser.add_argument(
        "--runs", 
        type=int, 
        default=1,
        help="Number of runs per model per GPU (default: 1)"
    )
    parser.add_argument(
        "--output", 
        default="./benchmark_results.json",
        help="Output file for benchmark results (default: ./benchmark_results.json)"
    )
    parser.add_argument(
        "--api-host",
        default=OLLAMA_API_HOST,
        help=f"Ollama API host (default: {OLLAMA_API_HOST})"
    )
    
    args = parser.parse_args()
    
    # Update API host if specified
    OLLAMA_API_HOST = args.api_host
    
    # Determine which models to use
    models_to_benchmark = []
    
    # First priority: models specified directly via --models
    if args.models is not None:
        models_to_benchmark = args.models
        print(f"Using {len(models_to_benchmark)} models specified via command line")
    
    # Second priority: models from JSON file
    elif args.models_file is not None:
        models_to_benchmark = load_models_from_json(args.models_file)
    
    # Third priority: try default models file
    elif os.path.exists(DEFAULT_MODELS_FILE):
        models_to_benchmark = load_models_from_json(DEFAULT_MODELS_FILE)
    
    # Fallback to hardcoded default models
    if not models_to_benchmark:
        models_to_benchmark = DEFAULT_MODELS
        print(f"Using {len(models_to_benchmark)} default models")
    
    # Get available GPUs
    available_gpus = get_available_gpus()
    
    # Use specified GPUs or all available GPUs
    gpus_to_use = args.gpus if args.gpus is not None else available_gpus
    
    # Validate GPUs
    if gpus_to_use:
        for gpu in gpus_to_use:
            if gpu not in available_gpus:
                print(f"Warning: GPU {gpu} is not available on this system.")
        # Filter to only available GPUs
        gpus_to_use = [gpu for gpu in gpus_to_use if gpu in available_gpus]
    
    if not gpus_to_use:
        print("No valid GPUs specified. Running on CPU.")
        gpus_to_use = [None]  # None represents CPU
    
    # Pull models if needed
    valid_models = []
    for model in models_to_benchmark:
        if pull_model(model):
            valid_models.append(model)
        else:
            print(f"Skipping model {model} due to pull failure.")
    
    if not valid_models:
        print("No models available for benchmarking. Exiting.")
        sys.exit(1)
    
    # Run benchmarks
    results = []
    for model in valid_models:
        for gpu in gpus_to_use:
            for run in range(args.runs):
                print(f"\nBenchmarking {model} on {'GPU ' + str(gpu) if gpu is not None else 'CPU'} (Run {run+1}/{args.runs})")
                result = run_inference(model, args.prompt, gpu)
                results.append(result)
    
    # Save results
    save_results(results, args.output)
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
