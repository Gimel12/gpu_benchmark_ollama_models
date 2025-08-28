#!/usr/bin/env python3
"""
BIZON GPU Benchmark Tool

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
    Returns a dict with temperature, utilization, memory, power, and GPU name.
    Returns None if nvidia-smi is not available or fails.
    """
    try:
        # Run nvidia-smi to get GPU metrics
        result = subprocess.run(
            [
                "nvidia-smi",
                # Query absolute memory used/total instead of percent; we'll compute percent ourselves.
                f"--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit",
                "--format=csv,noheader,nounits",
                f"--id={gpu_idx}"
            ],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output
        values = result.stdout.strip().split(", ")
        if len(values) >= 7:
            name = values[0].strip()
            temp_c = float(values[1])
            util_pct = float(values[2])
            mem_used_mib = float(values[3])
            mem_total_mib = float(values[4]) if float(values[4]) > 0 else None
            power_w = float(values[5])
            power_limit_w = float(values[6])
            mem_util_pct = (mem_used_mib / mem_total_mib * 100.0) if mem_total_mib else 0.0
            return {
                "gpu_name": name,
                "temperature_c": temp_c,
                "utilization_percent": util_pct,
                "memory_used_mib": mem_used_mib,
                "memory_total_mib": mem_total_mib,
                "memory_utilization_percent": mem_util_pct,
                "power_draw_w": power_w,
                "power_limit_w": power_limit_w,
            }
    except (subprocess.SubprocessError, ValueError, IndexError) as e:
        print(f"Error getting GPU metrics: {e}")
    
    return None


def run_inference(
    model_name: str,
    prompt: str,
    gpu_idx: Optional[int] = None,
    is_multi_gpu: bool = False,
    gpu_list: List[int] = None
) -> Dict[str, Union[float, str, int]]:
    """
    Run inference on a model and measure performance, including GPU metrics if available.
    """
    # Set environment variables for GPU selection
    env = os.environ.copy()
    
    # Handle GPU selection
    if is_multi_gpu and gpu_list:
        # For multi-GPU mode, use the environment variables set in main()
        # Just ensure OLLAMA_USE_GPU is set
        env["OLLAMA_USE_GPU"] = "1"
        print(f"Running inference on model {model_name} on multiple GPUs: {','.join(str(g) for g in gpu_list)}")
    elif gpu_idx is not None:
        if platform.system() == "Linux":
            # Set the specific GPU to use
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            # Also set OLLAMA_USE_GPU=1 to ensure GPU usage
            env["OLLAMA_USE_GPU"] = "1"
            print(f"Running inference on model {model_name} on GPU {gpu_idx}")
        elif platform.system() == "Darwin":
            # Apple Silicon doesn't need explicit GPU selection
            print(f"Running inference on model {model_name} on Apple Silicon GPU")
    else:
        print(f"Running inference on model {model_name} on CPU")

    # Prepare the request payload
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": 10000  # Limit token generation for benchmarking
        }
    }

    # GPU metrics sampling (NVIDIA only)
    gpu_metrics_samples = []
    
    # Determine which GPUs to sample metrics from
    gpus_to_sample = []
    if is_multi_gpu and gpu_list:
        # For multi-GPU mode, sample all GPUs in the list
        gpus_to_sample = gpu_list
    elif gpu_idx is not None:
        # For single GPU mode, sample just that GPU
        gpus_to_sample = [gpu_idx]
    
    # Sample metrics from all relevant GPUs
    if platform.system() == "Linux" and gpus_to_sample:
        try:
            for idx in gpus_to_sample:
                m = get_gpu_metrics_nvidia(idx)
                if m:
                    gpu_metrics_samples.append(m)
        except Exception as e:
            print(f"Error collecting GPU metrics: {e}")

    # Measure inference time
    start_time = time.time()
    try:
        response = requests.post(f"{OLLAMA_API_HOST}/api/generate", json=payload, stream=payload.get("stream", False))
        response_json = None
        generated_text = ""
        eval_count = 0
        eval_duration = 0
        if payload.get("stream", False):
            # Handle streaming response
            import json as _json
            last_sample_time = 0.0
            sample_interval = 1.0  # seconds
            for line in response.iter_lines():
                if line:
                    chunk = _json.loads(line.decode("utf-8"))
                    generated_text += chunk.get("response", "")
                    if "eval_count" in chunk:
                        eval_count = chunk["eval_count"]
                    if "eval_duration" in chunk:
                        eval_duration = chunk["eval_duration"]
                # Periodically sample GPU metrics during generation
                if platform.system() == "Linux" and gpus_to_sample:
                    now = time.time()
                    if now - last_sample_time >= sample_interval:
                        for idx in gpus_to_sample:
                            m = get_gpu_metrics_nvidia(idx)
                            if m:
                                gpu_metrics_samples.append(m)
                        last_sample_time = now
            response_json = {
                "response": generated_text,
                "eval_count": eval_count,
                "eval_duration": eval_duration
            }
        else:
            # Handle non-streaming response
            response_json = response.json()
            eval_count = response_json.get("eval_count", 0)
            eval_duration = response_json.get("eval_duration", 0)
        print("Ollama API response:", response_json)  # DEBUG
        end_time = time.time()

        # Sample GPU metrics again after inference
        try:
            if platform.system() == "Linux" and gpus_to_sample:
                for idx in gpus_to_sample:
                    m = get_gpu_metrics_nvidia(idx)
                    if m:
                        gpu_metrics_samples.append(m)
        except Exception as e:
            print(f"Error collecting GPU metrics after inference: {e}")

        # Extract metrics
        total_duration = end_time - start_time
        eval_count = response_json.get("eval_count", 0)
        eval_duration = response_json.get("eval_duration", 0)

        if eval_count > 0 and eval_duration > 0:
            tokens_per_second = eval_count / (eval_duration / 1e9)  # eval_duration is in nanoseconds
        else:
            tokens_per_second = 0

        # Aggregate GPU metrics if available
        gpu_avg_metrics = None
        if gpu_metrics_samples:
            n = len(gpu_metrics_samples)
            avg_temp = sum(m.get("temperature_c", 0) for m in gpu_metrics_samples) / n
            avg_util = sum(m.get("utilization_percent", 0) for m in gpu_metrics_samples) / n
            avg_mem_util = sum(m.get("memory_utilization_percent", 0) for m in gpu_metrics_samples) / n
            avg_mem_used_mib = sum(m.get("memory_used_mib", 0) for m in gpu_metrics_samples) / n
            peak_mem_used_mib = max(m.get("memory_used_mib", 0) for m in gpu_metrics_samples)
            # memory_total_mib might vary if sampling multiple GPUs (multi-GPU); take max as capacity indicator
            mem_total_mib_vals = [m.get("memory_total_mib") for m in gpu_metrics_samples if m.get("memory_total_mib")]
            mem_total_mib = max(mem_total_mib_vals) if mem_total_mib_vals else None
            avg_power = sum(m.get("power_draw_w", 0) for m in gpu_metrics_samples) / n
            peak_power_draw_w = max(m.get("power_draw_w", 0) for m in gpu_metrics_samples)
            power_limit_max = max(m.get("power_limit_w", 0) for m in gpu_metrics_samples)  # TDP (power limit)
            gpu_name = gpu_metrics_samples[0].get("gpu_name") if gpu_metrics_samples[0].get("gpu_name") else None
            gpu_avg_metrics = {
                "gpu_name": gpu_name,
                "avg_temperature_c": avg_temp,
                "avg_utilization_percent": avg_util,
                "avg_memory_utilization_percent": avg_mem_util,
                "avg_memory_used_mib": avg_mem_used_mib,
                "peak_memory_used_mib": peak_mem_used_mib,
                "memory_total_mib": mem_total_mib,
                "avg_power_draw_w": avg_power,
                "peak_power_draw_w": peak_power_draw_w,
                "power_limit_w": power_limit_max
            }

        # Create result dictionary
        tokens_generated = response_json.get("eval_count", 0)
        result = {
            "model": model_name,
            "gpu_idx": gpu_idx if not is_multi_gpu else ",".join(str(g) for g in gpu_list) if gpu_list else None,
            "total_duration_seconds": total_duration,
            "eval_count": eval_count,
            "eval_duration_ns": eval_duration,
            "tokens_per_second": tokens_per_second,
            "tokens_generated": tokens_generated,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "system_info": {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "processor": platform.processor()
            },
            "multi_gpu": is_multi_gpu
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
    parser = argparse.ArgumentParser(description="BIZON GPU Benchmark Tool")
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
    
    # Check if we're using multiple GPUs
    using_multiple_gpus = len(gpus_to_use) > 1 and all(gpu is not None for gpu in gpus_to_use)
    
    if using_multiple_gpus:
        # For multiple GPUs, set CUDA_VISIBLE_DEVICES once with all GPUs
        gpu_str = ",".join(str(gpu) for gpu in gpus_to_use)
        print(f"\nUsing multiple GPUs: {gpu_str}")
        
        # Set environment variable for all processes
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        os.environ["OLLAMA_USE_GPU"] = "1"
        
        # Run benchmarks on all GPUs
        for model in valid_models:
            for run in range(args.runs):
                print(f"\nBenchmarking {model} on GPUs {gpu_str} (Run {run+1}/{args.runs})")
                # Pass None as gpu_idx to avoid overriding the environment variable
                result = run_inference(model, args.prompt, None, is_multi_gpu=True, gpu_list=gpus_to_use)
                results.append(result)
    else:
        # Single GPU mode (or CPU)
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
