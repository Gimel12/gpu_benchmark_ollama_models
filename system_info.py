#!/usr/bin/env python3

import os
import sys
import platform
import subprocess
import json
import requests
from typing import Dict, List, Any, Optional
import shutil

def get_system_info() -> Dict[str, Any]:
    """
    Gather system information including OS, Python version, GPU details, and Ollama status.
    """
    info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "gpus": get_gpu_info(),
        "ollama_version": get_ollama_version(),
        "ollama_api_status": check_ollama_api(),
        "ollama_gpu_support": check_ollama_gpu_support(),
        "models": get_available_models()
    }
    return info

def get_gpu_info() -> List[Dict[str, Any]]:
    """
    Get information about available GPUs using nvidia-smi.
    """
    gpus = []
    
    # Check if nvidia-smi is available
    if not shutil.which('nvidia-smi'):
        return gpus
    
    try:
        # Run nvidia-smi to get GPU information
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,driver_version', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(', ')
                if len(parts) >= 4:
                    gpus.append({
                        "index": parts[0],
                        "name": parts[1],
                        "memory": f"{float(parts[2])/1024:.1f} GB",
                        "driver": parts[3],
                        "available": True
                    })
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    return gpus

def get_ollama_version() -> str:
    """
    Get the installed Ollama version.
    """
    try:
        result = subprocess.run(['ollama', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        return "Not found"
    except (subprocess.SubprocessError, FileNotFoundError):
        return "Not found"

def check_ollama_api(api_host: str = "http://localhost:11434") -> bool:
    """
    Check if the Ollama API is available.
    """
    try:
        response = requests.get(f"{api_host}/api/tags", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False

def check_ollama_gpu_support() -> bool:
    """
    Check if Ollama is configured to use GPU.
    """
    try:
        # First check if Ollama is running
        if not check_ollama_api():
            return False
        
        # Try to get GPU info from Ollama
        # This is a heuristic - we run a small model and check if GPU metrics are non-zero
        cmd = [
            'ollama', 'run', 'llama3:8b',
            'Write a one-word response: "test"',
            '--verbose'
        ]
        env = os.environ.copy()
        env['OLLAMA_USE_GPU'] = '1'  # Try to force GPU usage
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=30)
        
        # Check if the output mentions GPU
        return 'gpu' in result.stderr.lower() and not 'no gpu available' in result.stderr.lower()
    except (subprocess.SubprocessError, FileNotFoundError, TimeoutError):
        return False

def get_available_models() -> List[Dict[str, Any]]:
    """
    Get list of available models from Ollama.
    """
    models = []
    try:
        if not check_ollama_api():
            return models
        
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            for model in data.get('models', []):
                models.append({
                    "name": model.get('name', ''),
                    "size": f"{model.get('size', 0) / (1024*1024*1024):.1f} GB",
                    "modified": model.get('modified', '')
                })
    except requests.RequestException:
        pass
    
    return models

def test_gpu_inference() -> Dict[str, str]:
    """
    Run a test inference to check if GPU is being used.
    """
    output = "GPU Inference Test Results:\n\n"
    
    try:
        # Check if nvidia-smi is available
        if not shutil.which('nvidia-smi'):
            output += "❌ NVIDIA tools not found. GPU monitoring not available.\n"
        else:
            output += "✅ NVIDIA tools found.\n"
        
        # Check if Ollama is running
        if not check_ollama_api():
            output += "❌ Ollama API is not available. Please start Ollama service.\n"
            return {"output": output}
        
        output += "✅ Ollama API is available.\n"
        
        # Run a test inference with a small model
        output += "\nRunning test inference with llama3:8b...\n"
        
        cmd = [
            'ollama', 'run', 'llama3:8b',
            'Write a one-word response: "test"',
            '--verbose'
        ]
        env = os.environ.copy()
        env['OLLAMA_USE_GPU'] = '1'  # Try to force GPU usage
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=30)
        
        # Check GPU usage during inference
        gpu_cmd = ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv']
        gpu_result = subprocess.run(gpu_cmd, capture_output=True, text=True)
        
        if 'ollama' in gpu_result.stdout.lower():
            output += "✅ Ollama is using GPU for inference!\n"
        else:
            output += "❌ Ollama is NOT using GPU for inference.\n"
        
        # Extract relevant information from verbose output
        if 'gpu' in result.stderr.lower():
            output += "✅ GPU mentioned in Ollama output.\n"
        if 'no gpu available' in result.stderr.lower():
            output += "❌ Ollama reports no GPU available.\n"
        
        output += "\nRecommendations:\n"
        if 'ollama' not in gpu_result.stdout.lower():
            output += "1. Check if Ollama is installed with GPU support\n"
            output += "2. Set OLLAMA_USE_GPU=1 before starting Ollama\n"
            output += "3. Restart the Ollama service\n"
            output += "4. Verify GPU drivers are properly installed\n"
        
    except Exception as e:
        output += f"\nError during test: {str(e)}\n"
    
    return {"output": output}

def check_ollama_status() -> Dict[str, str]:
    """
    Check detailed Ollama status.
    """
    output = "Ollama Status Check:\n\n"
    
    try:
        # Check Ollama version
        version = get_ollama_version()
        if version != "Not found":
            output += f"✅ Ollama installed: {version}\n"
        else:
            output += "❌ Ollama not found in PATH\n"
        
        # Check if Ollama service is running
        if check_ollama_api():
            output += "✅ Ollama API is responding\n"
        else:
            output += "❌ Ollama API is not responding\n"
            output += "   Try starting Ollama with: ollama serve\n"
        
        # Check for available models
        models = get_available_models()
        if models:
            output += f"✅ {len(models)} models available\n"
            for model in models[:5]:  # Show only first 5 models
                output += f"   - {model['name']} ({model['size']})\n"
            if len(models) > 5:
                output += f"   - ... and {len(models) - 5} more\n"
        else:
            output += "❌ No models available\n"
        
        # Check for GPU support
        if check_ollama_gpu_support():
            output += "✅ Ollama appears to have GPU support enabled\n"
        else:
            output += "❌ Ollama does not appear to be using GPU\n"
            output += "\nTo enable GPU support:\n"
            output += "1. Stop Ollama service\n"
            output += "2. Set environment variable: export OLLAMA_USE_GPU=1\n"
            output += "3. Start Ollama: ollama serve\n"
        
    except Exception as e:
        output += f"\nError during check: {str(e)}\n"
    
    return {"output": output}

if __name__ == "__main__":
    # Print system info when run directly
    info = get_system_info()
    print(json.dumps(info, indent=2))
