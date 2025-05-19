# Ollama GPU Inference Benchmark Tool

This tool benchmarks inference performance of various LLM models using Ollama across available GPUs on your system.

## Features

- Automatically detects available GPUs (NVIDIA, AMD, or Apple Silicon)
- Downloads models from Ollama if not already available
- Runs inference with a specified prompt
- Measures and reports performance metrics (tokens/second)
- Supports running on multiple GPUs
- Saves detailed benchmark results to a JSON file

## Requirements

- Python 3.6+
- Ollama running (locally or remotely)
- NVIDIA, AMD, or Apple Silicon GPU(s) for GPU benchmarking

## Installation

1. Clone this repository or download the files
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Basic usage:

```bash
python benchmark.py
```

This will run the benchmark with default settings (a selection of popular models, running on all available GPUs).

### Command Line Options

- `--models`: List of models to benchmark (space-separated)
- `--models-file`: Path to a JSON file containing models to benchmark
- `--prompt`: Text prompt to use for inference
- `--gpus`: Specific GPU indices to use (space-separated)
- `--runs`: Number of runs per model per GPU
- `--output`: Output file path for benchmark results
- `--api-host`: Ollama API host address

### Examples

Run benchmark on specific models:

```bash
python benchmark.py --models llama3:8b phi3:14b
```

Run benchmark using models from a JSON file:

```bash
python benchmark.py --models-file custom_models.json
```

Run benchmark with a custom prompt:

```bash
python benchmark.py --prompt "Explain quantum computing in simple terms."
```

Run benchmark on specific GPUs:

```bash
python benchmark.py --gpus 0 1
```

Run multiple times per configuration for better statistics:

```bash
python benchmark.py --runs 3
```

Connect to a remote Ollama instance:

```bash
python benchmark.py --api-host http://your-ollama-server:11434
```

## Output

The benchmark results are saved to a JSON file (default: `benchmark_results.json`) and a summary is printed to the console.

## Example Output

```
BENCHMARK SUMMARY
==================================================

Model: llama3:8b
  GPU 0: 45.23 tokens/sec (avg of 1 runs)

Model: phi3:14b
  GPU 0: 32.17 tokens/sec (avg of 1 runs)
```

## Models JSON File Format

You can specify models to benchmark in a JSON file. The default file is `models.json` in the same directory as the script. The format is:

```json
{
  "models": [
    {
      "name": "llama3:8b",
      "description": "Meta's Llama 3 8B model"
    },
    {
      "name": "phi3:14b",
      "description": "Microsoft's Phi-3 14B model"
    }
  ]
}
```

The `description` field is optional and for documentation purposes only.

## Notes

- The benchmark limits token generation to 100 tokens to keep run times reasonable
- For NVIDIA GPUs, the script uses the `CUDA_VISIBLE_DEVICES` environment variable to select specific GPUs
- Apple Silicon is treated as a single GPU
- If no GPUs are detected, the benchmark will run on CPU
