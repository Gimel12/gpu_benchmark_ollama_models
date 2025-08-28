# Ollama GPU Inference Benchmark Tool

Web application for benchmarking inference performance of Ollama models across your GPUs, with a built‑in real‑time GPU monitor and basic advanced GPU controls.

## Features

- __Web UI__: Simple two‑pane interface to select models, configure run settings, and view results.
- __Model management__: Shows available models and whether they’re installed; one‑click install from the UI.
- __Run Settings__: Configure prompt, GPU indices, runs per model, and API host.
- __Always‑visible GPU panel__: Real‑time metrics per GPU (utilization, temperature, memory usage, power) updating every second from `/gpu/metrics`.
- __Advanced GPU controls__: Optional power limit and persistence mode controls per GPU via `nvidia-smi`.
- __Multi‑GPU__: Run the same model on multiple GPUs.
- __Results__: Saves detailed results to JSON and displays summaries in the UI.

## Quick Start (Web App)

1) Ensure Python 3 and Ollama are installed and that Ollama is running locally (or set a remote API host in the UI).
2) From the project root, run:

```bash
chmod +x install.sh
./install.sh
```

The script will create a virtualenv, install dependencies, start the Flask app in the background, wait for it to come up, and open http://localhost:5000.

To stop the app:

```bash
if [ -f app.pid ]; then kill $(cat app.pid) && rm -f app.pid; fi
```

Tail logs:

```bash
tail -f webapp.log
```

## Requirements

- Python 3.8+
- Ollama running (locally or remotely)
- NVIDIA, AMD, or Apple Silicon GPU(s)
- For Advanced GPU controls: NVIDIA GPU with `nvidia-smi` available in PATH and sufficient permissions

## Configuration

Models shown and installed by the UI come from `models.json`. You can edit it from the UI or directly in the file.

Benchmark results are stored in `benchmark_results.json` and `results_history/`.

## Usage (Web UI)

1) Open the app (http://localhost:5000)
2) Select one or more models in the left pane
3) Configure settings in the right pane (Run Settings)
4) Monitor GPU health and usage in the GPU panel under Run Settings
5) Click “Run Benchmark”

The GPU panel shows per‑GPU: name, temperature, power draw vs limit, utilization, and VRAM usage. The Advanced tab lets you set power limit and persistence mode.

Notes:
- Changing power limits/persistence may require admin privileges. The app surfaces any errors from `nvidia-smi`.
- GPU polling is 1s; you can continue to use the UI while metrics update.

## CLI Benchmark (optional)

All functionality can be driven from the Web UI, but a CLI is also available:

- `--models`: List of models to benchmark (space-separated)
- `--models-file`: Path to a JSON file containing models to benchmark
- `--prompt`: Text prompt to use for inference
- `--gpus`: Specific GPU indices to use (space-separated)
- `--runs`: Number of runs per model per GPU
- `--output`: Output file path for benchmark results
- `--api-host`: Ollama API host address

Examples:

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

- The benchmark limits token generation to keep run times reasonable.
- For NVIDIA GPUs, the CLI can use `CUDA_VISIBLE_DEVICES` to select GPUs; the Web UI uses indices.
- Apple Silicon is treated as a single GPU.
- If no GPUs are detected, the benchmark will run on CPU.

## Troubleshooting

- If the page doesn’t open automatically, visit http://localhost:5000.
- Check `webapp.log` for server errors.
- If GPU metrics show errors, ensure `nvidia-smi` is installed and in PATH (NVIDIA only).
- For remote Ollama, set the API Host under Run Settings.

## Security

Advanced GPU settings change device power and persistence state via `nvidia-smi`. Use caution, and ensure you have appropriate permissions.
