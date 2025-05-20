from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, Response
import subprocess
import os
import json
import csv
import io
import shutil
import uuid
from threading import Thread, Lock
from werkzeug.utils import secure_filename
from datetime import datetime

# Import system info module
from system_info import get_system_info, test_gpu_inference, check_ollama_status

def format_timestamp(value):
    try:
        dt = datetime.fromisoformat(value)
        return dt.strftime('%b/%d/%Y %I:%M%p').lower().replace('am', 'am').replace('pm', 'pm')
    except Exception:
        return value

app = Flask(__name__)
RESULTS_FILE = "benchmark_results.json"
RESULTS_HISTORY_DIR = "results_history"
PRESETS_FILE = "benchmark_presets.json"

# Create necessary directories
os.makedirs(RESULTS_HISTORY_DIR, exist_ok=True)

app.jinja_env.filters['format_timestamp'] = format_timestamp

# --- Benchmark process control ---
benchmark_process = None
benchmark_thread = None
process_lock = Lock()

LOG_FILE = "benchmark.log"

# --- Predefined presets ---
PREDEFINED_PRESETS = {
    "quick": {
        "name": "Quick Test",
        "selected_models": ["llama3:8b"],
        "models": "",
        "gpus": "0",
        "runs": 1,
        "prompt": "Write a short paragraph about AI."
    },
    "full": {
        "name": "Full Benchmark",
        "selected_models": [],  # Will use all available models
        "models": "",
        "gpus": "",  # Will use all available GPUs
        "runs": 1,
        "prompt": "Write an essay about the USA revolution."
    },
    "stress": {
        "name": "Stress Test",
        "selected_models": ["llama3:70b", "nemotron:70b"],
        "models": "",
        "gpus": "",  # Will use all available GPUs
        "runs": 3,
        "prompt": "Write a detailed analysis of quantum computing and its potential applications."
    }
}

@app.route('/')
def index():
    return render_template('index.html')

def load_models():
    models_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.json')
    if os.path.exists(models_file):
        with open(models_file) as f:
            data = json.load(f)
        return data.get('models', [])
    return []

def load_presets():
    """Load user-defined benchmark presets"""
    if os.path.exists(PRESETS_FILE):
        try:
            with open(PRESETS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading presets: {e}")
    return []

def save_presets(presets):
    """Save user-defined benchmark presets"""
    with open(PRESETS_FILE, 'w') as f:
        json.dump(presets, f, indent=2)

@app.route('/presets')
def presets():
    """View benchmark presets"""
    models_list = load_models()
    user_presets = load_presets()
    return render_template('presets.html', presets=user_presets, models=models_list)

@app.route('/save_preset', methods=['POST'])
def save_preset():
    """Save a benchmark preset"""
    preset_id = request.form.get('preset_id')
    preset = {
        'id': preset_id if preset_id else str(uuid.uuid4()),
        'name': request.form.get('name'),
        'selected_models': request.form.getlist('selected_models'),
        'models': request.form.get('models'),
        'prompt': request.form.get('prompt'),
        'gpus': request.form.get('gpus'),
        'runs': request.form.get('runs'),
        'api_host': request.form.get('api_host')
    }
    
    # Load existing presets
    presets = load_presets()
    
    # Update existing preset or add new one
    if preset_id:
        for i, p in enumerate(presets):
            if p.get('id') == preset_id:
                presets[i] = preset
                break
    else:
        presets.append(preset)
    
    # Save presets
    save_presets(presets)
    
    return redirect(url_for('presets'))

@app.route('/get_preset/<preset_id>')
def get_preset(preset_id):
    """Get a preset by ID"""
    # Check if it's a predefined preset
    if preset_id in PREDEFINED_PRESETS:
        preset = PREDEFINED_PRESETS[preset_id].copy()
        preset['id'] = preset_id
        return jsonify(preset)
    
    # Check user presets
    presets = load_presets()
    for preset in presets:
        if preset.get('id') == preset_id:
            return jsonify(preset)
    
    return jsonify({'error': 'Preset not found'}), 404

@app.route('/delete_preset/<preset_id>', methods=['POST'])
def delete_preset(preset_id):
    """Delete a benchmark preset"""
    presets = load_presets()
    presets = [p for p in presets if p.get('id') != preset_id]
    save_presets(presets)
    return redirect(url_for('presets'))

@app.route('/run', methods=['GET', 'POST'])
def run_benchmark():
    global benchmark_process, benchmark_thread
    models_list = load_models()
    
    # Handle preset loading from GET parameters
    preset_id = request.args.get('preset_id')
    preset_key = request.args.get('preset')
    
    # Initialize form data
    form_data = {
        'selected_models': [],
        'models': '',
        'prompt': '',
        'gpus': '',
        'runs': '1',
        'api_host': ''
    }
    
    # Load preset if specified
    if preset_id or preset_key:
        preset = None
        
        # Check predefined presets
        if preset_key and preset_key in PREDEFINED_PRESETS:
            preset = PREDEFINED_PRESETS[preset_key]
        
        # Check user presets
        elif preset_id:
            presets = load_presets()
            for p in presets:
                if p.get('id') == preset_id:
                    preset = p
                    break
        
        # Apply preset if found
        if preset:
            form_data['selected_models'] = preset.get('selected_models', [])
            form_data['models'] = preset.get('models', '')
            form_data['prompt'] = preset.get('prompt', '')
            form_data['gpus'] = preset.get('gpus', '')
            form_data['runs'] = preset.get('runs', '1')
            form_data['api_host'] = preset.get('api_host', '')
    
    if request.method == 'POST':
        # Get form values
        selected_models = request.form.getlist('selected_models')
        models_text = request.form.get('models')
        models = ' '.join(selected_models) if selected_models else models_text
        prompt = request.form.get('prompt')
        gpus = request.form.get('gpus')
        runs = request.form.get('runs')
        api_host = request.form.get('api_host')
        output = RESULTS_FILE
        
        # Build command
        cmd = [
            'python', 'benchmark.py',
            '--output', output
        ]
        if models:
            cmd += ['--models'] + models.split()
        if prompt:
            cmd += ['--prompt', prompt]
        if gpus:
            # Properly handle GPU indices - ensure each index is a separate argument
            gpu_indices = [idx.strip() for idx in gpus.split() if idx.strip()]
            if gpu_indices:
                cmd += ['--gpus'] + gpu_indices
        if runs:
            cmd += ['--runs', runs]
        if api_host:
            cmd += ['--api-host', api_host]
        
        # Run benchmark as subprocess
        def run_bench():
            global benchmark_process
            # Clear log file before starting
            open(LOG_FILE, 'w').close()
            with process_lock:
                benchmark_process = subprocess.Popen(
                    cmd,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
            # Stream output to log file in real time
            with open(LOG_FILE, 'a') as logf:
                for line in benchmark_process.stdout:
                    logf.write(line)
                    logf.flush()
            benchmark_process.wait()
            with process_lock:
                benchmark_process = None
        
        with process_lock:
            if benchmark_thread is not None and benchmark_thread.is_alive():
                return redirect(url_for('results'))  # Prevent double start
            benchmark_thread = Thread(target=run_bench)
            benchmark_thread.start()
        
        benchmark_thread.join()
        return redirect(url_for('results'))
    
    return render_template('run.html', models=models_list, form_data=form_data)

@app.route('/progress')
def progress():
    try:
        with open(LOG_FILE, 'r') as f:
            # Read the last 100 lines and strip whitespace
            lines = [line.strip() for line in f.readlines()[-100:]]
            # Filter out empty lines
            lines = [line for line in lines if line]
        return jsonify({'lines': lines})
    except Exception as e:
        print(f"Error reading log file: {e}")
        return jsonify({'lines': []})

@app.route('/stop', methods=['POST'])
def stop_benchmark():
    global benchmark_process
    with process_lock:
        if benchmark_process is not None:
            benchmark_process.terminate()
            benchmark_process = None
    return ('', 204)


@app.route('/models', methods=['GET', 'POST'])
def edit_models():
    models_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.json')
    models = load_models()
    if request.method == 'POST':
        # Handle add/remove
        if 'add' in request.form:
            models.append({'name': '', 'description': ''})
        elif 'remove' in request.form:
            idx = int(request.form['remove'])
            if 0 <= idx < len(models):
                models.pop(idx)
        else:
            # Save changes
            new_models = []
            for i in range(len(models)):
                name = request.form.get(f'name_{i}', '').strip()
                desc = request.form.get(f'desc_{i}', '').strip()
                if name:
                    new_models.append({'name': name, 'description': desc})
            models = new_models
            with open(models_file, 'w') as f:
                json.dump({'models': models}, f, indent=2)
        # Save after add/remove
        with open(models_file, 'w') as f:
            json.dump({'models': models}, f, indent=2)
    return render_template('models.html', models=models)

@app.route('/results')
def results():
    if not os.path.exists(RESULTS_FILE):
        return render_template('results.html', results=None, history=False)
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    return render_template('results.html', results=results, history=False)

@app.route('/save_results', methods=['POST'])
def save_results_to_history():
    if not os.path.exists(RESULTS_FILE):
        return jsonify({'success': False, 'error': 'No results to save'})
    
    # Generate a timestamp for the filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Get optional name from form
    name = request.form.get('name', '')
    if name:
        filename = f"{timestamp}_{secure_filename(name)}.json"
    else:
        filename = f"{timestamp}.json"
    
    # Copy the current results to the history directory
    history_file = os.path.join(RESULTS_HISTORY_DIR, filename)
    shutil.copy2(RESULTS_FILE, history_file)
    
    return jsonify({'success': True, 'filename': filename})

@app.route('/history')
def history():
    # Get all result files from history directory
    history_files = []
    if os.path.exists(RESULTS_HISTORY_DIR):
        for filename in os.listdir(RESULTS_HISTORY_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(RESULTS_HISTORY_DIR, filename)
                try:
                    with open(filepath) as f:
                        data = json.load(f)
                    
                    # Extract timestamp from filename or use file creation time
                    timestamp = filename.split('_')[0] if '_' in filename else os.path.getctime(filepath)
                    
                    # Get a name if it exists in the filename
                    name = filename[filename.find('_')+1:-5] if '_' in filename and filename.find('_') < len(filename)-5 else 'Unnamed'
                    
                    # Get first model name and count models for display
                    first_model = data[0]['model'] if data else 'Unknown'
                    model_count = len(set(r['model'] for r in data))
                    
                    history_files.append({
                        'filename': filename,
                        'timestamp': timestamp,
                        'name': name,
                        'model_sample': first_model,
                        'model_count': model_count,
                        'run_count': len(data)
                    })
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    # Sort by timestamp (newest first)
    history_files.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return render_template('history.html', history=history_files)

@app.route('/view_history/<filename>')
def view_history(filename):
    filepath = os.path.join(RESULTS_HISTORY_DIR, secure_filename(filename))
    if not os.path.exists(filepath):
        return redirect(url_for('history'))
    
    with open(filepath) as f:
        results = json.load(f)
    
    return render_template('results.html', results=results, history=True, history_file=filename)

@app.route('/delete_history/<filename>', methods=['POST'])
def delete_history(filename):
    filepath = os.path.join(RESULTS_HISTORY_DIR, secure_filename(filename))
    if os.path.exists(filepath):
        os.remove(filepath)
    
    return redirect(url_for('history'))

@app.route('/export_csv/<filename>')
def export_csv(filename):
    # Handle current results or history file
    if filename == 'current':
        filepath = RESULTS_FILE
    else:
        filepath = os.path.join(RESULTS_HISTORY_DIR, secure_filename(filename))
    
    if not os.path.exists(filepath):
        return "File not found", 404
    
    # Load the JSON data
    with open(filepath) as f:
        results = json.load(f)
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    header = ['Model', 'GPU', 'Tokens/sec', 'Duration (s)', 'Prompt', 
              'Avg Temp (°C)', 'Avg Util (%)', 'Avg Mem (%)', 'Avg Power (W)', 'Timestamp']
    writer.writerow(header)
    
    # Write data rows
    for r in results:
        row = [
            r.get('model', ''),
            r.get('gpu_idx', 'CPU'),
            r.get('tokens_per_second', ''),
            r.get('total_duration_seconds', ''),
            r.get('prompt', ''),
            r.get('gpu_avg_metrics', {}).get('avg_temperature_c', ''),
            r.get('gpu_avg_metrics', {}).get('avg_utilization_percent', ''),
            r.get('gpu_avg_metrics', {}).get('avg_memory_utilization_percent', ''),
            r.get('gpu_avg_metrics', {}).get('avg_power_draw_w', ''),
            r.get('timestamp', '')
        ]
        writer.writerow(row)
    
    # Prepare response
    output.seek(0)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename=benchmark_results_{timestamp}.csv"}
    )

@app.route('/download')
def download():
    if not os.path.exists(RESULTS_FILE):
        return "No results file found!", 404
    return send_file(RESULTS_FILE, as_attachment=True, download_name="benchmark_results.json")

@app.route('/system')
def system():
    """Display system information"""
    system_info = get_system_info()
    return render_template('system_info.html', system_info=system_info)

@app.route('/test_gpu')
def test_gpu():
    """Run a test to check if GPU is being used for inference"""
    result = test_gpu_inference()
    return jsonify(result)

@app.route('/check_ollama')
def check_ollama():
    """Check Ollama status"""
    result = check_ollama_status()
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
