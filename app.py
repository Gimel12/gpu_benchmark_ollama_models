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
import requests
import signal

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
INSTALL_LOG_FILE = "ollama_install.log"

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

# -------------------- Ollama helpers --------------------
def ollama_installed() -> bool:
    return shutil.which('ollama') is not None

def ollama_api_ok(host: str = "http://localhost:11434") -> bool:
    try:
        resp = requests.get(f"{host}/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False

def start_ollama_daemon():
    """Attempt to start ollama serve in the background if available."""
    if not ollama_installed():
        return
    # Best-effort: start if not responding
    if not ollama_api_ok():
        try:
            subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except Exception:
            pass

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

# -------------------- Ollama install/status endpoints --------------------
@app.route('/ollama/status')
def ollama_status_simple():
    """Lightweight status for front-end flow."""
    version = ""
    if ollama_installed():
        try:
            # Try several common flags for version output
            for cmd in (["ollama", "version"], ["ollama", "--version"], ["ollama", "-v"]):
                out = subprocess.run(cmd, capture_output=True, text=True)
                if out.returncode == 0 and out.stdout.strip():
                    version = out.stdout.strip()
                    break
        except Exception:
            pass
    return jsonify({
        'installed': ollama_installed(),
        'api': ollama_api_ok(),
        'version': version
    })

@app.route('/ollama/installed_models')
def ollama_installed_models():
    """Return a list of installed model tags from the local Ollama API."""
    models = []
    if ollama_api_ok():
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=3)
            if resp.ok:
                data = resp.json() or {}
                for m in data.get('models', []) or []:
                    name = m.get('name') or m.get('tag') or ''
                    if name:
                        models.append(name)
        except Exception:
            pass
    return jsonify({'models': models})

install_thread = None
install_lock = Lock()
pull_proc = None
pull_lock = Lock()
PULL_LOG_FILE = "ollama_pull.log"
current_pull_model = None

def _run_install_script():
    # Clear previous log
    open(INSTALL_LOG_FILE, 'w').close()
    with open(INSTALL_LOG_FILE, 'a') as lf:
        lf.write('Starting Ollama installation...\n')
        lf.flush()
        cmd = ['/bin/bash', '-lc', 'curl -fsSL https://ollama.com/install.sh | sh']
        try:
            import time
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

            last_update = time.time()
            hint_written = False

            def maybe_write_hint():
                nonlocal hint_written
                # Write a one-time hint if inactivity suggests sudo prompt
                if not hint_written:
                    lf.write("\n___NEED_SUDO___\n")
                    lf.write("No progress detected for a while. The installer may be waiting for sudo/password input.\n")
                    lf.write("If this machine requires sudo for /usr/local installs, please run the command below in a terminal, then click 'Run a Benchmark' again:\n")
                    lf.write("  curl -fsSL https://ollama.com/install.sh | sh\n\n")
                    lf.flush()
                    hint_written = True

            # Reader loop with inactivity checks
            while True:
                line = proc.stdout.readline()
                if line:
                    lf.write(line)
                    lf.flush()
                    last_update = time.time()
                    # check for permission-related errors that imply sudo
                    lower = line.lower()
                    if any(pat in lower for pat in (
                        'permission denied',
                        'operation not permitted',
                        'read-only file system',
                        'not writable',
                        'rm: cannot remove',
                        '/usr/local/'
                    )):
                        maybe_write_hint()
                elif proc.poll() is not None:
                    break
                else:
                    # no new data; check inactivity
                    if time.time() - last_update > 8:
                        maybe_write_hint()
                        # extend threshold so we don't spam
                        last_update = time.time()
                    time.sleep(0.5)

            proc.wait()
            lf.write(f"\nInstaller exit code: {proc.returncode}\n")
            lf.flush()
        except Exception as e:
            lf.write(f"Error running installer: {e}\n")
            lf.flush()
    # Try to start daemon and verify
    start_ollama_daemon()
    # Give it a moment to start
    try:
        for _ in range(10):
            if ollama_api_ok():
                break
            import time; time.sleep(1)
    except Exception:
        pass

@app.route('/ollama/install', methods=['POST'])
def ollama_install():
    global install_thread
    with install_lock:
        if install_thread is not None and install_thread.is_alive():
            return jsonify({'started': True})
        install_thread = Thread(target=_run_install_script, daemon=True)
        install_thread.start()
    return jsonify({'started': True})

@app.route('/ollama/install_progress')
def ollama_install_progress():
    try:
        if not os.path.exists(INSTALL_LOG_FILE):
            return jsonify({'lines': [], 'need_sudo': False, 'installed': ollama_installed(), 'api': ollama_api_ok()})
        with open(INSTALL_LOG_FILE, 'r') as f:
            lines = [line.rstrip('\n') for line in f.readlines()[-200:]]
        need_sudo = any('___NEED_SUDO___' in ln for ln in lines)
        # Filter sentinel from user-facing log
        filtered = [ln for ln in lines if '___NEED_SUDO___' not in ln]
        return jsonify({'lines': filtered, 'installed': ollama_installed(), 'api': ollama_api_ok(), 'need_sudo': need_sudo})
    except Exception:
        return jsonify({'lines': [], 'need_sudo': False})

@app.route('/ollama/start', methods=['POST'])
def ollama_start():
    start_ollama_daemon()
    return jsonify({'api': ollama_api_ok()})

@app.route('/ollama/install_sudo', methods=['POST'])
def ollama_install_sudo():
    """Run the installer with sudo using a password sent from the client.
    Security precautions:
    - Password is only read from JSON body and written to sudo stdin.
    - Never logged or stored; variable is cleared immediately after use.
    - Short timeout to avoid hanging.
    """
    data = request.get_json(silent=True) or {}
    password = data.get('password', '')
    if not isinstance(password, str) or not password:
        return jsonify({'started': False, 'error': 'Missing password'}), 400

    # Clear previous log and write a header
    open(INSTALL_LOG_FILE, 'w').close()
    with open(INSTALL_LOG_FILE, 'a') as lf:
        lf.write('Starting Ollama installation with sudo...\n')
        lf.flush()

    cmd = ['/bin/bash', '-lc', "sudo -S -k bash -lc 'curl -fsSL https://ollama.com/install.sh | sh'"]
    try:
        import time
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Feed password once; avoid storing beyond this scope
        try:
            proc.stdin.write(password + "\n")
            proc.stdin.flush()
        except Exception:
            pass
        finally:
            # Erase password variable
            password = ''

        # Stream output to log with masking
        start_time = time.time()
        TIMEOUT = 300  # 5 minutes max
        with open(INSTALL_LOG_FILE, 'a') as lf:
            while True:
                if proc.poll() is not None:
                    break
                line = proc.stdout.readline()
                if line:
                    low = line.lower()
                    # mask any sudo prompt lines
                    if 'password' in low:
                        line = '[sudo] password prompt...\n'
                    lf.write(line)
                    lf.flush()
                if time.time() - start_time > TIMEOUT:
                    # kill process group to avoid zombies
                    try:
                        proc.send_signal(signal.SIGTERM)
                    except Exception:
                        pass
                    lf.write('\nInstaller timed out.\n')
                    lf.flush()
                    break
            proc.wait(timeout=5)
            lf.write(f"\nInstaller exit code: {proc.returncode}\n")
            lf.flush()

        # Attempt to start daemon
        start_ollama_daemon()
        return jsonify({'started': True})
    except Exception as e:
        with open(INSTALL_LOG_FILE, 'a') as lf:
            lf.write(f"Error running sudo installer: {e}\n")
        return jsonify({'started': False, 'error': 'installer_failed'}), 500

# -------------------- Ollama pull endpoints --------------------
@app.route('/ollama/pull', methods=['POST'])
def ollama_pull():
    global pull_proc, current_pull_model
    data = request.get_json(silent=True) or {}
    model = (data.get('model') or '').strip()
    if not model:
        return jsonify({'started': False, 'error': 'missing_model'}), 400
    with pull_lock:
        if pull_proc is not None and pull_proc.poll() is None:
            return jsonify({'started': False, 'error': 'already_running'}), 409
        # Clear previous pull log and start process
        open(PULL_LOG_FILE, 'w').close()
        current_pull_model = model
        try:
            pull_proc = subprocess.Popen(
                ['ollama', 'pull', model],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
        except FileNotFoundError:
            return jsonify({'started': False, 'error': 'ollama_not_found'}), 500

        def stream_pull():
            with open(PULL_LOG_FILE, 'a') as lf:
                for line in pull_proc.stdout:
                    lf.write(line)
                    lf.flush()
            pull_proc.wait()
            # After pull, try to refresh daemon and status
            start_ollama_daemon()

        t = Thread(target=stream_pull, daemon=True)
        t.start()
        return jsonify({'started': True})

@app.route('/ollama/pull_progress')
def ollama_pull_progress():
    try:
        if not os.path.exists(PULL_LOG_FILE):
            return jsonify({'lines': [], 'running': False, 'done': False, 'success': False, 'model': current_pull_model, 'installed': ollama_installed(), 'api': ollama_api_ok()})
        with open(PULL_LOG_FILE, 'r') as f:
            lines = [ln.rstrip('\n') for ln in f.readlines()[-200:]]
        running = False
        with pull_lock:
            running = pull_proc is not None and pull_proc.poll() is None
            rc = None if running else (pull_proc.returncode if pull_proc is not None else None)
        success = (rc == 0)
        return jsonify({'lines': lines, 'running': running, 'done': not running and rc is not None, 'success': success, 'model': current_pull_model, 'installed': ollama_installed(), 'api': ollama_api_ok()})
    except Exception:
        return jsonify({'lines': [], 'running': False, 'done': False, 'success': False, 'model': current_pull_model})

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
              'Avg Temp (°C)', 'Avg Util (%)', 'Avg Mem (GiB)', 'Avg Power (W)', 'Timestamp']
    writer.writerow(header)
    
    # Write data rows
    for r in results:
        gpu_metrics = r.get('gpu_avg_metrics', {}) or {}
        avg_mem_gib = ''
        if 'avg_memory_used_mib' in gpu_metrics and gpu_metrics.get('avg_memory_used_mib') is not None:
            try:
                avg_mem_gib = float(gpu_metrics.get('avg_memory_used_mib')) / 1024.0
            except Exception:
                avg_mem_gib = ''
        row = [
            r.get('model', ''),
            r.get('gpu_idx', 'CPU'),
            r.get('tokens_per_second', ''),
            r.get('total_duration_seconds', ''),
            r.get('prompt', ''),
            gpu_metrics.get('avg_temperature_c', ''),
            gpu_metrics.get('avg_utilization_percent', ''),
            avg_mem_gib,
            gpu_metrics.get('avg_power_draw_w', ''),
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

@app.route('/gpu/metrics')
def gpu_metrics():
    """Return realtime GPU metrics from nvidia-smi.
    Fields: index, name, utilization_percent, memory_used_mib, memory_total_mib,
    temperature_c, power_draw_w, power_limit_w
    """
    try:
        # Query without units; returns one row per GPU
        cmd = [
            'nvidia-smi',
            '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit',
            '--format=csv,noheader,nounits'
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        if out.returncode != 0:
            return jsonify({'gpus': [], 'error': out.stderr.strip()}), 200
        gpus = []
        for line in out.stdout.strip().split('\n'):
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 8:
                continue
            try:
                gpus.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'utilization_percent': float(parts[2]),
                    'memory_used_mib': float(parts[3]),
                    'memory_total_mib': float(parts[4]),
                    'temperature_c': float(parts[5]),
                    'power_draw_w': float(parts[6]) if parts[6] not in ('N/A', '') else None,
                    'power_limit_w': float(parts[7]) if parts[7] not in ('N/A', '') else None,
                })
            except Exception:
                continue
        return jsonify({'gpus': gpus})
    except FileNotFoundError:
        return jsonify({'gpus': [], 'error': 'nvidia-smi not found'}), 200
    except Exception as e:
        return jsonify({'gpus': [], 'error': str(e)}), 200

@app.route('/gpu/list')
def gpu_list():
    """List GPUs with power limit metadata for the Advanced tab."""
    try:
        cmd = [
            'nvidia-smi',
            '--query-gpu=index,name,power.limit,power.max_limit,power.default_limit',
            '--format=csv,noheader,nounits'
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        if out.returncode != 0:
            return jsonify({'gpus': [], 'error': out.stderr.strip()}), 200
        gpus = []
        for line in out.stdout.strip().split('\n'):
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 5:
                continue
            def f(v):
                try:
                    return float(v)
                except Exception:
                    return None
            gpus.append({
                'index': int(parts[0]),
                'name': parts[1],
                'power_limit_w': f(parts[2]),
                'power_max_limit_w': f(parts[3]),
                'power_default_limit_w': f(parts[4]),
            })
        return jsonify({'gpus': gpus})
    except FileNotFoundError:
        return jsonify({'gpus': [], 'error': 'nvidia-smi not found'}), 200
    except Exception as e:
        return jsonify({'gpus': [], 'error': str(e)}), 200

@app.route('/gpu/settings_apply', methods=['POST'])
def gpu_settings_apply():
    """Apply basic GPU settings: power limit and persistence mode.
    Requires appropriate permissions; returns stderr in case of failure.
    Body JSON: { index: int, power_limit_w?: number, persistence?: '1'|'0' }
    """
    data = request.get_json(silent=True) or {}
    try:
        idx = int(data.get('index'))
    except Exception:
        return jsonify({'ok': False, 'error': 'invalid_index'}), 400
    power_limit = data.get('power_limit_w', None)
    persistence = data.get('persistence', None)
    outputs = []
    try:
        if power_limit is not None and str(power_limit).strip() != '':
            cmd_pl = ['nvidia-smi', '-i', str(idx), '-pl', str(int(float(power_limit)))]
            r = subprocess.run(cmd_pl, capture_output=True, text=True)
            outputs.append({'cmd': ' '.join(cmd_pl), 'rc': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr})
            if r.returncode != 0:
                return jsonify({'ok': False, 'step': 'power_limit', 'detail': r.stderr.strip()}), 200
        if persistence in ('0','1'):
            state = 'ENABLED' if persistence == '1' else 'DISABLED'
            cmd_pm = ['nvidia-smi', '-i', str(idx), '-pm', state]
            r = subprocess.run(cmd_pm, capture_output=True, text=True)
            outputs.append({'cmd': ' '.join(cmd_pm), 'rc': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr})
            if r.returncode != 0:
                return jsonify({'ok': False, 'step': 'persistence', 'detail': r.stderr.strip()}), 200
        return jsonify({'ok': True, 'steps': outputs})
    except FileNotFoundError:
        return jsonify({'ok': False, 'error': 'nvidia-smi not found'}), 200
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 200

@app.route('/gpu/settings_reset', methods=['POST'])
def gpu_settings_reset():
    """Reset power limit to GPU default limit for given index."""
    data = request.get_json(silent=True) or {}
    try:
        idx = int(data.get('index'))
    except Exception:
        return jsonify({'ok': False, 'error': 'invalid_index'}), 400
    try:
        # Read default limit
        out = subprocess.run([
            'nvidia-smi', '--query-gpu=power.default_limit', '--format=csv,noheader,nounits', '-i', str(idx)
        ], capture_output=True, text=True)
        if out.returncode != 0:
            return jsonify({'ok': False, 'error': out.stderr.strip()}), 200
        default_w = out.stdout.strip().split('\n')[0].strip()
        # Apply default
        r = subprocess.run(['nvidia-smi', '-i', str(idx), '-pl', default_w], capture_output=True, text=True)
        if r.returncode != 0:
            return jsonify({'ok': False, 'error': r.stderr.strip()}), 200
        return jsonify({'ok': True, 'power_limit_w': float(default_w)})
    except FileNotFoundError:
        return jsonify({'ok': False, 'error': 'nvidia-smi not found'}), 200
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 200

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
