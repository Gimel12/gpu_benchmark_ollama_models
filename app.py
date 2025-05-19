from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import subprocess
import os
import json
from threading import Thread, Lock
from werkzeug.utils import secure_filename

app = Flask(__name__)
RESULTS_FILE = "benchmark_results.json"

from datetime import datetime

def format_timestamp(value):
    try:
        dt = datetime.fromisoformat(value)
        return dt.strftime('%b/%d/%Y %I:%M%p').lower().replace('am', 'am').replace('pm', 'pm')
    except Exception:
        return value

app.jinja_env.filters['format_timestamp'] = format_timestamp

# --- Benchmark process control ---
benchmark_process = None
benchmark_thread = None
process_lock = Lock()

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

@app.route('/run', methods=['GET', 'POST'])
def run_benchmark():
    global benchmark_process, benchmark_thread
    models_list = load_models()
    if request.method == 'POST':
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
            cmd += ['--gpus'] + gpus.split()
        if runs:
            cmd += ['--runs', runs]
        if api_host:
            cmd += ['--api-host', api_host]
        # Run benchmark as subprocess
        def run_bench():
            global benchmark_process
            with process_lock:
                benchmark_process = subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
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
    return render_template('run.html', models=models_list)

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
        return render_template('results.html', results=None)
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    return render_template('results.html', results=results)

@app.route('/download')
def download():
    if not os.path.exists(RESULTS_FILE):
        return "No results file found!", 404
    return send_file(RESULTS_FILE, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
