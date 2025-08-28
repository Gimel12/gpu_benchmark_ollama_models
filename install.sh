#!/usr/bin/env bash
# Setup and launch the BIZON GPU Benchmark web app
# - Creates a Python virtual environment
# - Installs dependencies
# - Starts the Flask app in the background
# - Opens http://localhost:5000 in your default browser (if available)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

URL="http://localhost:5000"
PID_FILE="app.pid"
LOG_FILE="webapp.log"

# 1) Find Python 3
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "Error: Python 3 is required but was not found." >&2
  exit 1
fi

# 2) Create venv if missing
if [ ! -d .venv ]; then
  echo "[+] Creating virtual environment (.venv)"
  "$PY" -m venv .venv
fi

# 3) Activate venv
# shellcheck disable=SC1091
source .venv/bin/activate

# 4) Upgrade pip tools and install deps
python -m pip install --upgrade pip setuptools wheel
if [ -f requirements.txt ]; then
  echo "[+] Installing Python dependencies from requirements.txt"
  pip install -r requirements.txt
else
  echo "[!] requirements.txt not found. Skipping dependency install."
fi

# 5) Start the app in the background
if [ -f "$PID_FILE" ] && ps -p "$(cat "$PID_FILE" 2>/dev/null)" >/dev/null 2>&1; then
  echo "[i] App already running with PID $(cat "$PID_FILE")."
else
  echo "[+] Starting web app (logs: $LOG_FILE)"
  # Use nohup so it survives the terminal closing; write PID to file
  nohup python app.py >"$LOG_FILE" 2>&1 &
  APP_PID=$!
  echo "$APP_PID" > "$PID_FILE"
  disown "$APP_PID" 2>/dev/null || true
fi

# 6) Wait until the server responds (max 30s)
TIMEOUT=30
ELAPSED=0
printf "[i] Waiting for server to be ready"
until curl -sSf "$URL" >/dev/null 2>&1; do
  sleep 1
  ELAPSED=$((ELAPSED+1))
  printf "."
  if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "\n[!] Server did not respond within ${TIMEOUT}s. Check $LOG_FILE for details."
    break
  fi
done

echo "\n[✓] App is (likely) running at $URL"

# 7) Open the browser if possible
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$URL" >/dev/null 2>&1 || true
elif command -v sensible-browser >/dev/null 2>&1; then
  sensible-browser "$URL" >/dev/null 2>&1 || true
else
  echo "Open this link in your browser: $URL"
fi

cat <<EOF

How to manage the app:
- Logs: tail -f $LOG_FILE
- Stop:  if [ -f $PID_FILE ]; then kill \\$(cat $PID_FILE) && rm -f $PID_FILE; fi
- Restart: (run this script again)

EOF
