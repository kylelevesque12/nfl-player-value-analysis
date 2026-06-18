#!/usr/bin/env bash
# Launch the NFL Player Value & Fantasy Projection Lab Streamlit app locally.
#
# Usage:
#   ./run_app.sh
#
# Creates an isolated virtualenv on first run, installs dependencies, then
# starts the app. Re-running reuses the venv and skips reinstall.
set -euo pipefail

# Resolve repo root (directory this script lives in) so it works from anywhere.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

VENV_DIR=".venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Require Python 3.10+ (nflreadpy + the pinned scientific stack need it).
"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    sys.exit("Python 3.10+ is required; found %d.%d" % sys.version_info[:2])
PY

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtualenv in $VENV_DIR ..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Install deps once; touch a marker so repeat launches are fast.
if [ ! -f "$VENV_DIR/.deps-installed" ] || [ requirements.txt -nt "$VENV_DIR/.deps-installed" ]; then
  echo "Installing dependencies ..."
  pip install --upgrade pip >/dev/null
  pip install -r requirements.txt
  touch "$VENV_DIR/.deps-installed"
fi

echo "Starting Streamlit on http://localhost:8501 ..."
exec streamlit run app/streamlit_app.py
