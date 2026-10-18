#!/usr/bin/env bash
set -e

# Find virtualenv python
if [ -d "/home/anudeep/projects/ap hackathon/.venv" ]; then
    VENV_PATH="/home/anudeep/projects/ap hackathon/.venv"
elif [ -d ".venv" ]; then
    VENV_PATH=".venv"
else
    echo "Virtualenv not found!"
    exit 1
fi

PYTHON="${VENV_PATH}/bin/python"
UVICORN="${VENV_PATH}/bin/uvicorn"

echo "=== 1. Building frontend ==="
(cd web && npm run build)

echo "=== 2. Starting FastAPI server at http://localhost:8000 ==="
export PYTHONPATH=.
exec "$UVICORN" app.main:app --host 0.0.0.0 --port 8000
