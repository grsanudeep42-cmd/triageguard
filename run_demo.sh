#!/usr/bin/env bash
set -e

# Support dynamic port for deployment (e.g. Render, Railway, Hugging Face Spaces, Cloud Run)
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# Find virtualenv python and uvicorn
if [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/bin/uvicorn" ]; then
    UVICORN="$VIRTUAL_ENV/bin/uvicorn"
elif [ -f "/app/.venv/bin/uvicorn" ]; then
    UVICORN="/app/.venv/bin/uvicorn"
elif [ -f ".venv/bin/uvicorn" ]; then
    UVICORN=".venv/bin/uvicorn"
elif [ -f "/home/anudeep/projects/ap hackathon/.venv/bin/uvicorn" ]; then
    UVICORN="/home/anudeep/projects/ap hackathon/.venv/bin/uvicorn"
elif command -v uvicorn >/dev/null 2>&1; then
    UVICORN="uvicorn"
else
    UVICORN="python3 -m uvicorn"
fi

# Build web frontend if present and not prebuilt
if [ -d "web" ] && [ -f "web/package.json" ]; then
    if [ ! -d "web/dist" ]; then
        if command -v npm >/dev/null 2>&1; then
            echo "=== 1. Building frontend ==="
            (cd web && npm run build)
        else
            echo "=== 1. Warning: npm not found, skipping frontend build ==="
        fi
    else
        echo "=== 1. Frontend dist already built (web/dist) ==="
    fi
fi

# Locate application module
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
if [ -f "app/main.py" ]; then
    APP_MODULE="app.main:app"
elif [ -f "../tg-ui/app/main.py" ]; then
    export PYTHONPATH="../tg-ui:$PYTHONPATH"
    APP_MODULE="app.main:app"
elif [ -f "/home/anudeep/projects/tg-ui/app/main.py" ]; then
    export PYTHONPATH="/home/anudeep/projects/tg-ui:$PYTHONPATH"
    APP_MODULE="app.main:app"
else
    echo "Error: app/main.py not found in current directory or tg-ui."
    exit 1
fi

echo "=== 2. Starting TriageGuard FastAPI server at http://${HOST}:${PORT} ==="
exec "$UVICORN" "$APP_MODULE" --host "$HOST" --port "$PORT"
