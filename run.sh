#!/bin/bash

# Get the absolute path to the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set Python path to include the project root
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Default port if not set
PORT=${PORT:-5006}

echo "Starting CSO Analytics Dashboard on port $PORT..."
echo "Project root: ${PROJECT_ROOT}"
echo "Python path: ${PYTHONPATH}"

# Change to the project directory
cd "${PROJECT_ROOT}" || exit 1

# Run the application using the file path
python -m panel serve panel_app/app.py \
    --address=0.0.0.0 \
    --port=${PORT} \
    --allow-websocket-origin="*" \
    --autoreload \
    --show \
    --log-level=debug