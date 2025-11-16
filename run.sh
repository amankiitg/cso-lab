#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}"

# Default port if not set
PORT=${PORT:-5006}

echo "Starting CSO Analytics Dashboard on port $PORT..."
echo "Python path: $PYTHONPATH"

# Change to the project directory
cd "${SCRIPT_DIR}" || exit 1

# Run the Panel server with autoreload for development
python -m panel serve panel_app/app.py \
    --address=0.0.0.0 \
    --port=$PORT \
    --allow-websocket-origin="*" \
    --autoreload \
    --show \
    --log-level=debug