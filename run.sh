#!/bin/bash
# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Default port if not set
PORT=${PORT:-5006}

echo "Starting CSO Analytics Dashboard on port $PORT..."

# Run the Panel server with autoreload for development
panel serve panel_app/app.py \
    --address=0.0.0.0 \
    --port=$PORT \
    --allow-websocket-origin="*" \
    --autoreload \
    --show \
    --log-level=debug