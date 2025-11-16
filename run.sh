#!/bin/bash
panel serve panel_app/app.py --address=0.0.0.0 --port=$PORT --allow-websocket-origin="*"
