#!/bin/bash
panel serve panel/app.py --address=0.0.0.0 --port=$PORT --allow-websocket-origin="*"
