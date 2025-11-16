#!/bin/bash
panel serve cso-lab/panel/app.py --address=0.0.0.0 --port=$PORT --allow-websocket-origin="*"
