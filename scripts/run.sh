#!/usr/bin/env bash
set -euo pipefail
cd "${FORTE_HOME:-/home/ubuntu/forte-datathon}"
export FORTE_HOME="$PWD"
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
exec .venv/bin/python -m uvicorn forte.app:app --host 0.0.0.0 --port 8000 --workers 1 --no-access-log
