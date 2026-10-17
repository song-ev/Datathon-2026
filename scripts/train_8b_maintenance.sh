#!/usr/bin/env bash
# Bounded experiment on the existing GPU; always restore the demo.
set -euo pipefail
root=/home/ubuntu/forte-datathon
export FORTE_HOME="$root" PYTHONPATH="$root/src"
export MODEL_PATH="$root/models/Qwen3-VL-8B-Instruct"
"$root/.venv/bin/python" - <<'PY'
import json,urllib.request
j=json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health'))
if j.get('busy'):raise SystemExit('An active jury job is running; retry later')
PY
trap 'sudo systemctl start fortevision.service' EXIT
sudo systemctl stop fortevision.service
timeout --signal=TERM --kill-after=30s 300s "$root/.venv/bin/python" -u "$root/src/scripts/train_lora.py" --local-windows --trial-steps 4
timeout --signal=TERM --kill-after=30s 2400s "$root/.venv/bin/python" -u "$root/src/scripts/train_lora.py" --local-windows --epochs 4
