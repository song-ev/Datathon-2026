#!/usr/bin/env bash
set -euo pipefail
root="${FORTE_HOME:-/home/ubuntu/forte-datathon}"
if [[ "$root" != /home/ubuntu/forte-datathon ]]; then
  echo 'Review service paths before installing outside /home/ubuntu/forte-datathon' >&2
  exit 1
fi
sudo tee /etc/systemd/system/fortevision.service >/dev/null <<'UNIT'
[Unit]
Description=ForteVision local video analysis
After=network-online.target
Wants=network-online.target
[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/forte-datathon
Environment=FORTE_HOME=/home/ubuntu/forte-datathon
ExecStart=/bin/bash /home/ubuntu/forte-datathon/src/scripts/run.sh
Restart=on-failure
RestartSec=5
TimeoutStopSec=20
KillMode=control-group
UMask=0077
[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now fortevision.service
sudo systemctl is-active fortevision.service
