#!/usr/bin/env bash
set -euo pipefail
# Stop the guest after the agreed judging window. Verify Brev reports Stopped afterward.
sudo tee /etc/systemd/system/fortevision-budget-stop.service >/dev/null <<'UNIT'
[Unit]
Description=Stop hackathon VM after judging window
[Service]
Type=oneshot
ExecStart=/sbin/shutdown -h now
UNIT
sudo tee /etc/systemd/system/fortevision-budget-stop.timer >/dev/null <<'UNIT'
[Unit]
Description=ForteVision budget deadline: 19 October 2026 00:05 UTC+5
[Timer]
OnCalendar=2026-10-18 19:05:00 UTC
AccuracySec=1min
Persistent=false
Unit=fortevision-budget-stop.service
[Install]
WantedBy=timers.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now fortevision-budget-stop.timer
sudo systemctl list-timers fortevision-budget-stop.timer --no-pager
