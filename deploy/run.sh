#!/usr/bin/env bash
# ── Tennis Net Call 启动包装脚本 ──────────────────────────────────
# 供 systemd 调用，从环境变量读取配置后启动主程序。
# 也可手动执行: sudo /opt/tennis-net-call/deploy/run.sh

APP_DIR="/opt/tennis-net-call"
VENV_PYTHON="$APP_DIR/venv/bin/python"

CAMERA="${TENNIS_CAMERA:-0}"
NET_LINE="${TENNIS_NET_LINE:-0}"

exec "$VENV_PYTHON" "$APP_DIR/main.py" \
    --camera "$CAMERA" \
    --net-line "$NET_LINE"
