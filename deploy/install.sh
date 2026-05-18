#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# Trace Up — Tennis Net Call  Board Deployment Script
# ══════════════════════════════════════════════════════════════════
# Run on the Ubuntu board:
#   chmod +x install.sh && sudo ./install.sh
# ══════════════════════════════════════════════════════════════════

set -euo pipefail

APP_DIR="/opt/tennis-net-call"
VENV_DIR="$APP_DIR/venv"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

echo -e "${CYAN}"
echo "  ╔══════════════════════════════════════╗"
echo "  ║   🎾  Trace Up Tennis Net Call      ║"
echo "  ╚══════════════════════════════════════╝"
echo -e "${NC}"

# ── Root check ────────────────────────────────────────────────────
[[ $EUID -ne 0 ]] && { echo -e "${RED}Run as root: sudo ./install.sh${NC}"; exit 1; }

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── [1/4] System packages ─────────────────────────────────────────
echo -e "\n${YELLOW}[1/4]${NC} Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    python3 python3-pip python3-venv \
    v4l-utils ffmpeg \
    libgl1 libglib2.0-0 \
    alsa-utils

# ── [2/4] Copy app ────────────────────────────────────────────────
echo -e "${YELLOW}[2/4]${NC} Copying app to ${APP_DIR}..."
mkdir -p "$APP_DIR"
cp -r "${REPO_ROOT}"/*.py    "$APP_DIR/"
cp    "${REPO_ROOT}/requirements.txt" "$APP_DIR/"
cp -r "${REPO_ROOT}/audio"   "$APP_DIR/" 2>/dev/null || mkdir -p "$APP_DIR/audio"
cp -r "${REPO_ROOT}/deploy"  "$APP_DIR/"
chmod +x "$APP_DIR/deploy/run.sh"

# ── [3/4] Python venv ─────────────────────────────────────────────
echo -e "${YELLOW}[3/4]${NC} Setting up Python virtual environment..."
python3 -m venv "$VENV_DIR"
# 嵌入式板上使用 headless 版本，无需 GUI 依赖
"$VENV_DIR/bin/pip" install --quiet \
    "opencv-python-headless>=4.8.0" \
    "numpy>=1.24.0" \
    "pygame>=2.5.0"

# ── [4/4] systemd service ─────────────────────────────────────────
echo -e "${YELLOW}[4/4]${NC} Installing systemd service..."
cp "${REPO_ROOT}/deploy/tennis-net-call.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable tennis-net-call
systemctl restart tennis-net-call

# ── Done ──────────────────────────────────────────────────────────
echo -e "\n${GREEN}══════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅  Tennis Net Call Installed!${NC}"
echo -e "${GREEN}══════════════════════════════════════${NC}"
echo ""
echo -e "  安装目录 : ${CYAN}${APP_DIR}${NC}"
echo -e "  摄像头   : ${CYAN}/dev/video0${NC}  (改 TENNIS_CAMERA 修改)"
echo -e "  网线位置 : ${CYAN}自动标定${NC}      (改 TENNIS_NET_LINE 手动指定 Y 坐标)"
echo ""
echo -e "  ${YELLOW}状态:${NC}  systemctl status tennis-net-call"
echo -e "  ${YELLOW}日志:${NC}  journalctl -u tennis-net-call -f"
echo ""
echo -e "  ${YELLOW}配置网线位置（Y坐标，0=自动标定）:${NC}"
echo -e "  编辑 /etc/systemd/system/tennis-net-call.service"
echo -e "  修改 TENNIS_NET_LINE 值，然后: systemctl daemon-reload && systemctl restart tennis-net-call"
echo ""

# 检查摄像头
if ls /dev/video* &>/dev/null; then
    echo -e "  📷 检测到摄像头: $(ls /dev/video*)"
else
    echo -e "  ${RED}⚠  未检测到摄像头 (/dev/video*)，请检查设备连接${NC}"
fi
