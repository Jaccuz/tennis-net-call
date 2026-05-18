# 网球过网检测 — 语音播报系统
# Tennis Net Call — Real-time net clearance detection + voice feedback

## 概述

摄像头装在网柱上，朝向球场一侧。自动标定网线位置，实时追踪网球，检测球从网上方飞过时即时语音播报"好球"。

全程离线运行，无需联网。

## 硬件要求

- 摄像头：USB UVC 或 MIPI-CSI
- 工控板：Rockchip RK3576 / RK3588 或 Jetson 系列
- 音频：3.5mm 耳机孔或 HDMI 音频输出

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行
python main.py --camera 0
```

开机自动标定网线位置，2秒完成。之后每颗过网球都会即时语音播报。

## 项目结构

```
tennis-net-call/
├── main.py              # 主程序入口
├── court_calibrator.py  # 自动标定：检测网带、边线
├── ball_tracker.py      # 网球追踪：HSV颜色掩码 + Kalman滤波
├── shot_judge.py        # 好球判定：过网检测
├── audio_player.py      # 语音播报：预录音频即时播放
├── audio/               # 预录音频文件 (.wav)
├── deploy/
│   ├── install.sh               # 一键部署脚本
│   ├── run.sh                   # systemd 启动包装
│   └── tennis-net-call.service  # systemd 单元文件
├── requirements.txt
└── README.md
```

## 部署到 Ubuntu 工控板

### 一键部署

```bash
git clone <repo> /tmp/tennis-net-call
cd /tmp/tennis-net-call/deploy
chmod +x install.sh
sudo ./install.sh
```

脚本完成后服务自动启动并开机自启。

### 服务管理

```bash
sudo systemctl start   tennis-net-call   # 启动
sudo systemctl stop    tennis-net-call   # 停止
sudo systemctl restart tennis-net-call   # 重启
sudo systemctl status  tennis-net-call   # 状态
sudo journalctl -u tennis-net-call -f    # 实时日志
```

### 配置网线位置

编辑 `/etc/systemd/system/tennis-net-call.service`：

```ini
Environment=TENNIS_CAMERA=0       # 摄像头设备号
Environment=TENNIS_NET_LINE=0     # 0 = 自动标定，或手动填入 Y 坐标（如 180）
```

修改后：

```bash
sudo systemctl daemon-reload && sudo systemctl restart tennis-net-call
```

## 工作原理

```
摄像头帧 → 球检测(HSV掩码) → 轨迹追踪(Kalman) → 过网判定 → 语音播报
                ↓                                      ↓
           网线标定(霍夫)                         预录WAV池
```

网线位置在开机时自动标定一次，之后作为固定参照线。

## License

Internal use.
