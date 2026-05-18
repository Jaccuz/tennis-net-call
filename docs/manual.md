# Tennis Net Call — 操作手册

## 目录

1. [系统概述](#1-系统概述)
2. [环境准备](#2-环境准备)
3. [本地测试命令](#3-本地测试命令)
4. [main.py 参数详解](#4-mainpy-参数详解)
5. [test_video.py 参数详解](#5-test_videopy-参数详解)
6. [各模块功能说明](#6-各模块功能说明)
7. [调试技巧](#7-调试技巧)
8. [常见问题](#8-常见问题)

---

## 1. 系统概述

```
摄像头（网柱位置）
    │
    ▼
ball_tracker.py   — HSV 颜色掩码检测网球 + Kalman 滤波平滑轨迹
    │
    ▼
court_calibrator.py — 开机自动标定网线 Y 坐标（霍夫直线检测）
    │
    ▼
shot_judge.py     — 判断球是否从球员侧飞过网线（过网 = 好球）
    │
    ▼
audio_player.py   — 即时播放预录音频（无音频文件则 pygame 合成音）
```

**两个可运行入口：**

| 脚本 | 用途 | 输入源 |
|------|------|--------|
| `main.py` | 生产 / 板端运行 | 实时摄像头 |
| `test_video.py` | 开发调试 | 本地视频文件，支持导出标注视频 |

---

## 2. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# Windows 本机（非 WSL）需要先安装 pygame 依赖音频驱动
# 无额外操作

# WSL2 — 确认 WSLg 可显示（用于 --debug 窗口）
echo $DISPLAY   # 应输出 :0 或类似值
```

**确认摄像头可用（Linux / WSL）：**

```bash
ls /dev/video*          # 列出所有设备
v4l2-ctl --list-devices # 查看详细信息
```

---

## 3. 本地测试命令

### main.py — 摄像头实时检测

```bash
# ── 最常用：默认摄像头，自动标定网线 ──
python main.py

# ── 指定摄像头设备号 ──
python main.py --camera 1

# ── 手动指定网线 Y 坐标（跳过自动标定，立即开始检测）──
python main.py --net-line 200

# ── 开启调试画面（需要图形界面）──
python main.py --debug

# ── 调整分辨率（默认 640×480）──
python main.py --width 1280 --height 720

# ── 组合示例：指定摄像头 + 手动网线 + 调试画面 ──
python main.py --camera 0 --net-line 180 --debug
```

### test_video.py — 视频文件测试

```bash
# ── 指定视频文件（最常用）──
python test_video.py demo/1_analyzed.avi

# ── WSL 访问 Windows 路径 ──
python test_video.py "/mnt/d/tennis_vedio/test_12m_2k_60.avi"

# ── 不传参数：使用脚本顶部 VIDEO_PATH 默认值 ──
python test_video.py
```

> `test_video.py` 所有配置通过脚本顶部常量修改，不支持命令行参数。

---

## 4. main.py 参数详解

### `--camera`

```
类型：int   默认：0
```

摄像头设备号，对应系统 `/dev/videoN`。

```bash
python main.py --camera 0   # /dev/video0（通常是 USB 摄像头）
python main.py --camera 1   # /dev/video1（第二个摄像头或内置）
```

**查找正确设备号：**

```bash
v4l2-ctl --list-devices
# 输出类似：
# USB 2.0 Camera:
#     /dev/video0    ← 这就是设备号 0
```

---

### `--net-line`

```
类型：int   默认：0（0 = 自动标定）
```

网线在画面中的 **Y 坐标（像素）**，从画面顶端向下计算。

- `0`：自动标定。程序采集 30 帧，用霍夫直线检测网带白色横条，投票确定 Y 坐标（约耗时 2 秒）。需要摄像头朝向球网且网带白色清晰可见。
- `> 0`：跳过标定，直接使用该值。适合摄像头固定、网线位置已知的情况。

**如何找到正确值：**

```bash
# 先用 --debug 看画面，观察网线在屏幕上的像素位置
python main.py --camera 0 --debug
# 黄色虚线即当前网线标定结果，记录下 Y 值后续直接填入
```

```bash
python main.py --net-line 200   # 网线在距顶端 200px 处
```

---

### `--width` / `--height`

```
类型：int   默认：640 / 480
```

摄像头采集分辨率（实际能用的分辨率取决于硬件）。

```bash
python main.py --width 640 --height 480   # 标清，推荐默认
python main.py --width 1280 --height 720  # HD，帧率会下降
python main.py --width 320 --height 240   # 低分辨率，高帧率，适合性能弱的板
```

**注意：** 分辨率越高，处理延迟越大。嵌入式板推荐 `640×480`。

---

### `--debug`

```
类型：flag（开关）   默认：关
```

开启实时调试画面（需要图形界面）。画面上显示：
- 黄色虚线：当前网线位置
- 彩色轨迹：球的历史运动路径（蓝→绿渐变）
- 绿色圆圈：当前检测到的球位置
- 大字幕：好球 / 下网判定结果

```bash
python main.py --debug          # 开启画面
# 画面内按 q 退出
```

在无图形界面的服务器 / 板子上加 `--debug` 会自动降级为无头模式，不报错。

---

## 5. test_video.py 参数详解

`test_video.py` 不使用命令行参数，所有配置写在脚本顶部常量区（第 22–53 行）。修改后直接保存运行。

### `VIDEO_PATH`

```python
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "demo/1_analyzed.avi"
```

视频文件路径。命令行第一个参数会覆盖默认值。

---

### `DISPLAY_WIDTH` / `DISPLAY_HEIGHT`

```python
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540
```

**显示窗口**分辨率（不影响检测精度）。根据显示器尺寸调整。

---

### `PROCESS_WIDTH` / `PROCESS_HEIGHT`

```python
PROCESS_WIDTH = 320
PROCESS_HEIGHT = 180
```

**检测处理**分辨率。每帧先缩放到此尺寸再做 HSV 检测和 Kalman 追踪。

- 越小：速度越快，适合低性能机器；细节丢失，小球可能漏检
- 越大：检测精度高；速度慢

**推荐比例：** 保持与原视频宽高比一致，否则球的坐标映射会偏移。

---

### `NET_LINE_RATIO`

```python
NET_LINE_RATIO = 0.38
```

网线位置占画面高度的比例（从顶端算）。`0.38` 表示网线在画面 38% 处。

**调整方法：** 暂停视频（空格键），观察网线在画面中的大概位置，估算比例。

```python
NET_LINE_RATIO = 0.30   # 网线偏上（摄像头俯角大）
NET_LINE_RATIO = 0.50   # 网线居中
NET_LINE_RATIO = 0.45   # 网线偏下（摄像头仰角大）
```

---

### `SKIP_FRAMES`

```python
SKIP_FRAMES = 2
```

跳帧数。`2` 表示每 3 帧只处理 1 帧（有效帧率 = 原始 FPS ÷ (SKIP_FRAMES+1)）。

| 值 | 处理帧率（60fps 原视频） | 适用场景 |
|----|--------------------------|----------|
| 0  | 60 fps | 低分辨率 / 性能充足 |
| 1  | 30 fps | 常规使用 |
| 2  | 20 fps | **默认** |
| 3  | 15 fps | 性能不足时 |

**注意：** 跳帧过多会导致快速球漏检，建议不超过 3。

---

### `HSV_LOWER` / `HSV_UPPER`

```python
HSV_LOWER = np.array([20, 70, 45])   # [H_min, S_min, V_min]
HSV_UPPER = np.array([50, 255, 255]) # [H_max, S_max, V_max]
```

网球颜色的 HSV 范围（OpenCV HSV 色域：H 0–179，S/V 0–255）。

| 分量 | 含义 | 调整场景 |
|------|------|----------|
| **H（色相）** | 颜色种类（20–50 覆盖黄绿） | 换球品牌/颜色时调整 |
| **S（饱和度）** | 颜色纯度（70+ 过滤低饱和杂色） | 场地干扰多时调高；逆光时调低 |
| **V（明度）** | 亮度（45+ 允许阴影下的球） | 室内暗光调低；室外强光调高 |

**常见问题及调整：**

```python
# 室内灯光偏暖，球偏橙黄
HSV_LOWER = np.array([15, 60, 40])
HSV_UPPER = np.array([40, 255, 255])

# 室外强光反射，球色发白
HSV_LOWER = np.array([20, 50, 80])
HSV_UPPER = np.array([55, 255, 255])

# 误检多（场地绿色或黄色线干扰）→ 提高 S 下限
HSV_LOWER = np.array([25, 100, 60])
```

---

### `PLAYER_SIDE`

```python
PLAYER_SIDE = "bottom"  # 或 "top"
```

球员在画面中的位置（决定哪个方向算"好球"）。

```
摄像头安装在网柱上，朝向一侧半场：

  "bottom"（默认）：
  ┌─────────────────────┐  ← 对手侧（此处过来的球被忽略）
  │                     │
  │──── 网线 ───────────│  ← NET_LINE_RATIO 处
  │                     │
  │   球员在这里击球    │  ← 球从下往上飞过网线 = 好球
  └─────────────────────┘

  "top"：
  ┌─────────────────────┐
  │   球员在这里击球    │  ← 球从上往下飞过网线 = 好球
  │──── 网线 ───────────│
  │                     │
  └─────────────────────┘  ← 对手侧
```

---

### `SHOT_VX_SIGN`

```python
SHOT_VX_SIGN = 0   # 0=自动标定，1=向右，-1=向左
```

球击打后的水平飞行方向，用于过滤反向飞来的对手球。

| 值 | 含义 |
|----|------|
| `0` | **自动标定**（推荐）。程序统计前 5 次过网事件的水平速度，自动确定击球方向。标定期间允许少量误报。 |
| `1` | 手动指定向右（球员在画面左侧击球向右侧飞） |
| `-1` | 手动指定向左（球员在画面右侧击球向左侧飞） |

---

### `FLASH_FRAMES_GOOD` / `FLASH_FRAMES_NET`

```python
FLASH_FRAMES_GOOD = 45   # 好球字幕停留帧数
FLASH_FRAMES_NET  = 30   # 下网字幕停留帧数
```

判定结果大字幕在画面上停留的帧数。帧率 20fps 时：
- `45` 帧 ≈ 2.25 秒
- `30` 帧 ≈ 1.5 秒

---

### `TRAIL_LENGTH`

```python
TRAIL_LENGTH = 15
```

球运动轨迹保留的历史帧数。值越大，尾迹越长；值越小，响应越快。

---

### `EXPORT_VIDEO`

```python
EXPORT_VIDEO = "demo/output.avi"   # 或 None
```

标注结果视频的输出路径。设为 `None` 不导出。

导出视频包含：网线标注、球轨迹、好球/下网字幕。

---

## 6. 各模块功能说明

### `court_calibrator.py` — 网线自动标定

**何时运行：** `main.py --net-line 0` 时，开机采集 30 帧自动检测。

**原理：**
1. 截取画面上半部分（网一般在上方）
2. 白色阈值 > 200 提取白色区域（网带是白色）
3. Canny 边缘检测 + 霍夫直线检测水平线
4. 多帧投票取众数（容差 ±3px）确定最终 Y 坐标

**标定失败的原因：**
- 摄像头没有对准球网
- 网带不是白色或光线不足
- 画面中白色水平线太多（干扰）

**处理方法：** 失败后改用 `--net-line <Y值>` 手动指定。

---

### `ball_tracker.py` — 网球追踪

**检测方法：** HSV 颜色掩码 → 轮廓提取 → 圆形度过滤 → Kalman 滤波

**Kalman 状态向量：** `[x, y, vx, vy]`（位置 + 速度）

- `processNoiseCov = 0.01`：模型噪声小（信任匀速运动假设）
- `measurementNoiseCov = 0.1`：测量噪声小（信任 HSV 检测结果）
- 检测到球时：用真实检测坐标作为当前位置（消除 Kalman 预测延迟）
- 丢失球时：用 Kalman 外推最多 8 帧；超过 8 帧完全重置

**圆形度过滤公式：** `circularity = 4π·Area / Perimeter²`，阈值 `> 0.4`

---

### `shot_judge.py` — 好球判定

**判定流程：**

```
检测到球往上飞（vy < 0）且有挥拍动作
    ↓
追踪球是否穿越网线（prev_y > net_line_y ≥ current_y）
    ↓
穿越 → "great_shot"（好球）
球开始下落但还在网线下方 → "net"（下网）
```

**冷却时间 `cooldown = 1.5s`：** 避免同一击球触发多次判定。

---

### `audio_player.py` — 语音播报

**优先级：**
1. 播放 `audio/` 目录下的预录 WAV 文件（随机从音频池中选一个）
2. 找不到文件时：pygame 合成提示音（好球=上升音阶，下网=低沉音）

**预录文件命名规范：**

```
audio/
├── great_shot_01.wav   # 好球音效（可有多个，随机播放）
├── great_shot_02.wav
├── great_shot_03.wav
└── net_01.wav          # 下网音效
```

---

## 7. 调试技巧

### 确认 HSV 范围是否准确

```python
# 在 test_video.py 同目录新建临时脚本，用鼠标取色验证
import cv2, sys
img = cv2.imread(sys.argv[1])
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# 鼠标点击画面查看 HSV 值
```

或直接暂停 `test_video.py`（空格键），观察球被检测到时绿色圆圈是否准确套住球。

---

### 调整 Kalman 参数

| 场景 | 调整 |
|------|------|
| 快速球预测位置滞后 | 调高 `processNoiseCov`（如 1.0），让 Kalman 跟随更快 |
| 轨迹抖动严重 | 调低 `processNoiseCov`（如 0.001），增加平滑 |
| 频繁丢失追踪 | 调低 `measurementNoiseCov`（更信任检测） |

---

### 实时查看日志（板端）

```bash
sudo journalctl -u tennis-net-call -f
```

每次好球/下网都会有对应日志行：

```
[15:32:01] GOOD SHOT! (#3) pos=(145,92) vel=(12,-18)
[15:32:05] NET (#1)
```

---

### 手动验证摄像头帧

```bash
# 截取一帧保存为图片，确认画面内容
ffmpeg -i /dev/video0 -frames:v 1 /tmp/frame.jpg
```

---

## 8. 常见问题

**Q：启动后停在"正在检测球网位置..."**

自动标定需要 30 帧时间（约 1 秒），正常现象。如果超过 5 秒还在等，说明标定失败，改用手动：
```bash
python main.py --net-line 200
```

---

**Q：检测不到球（无输出）**

1. 先用 `--debug` 看画面，确认摄像头有画面
2. 检查 HSV 范围是否覆盖当前球的颜色（光线/场地不同，球色偏差大）
3. 球速过快、分辨率过低时容易漏帧，适当降低 `SKIP_FRAMES`

---

**Q：误报太多（对手球也触发好球）**

- 确认 `PLAYER_SIDE` 设置正确
- 等待 `SHOT_VX_SIGN` 自动标定完成（需要 5 次真实过网事件）
- 或手动设置 `SHOT_VX_SIGN = 1` 或 `-1`

---

**Q：无声音**

```bash
# 确认 ALSA 设备存在
aplay -l

# 测试音频输出
aplay /usr/share/sounds/alsa/Front_Left.wav

# WSL2 无音频设备时，改用 Windows TTS（见 test_video.py 中的 speak_windows）
```

---

**Q：WSL2 下 --debug 窗口打不开**

```bash
# 确认 WSLg 已启动
echo $DISPLAY   # 应输出 :0
export DISPLAY=:0
python main.py --debug
```
