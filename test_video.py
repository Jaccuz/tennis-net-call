#!/usr/bin/env python3
"""
网球过网检测 — 视频文件 / 实时摄像头 (WSL2 + WSLg)
摄像头装在右侧网柱，镜头朝球员方向。

判定原理：
  在球网上方自动框选"好球区"：
    球从左边界进入、右边界飞出 = 好球
    球从左边界进入、下边界落下 = 下网

自动标定：
  启动时扫描前 N 帧，找画面右侧白色网带的上沿，作为区域下边界。
  标注帧保存至 demo/zone_debug.jpg 供离线核验。

用法:
  python3 test_video.py [视频路径]
按键:
  q / ESC  退出   空格  暂停/继续
"""
import sys, os, subprocess, time
os.environ["DISPLAY"]                = ":0"
os.environ["OPENCV_LOG_LEVEL"]       = "ERROR"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
import numpy as np
import cv2
from collections import deque

# ═══════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════
VIDEO_PATH     = sys.argv[1] if len(sys.argv) > 1 \
                 else "/mnt/d/新建文件夹 (2)/摄像头测试/test_12m_2k_60.avi"
DISPLAY_WIDTH  = 960
DISPLAY_HEIGHT = 540
PROCESS_WIDTH  = 320
PROCESS_HEIGHT = 180
SKIP_FRAMES    = 2

# ── 网球 HSV ──
# 提高 S_min / V_min：树影偏暗（V<80）、低饱和（S<90），可有效过滤
HSV_LOWER = np.array([20,  90,  80])
HSV_UPPER = np.array([50, 255, 255])

# ── 自动标定参数 ──
CALIB_FRAMES      = 120    # 扫描帧数（约 2s@60fps）
NET_WHITE_THRESH  = 185    # 白色网带亮度阈值（光线暗时调低，如 165）
# 网带 Y 方向搜索区（ROI 右侧 55%，跳过天空）
NET_ROI_X_RATIO   = 0.55
NET_ROI_Y_START   = 0.30
NET_ROI_Y_END     = 0.92
NET_MIN_WHITE_COL  = 0.12   # 某行至少 12% 的列是白色，才算候选行
# ── 对比度校验：白带 + 正下方深色网眼 ──
# 这是核心判据：只有球网有"白带紧接深色网眼"的特征，建筑/围栏没有
NET_DARK_THRESH    = 70     # 网眼暗色阈值（低于此为深色像素）
NET_DARK_OFFSET    = 6      # 白带下方多少 px 开始检验深色
NET_DARK_WINDOW    = 20     # 深色检验区高度（px）
NET_DARK_MIN_RATIO = 0.22   # 深色区占比门槛，低于此说明下方不是网眼
# 网带 X 方向搜索（从帧左 35% 处向右扫）
NET_X_SEARCH_RATIO = 0.35
# ── 好球区参数 ──
ZONE_Y_TOP_MARGIN = 0.05   # 好球区上边界（帧高 × 5%，留天空余量）
# 好球区宽度（处理分辨率像素）：左边界 = 球网正上方，右边界 = 左 + ZONE_WIDTH
# 调大 → 球需飞更远才触发；调小 → 球过网即触发
# 球走抛物线，设太宽到不了右边界；建议 35-65px
ZONE_WIDTH        = 50

# ── 兜底默认区域（检测失败时使用，处理分辨率 320×180）──
DEFAULT_ZONE = (210, 9, 260, 95)  # (net_x, y_top, net_x+ZONE_WIDTH, net_top_y)

# ── 输出 ──
FLASH_FRAMES_GOOD = 45
FLASH_FRAMES_NET  = 30
TRAIL_LENGTH      = 15
EXPORT_VIDEO      = "demo/output.avi"   # None = 不导出
DEBUG_FRAME_PATH  = "demo/zone_debug.jpg"


# ═══════════════════════════════════════════
# Kalman 追踪
# ═══════════════════════════════════════════
class BallTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix    = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 1.0
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        self.kf.errorCovPost        = np.diag([10., 10., 100., 100.]).astype(np.float32)
        self.initialized = False
        self.last_pos    = None
        self.trajectory  = deque(maxlen=TRAIL_LENGTH)
        self.lost_frames = 0

    def update(self, detection=None):
        if detection is not None:
            x, y = detection
            if not self.initialized:
                self.kf.statePost    = np.array([[x],[y],[0],[0]], np.float32)
                self.kf.errorCovPost = np.diag([10., 10., 100., 100.]).astype(np.float32)
                self.initialized = True
            else:
                self.kf.correct(np.array([[x],[y]], np.float32))
            self.lost_frames = 0
            pred = self.kf.predict()
            self.last_pos = detection
            self.trajectory.append(detection)
            return detection, (float(pred[2][0]), float(pred[3][0]))
        else:
            self.lost_frames += 1
            if not self.initialized or self.lost_frames > 8:
                self.initialized = False
                self.last_pos = None
                return None
            pred = self.kf.predict()
            pos  = (int(pred[0][0]), int(pred[1][0]))
            self.last_pos = pos
            return pos, (float(pred[2][0]), float(pred[3][0]))


# ═══════════════════════════════════════════
# 好球区域检测器
# ═══════════════════════════════════════════
class ZoneJudge:
    """
    好球区域检测：摄像头在右侧网柱，球越网 = 从区域左边界进入、右边界飞出。

    防误报机制：
      1. 球必须在左侧连续存在 MIN_LEFT_FRAMES 帧才算"来自左侧"（过滤单帧树影跳变）
      2. 进入时 x 方向速度必须 >= MIN_ENTRY_VX px/帧（过滤缓慢漂移的影子）
      3. 右侧来球（球先在 x>x2 出现）会重置左侧计数，不会被误判为从左进入
    """
    COOLDOWN        = 30    # 判定后冷却帧数
    MAX_TRACKED     = 25    # 区域内追踪超时复位
    MIN_LEFT_FRAMES = 3     # 进入前球必须在左侧连续存在的最少帧数
    MIN_ENTRY_VX    = 4.0   # 进入时最小向右速度（px/处理帧），过滤慢漂移

    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.state       = 'idle'
        self.cooldown    = 0
        self.tracked     = 0
        self.left_frames = 0     # 球在左侧（x < x1）连续帧数
        self.pos_hist    = deque(maxlen=5)   # 最近位置，用于估算速度

    def update(self, ball_pos) -> str | None:
        if self.cooldown > 0:
            self.cooldown -= 1
            if ball_pos is not None:
                self.pos_hist.append((int(ball_pos[0]), int(ball_pos[1])))
            return None

        if ball_pos is None:
            if self.state == 'tracking':
                self.tracked += 1
                if self.tracked > self.MAX_TRACKED:
                    self._reset()
            # 短暂漏检不重置 left_frames（保留连续性）
            return None

        x, y = int(ball_pos[0]), int(ball_pos[1])
        self.pos_hist.append((x, y))
        in_y = self.y1 <= y <= self.y2

        if self.state == 'idle':
            # ── 维护左侧连续帧计数 ──
            if x < self.x1:
                self.left_frames += 1
            elif x > self.x2:
                # 球在区域右侧（来球方向），清零左侧计数
                self.left_frames = 0
            # 注意：x 在 [x1, x2] 时不清零（球正越过边界），允许自然穿越

            # ── 进入判定 ──
            if in_y and x >= self.x1 and self.left_frames >= self.MIN_LEFT_FRAMES:
                vx = self._entry_vx()
                if vx >= self.MIN_ENTRY_VX:
                    self.state       = 'tracking'
                    self.tracked     = 0
                    self.left_frames = 0
                # 速度不达标（慢漂移）：不复位 left_frames，等下一帧再判

        elif self.state == 'tracking':
            self.tracked += 1
            if self.tracked > self.MAX_TRACKED:
                self._reset()
            elif x >= self.x2:          # 从右侧飞出 → 好球
                self._reset()
                self.cooldown = self.COOLDOWN
                return 'great_shot'
            elif y > self.y2:           # 从下侧落入 → 下网
                self._reset()
                self.cooldown = self.COOLDOWN
                return 'net'
            elif x < self.x1 - 8:      # 退回左侧 → 复位
                self._reset()

        return None

    def _entry_vx(self) -> float:
        """用最近 pos_hist 估算 x 方向速度（px/帧）"""
        hist = list(self.pos_hist)
        if len(hist) < 2:
            return 0.0
        dx = hist[-1][0] - hist[0][0]
        dt = len(hist) - 1
        return dx / dt if dt > 0 else 0.0

    def _reset(self):
        self.state       = 'idle'
        self.tracked     = 0
        self.left_frames = 0


# ═══════════════════════════════════════════
# 自动标定：检测球网白带，计算好球区
# ═══════════════════════════════════════════

def _cluster_vote(values: list, tolerance: int = 4) -> int:
    """对候选值聚类投票，取最大簇中位数。"""
    values = sorted(values)
    clusters, cur = [], [values[0]]
    for v in values[1:]:
        if v - cur[-1] <= tolerance:
            cur.append(v)
        else:
            clusters.append(cur)
            cur = [v]
    clusters.append(cur)
    best = max(clusters, key=len)
    return int(np.median(best))


def _sample_court_color(cap, pw: int, ph: int) -> np.ndarray | None:
    """
    采样球场地面颜色（HSV 中位数）。
    取右下角区域（靠近摄像头的球场，颜色最可靠），用前 10 帧投票。
    """
    samples = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        small  = cv2.resize(frame, (pw, ph))
        # 右下角 30%×35% 区域（球场地面，近摄像头）
        patch  = small[int(ph * 0.70):, int(pw * 0.65):]
        hsv    = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        samples.append([np.median(hsv[:,:,0]),
                         np.median(hsv[:,:,1]),
                         np.median(hsv[:,:,2])])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not samples:
        return None
    return np.median(np.array(samples), axis=0)   # [H, S, V]


def _court_top_y(small, court_hsv: np.ndarray, roi_x: int) -> int | None:
    """
    在右侧 ROI 中，找球场地面颜色的最高行（= 球场边界，即球网所在的 Y 位置）。
    护栏在球场之外（更高位），因此只要找到球场的上沿即可锁定球网范围。
    """
    pw = small.shape[1]
    h, s, v = float(court_hsv[0]), float(court_hsv[1]), float(court_hsv[2])
    lo = np.array([max(0,   h - 22), max(0,   s - 60), max(0,   v - 65)], np.uint8)
    hi = np.array([min(179, h + 22), min(255, s + 60), min(255, v + 65)], np.uint8)

    hsv  = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lo, hi)

    # 每行在右侧 ROI 中有多少球场色像素
    row_cnt  = (mask[:, roi_x:] > 0).sum(axis=1)
    min_cols = int((pw - roi_x) * 0.15)    # 至少 15% 的列有球场色
    rows     = np.where(row_cnt >= min_cols)[0]

    # 要求至少 5 行连续的球场色，防止噪点
    if len(rows) < 5:
        return None
    return int(rows.min())


def detect_zone_auto(cap, n_frames: int = CALIB_FRAMES) -> tuple[int, int, int, int] | None:
    """
    三步法定位球网 → 确定好球区 (x1, y1, x2, y2)。

    Step 1  采样球场地面颜色（自适应，蓝/绿/红土均可）
    Step 2  在每帧找球场地面的上边界 court_top_y
              → 球网就在 court_top_y 正上方，护栏/建筑在更高处，天然被排除
    Step 3  在 [court_top_y-25, court_top_y+15] 的窄窗口内，
              找"白色网带 + 正下方深色网眼"的行 → net_top_y
              找该行最左"白+暗"的列 → net_x
    多帧投票取众数，确保稳定。
    """
    pw, ph = PROCESS_WIDTH, PROCESS_HEIGHT
    roi_x  = int(pw * NET_ROI_X_RATIO)
    x_srch = int(pw * NET_X_SEARCH_RATIO)
    d_off  = NET_DARK_OFFSET
    d_win  = NET_DARK_WINDOW

    # Step 1：采样球场颜色
    court_hsv = _sample_court_color(cap, pw, ph)
    if court_hsv is not None:
        print(f"[Calib] 球场颜色 H={court_hsv[0]:.0f} S={court_hsv[1]:.0f} V={court_hsv[2]:.0f}")
    else:
        print("[Calib] 球场颜色采样失败，将使用全范围搜索")

    net_y_cands = []
    net_x_cands = []

    saved = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        small   = cv2.resize(frame, (pw, ph))
        gray    = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        bright  = blurred > NET_WHITE_THRESH
        dark    = blurred < NET_DARK_THRESH

        # Step 2：找球场上边界，锁定白带搜索窗口
        if court_hsv is not None:
            cty = _court_top_y(small, court_hsv, roi_x)
        else:
            cty = None

        if cty is not None:
            # 球网白带在球场上边界正上方：向上留 25px 余量，向下留 15px 容错
            y_lo = max(int(ph * 0.10), cty - 25)
            y_hi = min(int(ph * 0.90), cty + 15)
        else:
            # 回退到全范围
            y_lo = int(ph * NET_ROI_Y_START)
            y_hi = int(ph * NET_ROI_Y_END)

        # Step 3a：在窗口内找"白×暗"最高分行
        best_y, best_score = -1, 0
        w_min = int((pw - roi_x) * NET_MIN_WHITE_COL)
        d_min = (pw - roi_x) * d_win * NET_DARK_MIN_RATIO

        for y in range(y_lo, min(y_hi, ph - d_off - d_win)):
            w = int(bright[y, roi_x:].sum())
            if w < w_min:
                continue
            dy1, dy2 = y + d_off, min(ph, y + d_off + d_win)
            d = int(dark[dy1:dy2, roi_x:].sum())
            if d < d_min:
                continue
            score = w * d
            if score > best_score:
                best_score, best_y = score, y

        if best_y < 0:
            continue
        net_y_cands.append(best_y)

        # Step 3b：找最左"白 + 下方暗"列
        dy1 = min(ph - 1, best_y + d_off)
        dy2 = min(ph,     best_y + d_off + d_win)
        w_cols = bright[max(0, best_y - 3): best_y + 5, x_srch:].sum(axis=0)
        d_cols = (dark[dy1:dy2, x_srch:].sum(axis=0)
                  if dy2 > dy1 else np.zeros(pw - x_srch, dtype=np.int32))
        col_d_min = (dy2 - dy1) * NET_DARK_MIN_RATIO
        valid = (w_cols >= 2) & (d_cols >= col_d_min)
        idxs  = np.where(valid)[0]
        if len(idxs) > 0:
            net_x_cands.append(x_srch + int(idxs.min()))

    cap.set(cv2.CAP_PROP_POS_FRAMES, saved)

    min_frames = max(5, n_frames // 6)
    if len(net_y_cands) < min_frames:
        print(f"[Calib] 网带检测帧数不足（Y:{len(net_y_cands)} / {n_frames}）")
        return None

    net_top_y = _cluster_vote(net_y_cands, tolerance=5)

    if len(net_x_cands) >= min_frames:
        net_x = _cluster_vote(net_x_cands, tolerance=6)
        print(f"[Calib] 网顶 Y={net_top_y}  网带左沿 X={net_x}  "
              f"（Y帧:{len(net_y_cands)}  X帧:{len(net_x_cands)}）")
    else:
        net_x = roi_x
        print(f"[Calib] X帧不足（{len(net_x_cands)}），回退 X={net_x}  网顶 Y={net_top_y}")

    zone_x1 = net_x
    zone_y1 = int(ph * ZONE_Y_TOP_MARGIN)
    zone_x2 = min(pw - 2, zone_x1 + ZONE_WIDTH)
    zone_y2 = net_top_y

    return zone_x1, zone_y1, zone_x2, zone_y2


def save_debug_frame(cap, zone: tuple[int, int, int, int], path: str = DEBUG_FRAME_PATH):
    """
    将第一帧 + 好球区标注保存为图片，供人工核验自动检测效果。
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not ret:
        return

    disp = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    sx   = DISPLAY_WIDTH  / PROCESS_WIDTH
    sy   = DISPLAY_HEIGHT / PROCESS_HEIGHT
    zx1, zy1, zx2, zy2 = zone
    dx1, dy1 = int(zx1*sx), int(zy1*sy)
    dx2, dy2 = int(zx2*sx), int(zy2*sy)

    # 半透明绿色填充
    ov = disp.copy()
    cv2.rectangle(ov, (dx1, dy1), (dx2, dy2), (0, 255, 80), -1)
    cv2.addWeighted(ov, 0.20, disp, 0.80, 0, disp)

    # 区域边框
    cv2.rectangle(disp, (dx1, dy1), (dx2, dy2), (0, 255, 80), 2)
    # 进入线（蓝色）
    cv2.line(disp, (dx1, dy1), (dx1, dy2), (80, 160, 255), 2)
    cv2.putText(disp, "ENTER", (dx1 + 4, dy1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 160, 255), 1)
    # 飞出线（绿色）
    cv2.line(disp, (dx2, dy1), (dx2, dy2), (0, 255, 80), 2)
    cv2.putText(disp, "EXIT", (max(dx2 - 55, 0), dy1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 1)
    # 穿越箭头
    cy = (dy1 + dy2) // 2
    if dx2 > dx1 + 20:
        cv2.arrowedLine(disp, (dx1 + 8, cy), (dx2 - 8, cy),
                        (255, 255, 255), 2, tipLength=0.3)
    # 标签
    cv2.putText(disp, "AUTO DETECTED ZONE",
                (dx1 + 4, max(dy1 - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 80), 2)
    cv2.putText(disp,
                f"net_top_y={zy2}  zone=({zx1},{zy1})-({zx2},{zy2})  proc {PROCESS_WIDTH}x{PROCESS_HEIGHT}",
                (10, disp.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    cv2.imwrite(path, disp)
    print(f"[Calib] 标注帧已保存 → {path}  （核验区域是否正确）")


# ═══════════════════════════════════════════
# HSV 网球检测
# ═══════════════════════════════════════════
def detect_ball_hsv(frame):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    mask = cv2.erode(mask,  None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    best, best_score = None, 0
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(c)
        if area < 10 or area > 2000:
            continue
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        circ = 4 * 3.14159 * area / (perim * perim)
        if circ < 0.5:
            continue
        if circ * area > best_score:
            best_score = circ * area
            best = c
    if best is None:
        return None
    M = cv2.moments(best)
    if M["m00"] == 0:
        return None
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


# ═══════════════════════════════════════════
# Windows TTS（同步）
# ═══════════════════════════════════════════
def speak_windows(text):
    ps_cmd = (f'Add-Type -AssemblyName System.Speech; '
              f'$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; '
              f'$s.Rate=0; $s.Volume=100; $s.Speak("{text}")')
    subprocess.run(
        ["/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
         "-Command", ps_cmd],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5
    )


# ═══════════════════════════════════════════
# 画面绘制
# ═══════════════════════════════════════════
def draw_frame(display_frame, zone_disp: tuple,
               tracker, flash_type, flash_remaining,
               good_shots, net_shots, frame_idx, total_frames, paused):
    h, w = display_frame.shape[:2]
    zx1, zy1, zx2, zy2 = zone_disp
    sx = DISPLAY_WIDTH  / PROCESS_WIDTH
    sy = DISPLAY_HEIGHT / PROCESS_HEIGHT

    # ── 好球区域 ──
    ov = display_frame.copy()
    cv2.rectangle(ov, (zx1, zy1), (zx2, zy2), (0, 255, 80), -1)
    cv2.addWeighted(ov, 0.12, display_frame, 0.88, 0, display_frame)
    cv2.rectangle(display_frame, (zx1, zy1), (zx2, zy2), (0, 255, 80), 2)
    cv2.line(display_frame, (zx1, zy1), (zx1, zy2), (80, 160, 255), 2)   # 进入线
    cy = (zy1 + zy2) // 2
    if zx2 > zx1 + 20:
        cv2.arrowedLine(display_frame, (zx1 + 8, cy), (zx2 - 8, cy),
                        (200, 255, 200), 1, tipLength=0.25)
    cv2.putText(display_frame, "GOOD SHOT ZONE",
                (zx1 + 4, max(zy1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 80), 1)

    # ── 球轨迹 ──
    trail = list(tracker.trajectory)
    for i, pos in enumerate(trail):
        alpha  = (i + 1) / len(trail)
        radius = int(3 + alpha * 6)
        color  = (int(255 * alpha), int(120 * alpha), 0)
        cv2.circle(display_frame,
                   (int(pos[0] * sx), int(pos[1] * sy)), radius, color, -1)

    # ── 当前球位置 ──
    if tracker.last_pos is not None:
        bx = int(tracker.last_pos[0] * sx)
        by = int(tracker.last_pos[1] * sy)
        cv2.circle(display_frame, (bx, by), 10, (0, 255, 0), 2)
        cv2.circle(display_frame, (bx, by),  3, (0, 255, 0), -1)

    # ── 好球 / 下网 大字幕 ──
    if flash_remaining > 0:
        if flash_type == 'great_shot':
            text, fg, fs, th = "GOOD SHOT!", (0, 255, 100), 2.2, 4
        else:
            text, fg, fs, th = "NET",        (50, 100, 255), 2.0, 3
        ov = display_frame.copy()
        cv2.rectangle(ov, (w//2-200, h//2-55), (w//2+200, h//2+45), (0,0,0), -1)
        cv2.addWeighted(ov, 0.6, display_frame, 0.4, 0, display_frame)
        (tw, tth), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        cv2.putText(display_frame, text,
                    (w//2 - tw//2, h//2 + tth//2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, fg, th)

    # ── 状态栏 ──
    ov = display_frame.copy()
    cv2.rectangle(ov, (0, h-40), (w, h), (0,0,0), -1)
    cv2.addWeighted(ov, 0.5, display_frame, 0.5, 0, display_frame)
    status = f" GOOD:{good_shots}  NET:{net_shots}  Frame:{frame_idx}/{total_frames}"
    if paused:
        status += "  [PAUSED]"
    cv2.putText(display_frame, status, (10, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return display_frame


# ═══════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════
def main():
    video_path = VIDEO_PATH
    if not os.path.exists(video_path):
        print(f"[ERROR] 视频不存在: {video_path}")
        sys.exit(1)

    print("=" * 55)
    print("  Tennis Net Call — Auto Zone Detection (WSL2+WSLg)")
    print("=" * 55)
    print(f"[Video] {video_path}")

    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    print(f"[Info]  {total_frames} frames  {fps:.1f} FPS  {total_frames/fps/60:.1f} min\n")

    # ── 自动标定好球区 ──
    print(f"[Calib] 扫描前 {CALIB_FRAMES} 帧检测白色网带...")
    zone = detect_zone_auto(cap, CALIB_FRAMES)

    if zone is not None:
        zx1, zy1, zx2, zy2 = zone
        print(f"[Calib] 检测成功：网顶 Y={zy2}  好球区 ({zx1},{zy1})-({zx2},{zy2})")
    else:
        zone = DEFAULT_ZONE
        zx1, zy1, zx2, zy2 = zone
        print(f"[Calib] 检测失败，使用默认区域 ({zx1},{zy1})-({zx2},{zy2})")

    save_debug_frame(cap, zone, DEBUG_FRAME_PATH)

    sx = DISPLAY_WIDTH  / PROCESS_WIDTH
    sy = DISPLAY_HEIGHT / PROCESS_HEIGHT
    zone_disp = (int(zx1*sx), int(zy1*sy), int(zx2*sx), int(zy2*sy))

    tracker = BallTracker()
    judge   = ZoneJudge(zx1, zy1, zx2, zy2)

    frame_idx       = 0
    proc_idx        = 0
    good_shots      = 0
    net_shots       = 0
    flash_type      = None
    flash_remaining = 0
    paused          = False

    cv2.namedWindow("Tennis Net Call | q=quit SPACE=pause", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tennis Net Call | q=quit SPACE=pause", DISPLAY_WIDTH, DISPLAY_HEIGHT)

    writer = None
    if EXPORT_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(EXPORT_VIDEO, fourcc,
                                 fps / (SKIP_FRAMES + 1),
                                 (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        print(f"[Export] → {EXPORT_VIDEO}")

    print("[System] Running... q/ESC=quit  SPACE=pause\n")

    start_time = time.time()
    try:
        while True:
            if not paused:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                frame_idx += 1
                display = cv2.resize(frame_bgr, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

                # 跳帧：只显示，不判定
                if frame_idx % (SKIP_FRAMES + 1) != 0:
                    draw_frame(display, zone_disp, tracker,
                               flash_type, flash_remaining,
                               good_shots, net_shots, frame_idx, total_frames, paused)
                    cv2.imshow("Tennis Net Call | q=quit SPACE=pause", display)
                    if writer: writer.write(display)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27): raise KeyboardInterrupt
                    elif key == ord(' '): paused = not paused
                    continue

                proc_idx += 1
                frame    = cv2.resize(frame_bgr, (PROCESS_WIDTH, PROCESS_HEIGHT))

                # 球检测 + Kalman
                ball_det = detect_ball_hsv(frame)
                tracked  = tracker.update(ball_det)

                # 优先实际检测，Kalman 最多外推 3 帧
                if ball_det is not None:
                    judge_pos = ball_det
                elif tracked is not None and tracker.lost_frames <= 3:
                    judge_pos = tracked[0]
                else:
                    judge_pos = None

                result = judge.update(judge_pos)

                if result == 'great_shot':
                    good_shots += 1
                    flash_type, flash_remaining = 'great_shot', FLASH_FRAMES_GOOD
                    pos_s = f"({ball_det[0]},{ball_det[1]})" if ball_det else "?"
                    print(f"  [{time.strftime('%H:%M:%S')}] GOOD SHOT! (#{good_shots}) ball={pos_s}", flush=True)
                    osd = draw_frame(display.copy(), zone_disp, tracker,
                                     flash_type, flash_remaining,
                                     good_shots, net_shots, frame_idx, total_frames, paused)
                    cv2.imshow("Tennis Net Call | q=quit SPACE=pause", osd)
                    if writer: writer.write(osd)
                    cv2.waitKey(1)
                    speak_windows("好球")

                elif result == 'net':
                    net_shots += 1
                    flash_type, flash_remaining = 'net', FLASH_FRAMES_NET
                    print(f"  [{time.strftime('%H:%M:%S')}] NET (#{net_shots})", flush=True)

                if flash_remaining > 0:
                    flash_remaining -= 1
                    if flash_remaining == 0:
                        flash_type = None

                draw_frame(display, zone_disp, tracker,
                           flash_type, flash_remaining,
                           good_shots, net_shots, frame_idx, total_frames, paused)

            cv2.imshow("Tennis Net Call | q=quit SPACE=pause", display)
            if writer: writer.write(display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27): raise KeyboardInterrupt
            elif key == ord(' '): paused = not paused

    except KeyboardInterrupt:
        print("\n[Interrupted]")
    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"[Export] Saved → {EXPORT_VIDEO}")
        cv2.destroyAllWindows()
        elapsed = time.time() - start_time
        print(f"\n{'='*55}")
        print(f"  GOOD SHOTS : {good_shots}")
        print(f"  NET        : {net_shots}")
        print(f"  Frames     : {proc_idx}  Elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
