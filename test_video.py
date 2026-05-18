#!/usr/bin/env python3
"""
网球过网检测 — 视频文件测试 (WSL2 + WSLg 显示)
从视频文件读取帧 → HSV网球追踪 → 过网判定 → 画面标注 + Windows端语音同步播报

用法:
  python3 test_video.py [视频路径]
  python3 test_video.py "/mnt/d/新建文件夹 (2)/摄像头测试/test_12m_2k_60.avi"
  
按键:
  q / ESC  退出
  空格     暂停/继续
"""
import sys, os, subprocess, time
os.environ["DISPLAY"] = ":0"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
import numpy as np
import cv2
from collections import deque

# ═══════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/新建文件夹 (2)/摄像头测试/test_12m_2k_60.avi"
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540
PROCESS_WIDTH = 320
PROCESS_HEIGHT = 180
NET_LINE_RATIO = 0.38
SKIP_FRAMES = 2
# 通用网球黄绿色 — 适应不同场地/光线
# H: 20-50 覆盖黄绿到纯绿，不同光照下球色会偏移
# S: 70+ 过滤掉场地低饱和度杂色（灰白线、地面反光）
# V: 45+ 允许阴影下的球被检测到
HSV_LOWER = np.array([20, 70, 45])
HSV_UPPER = np.array([50, 255, 255])
# ── 方向识别 ──
# 摄像头在网柱，朝向一侧半场。球员在哪一侧？
# "bottom": 球员在网线下方（画面下半部），向上击球（球 y 减小穿过网线）
# "top":    球员在网线上方（画面上半部），向下击球（球 y 增大穿过网线）
# 对面的球（从非球员侧飞来）会被直接忽略
PLAYER_SIDE = "bottom"
# ── 水平方向过滤 ──
#   0  = 开机自动标定（推荐，前 5 次过网事件自动确定方向）
#   1  = 手动指定向右（player 在左侧击向右侧对手）
#  -1  = 手动指定向左（player 在右侧击向左侧对手）
SHOT_VX_SIGN = 0
# ── 字幕参数 ──
FLASH_FRAMES_GOOD = 45   # 好球文字停留帧数
FLASH_FRAMES_NET = 30    # 下网文字停留帧数
TRAIL_LENGTH = 15        # 球轨迹长度
EXPORT_VIDEO = "demo/output.avi"       # 导出路径 — None=不导出

# ═══════════════════════════════════════════
#  Kalman 追踪
# ═══════════════════════════════════════════
class BallTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        # 速度初始不确定度设大（100），让 Kalman 在前几帧快速学到真实速度
        # np.eye 的默认值 1.0 太小，会导致打击后速度收敛慢、预测位置滞后
        self.kf.errorCovPost = np.diag([10.0, 10.0, 100.0, 100.0]).astype(np.float32)
        self.initialized = False
        self.last_pos = None
        self.trajectory = deque(maxlen=TRAIL_LENGTH)
        self.lost_frames = 0

    def update(self, detection=None):
        if detection is not None:
            x, y = detection
            if not self.initialized:
                self.kf.statePost = np.array([[x],[y],[0],[0]], np.float32)
                # 每次重新初始化都重置误差协方差
                self.kf.errorCovPost = np.diag([10.0, 10.0, 100.0, 100.0]).astype(np.float32)
                self.initialized = True
            else:
                self.kf.correct(np.array([[x],[y]], np.float32))
            self.lost_frames = 0
            pred = self.kf.predict()
            vel = (pred[2][0], pred[3][0])
            # 用原始检测坐标作为当前位置：消除 Kalman t+1 预测带来的位移延迟
            # 只有在球消失时才用 Kalman 外推
            pos = detection
            self.last_pos = pos
            self.trajectory.append(pos)
            return pos, vel
        else:
            self.lost_frames += 1
            if not self.initialized:
                return None
            if self.lost_frames > 8:
                self.initialized = False
                self.last_pos = None
                return None
            # 球短暂丢失时用 Kalman 外推，但不加入轨迹（避免漂移）
            pred = self.kf.predict()
            pos = (int(pred[0][0]), int(pred[1][0]))
            vel = (pred[2][0], pred[3][0])
            self.last_pos = pos
            return pos, vel

# ═══════════════════════════════════════════
# 好球判定 — 轨迹拟合版
# ═══════════════════════════════════════════
class ShotJudge:
    """
    对最近 FIT_WINDOW 个检测点拟合抛物线，通过曲线预测穿越时刻。
    优势：
      - 容忍 1-3 帧漏检（不要求连续帧）
      - 最多提前 PREDICT_AHEAD 帧预判快速穿越，避免球飞出画面漏报
      - 来球落地弹起保护：检测到来球后短期内要求更高的向上速度
    """
    FIT_WINDOW      = 8     # 拟合用的最近检测点数
    MIN_POINTS       = 4     # 开始拟合的最少点数
    VY_THRESHOLD     = -3.0  # 穿越时向上速度下限（px/frame，处理分辨率下）
    PREDICT_AHEAD    = 6     # 最多提前几帧预判穿越
    INCOMING_WINDOW  = 8     # 来球落地后的保护期（帧）
    INCOMING_STRICT  = 2.0   # 保护期内速度门槛倍数（要求球员主动击球）
    CALIB_SHOTS      = 5     # 自动标定需要收集的过网次数

    def __init__(self, net_line_y, player_side="bottom", shot_vx_sign=0):
        self.net_line_y          = net_line_y
        self.player_side         = player_side
        self.pts                 = deque(maxlen=self.FIT_WINDOW)  # (frame_idx, x, y)
        self.frame_idx           = 0
        self.cooldown            = 0
        self.last_incoming_frame = -999
        # ── 击球方向：手动指定或开机自动标定 ──
        # shot_vx_sign=0 → 前 CALIB_SHOTS 次过网事件自动确定方向
        # shot_vx_sign=1/-1 → 直接使用（手动覆盖，跳过标定）
        self.shot_vx_sign = shot_vx_sign
        self.calibrated   = (shot_vx_sign != 0)
        self._calib_vx    = []   # 标定期间收集的 vx 样本

    def update(self, ball_pos, ball_vel):
        self.frame_idx += 1

        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        if ball_pos is None:
            return None  # 漏检帧：保留历史点，不影响拟合

        x, y = ball_pos[0], ball_pos[1]
        self.pts.append((self.frame_idx, x, y))

        # 检测来球穿越（对侧 → 球员侧），更新保护计时
        if len(self.pts) >= 2:
            prev_y = self.pts[-2][2]
            if self.player_side == "bottom":
                incoming = prev_y <= self.net_line_y and y > self.net_line_y
            else:
                incoming = prev_y >= self.net_line_y and y < self.net_line_y
            if incoming:
                self.last_incoming_frame = self.frame_idx

        if len(self.pts) < self.MIN_POINTS:
            return None

        if self._detect_crossing():
            self.cooldown = 30
            self.pts.clear()
            return 'great_shot'
        return None

    def _detect_crossing(self):
        frames = np.array([p[0] for p in self.pts], dtype=np.float64)
        xs     = np.array([p[1] for p in self.pts], dtype=np.float64)
        ys     = np.array([p[2] for p in self.pts], dtype=np.float64)
        t      = frames - frames[0]   # 相对帧号，数值稳定
        t_now  = t[-1]

        # 1. 至少一半点在球员侧（否则是来球轨迹，直接忽略）
        on_player = (ys > self.net_line_y) if self.player_side == "bottom" \
                    else (ys < self.net_line_y)
        if on_player.sum() < max(2, len(ys) // 2):
            return False

        # 2. 拟合抛物线 y = a*t² + b*t + c
        try:
            a, b, c = np.polyfit(t, ys, 2)
        except Exception:
            return False

        # 3. 当前拟合速度必须朝向对侧
        vy_now    = 2 * a * t_now + b
        threshold = self.VY_THRESHOLD
        # 来球落地保护期内：提高速度门槛，过滤自然弹起（自然弹起 < 2x 门槛）
        if (self.frame_idx - self.last_incoming_frame) < self.INCOMING_WINDOW:
            threshold *= self.INCOMING_STRICT
        if self.player_side == "bottom" and vy_now > threshold:
            return False
        if self.player_side == "top"    and vy_now < -threshold:
            return False

        # 4. 求抛物线与网线的交点 a*t² + b*t + (c - net) = 0
        if abs(a) < 1e-8:
            if abs(b) < 1e-8:
                return False
            t_cross = (self.net_line_y - c) / b
        else:
            disc = b * b - 4 * a * (c - self.net_line_y)
            if disc < 0:
                # 抛物线未触及网线 —— 判断球是否已经越过
                y_now = a * t_now**2 + b * t_now + c
                already = (y_now <= self.net_line_y) if self.player_side == "bottom" \
                          else (y_now >= self.net_line_y)
                if not already:
                    return False
                t_cross = t_now
            else:
                sqrt_d = disc ** 0.5
                t1 = (-b + sqrt_d) / (2 * a)
                t2 = (-b - sqrt_d) / (2 * a)
                # 取距当前帧最近、且在合理预判范围内的穿越时刻
                candidates = [tc for tc in (t1, t2)
                              if t_now - 1 <= tc <= t_now + self.PREDICT_AHEAD]
                if not candidates:
                    return False
                t_cross = min(candidates, key=lambda tc: abs(tc - t_now))

        # 5. 穿越前球必须在球员侧（排除来球高弹误判）
        t_before = max(t_cross - 4, 0.0)
        y_before = a * t_before**2 + b * t_before + c
        if self.player_side == "bottom":
            if y_before <= self.net_line_y:
                return False
        else:
            if y_before >= self.net_line_y:
                return False

        # 6. 水平方向校验（自动标定 + 过滤迎面来球）
        vx_avg = (xs[-1] - xs[0]) / t_now if t_now > 0 else 0.0

        if not self.calibrated:
            # 标定期：收集 vx 样本，暂不过滤（允许误报）
            self._calib_vx.append(vx_avg)
            if len(self._calib_vx) >= self.CALIB_SHOTS:
                median_vx = sorted(self._calib_vx)[len(self._calib_vx) // 2]
                self.shot_vx_sign = 1 if median_vx >= 0 else -1
                self.calibrated = True
                arrow = '→' if self.shot_vx_sign > 0 else '←'
                print(f"[方向标定] 完成：击球方向 {arrow}（采样 {len(self._calib_vx)} 次）", flush=True)
        else:
            # 标定完成：方向明显相反则拒绝（-1.0 容差保留直线球）
            if vx_avg * self.shot_vx_sign < -1.0:
                return False

        return True

# ═══════════════════════════════════════════
# HSV 网球检测
# ═══════════════════════════════════════════
def detect_ball_hsv(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # 按面积排序，取最大的几个候选
    candidates = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    # 筛选：面积适中 + 圆形度 > 0.4（过滤条状/不规则噪声）
    best = None
    for c in candidates:
        area = cv2.contourArea(c)
        if area < 3 or area > 2000:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * 3.14159 * area / (perimeter * perimeter)
        if circularity < 0.4:  # 非圆形，跳过
            continue
        best = c
        break
    if best is None:
        return None
    M = cv2.moments(best)
    if M["m00"] == 0:
        return None
    return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

# ═══════════════════════════════════════════
# Windows TTS (同步)
# ═══════════════════════════════════════════
def speak_windows(text):
    ps_cmd = f'Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Rate=0; $s.Volume=100; $s.Speak("{text}")'
    subprocess.run(
        ["/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe", "-Command", ps_cmd],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5
    )

# ═══════════════════════════════════════════
# 画面绘制
# ═══════════════════════════════════════════
def draw_frame(display_frame, net_line_y_disp, tracker, flash_type, flash_remaining, good_shots, net_shots, frame_idx, total_frames, paused):
    h, w = display_frame.shape[:2]

    # ── 网线 (黄色虚线) ──
    for i in range(0, w, 30):
        cv2.line(display_frame, (i, net_line_y_disp), (i+15, net_line_y_disp), (0, 215, 255), 2)

    # ── 球轨迹 ──
    trail = list(tracker.trajectory)
    scale_x = DISPLAY_WIDTH / PROCESS_WIDTH
    scale_y = DISPLAY_HEIGHT / PROCESS_HEIGHT
    
    # 从旧到新绘制
    for i, pos in enumerate(trail):
        alpha = (i + 1) / len(trail) if trail else 0
        radius = int(3 + alpha * 6)
        color = (int(255 * alpha), int(120 * alpha), 0)  # 蓝色渐变，避免和黄色网线撞色
        dx = int(pos[0] * scale_x)
        dy = int(pos[1] * scale_y)
        cv2.circle(display_frame, (dx, dy), radius, color, -1)

    # ── 当前球位置 (大绿圈) ──
    if tracker.last_pos is not None:
        bx = int(tracker.last_pos[0] * scale_x)
        by = int(tracker.last_pos[1] * scale_y)
        cv2.circle(display_frame, (bx, by), 10, (0, 255, 0), 2)
        cv2.circle(display_frame, (bx, by), 3, (0, 255, 0), -1)

    # ── 好球/下网 大字幕 (英文, OpenCV putText 原生支持) ──
    if flash_remaining > 0:
        if flash_type == 'great_shot':
            text = "GOOD SHOT!"
            fg = (0, 255, 100)
            font_scale = 2.2
            thickness = 4
        else:
            text = "NET"
            fg = (50, 100, 255)
            font_scale = 2.0
            thickness = 3
        
        # 背景半透明条
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (w//2-200, h//2-55), (w//2+200, h//2+45), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
        
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tx = w//2 - tw//2
        ty = h//2 + th//2
        cv2.putText(display_frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, fg, thickness)

    # ── 状态栏 ──
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0, h-40), (w, h), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)
    
    status = f" GOOD: {good_shots}  |  NET: {net_shots}  |  Frame: {frame_idx}/{total_frames}"
    if paused:
        status += "  [PAUSED]"
    cv2.putText(display_frame, status, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    # ── 网线标签 (纯 ASCII) ──
    cv2.putText(display_frame, "--- NET ---", (10, net_line_y_disp - 8), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1)

    return display_frame

# ═══════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════
def main():
    video_path = VIDEO_PATH
    if not os.path.exists(video_path):
        print(f"[ERROR] 视频文件不存在: {video_path}")
        sys.exit(1)

    print("=" * 55)
    print("  Tennis Net Call Detection — Video Test (WSL2+WSLg)")
    print("=" * 55)
    print(f"[Video] {video_path}")
    print(f"[Display] {DISPLAY_WIDTH}x{DISPLAY_HEIGHT} (processing {PROCESS_WIDTH}x{PROCESS_HEIGHT})")
    print(f"[Sample] 1 frame every {SKIP_FRAMES+1}")
    print(f"[Audio] Windows TTS (sync)")
    print(f"[Keys]  q/ESC=quit  SPACE=pause")
    print()

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[Info] {total_frames} frames, {fps:.1f} FPS, {total_frames/fps/60:.1f} min")
    
    net_line_y = int(PROCESS_HEIGHT * NET_LINE_RATIO)
    net_line_y_disp = int(DISPLAY_HEIGHT * NET_LINE_RATIO)
    print(f"[Calib] Net line Y={net_line_y} ({NET_LINE_RATIO*100:.0f}% of frame)")
    print(f"[Side]  Player is on {PLAYER_SIDE} side (opponent balls ignored)")
    print()

    tracker = BallTracker()
    judge = ShotJudge(net_line_y, PLAYER_SIDE, SHOT_VX_SIGN)
    
    frame_idx = 0
    proc_idx = 0
    good_shots = 0
    net_shots = 0
    flash_type = None
    flash_remaining = 0
    paused = False
    
    cv2.namedWindow("Tennis Net Call | q=quit SPACE=pause", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tennis Net Call | q=quit SPACE=pause", DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
    # 导出视频
    writer = None
    if EXPORT_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(EXPORT_VIDEO, fourcc, fps / (SKIP_FRAMES + 1), (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        print(f"[Export] Saving to {EXPORT_VIDEO}")
    
    print("[System] Ball crosses net from player side -> GOOD SHOT!")
    print("[System] Press q or ESC to quit\n")

    start_time = time.time()
    try:
        while True:
            if not paused:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                frame_idx += 1

                # 跳帧
                if frame_idx % (SKIP_FRAMES + 1) != 0:
                    # 仍然显示视频帧（无处理叠加）
                    display = cv2.resize(frame_bgr, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                    cv2.line(display, (0, net_line_y_disp), (DISPLAY_WIDTH, net_line_y_disp), (0, 180, 240), 1)
                    draw_frame(display, net_line_y_disp, tracker, flash_type, flash_remaining, good_shots, net_shots, frame_idx, total_frames, paused)
                    cv2.imshow("Tennis Net Call | q=quit SPACE=pause", display)
                    if writer:
                        writer.write(display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        raise KeyboardInterrupt
                    elif key == ord(' '):
                        paused = not paused
                    continue

                proc_idx += 1

                # 缩放
                frame = cv2.resize(frame_bgr, (PROCESS_WIDTH, PROCESS_HEIGHT))
                display = cv2.resize(frame_bgr, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

                # HSV 检测网球
                ball_det = detect_ball_hsv(frame)

                # Kalman 追踪
                tracked = tracker.update(ball_det)

                # 过网判定
                # 用原始检测坐标而非 Kalman 预测位置，避免高速球打出后位置滞后漏判
                if tracked is not None:
                    pos, vel = tracked
                    judge_pos = ball_det if ball_det is not None else pos
                    result = judge.update(judge_pos, vel)

                    if result == 'great_shot':
                        good_shots += 1
                        flash_type = 'great_shot'
                        flash_remaining = FLASH_FRAMES_GOOD
                        ts = time.strftime("%H:%M:%S")
                        print(f"  [{ts}] GOOD SHOT! (#{good_shots}) pos=({pos[0]},{pos[1]}) vel=({vel[0]:.0f},{vel[1]:.0f}) vx_sign={'→' if vel[0]>0 else '←'}", flush=True)
                        # 先绘帧 → TTS → 继续显示
                        display_with_osd = draw_frame(display.copy(), net_line_y_disp, tracker, flash_type, flash_remaining, good_shots, net_shots, frame_idx, total_frames, paused)
                        cv2.imshow("Tennis Net Call | q=quit SPACE=pause", display_with_osd)
                        if writer:
                            writer.write(display_with_osd)
                        cv2.waitKey(1)
                        speak_windows("好球")

                    elif result == 'net':
                        net_shots += 1
                        flash_type = 'net'
                        flash_remaining = FLASH_FRAMES_NET
                        ts = time.strftime("%H:%M:%S")
                        print(f"  [{ts}] NET (#{net_shots})", flush=True)

                # 递减字幕帧数
                if flash_remaining > 0:
                    flash_remaining -= 1
                    if flash_remaining == 0:
                        flash_type = None

                # 绘制
                display = draw_frame(display, net_line_y_disp, tracker, flash_type, flash_remaining, good_shots, net_shots, frame_idx, total_frames, paused)
            else:
                # 暂停时也显示
                pass

            # 击球方向标定状态（左上角小字）
            if not judge.calibrated:
                remaining = judge.CALIB_SHOTS - len(judge._calib_vx)
                calib_text = f"DIR CAL: {remaining} shots left"
                cv2.putText(display, calib_text, (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
            else:
                arrow = "->" if judge.shot_vx_sign > 0 else "<-"
                cv2.putText(display, f"DIR: {arrow}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1)

            cv2.imshow("Tennis Net Call | q=quit SPACE=pause", display)
            if writer:
                writer.write(display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                raise KeyboardInterrupt
            elif key == ord(' '):
                paused = not paused

    except KeyboardInterrupt:
        print("\n[Interrupted] User stopped")

    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"[Export] Saved to {EXPORT_VIDEO}")
        cv2.destroyAllWindows()
        elapsed = time.time() - start_time
        print(f"\n{'='*55}")
        print(f"  Detection Complete")
        print(f"{'='*55}")
        print(f"  Processed frames: {proc_idx}")
        print(f"  Elapsed: {elapsed:.0f}s")
        print(f"  GOOD SHOTS: {good_shots}")
        print(f"  NET: {net_shots}")

if __name__ == "__main__":
    main()
