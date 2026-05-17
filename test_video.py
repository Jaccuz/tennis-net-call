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
# ── 字幕参数 ──
FLASH_FRAMES_GOOD = 45   # 好球文字停留帧数
FLASH_FRAMES_NET = 30    # 下网文字停留帧数
TRAIL_LENGTH = 15        # 球轨迹长度

# ═══════════════════════════════════════════
#  Kalman 追踪
# ═══════════════════════════════════════════
class BallTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1.0   # 提高响应速度
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1  # 更信任检测结果
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False
        self.last_pos = None
        self.trajectory = deque(maxlen=TRAIL_LENGTH)
        self.lost_frames = 0   # 连续丢失帧数

    def update(self, detection=None):
        if detection is not None:
            x, y = detection
            if not self.initialized:
                self.kf.statePost = np.array([[x],[y],[0],[0]], np.float32)
                self.initialized = True
            else:
                self.kf.correct(np.array([[x],[y]], np.float32))
            self.lost_frames = 0
            pred = self.kf.predict()
            pos = (int(pred[0][0]), int(pred[1][0]))
            vel = (pred[2][0], pred[3][0])
            self.last_pos = pos
            self.trajectory.append(pos)  # 只记录有检测的位置
            return pos, vel
        else:
            self.lost_frames += 1
            if not self.initialized:
                return None
            if self.lost_frames > 8:
                self.initialized = False  # 丢失太久，强制重新初始化
                self.last_pos = None
                return None
            # 预测但不加入轨迹（避免漂移）
            pred = self.kf.predict()
            pos = (int(pred[0][0]), int(pred[1][0]))
            vel = (pred[2][0], pred[3][0])
            self.last_pos = pos
            return pos, vel

# ═══════════════════════════════════════════
# 好球判定（带方向识别）
# ═══════════════════════════════════════════
class ShotJudge:
    def __init__(self, net_line_y, player_side="bottom"):
        self.net_line_y = net_line_y
        self.player_side = player_side  # "bottom" 或 "top"
        self.last_side = None   # 球上一帧在哪侧: "bottom" / "top" / None
        self.cooldown = 0

    def update(self, ball_pos, ball_vel):
        if ball_pos is None:
            return None  # 保留 last_side，检测恢复后方向信息仍在
        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        x, y = ball_pos
        current_side = "bottom" if y > self.net_line_y else "top"

        result = None
        if self.last_side is not None and current_side != self.last_side:
            # 球穿越了网线！
            if self.last_side == self.player_side:
                # 球从球员侧飞来 → 是球员击出的球
                result = 'great_shot'
            # else: 球从对面飞来 → 忽略（对手的球）

        self.last_side = current_side
        if result:
            self.cooldown = 25
        return result

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
    judge = ShotJudge(net_line_y, PLAYER_SIDE)
    
    frame_idx = 0
    proc_idx = 0
    good_shots = 0
    net_shots = 0
    flash_type = None
    flash_remaining = 0
    paused = False
    
    cv2.namedWindow("Tennis Net Call | q=quit SPACE=pause", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tennis Net Call | q=quit SPACE=pause", DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
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
                if tracked is not None:
                    pos, vel = tracked
                    result = judge.update(pos, vel)

                    if result == 'great_shot':
                        good_shots += 1
                        flash_type = 'great_shot'
                        flash_remaining = FLASH_FRAMES_GOOD
                        ts = time.strftime("%H:%M:%S")
                        print(f"  [{ts}] GOOD SHOT! (#{good_shots}) pos=({pos[0]},{pos[1]}) vel=({vel[0]:.0f},{vel[1]:.0f})", flush=True)
                        # 先绘帧 → TTS → 继续显示
                        display_with_osd = draw_frame(display.copy(), net_line_y_disp, tracker, flash_type, flash_remaining, good_shots, net_shots, frame_idx, total_frames, paused)
                        cv2.imshow("Tennis Net Call | q=quit SPACE=pause", display_with_osd)
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
            
            cv2.imshow("Tennis Net Call | q=quit SPACE=pause", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                raise KeyboardInterrupt
            elif key == ord(' '):
                paused = not paused

    except KeyboardInterrupt:
        print("\n[Interrupted] User stopped")

    finally:
        cap.release()
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
