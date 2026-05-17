#!/usr/bin/env python3
"""
网球过网检测 — 通用测试版
==========================
不需要网球、不需要球场、不需要摄像头。
任意物体从"网线"上方飞过 → 播报"好球"。

用法：
  # 模拟模式（画面里自动生成移动圆球）
  python3 test_demo.py --sim

  # 摄像头模式（Linux）
  python3 test_demo.py --camera 0 --net-line 200

  # TCP流模式（从Windows ffmpeg推流过来）
  python3 test_demo.py --source tcp://127.0.0.1:9999 --net-line 200
"""

import sys
import time
import argparse
import threading
import socket
import struct
import io

import cv2
import numpy as np
import pygame


# ═══════════════════════════════════════════
#  运动物体检测器（替代 HSV 网球检测）
# ═══════════════════════════════════════════

class MotionTracker:
    """MOG2 背景减除 + 轮廓追踪任意运动物体"""

    def __init__(self, history=50, var_threshold=40):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )
        self.trail = []          # 最近 N 帧的质心轨迹
        self.positions = []      # 平滑后的位置
        self.max_trail = 30
        self.last_centroid = None
        self.lost_frames = 0

    def detect(self, frame):
        """
        检测画面中最大的运动物体，返回 (cx, cy, area) 或 None。
        """
        fg_mask = self.bg_subtractor.apply(frame)

        # 膨胀 + 腐蚀 去噪
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = 200  # 最小面积阈值

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < best_area:
                continue

            # 过滤太大（可能是整个画面变化）
            h, w = frame.shape[:2]
            if area > (w * h * 0.3):
                continue

            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if area > best_area:
                    best_area = area
                    best = (cx, cy, int(np.sqrt(area)))

        if best:
            self.last_centroid = (best[0], best[1])
            self.trail.append(self.last_centroid)
            if len(self.trail) > self.max_trail:
                self.trail.pop(0)

            # 平滑
            self.positions.append(self.last_centroid)
            if len(self.positions) > 10:
                self.positions.pop(0)
            self.lost_frames = 0
        else:
            self.lost_frames += 1
            if self.lost_frames > 30:
                self.positions.clear()
                self.trail.clear()

        return best

    def get_velocity(self):
        """像素/帧速度"""
        if len(self.positions) < 3:
            return 0, 0
        p1 = self.positions[-3]
        p2 = self.positions[-1]
        return p2[0] - p1[0], p2[1] - p1[1]

    def get_position(self):
        if self.positions:
            return self.positions[-1]
        return self.last_centroid

    def draw(self, frame):
        """绘制轨迹"""
        for i, pos in enumerate(self.trail):
            alpha = (i + 1) / len(self.trail) if self.trail else 0
            color = (0, int(255 * alpha), int(255 * (1 - alpha)))
            cv2.circle(frame, pos, 3, color, -1)

        pos = self.get_position()
        if pos:
            cv2.circle(frame, pos, 8, (0, 255, 0), 2)
        return frame


# ═══════════════════════════════════════════
#  过线判定器
# ═══════════════════════════════════════════

class LineJudge:
    """判断物体是否从参考线上方飞过"""

    def __init__(self, line_y: int):
        self.line_y = line_y
        self.prev_y = None
        self.crossing_up = False
        self.shot_in_progress = False
        self.last_result_time = 0
        self.cooldown = 1.5

    def update(self, pos, velocity):
        """返回 'great_shot' | 'net' | 'waiting'"""
        if pos is None:
            return self._reset()

        _, y = pos
        _, vy = velocity

        now = time.time()
        cooled = (now - self.last_result_time) > self.cooldown

        if not self.shot_in_progress:
            if vy < -3 and y > self.line_y and cooled:
                self.shot_in_progress = True
                self.crossing_up = True
                self.prev_y = y
            return 'waiting'

        if self.crossing_up and self.prev_y is not None:
            if self.prev_y > self.line_y >= y:
                return self._finalize('great_shot')

            if vy >= 0 and y > self.line_y:
                return self._finalize('net')

        self.prev_y = y

        # 超时保护 3 秒
        if self.shot_in_progress and (now - self.last_result_time > 3):
            return self._reset()

        return 'waiting'

    def _finalize(self, result):
        self.shot_in_progress = False
        self.crossing_up = False
        self.prev_y = None
        self.last_result_time = time.time()
        return result

    def _reset(self):
        self.shot_in_progress = False
        self.crossing_up = False
        self.prev_y = None
        return 'waiting'


# ═══════════════════════════════════════════
#  音频播放器
# ═══════════════════════════════════════════

class Beeper:
    """pygame 合成提示音，无音频设备时降级为纯控制台输出"""

    def __init__(self):
        self.initialized = False
        self.no_audio = False
        self.sample_rate = 22050

    def init(self):
        if self.initialized:
            return
        try:
            import os
            # WSL2 无音频设备时跳过
            if not os.path.exists('/dev/snd') and not os.path.exists('/dev/dsp'):
                print("[语音] 无音频设备，仅控制台输出")
                self.no_audio = True
                self.initialized = True
                return
            pygame.mixer.init(frequency=self.sample_rate, size=-16,
                              channels=1, buffer=512)
            self.initialized = True
        except Exception as e:
            print(f"[语音] 初始化失败: {e}，仅控制台输出")
            self.no_audio = True
            self.initialized = True

    def play(self, result_type):
        if not self.initialized:
            self.init()
        if self.no_audio:
            return  # 无音频设备，已经在控制台打印了
        duration = 0.25
        duration = 0.25
        if result_type == 'great_shot':
            freqs = [523, 659, 784, 1047]  # C E G C → "好球！"
            dur_each = duration / len(freqs)
        elif result_type == 'net':
            freqs = [200, 150]  # 低沉 → "下网…"
            dur_each = duration / len(freqs)
        else:
            return

        t = np.linspace(0, dur_each, int(self.sample_rate * dur_each), False)

        for freq in freqs:
            wave = np.sin(2 * np.pi * freq * t) * 0.4
            wave = (wave * 32767).astype(np.int16)
            sound = pygame.sndarray.make_sound(wave)
            sound.play()
            pygame.time.wait(int(dur_each * 1000))

    def cleanup(self):
        pygame.mixer.quit()


# ═══════════════════════════════════════════
#  模拟模式：自动生成圆球上下弹跳
# ═══════════════════════════════════════════

class SimBall:
    """画面中模拟一个黄绿色圆球，在网线附近弹跳"""

    def __init__(self, w, h, net_y):
        self.w = w
        self.h = h
        self.net_y = net_y
        self.x = w // 2
        self.y = h - 100
        self.vy = 0
        self.radius = 15
        self.launching = False
        self.launch_timer = 0

    def update(self):
        """返回 (x, y, r) 当前球的位置"""
        if not self.launching:
            self.launch_timer += 1
            if self.launch_timer > 20:  # 20帧 ≈ 1秒
                self.launch()
                self.launch_timer = 0
            else:
                # 等待期间保持不动，防止重力拉出屏幕
                return int(self.x), int(self.y), self.radius
        
        # 发射后的物理模拟
        self.y += self.vy
        self.vy += 0.8  # 重力

        # 球掉到画面底部 → 重置
        if self.y > self.h + 50:
            self.reset()

        # 球飞出顶部 → 继续飞
        if self.y < -50:
            self.reset()

        return int(self.x), int(self.y), self.radius

    def launch(self):
        """发射球：从底部往上飞"""
        self.x = np.random.randint(self.w // 4, 3 * self.w // 4)
        self.y = self.h - 50
        # v² = 2*g*d, d = net_y - launch_y ≈ 430-160 = 270
        # v_min = sqrt(2*0.8*270) ≈ 21, 加余量到 23-28
        self.vy = np.random.randint(-28, -22)
        self.launching = True

    def reset(self):
        self.x = np.random.randint(self.w // 4, 3 * self.w // 4)
        self.y = self.h - 50
        self.vy = 0
        self.launching = False
        self.launch_timer = 0

    def draw(self, frame):
        """在画面上画球"""
        x, y, r = self.update()
        cv2.circle(frame, (x, y), r, (0, 240, 255), -1)
        cv2.circle(frame, (x, y), r, (0, 180, 200), 2)
        return frame, (x, y, r)


# ═══════════════════════════════════════════
#  主程序
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="网球过网检测 — 测试版")
    parser.add_argument("--sim", action="store_true",
                        help="模拟模式：画面中自动生成弹跳球")
    parser.add_argument("--camera", type=int, default=-1,
                        help="摄像头设备号（Linux: /dev/videoN）")
    parser.add_argument("--source", type=str, default="",
                        help="TCP视频流 tcp://host:port")
    parser.add_argument("--net-line", type=int, default=0,
                        help="网线Y坐标（0=画面1/3处）")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--debug", action="store_true",
                        help="显示调试画面（需要图形界面）")
    parser.add_argument("--headless", action="store_true",
                        help="无头模式（仅控制台输出+音频）")
    args = parser.parse_args()

    # WSL2 默认无头模式
    has_display = False
    try:
        has_display = (cv2.imshow.__name__ is not None) and \
                       (cv2.waitKey.__name__ is not None)
    except Exception:
        pass

    if args.debug and not has_display:
        # 尝试检查 DISPLAY 环境变量
        import os
        if not os.environ.get("DISPLAY"):
            print("[警告] 无图形界面，自动切换到无头模式")
            print("[提示] 音频播报正常工作，按 Ctrl+C 退出\n")
            args.headless = True
            args.debug = False

    # ── 确定输入源 ──
    if args.sim:
        mode = "sim"
        source = None
        print("[模式] 模拟模式 — 画面自动生成弹跳球")
    elif args.source:
        mode = "tcp"
        source = args.source
        print(f"[模式] TCP流 — {args.source}")
    elif args.camera >= 0:
        mode = "camera"
        source = args.camera
        print(f"[模式] 摄像头 — /dev/video{args.camera}")
    else:
        # 默认：试摄像头，不行就模拟
        mode = "auto"
        source = 0
        print("[模式] 自动检测 — 优先摄像头，失败转模拟")

    # ── 打开视频源 ──
    cap = None
    sim_ball = None
    width, height = args.width, args.height

    if mode in ("camera", "auto"):
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not cap.isOpened() and mode == "auto":
            print("[摄像头] 不可用，切换到模拟模式")
            mode = "sim"
            cap = None
        elif not cap.isOpened():
            print("[摄像头] ❌ 无法打开")
            sys.exit(1)
        else:
            print("[摄像头] ✅ 就绪")

    if mode == "tcp":
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[TCP] ❌ 无法连接 {args.source}")
            print("[提示] 先在 Windows 端启动推流：")
            print('  ffmpeg -f dshow -i video="USB Camera" -f mjpeg -q:v 3 tcp://WSL_IP:9999?listen')
            sys.exit(1)
        print("[TCP] ✅ 已连接")

    # ── 网线位置 ──
    if args.net_line > 0:
        net_line_y = args.net_line
    else:
        net_line_y = height // 3  # 默认画面 1/3 处
    print(f"[标定] 网线 Y={net_line_y} (画面 {net_line_y/height:.0%} 处)")

    # ── 初始化模块 ──
    tracker = MotionTracker()
    judge = LineJudge(net_line_y)
    beeper = Beeper()
    beeper.init()

    if mode == "sim":
        sim_ball = SimBall(width, height, net_line_y)

    print("[系统] 实时检测中，物体从下往上飞过黄线 → 好球！")
    print("[系统] 按 Q 或 Ctrl+C 退出\n")

    # ── 主循环 ──
    result_display = "..."
    result_timer = 0

    try:
        while True:
            if mode == "sim":
                # 生成模拟帧
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame[:, :] = (50, 50, 50)  # 深灰背景
                # 画网线
                cv2.line(frame, (0, net_line_y), (width, net_line_y),
                         (0, 255, 255), 2)
                cv2.putText(frame, "NET LINE", (10, net_line_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                # 画球
                frame, ball_data = sim_ball.draw(frame)
                ball_circle = ball_data
                ball_pos = (ball_data[0], ball_data[1])
                ball_vel = (0, sim_ball.vy)
                has_swing = sim_ball.launching and sim_ball.vy < -5
            else:
                ret, frame = cap.read()
                if not ret:
                    print("[视频] 读取帧失败")
                    time.sleep(0.1)
                    continue

                # 运动检测
                ball_circle = tracker.detect(frame)
                ball_pos = tracker.get_position()
                ball_vel = tracker.get_velocity()
                has_swing = (ball_vel[1] < -5)

            # ── 好球判定 ──
            result = judge.update(ball_pos, ball_vel)

            if result == 'great_shot':
                print("  🎾 好球!!!", flush=True)
                beeper.play('great_shot')
                result_display = "GOOD!"
                result_timer = 30
            elif result == 'net':
                print("  ⬇ 下网", flush=True)
                beeper.play('net')
                result_display = "NET..."
                result_timer = 30

            # ── 调试画面（仅在有图形界面时显示）──
            show_debug = (args.debug or mode == "sim") and not args.headless
            
            if mode == "sim" and show_debug:
                # 模拟模式默认画调试帧（如果有 GUI）
                pass  # frame 已经在上面画好了
            
            if show_debug:
                if mode != "sim":
                    if ball_circle:
                        x, y, a = ball_circle
                        cv2.circle(frame, (x, y), 10, (0, 255, 0), 2)
                    frame = tracker.draw(frame)

                    # 画网线
                    cv2.line(frame, (0, net_line_y), (width, net_line_y),
                             (0, 255, 255), 2)
                    cv2.putText(frame, "NET", (10, net_line_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # 状态文字
                if result_timer > 0:
                    result_timer -= 1
                else:
                    result_display = "..."

                color = (0, 255, 0) if "GOOD" in result_display else \
                        (0, 0, 255) if "NET" in result_display else (200, 200, 200)
                cv2.putText(frame, result_display, (width // 2 - 50, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

                # 操作提示
                cv2.putText(frame, "Q=Quit", (width - 100, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

                cv2.imshow("Tennis Net Call - Test", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                # 空格键手动发射（模拟模式）
                if key == ord(' ') and sim_ball:
                    sim_ball.launch()

            # 模拟模式限帧
            if mode == "sim":
                time.sleep(0.05)  # ~20fps

    except KeyboardInterrupt:
        print("\n[系统] 用户退出")

    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        beeper.cleanup()
        print("[系统] 已关闭")


if __name__ == "__main__":
    main()
