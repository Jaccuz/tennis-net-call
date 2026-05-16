#!/usr/bin/env python3
"""
网球过网检测 — 实时语音播报
=============================
摄像头装在网柱上，自动标定网线位置。
实时追踪网球，球从网上方飞过时即时语音播报"好球"。
"""
import sys
import time
import argparse
import cv2

from court_calibrator import CourtCalibrator
from ball_tracker import BallTracker
from shot_judge import ShotJudge
from audio_player import AudioPlayer


def main():
    parser = argparse.ArgumentParser(description="网球过网检测 - 语音播报")
    parser.add_argument("--camera", type=int, default=0, help="摄像头设备号")
    parser.add_argument("--width", type=int, default=640, help="画面宽度")
    parser.add_argument("--height", type=int, default=480, help="画面高度")
    parser.add_argument("--debug", action="store_true", help="显示调试画面")
    parser.add_argument("--net-line", type=int, default=0, 
                        help="手动指定网线Y坐标（0=自动标定）")
    args = parser.parse_args()
    
    print("=" * 50)
    print("  🎾 网球过网检测系统")
    print("  Tennis Net Call v1.0")
    print("=" * 50)
    
    # ── 1. 打开摄像头 ──
    print(f"[摄像头] 正在打开设备 {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    if not cap.isOpened():
        print("[摄像头] ❌ 无法打开摄像头，请确认设备已连接")
        sys.exit(1)
    print("[摄像头] ✅ 就绪")
    
    # ── 2. 自动标定网线位置 ──
    if args.net_line > 0:
        net_line_y = args.net_line
        print(f"[标定] 使用手动网线位置: Y={net_line_y}")
    else:
        calibrator = CourtCalibrator()
        if not calibrator.calibrate(cap):
            print("[标定] ❌ 自动标定失败，请用 --net-line 手动指定")
            cap.release()
            sys.exit(1)
        net_line_y = calibrator.net_line_y
    
    # ── 3. 初始化各模块 ──
    tracker = BallTracker()
    judge = ShotJudge(net_line_y)
    audio = AudioPlayer()
    audio.init()
    
    print(f"[系统] 网线 Y={net_line_y} | 实时检测中...")
    print("[系统] 按 Ctrl+C 退出\n")
    
    # ── 主循环 ──
    fps_counter = []
    last_swing_time = 0  # 简单的挥拍检测：上次球向上飞的时间
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[摄像头] 读取帧失败")
                break
            
            t_start = time.time()
            
            # 球检测
            ball_circle = tracker.detect(frame)
            ball_pos = tracker.get_position() if ball_circle else None
            ball_vel = tracker.get_velocity()
            
            # 简易挥拍检测：球速度突然向上
            _, vy = ball_vel
            has_swing = (vy < -5)  # Y负方向 = 向上飞
            
            # 好球判定
            result = judge.update(ball_pos, ball_vel, has_swing)
            
            if result == 'great_shot':
                print("  🎾 好球！")
                audio.play('great_shot')
            elif result == 'net':
                print("  ⬇ 下网")
                audio.play('net')
            
            # 调试显示
            if args.debug:
                if ball_circle:
                    x, y, r = ball_circle
                    cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                frame = tracker.draw(frame)
                
                # 画网线
                h, w = frame.shape[:2]
                cv2.line(frame, (0, net_line_y), (w, net_line_y),
                         (0, 255, 255), 2)
                
                # 状态文字
                status = result if result != 'waiting' else '...'
                cv2.putText(frame, f"Status: {status}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Tennis Net Call", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # FPS
            fps_counter.append(time.time() - t_start)
            if len(fps_counter) > 50:
                fps_counter.pop(0)
                avg_fps = 1.0 / (sum(fps_counter) / len(fps_counter))
                if args.debug and len(fps_counter) % 50 == 0:
                    print(f"  FPS: {avg_fps:.1f}")
    
    except KeyboardInterrupt:
        print("\n[系统] 用户退出")
    
    finally:
        cap.release()
        if args.debug:
            cv2.destroyAllWindows()
        audio.cleanup()
        print("[系统] 已关闭")


if __name__ == "__main__":
    main()
