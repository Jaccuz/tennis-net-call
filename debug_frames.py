#!/usr/bin/env python3
"""
问题帧诊断工具
提取指定帧并显示球检测状态、网线位置、速度向量
用法: python debug_frames.py [视频路径]
"""
import sys, os
os.environ["OPENCV_LOG_LEVEL"]      = "ERROR"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
import cv2
import numpy as np
from collections import deque

VIDEO_PATH  = sys.argv[1] if len(sys.argv) > 1 else r"demo\test_12m_2k_60.avi"
OUT_DIR     = "debug_frames"
PROCESS_W   = 320
PROCESS_H   = 180
NET_RATIO   = 0.38
HSV_LOWER   = np.array([20, 70, 45])
HSV_UPPER   = np.array([50, 255, 255])
TRAIL_LEN   = 20

# 需要检查的帧 (±WINDOW 帧范围均输出)
PROBLEM_FRAMES = [2364, 2601, 2826, 3243, 3819, 3927, 4251, 4662, 5065]
WINDOW = 10   # 每个问题帧前后各输出 WINDOW 帧

def detect_ball(frame):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask
    best = None
    best_score = 0
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(c)
        if area < 3 or area > 2000:
            continue
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        circ = 4 * 3.14159 * area / (perim * perim)
        if circ < 0.4:
            continue
        score = circ * area
        if score > best_score:
            best_score = score
            best = c
    if best is None:
        return None, mask
    M = cv2.moments(best)
    if M["m00"] == 0:
        return None, mask
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask


class KalmanTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix  = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix   = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov    = np.eye(4, dtype=np.float32) * 1.0
        self.kf.measurementNoiseCov= np.eye(2, dtype=np.float32) * 0.1
        self.kf.errorCovPost       = np.diag([10.0,10.0,100.0,100.0]).astype(np.float32)
        self.init  = False
        self.lost  = 0
        self.trail = deque(maxlen=TRAIL_LEN)
        self.last_pos = None

    def update(self, det):
        if det is not None:
            x, y = det
            if not self.init:
                self.kf.statePost = np.array([[x],[y],[0],[0]], np.float32)
                self.kf.errorCovPost = np.diag([10.0,10.0,100.0,100.0]).astype(np.float32)
                self.init = True
            else:
                self.kf.correct(np.array([[x],[y]], np.float32))
            self.lost = 0
            pred = self.kf.predict()
            self.last_pos = det
            self.trail.append(det)
            return det, (float(pred[2]), float(pred[3]))
        else:
            self.lost += 1
            if not self.init or self.lost > 8:
                self.init = False
                self.last_pos = None
                return None
            pred = self.kf.predict()
            pos = (int(pred[0]), int(pred[1]))
            self.last_pos = pos
            return pos, (float(pred[2]), float(pred[3]))


def annotate(frame_bgr, det, tracked, vel, net_y_proc, net_y_disp,
             frame_idx, note=""):
    disp = cv2.resize(frame_bgr, (960, 540))
    sh, sw = disp.shape[:2]
    scale_x = sw / PROCESS_W
    scale_y = sh / PROCESS_H

    # 网线
    cv2.line(disp, (0, net_y_disp), (sw, net_y_disp), (0, 215, 255), 2)
    cv2.putText(disp, "NET", (10, net_y_disp - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,215,255), 1)

    # 球轨迹 (从 tracker)
    if tracked:
        pos, _ = tracked
        bx = int(pos[0] * scale_x)
        by = int(pos[1] * scale_y)
        cv2.circle(disp, (bx, by), 12, (0, 255, 0), 2)
        cv2.circle(disp, (bx, by), 3, (0, 255, 0), -1)

    if det is not None:
        dx = int(det[0] * scale_x)
        dy = int(det[1] * scale_y)
        cv2.circle(disp, (dx, dy), 8, (0, 0, 255), 2)  # 红圈 = 原始检测

    if vel:
        vx, vy = vel
        cv2.putText(disp, f"vx={vx:.1f}  vy={vy:.1f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 50), 2)
        # 速度箭头
        if tracked:
            pos, _ = tracked
            bx = int(pos[0] * scale_x)
            by = int(pos[1] * scale_y)
            ev = int(vx * scale_x * 3), int(vy * scale_y * 3)
            cv2.arrowedLine(disp, (bx, by), (bx + ev[0], by + ev[1]),
                            (255, 100, 0), 2, tipLength=0.3)

    ball_y_proc = tracked[0][1] if tracked else -1
    side_txt = "PLAYER SIDE" if ball_y_proc > net_y_proc else ("OPPONENT SIDE" if ball_y_proc >= 0 else "NO BALL")
    col = (0, 255, 0) if ball_y_proc > net_y_proc else (0, 100, 255)
    cv2.putText(disp, f"Frame {frame_idx}  {side_txt}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    if note:
        cv2.putText(disp, note, (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 255), 2)

    # HSV 掩码缩略图 (右下角)
    return disp


def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] 视频不存在: {VIDEO_PATH}")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频: {total} 帧, {fps:.0f} fps")

    net_y_proc = int(PROCESS_H * NET_RATIO)
    net_y_disp = int(540 * NET_RATIO)
    print(f"网线 Y (处理分辨率): {net_y_proc}")

    # 收集所有需要输出的帧号
    targets = set()
    for f in PROBLEM_FRAMES:
        for d in range(-WINDOW, WINDOW + 1):
            fi = f + d
            if 0 < fi <= total:
                targets.add(fi)

    notes = {f: "!! PROBLEM FRAME" for f in PROBLEM_FRAMES}

    tracker = KalmanTracker()
    frame_idx = 0
    saved = 0

    print("提取帧中...")
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 缩放到处理分辨率做检测
        proc = cv2.resize(frame_bgr, (PROCESS_W, PROCESS_H))
        det, mask = detect_ball(proc)
        tracked = tracker.update(det)

        vel = None
        if tracked:
            _, vel = tracked

        if frame_idx in targets:
            note = notes.get(frame_idx, "")
            disp = annotate(frame_bgr, det, tracked, vel,
                            net_y_proc, net_y_disp, frame_idx, note)

            # 附上 HSV 掩码缩略图到右下角
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mh, mw = 90, 160
            mask_small = cv2.resize(mask_rgb, (mw, mh))
            sh, sw = disp.shape[:2]
            disp[sh-mh:sh, sw-mw:sw] = mask_small
            cv2.rectangle(disp, (sw-mw, sh-mh), (sw, sh), (200,200,200), 1)
            cv2.putText(disp, "HSV mask", (sw-mw+4, sh-mh+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

            # 打印检测信息
            ball_y = tracked[0][1] if tracked else -1
            vx_s = f"{vel[0]:.1f}" if vel else "N/A"
            vy_s = f"{vel[1]:.1f}" if vel else "N/A"
            side = "player" if ball_y > net_y_proc else "opponent" if ball_y >= 0 else "none"
            det_s = f"({det[0]},{det[1]})" if det else "None"
            print(f"  [{frame_idx:5d}]{notes.get(frame_idx,''):18s}  "
                  f"det={det_s:12s}  y={ball_y:4.0f}  "
                  f"vx={vx_s:6s}  vy={vy_s:6s}  side={side}")

            fname = f"{OUT_DIR}/f{frame_idx:05d}{'_PROBLEM' if frame_idx in PROBLEM_FRAMES else ''}.jpg"
            cv2.imwrite(fname, disp)
            saved += 1

    cap.release()
    print(f"\n已保存 {saved} 帧到 {OUT_DIR}/")
    print("红圈=原始HSV检测  绿圈=Kalman平滑位置  蓝色箭头=速度方向")


if __name__ == "__main__":
    main()
