"""
网球追踪
========
HSV颜色掩码检测网球（黄绿色），Kalman滤波平滑轨迹。
网球在画面中通常是唯一的黄绿色高速移动物体。
"""
import cv2
import numpy as np
from collections import deque

# ── 网球 HSV 颜色范围（黄绿色） ──
BALL_HSV_RANGES = [
    # (H_low, H_high, S_low, S_high, V_low, V_high)
    (25, 45, 80, 255, 100, 255),   # 标准黄绿
    (20, 50, 60, 255, 80, 255),    # 宽松范围（阴天/逆光）
]

class BallTracker:
    """网球检测 + Kalman轨迹追踪"""
    
    def __init__(self):
        self.trail = deque(maxlen=30)  # 最近30帧的轨迹
        self.positions = deque(maxlen=10)
        self.kalman = None
        self.kalman_initialized = False
        self.last_position = None
        
    def detect(self, frame):
        """
        在帧中检测网球，返回 (x, y, radius) 或 None。
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        best_circle = None
        best_score = 0
        
        for h_low, h_high, s_low, s_high, v_low, v_high in BALL_HSV_RANGES:
            mask = cv2.inRange(hsv, (h_low, s_low, v_low), (h_high, s_high, v_high))
            
            # 开运算去噪
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 5 or area > 500:  # 网球在画面中的合理大小
                    continue
                
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                if radius < 3 or radius > 25:
                    continue
                
                # 圆度检查
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                score = circularity * area
                if score > best_score:
                    best_score = score
                    best_circle = (int(x), int(y), int(radius))
        
        if best_circle:
            self.last_position = (best_circle[0], best_circle[1])
            self.trail.append(self.last_position)
            self._update_kalman(best_circle[0], best_circle[1])
        
        return best_circle
    
    def _update_kalman(self, x, y):
        """Kalman滤波平滑轨迹"""
        if not self.kalman_initialized:
            self.kalman = cv2.KalmanFilter(4, 2)
            self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
            self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
            self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
            self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
            self.kalman.statePre = np.array([[x],[y],[0],[0]], np.float32)
            self.kalman.statePost = np.array([[x],[y],[0],[0]], np.float32)
            self.kalman_initialized = True
        
        measurement = np.array([[np.float32(x)],[np.float32(y)]])
        self.kalman.correct(measurement)
        prediction = self.kalman.predict()
        
        self.positions.append((int(prediction[0]), int(prediction[1])))
    
    def get_velocity(self):
        """计算球的速度（像素/帧）"""
        if len(self.positions) < 3:
            return 0, 0
        p1 = self.positions[-3]
        p2 = self.positions[-1]
        return p2[0] - p1[0], p2[1] - p1[1]
    
    def get_position(self):
        """获取当前平滑位置"""
        if self.positions:
            return self.positions[-1]
        return self.last_position
    
    def draw(self, frame):
        """在画面上标出球的位置和轨迹（调试用）"""
        for i, pos in enumerate(self.trail):
            alpha = (i + 1) / len(self.trail)
            cv2.circle(frame, pos, 3, (0, int(255*alpha), int(255*(1-alpha))), -1)
        
        pos = self.get_position()
        if pos:
            cv2.circle(frame, pos, 8, (0, 255, 0), 2)
        
        return frame
