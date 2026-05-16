"""
自动标定：检测网带位置
======================
摄像头固定在网柱上后，开机自动检测白色网带在画面中的位置。
使用霍夫直线检测 + 多帧投票，2秒内完成标定。
"""
import cv2
import numpy as np
from collections import Counter

class CourtCalibrator:
    """自动标定球网和场地线"""
    
    def __init__(self):
        self.net_line_y: int = 0          # 网带顶端的Y坐标
        self.net_line_angle: float = 0.0  # 网带倾斜角度
        self.calibrated: bool = False
        self.confidence: float = 0.0
        
    def calibrate(self, camera):
        """
        采集30帧，检测网带位置，投票确定最终Y坐标。
        返回 True 表示标定成功。
        """
        net_y_candidates = []
        print("[标定] 正在检测球网位置...")
        
        for i in range(30):
            ret, frame = camera.read()
            if not ret:
                continue
            y = self._detect_net_line(frame)
            if y > 0:
                net_y_candidates.append(y)
            
            if i % 10 == 0:
                print(f"[标定] 进度: {i}/30")
        
        if len(net_y_candidates) < 5:
            print("[标定] ❌ 未检测到球网，请确认摄像头朝向球场")
            return False
        
        # 投票：取众数（容忍±3像素）
        self.net_line_y = self._vote_y(net_y_candidates, tolerance=3)
        self.confidence = len(net_y_candidates) / 30.0
        self.calibrated = True
        
        print(f"[标定] ✅ 完成 — 网线 Y={self.net_line_y} (置信度 {self.confidence:.0%})")
        return True
    
    def _detect_net_line(self, frame) -> int:
        """
        在画面中上1/3区域检测白色水平横条（网带）。
        返回网带顶端的Y坐标，检测不到返回 -1。
        """
        h, w = frame.shape[:2]
        # 只分析画面上半部分（网在上方）
        roi = frame[0:h//2, :]
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 白色区域阈值（网带是白色的）
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # 边缘检测
        edges = cv2.Canny(white_mask, 50, 150)
        
        # 霍夫直线检测
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                 minLineLength=w//3, maxLineGap=20)
        
        if lines is None:
            return -1
        
        # 筛选接近水平的线（±10度内）
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 10:  # 接近水平
                horizontal_lines.append((y1 + y2) // 2)
        
        if not horizontal_lines:
            return -1
        
        # 返回最上方的那条（网带顶端）
        return min(horizontal_lines)
    
    def _vote_y(self, candidates: list, tolerance: int) -> int:
        """对候选Y坐标做容差投票"""
        if not candidates:
            return 0
        
        # 对候选值做聚类
        candidates.sort()
        clusters = []
        current = [candidates[0]]
        
        for y in candidates[1:]:
            if y - current[-1] <= tolerance:
                current.append(y)
            else:
                clusters.append(current)
                current = [y]
        clusters.append(current)
        
        # 取最大簇的中位数
        best = max(clusters, key=len)
        return int(np.median(best))
    
    def draw_net_line(self, frame):
        """在画面上标出网线位置（调试用）"""
        if not self.calibrated:
            return frame
        h, w = frame.shape[:2]
        cv2.line(frame, (0, self.net_line_y), (w, self.net_line_y), 
                 (0, 255, 255), 2)
        cv2.putText(frame, "NET", (10, self.net_line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame
