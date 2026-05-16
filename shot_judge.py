"""
好球判定
========
判断球是否从网上方飞过，返回判定结果：
  - 'waiting'      : 还没发生
  - 'great_shot'   : 球过网了 → 好球！
  - 'net'          : 球碰到网 / 下网
  - 'whiff'        : 挥空了（球没动）
"""
import time

class ShotJudge:
    """网球过网判定器"""
    
    def __init__(self, net_line_y: int):
        self.net_line_y = net_line_y
        self.prev_y = None
        self.crossing_up = False       # 球在往上飞
        self.above_net = False         # 球已经到了网上方
        self.shot_in_progress = False  # 正在进行一次击球判定
        self.last_result = None
        self.last_result_time = 0
        self.cooldown = 1.5            # 两次判定之间的冷却时间（秒）
        
    def update(self, ball_pos, ball_vel, has_swing: bool) -> str:
        """
        每帧调用，返回判定结果字符串。
        ball_pos: (x, y) 球当前像素坐标
        ball_vel: (vx, vy) 球速度
        has_swing: 是否检测到挥拍动作
        """
        if ball_pos is None:
            return self._reset('waiting')
        
        x, y = ball_pos
        vx, vy = ball_vel
        
        cooldown_ok = (time.time() - self.last_result_time) > self.cooldown
        
        # ── 阶段1：球在网下方，等待上飞 ──
        if not self.shot_in_progress:
            if has_swing and vy < 0 and cooldown_ok:
                # 挥拍 + 球往上飞 → 开始判定
                self.shot_in_progress = True
                self.crossing_up = True
                self.above_net = False
                self.prev_y = y
            return 'waiting'
        
        # ── 阶段2：追踪球是否过网 ──
        if self.crossing_up:
            # 球还在往上/往前飞
            if vy >= 0:
                # 球开始下落 → 可能已经过网或下网
                self.crossing_up = False
                
                if y <= self.net_line_y:
                    # 球在网线之上开始下落 → 过网了
                    self.above_net = True
                else:
                    # 球在网线之下开始下落 → 没过网
                    return self._finalize('net')
            
            # 检查球是否正在穿越网线
            if self.prev_y is not None:
                if self.prev_y > self.net_line_y >= y:
                    # 上帧在网下，当前帧在网上 → 过网了
                    self.above_net = True
        
        # ── 阶段3：球过网后的判定 ──
        if self.above_net and not self.crossing_up:
            # 球已经过网，确认判定
            return self._finalize('great_shot')
        
        # 超时保护：如果球丢失超过2秒，重置
        self.prev_y = y
        return 'waiting'
    
    def _finalize(self, result: str) -> str:
        """完成一次判定，返回结果并重置状态"""
        self.shot_in_progress = False
        self.crossing_up = False
        self.above_net = False
        self.prev_y = None
        
        self.last_result = result
        self.last_result_time = time.time()
        
        return result
    
    def _reset(self, default='waiting') -> str:
        """完全重置判定状态"""
        self.shot_in_progress = False
        self.crossing_up = False
        self.above_net = False
        self.prev_y = None
        return default
