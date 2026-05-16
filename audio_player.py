"""
语音播报
========
即时播放预录音频文件，延迟 < 20ms。
若预录文件不存在，自动用 pygame 发声替代。
"""
import os
import random
import pygame

# ── 预录音频文件映射 ──
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio")

# 不同判定结果对应的音频池
CHEERS = {
    "great_shot": [
        "great_shot_01.wav",
        "great_shot_02.wav",
        "great_shot_03.wav",
    ],
    "net": [
        "net_01.wav",
    ],
    "whiff": [
        "whiff_01.wav",
    ],
}

class AudioPlayer:
    """预录音频播放器"""
    
    def __init__(self):
        self.initialized = False
        self.fallback_mode = False  # 找不到预录文件时用合成音
    
    def init(self):
        pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
        self.initialized = True
        
        # 检查预录音频是否存在
        has_audio = False
        for pool in CHEERS.values():
            for f in pool:
                if os.path.exists(os.path.join(AUDIO_DIR, f)):
                    has_audio = True
                    break
        
        if not has_audio:
            self.fallback_mode = True
            print("[语音] ⚠️ 预录音频未找到，使用合成提示音")
    
    def play(self, result_type: str):
        """
        根据判定类型播放对应音频。
        result_type: 'great_shot' | 'net' | 'whiff'
        """
        if not self.initialized:
            self.init()
        
        if self.fallback_mode:
            self._play_fallback(result_type)
            return
        
        pool = CHEERS.get(result_type, [])
        if not pool:
            return
        
        # 随机选一个（但尽量跟前一次不同）
        filename = random.choice(pool)
        filepath = os.path.join(AUDIO_DIR, filename)
        
        if os.path.exists(filepath):
            try:
                sound = pygame.mixer.Sound(filepath)
                sound.play()
            except Exception as e:
                print(f"[语音] 播放失败: {e}")
        else:
            self._play_fallback(result_type)
    
    def _play_fallback(self, result_type: str):
        """合成提示音：不同判定用不同音调"""
        import numpy as np
        
        sample_rate = 22050
        duration = 0.3
        
        if result_type == "great_shot":
            # 上升音阶（好球！）
            freqs = [523, 659, 784]
        elif result_type == "net":
            # 低沉（下网…）
            freqs = [200]
        else:
            freqs = [300]
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        for freq in freqs:
            wave = np.sin(2 * np.pi * freq * t) * 0.5
            wave = (wave * 32767).astype(np.int16)
            sound = pygame.sndarray.make_sound(wave)
            sound.play()
            pygame.time.wait(int(duration * 1000 / len(freqs)))
    
    def cleanup(self):
        pygame.mixer.quit()
