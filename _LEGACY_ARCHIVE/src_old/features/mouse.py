import pyautogui
import numpy as np
import time
from core.one_euro_filter import PointFilter
from core.kinematics import ScreenMapper

class MouseController:
    def __init__(self):
        # TUNING
        self.filter = PointFilter(min_cutoff=0.01, beta=0.2)
        self.mapper = ScreenMapper()
        self.screen_w, self.screen_h = pyautogui.size()
        pyautogui.FAILSAFE = False 
        self.last_pos = (0, 0)

    def process_hand(self, hand_lms, gesture_state):
        # 1. RAW COORDS (Index Tip #8)
        raw_x = hand_lms.landmark[8].x
        raw_y = hand_lms.landmark[8].y
        
        # 2. SMOOTHING
        smooth_x, smooth_y = self.filter.process([raw_x, raw_y])
        
        # 3. MAPPING (Fixed: Passing screen dims)
        cursor_x, cursor_y = self.mapper.map(smooth_x, smooth_y, self.screen_w, self.screen_h)
        
        # 4. MOVE
        try:
            pyautogui.moveTo(cursor_x, cursor_y, _pause=False)
        except:
            pass