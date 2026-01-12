import pyautogui
import numpy as np
from core.one_euro_filter import PointFilter
from core.kinematics import ScreenMapper

class MouseController:
    def __init__(self):
        self.filter = PointFilter(min_cutoff=0.1, beta=2.0) # Slower = Smoother
        self.mapper = ScreenMapper()
        self.screen_w, self.screen_h = pyautogui.size()
        pyautogui.FAILSAFE = False
        
        self.is_clicking = False # Left
        self.is_r_clicking = False # Right

    def process_hand(self, hand_lms, gesture_state):
        # 1. TRACKING ANCHOR (Index Base - Landmark 5 - is more stable than tip)
        raw_x = hand_lms.landmark[5].x
        raw_y = hand_lms.landmark[5].y
        
        # 2. STABILIZE
        smooth_x, smooth_y = self.filter.process([raw_x, raw_y])
        
        # 3. MOVE CURSOR
        cursor_x, cursor_y = self.mapper.map(smooth_x, smooth_y, self.screen_w, self.screen_h)
        pyautogui.moveTo(cursor_x, cursor_y, _pause=False)

    def check_clicks(self, hand_lms, pinch_thresh):
        # --- LEFT CLICK (Index + Thumb) ---
        p4 = np.array([hand_lms.landmark[4].x, hand_lms.landmark[4].y])
        p8 = np.array([hand_lms.landmark[8].x, hand_lms.landmark[8].y])
        dist_left = np.linalg.norm(p4 - p8) * 100
        
        if not self.is_clicking and dist_left < pinch_thresh:
            pyautogui.mouseDown()
            self.is_clicking = True
        elif self.is_clicking and dist_left > pinch_thresh * 1.5:
            pyautogui.mouseUp()
            self.is_clicking = False

        # --- RIGHT CLICK (Middle + Thumb) ---
        p12 = np.array([hand_lms.landmark[12].x, hand_lms.landmark[12].y])
        dist_right = np.linalg.norm(p4 - p12) * 100
        
        if not self.is_r_clicking and dist_right < pinch_thresh:
            pyautogui.click(button='right') # Instant click, not drag
            self.is_r_clicking = True
        elif self.is_r_clicking and dist_right > pinch_thresh * 1.5:
            self.is_r_clicking = False

    def scroll(self, hand_lms):
        # Scroll based on Index Tip Height
        # High = Scroll Up, Low = Scroll Down
        y = hand_lms.landmark[8].y
        if y < 0.3: pyautogui.scroll(20)   # Up
        if y > 0.7: pyautogui.scroll(-20)  # Down