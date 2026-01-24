import time
import pyautogui
from collections import deque
from src.config import CONFIG

class SequenceEngine:
    def __init__(self, state):
        self.state = state
        self.history = deque(maxlen=20)
        
        # Timers
        self.gesture_start_time = 0
        self.last_trigger_time = 0
        
        # Motion Tracking for Select All
        self.palm_motion_path = [] # Stores (x, y) tuples

    # UPDATE: Now accepts 'lms' (Landmarks)
    def process(self, gesture, lms=None):
        now = time.time()
        
        # 1. Update History
        if not self.history or self.history[-1] != gesture:
            self.history.append(gesture)
            self.gesture_start_time = now
            # Reset motion path on new gesture
            self.palm_motion_path = [] 
        
        hold_duration = now - self.gesture_start_time
        
        # Cooldown
        if (now - self.last_trigger_time) < 0.5:
            return

        # --- MOTION TRACKING (For PALM) ---
        if gesture == "PALM" and lms:
            # Record hand position (using Middle Finger Base 9)
            self.palm_motion_path.append((lms[9].x, lms[9].y))
            
            # Keep path manageable (last 60 frames / ~1 sec)
            if len(self.palm_motion_path) > 60:
                self.palm_motion_path.pop(0)

        # --- A. LOOP SEQUENCES ---
        if self._check_pattern(["DELETE_0", "DELETE_1"] * 2):
            print("🗑️ DELETE TRIGGERED"); pyautogui.press('delete'); self._reset_trigger(now); return

        if self._check_pattern(["NEXT_1", "NEXT_0"] * 3):
            print("➡️ NEXT TRIGGERED"); pyautogui.hotkey('alt', 'right'); self._reset_trigger(now); return

        if self._check_pattern(["PREVIOUS_1", "PREVIOUS_0"] * 3):
            print("⬅️ PREV TRIGGERED"); pyautogui.hotkey('alt', 'left'); self._reset_trigger(now); return

        # --- B. TRANSITION SEQUENCES ---
        if self._check_pattern(["TAB_CENTER", "TAB_LEFT"]):
            print("Tab Left"); pyautogui.hotkey('alt', 'shift', 'tab'); self._reset_trigger(now)
            
        if self._check_pattern(["TAB_CENTER", "TAB_RIGHT"]):
            print("Tab Right"); pyautogui.hotkey('alt', 'tab'); self._reset_trigger(now)

        # --- C. COPY / PASTE / SELECT ALL ---
        
        # SELECT ALL (Circular Motion Check)
        if gesture == "PALM":
            if hold_duration > CONFIG["SELECT_ALL_HOLD"]:
                # CALCULATE DIAMETER OF MOVEMENT
                if len(self.palm_motion_path) > 10:
                    xs = [p[0] for p in self.palm_motion_path]
                    ys = [p[1] for p in self.palm_motion_path]
                    
                    width = max(xs) - min(xs)
                    height = max(ys) - min(ys)
                    
                    # Logic: Must cover area > Threshold
                    if width > CONFIG["SELECT_ALL_DIAMETER"] or height > CONFIG["SELECT_ALL_DIAMETER"]:
                        print(f"🟦 SELECT ALL (Size: {width:.2f})")
                        pyautogui.hotkey('ctrl', 'a')
                        self._reset_trigger(now)
                        # Prevents looping
                        self.gesture_start_time = now + 5 

        # COPY (Palm -> Fist)
        if gesture == "FIST":
            if len(self.history) >= 2 and self.history[-2] == "PALM":
                if hold_duration > 0.1:
                    print("📄 COPY")
                    pyautogui.hotkey('ctrl', 'c')
                    self._reset_trigger(now)
        
        # PASTE (Fist -> Palm)
        if gesture == "PALM":
             if len(self.history) >= 2 and self.history[-2] == "FIST":
                 if hold_duration > 0.1:
                    print("📋 PASTE")
                    pyautogui.hotkey('ctrl', 'v')
                    self._reset_trigger(now)

    def _check_pattern(self, pattern):
        if len(self.history) < len(pattern): return False
        recent = list(self.history)[-len(pattern):]
        return recent == pattern

    def _reset_trigger(self, t):
        self.last_trigger_time = t
        self.history.clear()
        self.palm_motion_path = []