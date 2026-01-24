import time
import pyautogui
from src.config import CONFIG

class SystemEngine:
    def __init__(self, state):
        self.state = state
        self.lock_start_time = 0
        self.vol_anchor_y = None

    def update(self, gesture, lms):
        now = time.time()

        # --- 1. LOCKING (GEN_Z) ---
        if gesture == "LOCK":
            if self.lock_start_time == 0:
                self.lock_start_time = now
            
            elapsed = now - self.lock_start_time
            self.state.lock_progress = elapsed / CONFIG["LOCK_TIME"]
            
            if elapsed > CONFIG["LOCK_TIME"]:
                self.state.locked = not self.state.locked
                print(f"🔒 LOCK STATE: {self.state.locked}")
                self.lock_start_time = 0
                self.state.lock_progress = 0
                time.sleep(1) # Safety pause
        else:
            self.lock_start_time = 0
            self.state.lock_progress = 0

        # --- 2. WAVE EXIT (Only when locked) ---
        if self.state.locked:
            cx = lms[9].x
            speed = abs(cx - self.state.last_x)
            
            if gesture == "PALM" and speed > 0.05:
                self.state.wave_energy += 1
            else:
                self.state.wave_energy = max(0, self.state.wave_energy - 1)
            
            self.state.last_x = cx
            if self.state.wave_energy > 40:
                print("👋 WAVE EXIT")
                self.state.running = False

        # --- 3. VOLUME ---
        if gesture == "VOLUME" and not self.state.locked:
            y = lms[4].y
            if self.vol_anchor_y is None: self.vol_anchor_y = y
            
            dist = self.vol_anchor_y - y
            if abs(dist) > 0.05:
                # Acceleration
                steps = int(abs(dist) * 10)
                if dist > 0: 
                    for _ in range(steps): pyautogui.press('volumeup')
                else:
                    for _ in range(steps): pyautogui.press('volumedown')
        else:
            self.vol_anchor_y = None
            
        # --- 4. CUT ---
        # Reuse Copy logic usually, but handled here for specific triggers
        if gesture == "CUT":
             # Use sequence logic or simple timer
             pass