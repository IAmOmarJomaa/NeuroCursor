import time
import math
import numpy as np
import pyautogui
from collections import deque
from src.control.mouse import WindowsMouse
from src.config import CONFIG

class CursorEngine:
    def __init__(self, state, mouse: WindowsMouse):
        self.state = state
        self.mouse = mouse
        
        self.pos_history = deque(maxlen=10)
        self.bone_history = deque(maxlen=10)
        
        self.click_state = "IDLE"           
        self.click_start_time = 0
        self.anchor_x = 0
        self.anchor_y = 0
        self.release_buffer = 0  
        
        self.scroll_anchor_y = None
        self.right_click_start = 0
        self.has_triggered_right = False

    def _map_coordinates(self, raw_x, raw_y):
        # 1. Calculate Active Box
        # A Sensitivity of 2.0 means we only use the middle 50% of the camera
        box_size = 1.0 / CONFIG["DPI_SENSITIVITY"]
        
        # 2. Center the box (0.5) and apply 0 offset
        center_x = 0.5 + CONFIG["X_OFFSET"]
        center_y = 0.5 + CONFIG["Y_OFFSET"]
        
        # 3. Define Box Edges
        x1 = center_x - (box_size / 2)
        y1 = center_y - (box_size / 2)
        
        # 4. Map Input to 0-1 range relative to the box
        norm_x = (raw_x - x1) / box_size
        norm_y = (raw_y - y1) / box_size
        
        # 5. Clamp to screen edges so cursor doesn't fly off
        return np.clip(norm_x, 0, 1), np.clip(norm_y, 0, 1)

    def _calculate_stability_score(self, lms):
        # Bone consistency check (Wrist to Middle Knuckle)
        bone_len = math.hypot(lms[0].x - lms[9].x, lms[0].y - lms[9].y)
        self.bone_history.append(bone_len)
        if len(self.bone_history) < 5: return 1.0
        variance = np.var(list(self.bone_history)[-5:]) * 10000 
        return 1.0 + min(variance, 10.0)

    def update(self, gesture, lms, secondary_lms=None):
        # Stability check
        stability_mult = self._calculate_stability_score(lms)
        current_smooth = CONFIG["SMOOTHING"] * stability_mult

        # --- THE DIRECT MAPPING ---
        # We use Landmark 8 (INDEX_FINGER_TIP) directly.
        nx, ny = self._map_coordinates(lms[8].x, lms[8].y)
        
        raw_x = int(nx * self.mouse.screen_w)
        raw_y = int(ny * self.mouse.screen_h)
        self.pos_history.append((raw_x, raw_y))

        # --- CLICK STATE MACHINE ---
        if gesture == "CLICK":
            self.release_buffer = 0 
            
            if self.click_state == "IDLE":
                self.click_state = "PRE_CLICK"
                self.click_start_time = time.time()
                # Lock Anchor to previous stable position
                if len(self.pos_history) >= 4:
                    self.anchor_x, self.anchor_y = self.pos_history[-4]
                else:
                    self.anchor_x, self.anchor_y = raw_x, raw_y
            
            elif self.click_state == "PRE_CLICK":
                duration = time.time() - self.click_start_time
                
                # Freeze at anchor
                self.state.curr_x = self.anchor_x
                self.state.curr_y = self.anchor_y
                self.mouse.move(self.anchor_x, self.anchor_y)
                
                if duration > 0.05 and self.state.hold_timer == 0:
                    print("🖱️ DOWN")
                    self.mouse.down()
                    self.state.hold_timer = 1 

                dist = math.hypot(raw_x - self.anchor_x, raw_y - self.anchor_y)
                
                # We relax the drag delay slightly to make it feel snappier
                if duration > 0.15 and dist > CONFIG["CLICK_DEADZONE"]:
                    self.click_state = "DRAGGING"
                    print("🚀 DRAG MODE")

            elif self.click_state == "DRAGGING":
                self.state.curr_x += (raw_x - self.state.curr_x) / current_smooth
                self.state.curr_y += (raw_y - self.state.curr_y) / current_smooth
                self.mouse.move(self.state.curr_x, self.state.curr_y)

        elif gesture == "POINTER":
            if self.click_state != "IDLE":
                self.release_buffer += 1
                if self.release_buffer > 5: 
                    print("👆 UP")
                    self.mouse.up()
                    self.click_state = "IDLE"
                    self.state.hold_timer = 0
                    self.release_buffer = 0
            else:
                self.release_buffer = 0
                self.state.curr_x += (raw_x - self.state.curr_x) / current_smooth
                self.state.curr_y += (raw_y - self.state.curr_y) / current_smooth
                self.mouse.move(self.state.curr_x, self.state.curr_y)

        # --- AUXILIARY ---
        elif gesture == "SCROLL":
            self._handle_scroll(lms)
        elif gesture == "RIGHT_CLICK":
            self._handle_right_click()
        else:
            self.scroll_anchor_y = None
            self.right_click_start = 0
            self.has_triggered_right = False

    def _handle_scroll(self, lms):
        curr_y = lms[9].y
        if self.scroll_anchor_y is None: self.scroll_anchor_y = curr_y
        dist = self.scroll_anchor_y - curr_y
        if abs(dist) > 0.02:
            speed = int(dist * 100 * CONFIG["SCROLL_SPEED_BASE"])
            self.mouse.scroll(speed)

    def _handle_right_click(self):
        if self.right_click_start == 0: self.right_click_start = time.time()
        if (time.time() - self.right_click_start) > CONFIG["RIGHT_HOLD_MIDDLE"] and not self.has_triggered_right:
            print("🖱️ MIDDLE CLICK")
            pyautogui.middleClick()
            self.has_triggered_right = True