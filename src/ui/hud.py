"""
NeuroCursor HUD (God Tier).
Visualizes the Physics Engine: Dynamic Deadzones and Z-Depth.
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from src.core.kinematics import KinematicsEngine

class HUD:
    def __init__(self):
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.physics_engine = KinematicsEngine() # For visual calculations
        
        # --- THEME COLORS (BGR) ---
        self.C_CYAN   = (255, 255, 0)    # Standard UI
        self.C_RED    = (0, 0, 255)      # Locked / Critical
        self.C_ORANGE = (0, 165, 255)    # Dragging / Active
        self.C_GREEN  = (0, 255, 0)      # Success / Unlocked
        self.C_PURPLE = (255, 0, 255)    # Physics / Debug
        self.C_DARK   = (20, 20, 20)     # Backgrounds
        
        # --- ANIMATION STATE ---
        self.scan_line_y = 0

    def _draw_glass_panel(self, img, x, y, w, h, color, alpha=0.6):
        """Draws a semi-transparent 'Glass' background."""
        # Safety check for image bounds
        if y+h > img.shape[0] or x+w > img.shape[1] or x < 0 or y < 0: return
        
        sub_img = img[y:y+h, x:x+w]
        white_rect = np.full(sub_img.shape, color, dtype=np.uint8)
        res = cv2.addWeighted(sub_img, 1 - alpha, white_rect, alpha, 1.0)
        img[y:y+h, x:x+w] = res
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 1)

    def render(self, frame, pilot, hand_data, stabilizer):
        h, w, _ = frame.shape
        label, conf, lms = hand_data
        
        # 1. DETERMINE SYSTEM STATE & COLOR
        ui_color = self.C_CYAN
        status_msg = "NEURO_CORE // ONLINE"
        
        if pilot.state.is_locked:
            ui_color = self.C_RED
            status_msg = "SYSTEM LOCKED // GEN_Z TO AUTH"
        elif pilot.state.is_dragging:
            ui_color = self.C_ORANGE
            status_msg = "ACTUATOR ENGAGED // DRAGGING"
        elif stabilizer.locked:
            ui_color = self.C_GREEN
            status_msg = "STABILIZER ACTIVE // ANCHORED"

        # 2. DRAW PHYSICS VIZ (The Bunker Ring)
        # We draw this *around* the hand input to show the noise threshold
        phys = pilot.last_physics
        if lms and not pilot.state.is_locked:
            # A. Get Dynamic Deadzone Radius (Normalized)
            dz_norm = self.physics_engine.get_dynamic_deadzone(phys.smooth_speed)
            
            # B. Convert to Pixels (Scale to current frame width)
            # The deadzone is relative to the ref width (1920), so we scale it down to `w`
            radius_px = int(dz_norm * w)
            
            # C. Hand Center (Input)
            cx, cy = int(phys.raw_x * w), int(phys.raw_y * h)
            
            # Draw the Deadzone Ring
            # If speed is low, ring is large (Stable). If speed is high, ring is small (Agile).
            cv2.circle(frame, (cx, cy), radius_px, self.C_PURPLE, 1)
            
            # Optional: Draw Velocity Vector
            # end_x = cx + int(phys.raw_x - pilot.prev_x) * 500 # Amplified
            # cv2.line(frame, (cx, cy), (end_x, cy), self.C_PURPLE, 1)

        # 3. DRAW WORKSPACE BOX
        if pilot.show_box:
            cx, cy = pilot.BOX_CENTER_X, pilot.BOX_CENTER_Y
            bw, bh = pilot.BOX_WIDTH, pilot.BOX_HEIGHT
            
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            
            l = 30 # Line length
            t = 2  # Thickness
            
            # Draw Corners
            pts = [((x1, y1), (1,0), (0,1)), ((x2, y1), (-1,0), (0,1)),
                   ((x1, y2), (1,0), (0,-1)), ((x2, y2), (-1,0), (0,-1))]
            
            for p, dx, dy in pts:
                cv2.line(frame, p, (p[0]+dx[0]*l, p[1]+dx[1]*l), ui_color, t)
                cv2.line(frame, p, (p[0]+dy[0]*l, p[1]+dy[1]*l), ui_color, t)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), ui_color, 1)

        # 4. DRAW SKELETON
        if lms:
            self.mp_draw.draw_landmarks(
                frame, lms, self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=self.C_DARK, thickness=4, circle_radius=2),
                self.mp_draw.DrawingSpec(color=ui_color, thickness=2, circle_radius=2)
            )

        # 5. ENERGY BAR (Confidence / Lock Timer)
        bar_w = 200
        bar_h = 10
        bar_x = (w - bar_w) // 2
        bar_y = h - 40
        
        self._draw_glass_panel(frame, bar_x, bar_y, bar_w, bar_h, self.C_DARK, 0.8)
        fill_w = int(bar_w * conf)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), ui_color, -1)
        
        # 6. STATUS BAR
        self._draw_glass_panel(frame, 20, 20, 400, 50, self.C_DARK, 0.4)
        
        display_label = str(label)
        if "." in display_label: 
             display_label = display_label.split(".")[-1].strip()
             
        cv2.putText(frame, display_label, (35, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, ui_color, 2)
        
        cv2.putText(frame, status_msg, (w//2 - 150, 40), 
                    cv2.FONT_HERSHEY_PLAIN, 1.2, ui_color, 1)

        # 7. SCAN LINE (Idle Animation)
        if pilot.state.is_locked:
            self.scan_line_y += 5
            if self.scan_line_y > h: self.scan_line_y = 0
            cv2.line(frame, (0, self.scan_line_y), (w, self.scan_line_y), self.C_RED, 1)

    def draw_fps(self, frame, fps):
        cv2.putText(frame, f"{int(fps)} FPS", (frame.shape[1]-100, 40), 
                    cv2.FONT_HERSHEY_PLAIN, 1.2, self.C_GREEN, 1)