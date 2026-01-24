import cv2
import mediapipe as mp
import sys
import os
import math
import time
import numpy as np
import random

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.gesture_engine import NeuroCursorBrain
from src.config import CONFIG

class ScenarioSim:
    def __init__(self):
        self.brain = NeuroCursorBrain()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, model_complexity=0, min_detection_confidence=0.5)
        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        # Sim State
        self.cursor_x, self.cursor_y = 0, 0
        self.anchor_x, self.anchor_y = 0, 0
        self.is_dragging = False
        
        # Game State
        self.scenario = 1
        self.score = 0
        self.target_pos = (300, 300)
        self.drag_obj_pos = [100, 300]
        self.hover_timer = 0
        self.start_msg_timer = time.time()

    def update_physics(self, lms, w, h):
        # 1. Map
        idx_x, idx_y = lms.landmark[8].x, lms.landmark[8].y
        sens = CONFIG.get("DPI_SENSITIVITY", 1.5)
        idx_x = 0.5 + (idx_x - 0.5) * sens
        idx_y = 0.5 + (idx_y - 0.5) * sens
        
        target_x = np.clip(idx_x * w, 0, w)
        target_y = np.clip(idx_y * h, 0, h)
        
        # 2. Get Gesture
        label, conf = self.brain.predict(lms.landmark)
        
        # 3. Physics Logic (Replicating NeuroTuner logic)
        smooth = CONFIG.get("SMOOTHING", 5.0)
        deadzone = CONFIG.get("CLICK_DEADZONE", 35)
        
        if label == "THE_CLICK":
            if not self.is_dragging:
                self.anchor_x, self.anchor_y = self.cursor_x, self.cursor_y
                self.is_dragging = True
            
            dist = math.hypot(target_x - self.anchor_x, target_y - self.anchor_y)
            if dist > deadzone:
                self.cursor_x += (target_x - self.cursor_x) / smooth
                self.cursor_y += (target_y - self.cursor_y) / smooth
            # Else: Frozen in deadzone
        else:
            self.is_dragging = False
            self.cursor_x += (target_x - self.cursor_x) / smooth
            self.cursor_y += (target_y - self.cursor_y) / smooth
            
        return label

    def draw_game(self, frame, gesture):
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0,0), (w, h), (20,20,20), -1)
        
        # --- SCENARIO 1: THE SNIPER ---
        if self.scenario == 1:
            cv2.putText(frame, "TEST 1: MOVE TO RED DOT AND CLICK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Score: {self.score}/5", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            
            cv2.circle(frame, self.target_pos, 25, (0, 0, 255), -1)
            
            dist = math.hypot(self.cursor_x - self.target_pos[0], self.cursor_y - self.target_pos[1])
            if dist < 25 and gesture == "THE_CLICK":
                self.score += 1
                self.target_pos = (random.randint(100, w-100), random.randint(100, h-100))
                if self.score >= 5: 
                    self.scenario = 2
                    self.score = 0
                    self.start_msg_timer = time.time()

        # --- SCENARIO 2: THE CARGO ---
        elif self.scenario == 2:
            cv2.putText(frame, "TEST 2: DRAG BLUE BOX TO GREEN ZONE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Score: {self.score}/3", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

            # Goal
            cv2.rectangle(frame, (w-150, h-150), (w-50, h-50), (0, 255, 0), 2)
            
            # Object
            bx, by = int(self.drag_obj_pos[0]), int(self.drag_obj_pos[1])
            col = (255, 200, 0)
            
            # Logic
            if abs(self.cursor_x - bx) < 40 and abs(self.cursor_y - by) < 40 and self.is_dragging:
                col = (0, 255, 255) # Grabbed
                self.drag_obj_pos[0] = self.cursor_x
                self.drag_obj_pos[1] = self.cursor_y
            
            cv2.rectangle(frame, (bx-40, by-40), (bx+40, by+40), col, -1)
            
            if bx > (w-150) and by > (h-150) and not self.is_dragging:
                self.score += 1
                self.drag_obj_pos = [100, 300]
                if self.score >= 3: self.scenario = 3

        # --- SCENARIO 3: THE STATUE ---
        elif self.scenario == 3:
            cv2.putText(frame, "TEST 3: HOVER IN BOX - DO NOT CLICK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.rectangle(frame, (200, 200), (w-200, h-200), (255, 255, 255), 2)
            
            in_zone = 200 < self.cursor_x < w-200 and 200 < self.cursor_y < h-200
            
            if in_zone:
                if gesture == "THE_CLICK":
                    cv2.putText(frame, "FALSE CLICK! RESETTING...", (220, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    self.hover_timer = time.time()
                else:
                    elapsed = time.time() - self.hover_timer
                    cv2.putText(frame, f"HOLD: {elapsed:.1f}s", (220, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if elapsed > 5.0:
                        cv2.putText(frame, "ALL TESTS PASSED!", (200, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            else:
                self.hover_timer = time.time()

        # Draw Cursor
        ccol = (0, 255, 0) if self.is_dragging else (0, 255, 255)
        cv2.circle(frame, (int(self.cursor_x), int(self.cursor_y)), 10, ccol, -1)
        if self.is_dragging:
             cv2.circle(frame, (int(self.anchor_x), int(self.anchor_y)), 5, (50,50,50), 1)

    def run(self):
        print("🎮 SCENARIO SIMULATOR STARTING...")
        while True:
            ret, frame = self.cam.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            gesture = "NONE"
            if results.multi_hand_landmarks:
                gesture = self.update_physics(results.multi_hand_landmarks[0], w, h)
                
            self.draw_game(frame, gesture)
            cv2.imshow("NeuroSimulator", frame)
            
            if cv2.waitKey(1) == 27: break
            
        self.cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = ScenarioSim()
    app.run()