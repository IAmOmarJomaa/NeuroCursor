import cv2
import mediapipe as mp
import sys
import os
import time
import numpy as np
import math
from collections import deque

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.camera import FastCamera
from src.vision.logic_brain import LogicBrain
from src.config import CONFIG

class NeuroFlightDeck:
    def __init__(self):
        self.cam = FastCamera()
        self.brain = LogicBrain()
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, model_complexity=0, min_detection_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

        # --- LOGIC STATE ---
        self.timers = {
            "PALM_CIRCLE": 0, # Select All
            "FIST_HOLD": 0,   # Delete
            "LOCK": 0,        # Gen Z
            "RIGHT": 0,       # Right Click
            "VOLUME": 0,      # Shaka
        }
        
        # Motion Tracking for Circular Gesture
        self.wrist_history = deque(maxlen=20)
        self.circle_score = 0.0
        
        self.triggered_action = None
        self.action_display_timer = 0
        self.event_log = []

    def detect_circular_motion(self, current_x, current_y):
        # Store normalized coordinates
        self.wrist_history.append((current_x, current_y))
        if len(self.wrist_history) < 10: return 0.0
        
        # Calculate bounding box of movement
        xs = [p[0] for p in self.wrist_history]
        ys = [p[1] for p in self.wrist_history]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        
        # A circle needs movement in both X and Y roughly equal
        movement_size = (width + height) / 2
        if movement_size < 0.05: return 0.0 # Not moving enough
        
        aspect_ratio = width / (height + 1e-6)
        
        # Perfect circle has aspect ratio ~1.0
        if 0.5 < aspect_ratio < 2.0:
            return min(1.0, movement_size * 5) # Score based on size
        return 0.0

    def draw_progress_bar(self, frame, label, progress, x, y, color=(0, 255, 0)):
        cv2.rectangle(frame, (x, y), (x + 200, y + 20), (50, 50, 50), -1)
        fill_width = int(progress * 200)
        cv2.rectangle(frame, (x, y), (x + fill_width, y + 20), color, -1)
        cv2.putText(frame, label, (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def update_logic(self, gesture, lms, dt):
        wrist = lms.landmark[0]
        
        # 1. PALM + CIRCLE (SELECT ALL)
        if gesture == "PALM":
            # Check for circular motion
            self.circle_score = self.detect_circular_motion(wrist.x, wrist.y)
            if self.circle_score > 0.5:
                self.timers["PALM_CIRCLE"] += dt
                if self.timers["PALM_CIRCLE"] > CONFIG["SELECT_ALL_HOLD"]:
                    return "SELECT ALL (Ctrl+A)", (0, 255, 255)
            else:
                self.timers["PALM_CIRCLE"] = max(0, self.timers["PALM_CIRCLE"] - dt)
        else:
            self.timers["PALM_CIRCLE"] = 0
            self.wrist_history.clear()

        # 2. FIST (DELETE)
        if gesture == "FIST":
            self.timers["FIST_HOLD"] += dt
            if self.timers["FIST_HOLD"] > CONFIG["DELETE_LOOPS"] * 0.5: 
                return "DELETE / CLOSE", (0, 0, 255)
        else:
            self.timers["FIST_HOLD"] = 0

        # 3. GEN_Z (LOCK SYSTEM)
        if gesture == "GEN_Z" or gesture == "LOCK": # Handling both names
            self.timers["LOCK"] += dt
            if self.timers["LOCK"] > CONFIG["LOCK_TIME"]:
                return "SYSTEM LOCK/UNLOCK", (255, 0, 255)
        else:
            self.timers["LOCK"] = 0

        # 4. VOLUME (SHAKA)
        if gesture == "VOLUME":
            # Just visualizing detection, volume is usually continuous
            return "VOLUME CONTROL ACTIVE", (255, 100, 0)
            
        # 5. RIGHT CLICK (Any Variant)
        if gesture == "RIGHT_CLICK":
            self.timers["RIGHT"] += dt
            if self.timers["RIGHT"] > CONFIG["RIGHT_HOLD_MIDDLE"]:
                return "RIGHT CLICK", (0, 100, 255)
        else:
            self.timers["RIGHT"] = 0

        return None, None

    def run(self):
        print("✈️ NEURO FLIGHT DECK V2")
        print("   -> Testing YOUR Specific Logic.")
        
        prev_time = time.time()
        
        while True:
            now = time.time()
            dt = now - prev_time
            prev_time = now

            frame = self.cam.read()
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            gesture = "NOISE"
            pinch_dist = 0
            
            # HUD Background
            cv2.rectangle(frame, (0, 0), (300, h), (30, 30, 30), -1)

            if results.multi_hand_landmarks:
                lms = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, lms, self.mp_hands.HAND_CONNECTIONS)
                
                # PREDICT
                gesture, debug = self.brain.predict(lms.landmark)
                pinch_dist = debug.get("pinch_dist", 0)

                # UPDATE SIMULATOR
                action, color = self.update_logic(gesture, lms, dt)
                
                if action:
                    self.triggered_action = action
                    self.action_display_timer = time.time()
                    if not self.event_log or self.event_log[-1] != action:
                        self.event_log.append(f"{time.strftime('%H:%M:%S')} - {action}")
                        if len(self.event_log) > 10: self.event_log.pop(0)

            # --- DRAW HUD ---
            y = 40
            cv2.putText(frame, "GESTURE MONITOR", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            y += 40
            
            # Current Gesture
            g_color = (0, 255, 0) if gesture != "NOISE" else (100, 100, 100)
            cv2.putText(frame, f"{gesture}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, g_color, 2)
            y += 30
            
            # 1. Palm Circle (Select All)
            palm_prog = min(1.0, self.timers["PALM_CIRCLE"] / CONFIG["SELECT_ALL_HOLD"])
            # Show Circle Score so you know if you are moving enough
            label = f"SELECT ALL (Circle: {int(self.circle_score*100)}%)"
            self.draw_progress_bar(frame, label, palm_prog, 10, y, (0, 255, 255))
            y += 40
            
            # 2. Delete (Fist)
            del_prog = min(1.0, self.timers["FIST_HOLD"] / (CONFIG["DELETE_LOOPS"] * 0.5))
            self.draw_progress_bar(frame, "DELETE (Hold Fist)", del_prog, 10, y, (0, 0, 255))
            y += 40

            # 3. Lock (Gen Z)
            lock_prog = min(1.0, self.timers["LOCK"] / CONFIG["LOCK_TIME"])
            self.draw_progress_bar(frame, "LOCK (Gen Z)", lock_prog, 10, y, (255, 0, 255))
            y += 40

            # 4. Right Click
            rc_prog = min(1.0, self.timers["RIGHT"] / CONFIG["RIGHT_HOLD_MIDDLE"])
            self.draw_progress_bar(frame, "RIGHT CLICK", rc_prog, 10, y, (0, 100, 255))
            y += 60

            # Action Flash
            if time.time() - self.action_display_timer < 1.0 and self.triggered_action:
                text_size = cv2.getTextSize(self.triggered_action, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                cx, cy = (w + 300)//2, h//2
                cv2.putText(frame, self.triggered_action, (cx - text_size[0]//2, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

            cv2.imshow("Neuro Flight Deck", frame)
            if cv2.waitKey(1) == 27: break

        self.cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = NeuroFlightDeck()
    app.run()