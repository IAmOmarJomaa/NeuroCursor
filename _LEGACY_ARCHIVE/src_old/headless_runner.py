import cv2
import mediapipe as mp
import time
import pickle
import numpy as np
import pyautogui
import threading
import sys
import os
import pandas as pd
from pathlib import Path

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import PATHS

# --- HYPER PERFORMANCE CONFIG ---
# Disables fail-safes for maximum speed
pyautogui.PAUSE = 0
pyautogui.MINIMUM_DURATION = 0
pyautogui.FAILSAFE = False

# MOUSE SMOOTHING (Lower = Faster/Jittery, Higher = Smoother/Slower)
# 1.5 is the sweet spot for gaming-grade response
SMOOTHING = 2.0 

# LOAD THE 99% ACCURACY BRAIN
try:
    with open(PATHS["MODELS_DIR"] / "neurocursor_model.pkl", 'rb') as f:
        MODEL = pickle.load(f)
    print("🧠 BRAIN LOADED (99.8% Accuracy Mode)")
except:
    sys.exit("❌ CRITICAL: Model not found. Run pipeline/04_train_model.py")

# FEATURE NAMES (Must match training)
FEATURE_COLUMNS = [
    'thumb_bend', 'index_bend', 'mid_bend', 'ring_bend', 'pinky_bend',
    'pinch_dist', 'thumb_spread', 'mid_palm_dist', 
    'orientation_y', 'orientation_x'
]

class FeatureExtractor:
    """The Golden Formula Engine (Pure Math)"""
    @staticmethod
    def extract(lms):
        # Optimized Numpy Math for speed
        coords = np.array([[lm.x, lm.y, lm.z] for lm in lms])
        
        def get_ang(i1, i2, i3):
            v1 = coords[i2] - coords[i1]
            v2 = coords[i3] - coords[i2]
            dot = np.dot(v1, v2)
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            return np.degrees(np.arccos(np.clip(dot/(norm+1e-6), -1.0, 1.0)))

        def get_dist(i1, i2, norm_factor):
            return np.linalg.norm(coords[i1] - coords[i2]) / norm_factor

        palm_size = np.linalg.norm(coords[9] - coords[0]) + 1e-6
        
        # Calculate only what we need
        data = {
            'thumb_bend': get_ang(1, 2, 3),
            'index_bend': get_ang(5, 6, 7),
            'mid_bend':   get_ang(9, 10, 11),
            'ring_bend':  get_ang(13, 14, 15),
            'pinky_bend': get_ang(17, 18, 19),
            'pinch_dist': get_dist(4, 8, palm_size),
            'thumb_spread': get_dist(4, 5, palm_size),
            'mid_palm_dist': get_dist(12, 0, palm_size),
            'orientation_y': coords[9][1] - coords[0][1],
            'orientation_x': coords[9][0] - coords[0][0]
        }
        return pd.DataFrame([data], columns=FEATURE_COLUMNS)

class NeuroCore:
    def __init__(self):
        # 1. CAMERA SETUP (Reduced Resolution = 3x Faster Processing)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        
        # 2. MEDIAPIPE LITE (The fastest model available)
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1, # Tracking 1 hand is much faster
            model_complexity=0, # 0=Lite, 1=Full. 0 is sufficient for cursor.
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.screen_w, self.screen_h = pyautogui.size()
        self.running = True
        self.locked = True
        
        # Physics State
        self.curr_x, self.curr_y = 0, 0
        self.hold_timer = 0
        self.scroll_anchor = None
        self.wave_energy = 0
        self.last_x = 0

    def run(self):
        print("🚀 NEUROCURSOR HEADLESS (NO UI)")
        print("   -> Latency Optimized.")
        print("   -> Waiting for UNLOCK gesture (Gen Z Heart)...")
        
        while self.running:
            # 1. READ (Blocking call, but fast at 640x480)
            ret, frame = self.cap.read()
            if not ret: continue

            # 2. PROCESS
            # We skip cv2.flip/cvtColor if possible? No, MP needs RGB.
            # But we DON'T create a display buffer.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            if results.multi_hand_landmarks:
                # We only process the FIRST hand for maximum speed
                lms = results.multi_hand_landmarks[0]
                
                # 3. EXTRACT & PREDICT
                feats = FeatureExtractor.extract(lms.landmark)
                gesture = MODEL.predict(feats)[0]
                
                # 4. EXECUTE
                self.execute(gesture, lms.landmark)
            
            # NO cv2.imshow() -> This saves ~15ms per frame!
            
            # Emergency Escape Key check (requires window focus usually, 
            # so we might rely on the terminal Ctrl+C or the Wave Gesture)

    def execute(self, gesture, lms):
        # --- GLOBAL LOCK ---
        if gesture == "LOCK":
            self.hold_timer += 1
            if self.hold_timer > 10:
                self.locked = not self.locked
                self.hold_timer = 0
                state = "LOCKED" if self.locked else "ACTIVE"
                print(f"🔒 SYSTEM STATE: {state}")
                time.sleep(0.5)
            return

        # --- LOCKED LOGIC (Wave to Quit) ---
        if self.locked:
            cx = lms[9].x
            speed = abs(cx - self.last_x)
            if gesture in ["NOISE", "PALM_OPEN", "SHAKA"] and speed > 0.03:
                self.wave_energy += 1
            else:
                self.wave_energy = max(0, self.wave_energy - 1)
            
            self.last_x = cx
            if self.wave_energy > 20:
                print("👋 WAVE DETECTED. SHUTTING DOWN.")
                self.running = False
                sys.exit()
            return

        # --- ACTIVE LOGIC ---
        
        # 1. POINTER (The Bread & Butter)
        if gesture == "POINTER":
            # Map coordinates (Mirroring math inline)
            target_x = (1.0 - lms[8].x) * self.screen_w
            target_y = lms[8].y * self.screen_h
            
            # Fast Smoothing
            self.curr_x += (target_x - self.curr_x) / SMOOTHING
            self.curr_y += (target_y - self.curr_y) / SMOOTHING
            
            pyautogui.moveTo(self.curr_x, self.curr_y, _pause=False)
            self.hold_timer = 0

        # 2. CLICK & DRAG
        elif gesture == "PINCH":
            target_x = (1.0 - lms[8].x) * self.screen_w
            target_y = lms[8].y * self.screen_h
            
            # Follow hand while pinching
            pyautogui.moveTo(target_x, target_y, _pause=False)
            
            self.hold_timer += 1
            if self.hold_timer == 2: # Trigger on 2nd frame to prevent misfires
                pyautogui.mouseDown()

        # 3. SCROLL
        elif gesture == "SCROLL":
            curr_y = lms[9].y
            if self.scroll_anchor is None: self.scroll_anchor = curr_y
            
            delta = (self.scroll_anchor - curr_y) * 150
            if abs(delta) > 5:
                pyautogui.scroll(int(delta))

        # 4. CLICK TRIGGERS
        elif gesture == "RIGHT_CLICK" and self.hold_timer == 0:
            pyautogui.rightClick()
            self.hold_timer = 15 # Cooldown frames
            print("🖱️ RIGHT CLICK")

        # RESET LOGIC
        if gesture != "PINCH":
            if self.hold_timer > 2: pyautogui.mouseUp()
            if gesture != "SCROLL": self.scroll_anchor = None
            if gesture not in ["RIGHT_CLICK", "ALT_TAB", "LOCK"]:
                self.hold_timer = 0

if __name__ == "__main__":
    NeuroCore().run()