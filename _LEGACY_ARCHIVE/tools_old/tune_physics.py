import cv2
import mediapipe as mp
import math
import sys
import os
import time
import numpy as np

# Load YOUR Config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.gesture_engine import NeuroCursorBrain
from src.config import CONFIG

def nothing(x): pass

def run_physics_lab():
    print("🧪 PHYSICS LAB: TUNING HYSTERESIS & SPEED")
    print("   [S] Save & Print  |  [ESC] Exit")
    
    cv2.namedWindow("PhysicsLab")
    cv2.resizeWindow("PhysicsLab", 600, 650) # Taller window for instructions

    # 1. Load Current Config (Short Names)
    # Scale: 0.054 -> 54
    cv2.createTrackbar("START", "PhysicsLab", int(CONFIG["PINCH_START"]*1000), 150, nothing)
    cv2.createTrackbar("STOP", "PhysicsLab", int(CONFIG["PINCH_STOP"]*1000), 150, nothing)
    cv2.createTrackbar("DYN", "PhysicsLab", int(CONFIG["PINCH_DYNAMIC_SCALE"]*100), 100, nothing)
    cv2.createTrackbar("MAX", "PhysicsLab", int(CONFIG.get("PINCH_STOP_MAX", 0.25)*1000), 300, nothing)

    brain = NeuroCursorBrain()
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, model_complexity=0)
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    prev_x, prev_y = 0, 0
    is_pinching = False

    while True:
        # Live Values
        p_start = cv2.getTrackbarPos("START", "PhysicsLab") / 1000.0
        p_stop  = cv2.getTrackbarPos("STOP", "PhysicsLab") / 1000.0
        p_scale = cv2.getTrackbarPos("DYN", "PhysicsLab") / 100.0
        p_max   = cv2.getTrackbarPos("MAX", "PhysicsLab") / 1000.0

        if p_stop < p_start: p_stop = p_start # Sanity check

        ret, frame = cam.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # --- DRAW INTERFACE (The Legend) ---
        # Dark panel at the bottom
        panel_h = 220
        cv2.rectangle(frame, (0, h-panel_h), (w, h), (30, 30, 30), -1)
        
        y_txt = h - panel_h + 30
        cv2.putText(frame, f"START ({p_start:.3f}): Hardness to CLICK. Lower = Harder.", (20, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"STOP  ({p_stop:.3f}): Relax to RELEASE. Higher = Easier Hold.", (20, y_txt+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"DYN   ({p_scale:.2f}): Speed Boost. Higher = Sticky Fast Drag.", (20, y_txt+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"MAX   ({p_max:.3f}): Max Opening allowed at full speed.", (20, y_txt+90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            
            # 1. Calculate Speed
            raw_x = lms.landmark[8].x
            raw_y = lms.landmark[8].y
            speed = math.hypot(raw_x - prev_x, raw_y - prev_y)
            prev_x, prev_y = raw_x, raw_y

            # 2. Calculate Dynamic Threshold
            dynamic_boost = speed * p_scale
            current_stop_thresh = min(p_stop + dynamic_boost, p_max)

            # 3. Calculate Pinch
            tx, ty = lms.landmark[4].x, lms.landmark[4].y
            pinch_dist = math.hypot(raw_x - tx, raw_y - ty)

            # 4. State Logic
            if is_pinching:
                is_pinching = (pinch_dist < current_stop_thresh)
            else:
                is_pinching = (pinch_dist < p_start)

            # --- VISUALIZATION ---
            # Pinch Bar (Actual distance)
            bar_len = int(np.clip(pinch_dist * 1000, 0, 400))
            col = (0, 255, 0) if is_pinching else (0, 0, 255)
            cv2.rectangle(frame, (50, 100), (50+bar_len, 130), col, -1)
            cv2.putText(frame, f"PINCH: {pinch_dist:.3f}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

            # Threshold Marker (The moving target)
            thresh_len = int(np.clip((current_stop_thresh if is_pinching else p_start) * 1000, 0, 400))
            cv2.line(frame, (50+thresh_len, 80), (50+thresh_len, 150), (255, 255, 0), 3)
            cv2.putText(frame, "THRESH", (50+thresh_len-20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Speed Indicator
            cv2.putText(frame, f"SPEED: {speed:.3f}", (400, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            status = "HOLDING" if (is_pinching and pinch_dist > p_start) else ("CLICKED" if is_pinching else "OPEN")
            if is_pinching:
                if pinch_dist > p_start: status += " (Dynamic Boost Active)"
            
            cv2.putText(frame, status, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

        cv2.imshow("PhysicsLab", frame)
        k = cv2.waitKey(1)
        if k == 27: break
        if k == ord('s'):
            print("\n" + "="*40)
            print("👇 COPY TO CONFIG (CLICK PHYSICS) 👇")
            print(f'    "PINCH_START": {p_start:.3f},')
            print(f'    "PINCH_STOP": {p_stop:.3f},')
            print(f'    "PINCH_DYNAMIC_SCALE": {p_scale:.2f},')
            print(f'    "PINCH_STOP_MAX": {p_max:.3f},')
            print("="*40 + "\n")

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_physics_lab()