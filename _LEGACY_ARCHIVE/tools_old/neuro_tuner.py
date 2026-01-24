import cv2
import mediapipe as mp
import sys
import os
import math
import numpy as np

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.gesture_engine import NeuroCursorBrain
from src.config import CONFIG

def nothing(x): pass

def run_tuner():
    print("🎛️ NEURO-TUNER V7: PHYSICS ONLY")
    print("   -> Loaded values from src/config.py")
    print("   -> Press 'S' to save PHYSICS values (Box settings ignored).")
    
    cv2.namedWindow("NeuroTuner")
    cv2.resizeWindow("NeuroTuner", 600, 500)

    # 1. LOAD DEFAULTS FROM CONFIG
    def_dead      = int(CONFIG.get("CLICK_DEADZONE", 35))
    def_smooth    = int(CONFIG.get("SMOOTHING", 5.0) * 10)
    def_p_start   = int(CONFIG.get("PINCH_START", 0.05) * 1000)
    def_p_stop    = int(CONFIG.get("PINCH_STOP", 0.10) * 1000)

    # 2. CREATE SLIDERS (PHYSICS ONLY)
    cv2.createTrackbar("Deadzone (px)", "NeuroTuner", def_dead, 150, nothing)
    cv2.createTrackbar("Smoothing", "NeuroTuner", def_smooth, 100, nothing)
    cv2.createTrackbar("Click START", "NeuroTuner", def_p_start, 200, nothing)
    cv2.createTrackbar("Click MAINTAIN", "NeuroTuner", def_p_stop, 200, nothing)

    brain = NeuroCursorBrain()
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, model_complexity=0)
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # State
    anchor_raw = None
    is_dragging = False
    is_clicked_state = False
    
    while True:
        # 3. READ SLIDERS LIVE
        val_dead   = cv2.getTrackbarPos("Deadzone (px)", "NeuroTuner")
        val_smooth = cv2.getTrackbarPos("Smoothing", "NeuroTuner") / 10.0
        val_p_start= cv2.getTrackbarPos("Click START", "NeuroTuner") / 1000.0
        val_p_stop = cv2.getTrackbarPos("Click MAINTAIN", "NeuroTuner") / 1000.0
        
        # Sanity Checks
        if val_smooth < 1: val_smooth = 1.0
        if val_p_stop < val_p_start: val_p_stop = val_p_start

        ret, frame = cam.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            
            # Predict
            label, conf = brain.predict(lms.landmark)
            
            # --- CALCULATE PINCH ---
            raw_x = int(lms.landmark[8].x * w)
            raw_y = int(lms.landmark[8].y * h)
            t_x = int(lms.landmark[4].x * w)
            t_y = int(lms.landmark[4].y * h)
            
            norm_dist = math.hypot(lms.landmark[8].x - lms.landmark[4].x, lms.landmark[8].y - lms.landmark[4].y)
            
            # --- HYSTERESIS SIMULATION ---
            if is_clicked_state:
                if norm_dist < val_p_stop: is_clicked_state = True
                else: is_clicked_state = False
            else:
                if norm_dist < val_p_start: is_clicked_state = True
                else: is_clicked_state = False

            # --- VISUALIZATION ---
            # 1. Red Dot (Index Tip)
            cv2.circle(frame, (raw_x, raw_y), 6, (0, 0, 255), -1)
            
            # 2. Pinch Line
            line_col = (0, 255, 0) if is_clicked_state else (0, 255, 255)
            cv2.line(frame, (raw_x, raw_y), (t_x, t_y), line_col, 2)
            
            # 3. Deadzone Logic
            if is_clicked_state:
                if anchor_raw is None: anchor_raw = (raw_x, raw_y)
                
                # Draw Deadzone Circle
                cv2.circle(frame, anchor_raw, val_dead, (255, 0, 0), 2)
                
                dist_from_anchor = math.hypot(raw_x - anchor_raw[0], raw_y - anchor_raw[1])
                
                if dist_from_anchor > val_dead:
                    is_dragging = True
                    cv2.putText(frame, "DRAGGING", (anchor_raw[0], anchor_raw[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.line(frame, anchor_raw, (raw_x, raw_y), (0, 255, 0), 2)
                else:
                    is_dragging = False
                    cv2.putText(frame, "FROZEN", (anchor_raw[0], anchor_raw[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                anchor_raw = None
                is_dragging = False

            # Info
            status = "CLICKED" if is_clicked_state else "OPEN"
            cv2.putText(frame, f"STATUS: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_col, 2)
            cv2.putText(frame, f"Dist: {norm_dist:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
            cv2.putText(frame, f"Start: <{val_p_start:.2f} | Maintain: <{val_p_stop:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        cv2.imshow("NeuroTuner", frame)
        key = cv2.waitKey(1)
        if key == 27: break
        if key == ord('s'):
            print("\n" + "="*50)
            print("✅ NEW PHYSICS VALUES (Copy to src/config.py):")
            print("-" * 50)
            print(f'    "CLICK_DEADZONE": {val_dead},')
            print(f'    "SMOOTHING": {val_smooth},')
            print(f'    "PINCH_START": {val_p_start:.3f},')
            print(f'    "PINCH_STOP": {val_p_stop:.3f},')
            print("-" * 50)
            print("⚠️  NOTE: Box settings are NOT changed here. Adjust them in MAIN.")
            print("="*50 + "\n")

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tuner()