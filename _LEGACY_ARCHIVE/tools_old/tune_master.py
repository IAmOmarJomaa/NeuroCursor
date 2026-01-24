import cv2
import mediapipe as mp
import math
import numpy as np
import sys
import os

# Align with project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.gesture_engine import NeuroCursorBrain
from src.config import CONFIG

def nothing(x): pass

def run_tune_master():
    print("🔬 TUNE MASTER: Anchored Deadzone & Hysteresis Simulator")
    print("   [S] Save to Terminal  |  [ESC] Exit")
    
    cv2.namedWindow("TUNE_MASTER", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("TUNE_MASTER", 800, 800)

    # Trackbars mapped directly to CONFIG names
    cv2.createTrackbar("DEADZONE", "TUNE_MASTER", int(CONFIG["CLICK_DEADZONE"]), 200, nothing)
    cv2.createTrackbar("P_START", "TUNE_MASTER", int(CONFIG["PINCH_START"]*1000), 100, nothing)
    cv2.createTrackbar("P_STOP", "TUNE_MASTER", int(CONFIG["PINCH_STOP"]*1000), 150, nothing)
    cv2.createTrackbar("F_SENS", "TUNE_MASTER", int(CONFIG["CLICK_FREEZE_SENSITIVITY"]*1000), 100, nothing)

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, model_complexity=0)

    # State for the sim
    physical_anchor = None
    is_pinching = False

    while True:
        ret, frame = cam.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Get live trackbar values
        dz_px = cv2.getTrackbarPos("DEADZONE", "TUNE_MASTER")
        p_start = cv2.getTrackbarPos("P_START", "TUNE_MASTER") / 1000.0
        p_stop = cv2.getTrackbarPos("P_STOP", "TUNE_MASTER") / 1000.0
        f_sens = cv2.getTrackbarPos("F_SENS", "TUNE_MASTER") / 1000.0

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            # RAW index and thumb tip coordinates in pixels
            ix, iy = int(lms.landmark[8].x * w), int(lms.landmark[8].y * h)
            tx, ty = int(lms.landmark[4].x * w), int(lms.landmark[4].y * h)
            
            # Normalized distance for hysteresis check
            dist = math.hypot(lms.landmark[8].x - lms.landmark[4].x, 
                              lms.landmark[8].y - lms.landmark[4].y)

            # --- CLICK DETECTION SIM ---
            if dist < p_start and not is_pinching:
                is_pinching = True
                physical_anchor = (ix, iy) # Lock the deadzone to RAW index tip
            
            if dist > p_stop:
                is_pinching = False
                physical_anchor = None

            # --- POINT FREEZE PREDICTION ---
            # If distance is approaching p_start, show prediction highlight
            if not is_pinching and dist < (p_start + f_sens):
                cv2.circle(frame, (ix, iy), 15, (0, 165, 255), 2)
                cv2.putText(frame, "FREEZE ACTIVE", (ix+20, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            # --- VISUALIZATION ---
            # Draw real-time landmarks
            cv2.circle(frame, (ix, iy), 6, (0, 255, 0), -1)
            cv2.circle(frame, (tx, ty), 6, (255, 0, 0), -1)
            
            if physical_anchor:
                # Draw the static deadzone circle on the anchor
                cv2.circle(frame, physical_anchor, dz_px, (255, 0, 0), 2)
                # Line showing the index "dip" away from the anchor
                cv2.line(frame, physical_anchor, (ix, iy), (0, 255, 255), 1)
                dip_val = math.hypot(ix - physical_anchor[0], iy - physical_anchor[1])
                cv2.putText(frame, f"DIP: {int(dip_val)}px", (physical_anchor[0], physical_anchor[1]-dz_px-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Pinch Distance Bar
            bar_w = int(np.clip(dist * 1000, 0, 200))
            cv2.rectangle(frame, (20, h-40), (20+bar_w, h-20), (0, 255, 0) if is_pinching else (0, 0, 255), -1)
            cv2.putText(frame, f"PINCH DIST: {dist:.3f}", (20, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("TUNE_MASTER", frame)
        key = cv2.waitKey(1)
        if key == 27: break
        elif key == ord('s'):
            print("\n" + "="*40)
            print("📋 RECOMMENDED VALUES FOR CONFIG.PY:")
            print(f'   "CLICK_DEADZONE": {dz_px},')
            print(f'   "PINCH_START": {p_start:.3f},')
            print(f'   "PINCH_STOP": {p_stop:.3f},')
            print(f'   "CLICK_FREEZE_SENSITIVITY": {f_sens:.3f},')
            print("="*40 + "\n")

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tune_master()