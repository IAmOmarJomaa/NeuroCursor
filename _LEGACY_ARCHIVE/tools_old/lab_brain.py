import cv2
import numpy as np
import math
import sys, os
from collections import deque
import mediapipe as mp

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.gesture_engine import NeuroCursorBrain
from src.config import CONFIG

def nothing(x): pass

def run_brain_lab():
    # Setup Window
    WINDOW_W, WINDOW_H = 1024, 768
    window_name = "Brain Lab: Input Stability"
    cv2.namedWindow(window_name)

    # 1. BRAIN CONTROLS
    # Buffer: 1 (raw) to 10 (very heavy)
    cv2.createTrackbar("BUFFER (Size)", window_name, CONFIG.get("SKELETON_BUFFER_SIZE", 5), 10, nothing)
    # Consensus: 0 = Average, 1 = Median (Median is usually superior for jitter)
    cv2.createTrackbar("USE MEDIAN", window_name, 1 if CONFIG.get("USE_MEDIAN_CONSENSUS", True) else 0, 1, nothing)
    # Outlier: Pixel jump limit
    cv2.createTrackbar("OUTLIER (Px)", window_name, CONFIG.get("OUTLIER_REJECTION_PX", 80), 200, nothing)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    brain = NeuroCursorBrain()
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, model_complexity=0)

    print("🧠 BRAIN LAB STARTED: Focus on the skeleton stability.")

    while True:
        # Update Config Live
        new_size = max(1, cv2.getTrackbarPos("BUFFER (Size)", window_name))
        CONFIG["SKELETON_BUFFER_SIZE"] = new_size
        CONFIG["USE_MEDIAN_CONSENSUS"] = True if cv2.getTrackbarPos("USE MEDIAN", window_name) == 1 else False
        CONFIG["OUTLIER_REJECTION_PX"] = cv2.getTrackbarPos("OUTLIER (Px)", window_name)
        
        # Sync Brain buffer size
        if brain.buffer_size != new_size:
            brain.buffer_size = new_size
            brain.skeleton_buffer = deque(maxlen=new_size)

        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WINDOW_W, WINDOW_H))
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            
            # --- 1. RAW SKELETON (Red - What the camera sees) ---
            raw_ix = int(lms.landmark[8].x * WINDOW_W)
            raw_iy = int(lms.landmark[8].y * WINDOW_H)
            cv2.circle(frame, (raw_ix, raw_iy), 4, (0, 0, 255), -1)

            # --- 2. STABILIZED SKELETON (Green - The Brain's output) ---
            # We use the internal brain method to get the filtered coords
            # Pass width/height for outlier logic
            stable_coords = brain._get_consensus_skeleton(lms.landmark)
            
            # Index Tip is at index 8 of the skeleton
            stab_ix = int(stable_skeleton[8][0] * WINDOW_W)
            stab_iy = int(stable_skeleton[8][1] * WINDOW_H)
            
            # Draw stabilized point
            cv2.circle(frame, (stab_ix, stab_iy), 8, (0, 255, 0), -1)
            
            # Draw the 'jitter range' (The gap the brain is fixing)
            cv2.line(frame, (raw_ix, raw_iy), (stab_ix, stab_iy), (255, 255, 255), 1)

            # --- 3. METRICS ---
            error = math.hypot(raw_ix - stab_ix, raw_iy - stab_iy)
            cv2.putText(frame, f"JITTER SUPPRESSED: {error:.1f}px", (20, WINDOW_H - 40), 1, 1.5, (0, 255, 0), 2)

        # Instructions
        cv2.putText(frame, "RED = Raw Shaky Hand", (20, 30), 1, 1, (0, 0, 255), 1)
        cv2.putText(frame, "GREEN = Stabilized Brain Output", (20, 60), 1, 1, (0, 255, 0), 1)
        cv2.putText(frame, "Goal: Make GREEN stay still while RED shakes.", (20, 90), 1, 1, (200, 200, 200), 1)

        cv2.imshow(window_name, frame)
        k = cv2.waitKey(1)
        if k == 27: break
        elif k == ord('s'):
            print(f"\n✅ BRAIN SETTINGS SAVED:")
            print(f'   "SKELETON_BUFFER_SIZE": {CONFIG["SKELETON_BUFFER_SIZE"]},')
            print(f'   "USE_MEDIAN_CONSENSUS": {CONFIG["USE_MEDIAN_CONSENSUS"]},')
            print(f'   "OUTLIER_REJECTION_PX": {CONFIG["OUTLIER_REJECTION_PX"]}')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_brain_lab()