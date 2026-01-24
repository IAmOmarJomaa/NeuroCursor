import cv2
import mediapipe as mp
import math
import sys
import os
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.gesture_engine import NeuroCursorBrain
from src.config import CONFIG

def nothing(x): pass

def run_gesture_gym():
    print("🤸 GESTURE GYM: SHORTCUTS & COMBOS")
    print("   [S] Save & Print  |  [ESC] Exit")
    
    cv2.namedWindow("GestureGym")
    cv2.resizeWindow("GestureGym", 800, 800)

    # Load Config
    def_win = int(CONFIG.get("COPY_WINDOW", 0.30) * 100)
    def_c_time = int(CONFIG.get("SELECT_ALL_WINDOW", 1.0) * 100)
    def_c_size = int(CONFIG.get("SELECT_ALL_DIAMETER", 0.05) * 1000)
    def_vol = int(CONFIG.get("VOLUME_SENSITIVITY", 15.0))

    cv2.createTrackbar("COMBO", "GestureGym", def_win, 100, nothing)
    cv2.createTrackbar("C_TIME", "GestureGym", def_c_time, 200, nothing)
    cv2.createTrackbar("C_SIZE", "GestureGym", def_c_size, 200, nothing)
    cv2.createTrackbar("VOL", "GestureGym", def_vol, 50, nothing)

    brain = NeuroCursorBrain()
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, model_complexity=0)
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # State
    last_palm = 0
    last_fist = 0
    palm_path = []
    palm_start = 0
    
    msg = ""
    msg_timer = 0
    msg_color = (0, 255, 255)

    vol_sim = 50.0 
    vol_anchor = 0
    prev_gesture = "RESTING"

    while True:
        # Values
        val_win = cv2.getTrackbarPos("COMBO", "GestureGym") / 100.0
        val_c_time = cv2.getTrackbarPos("C_TIME", "GestureGym") / 100.0
        val_c_size = cv2.getTrackbarPos("C_SIZE", "GestureGym") / 1000.0
        val_vol = cv2.getTrackbarPos("VOL", "GestureGym")

        ret, frame = cam.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # --- DRAW LEGEND ---
        panel_h = 200
        cv2.rectangle(frame, (0, h-panel_h), (w, h), (30, 30, 30), -1)
        y_txt = h - panel_h + 30
        
        cv2.putText(frame, f"COMBO  ({val_win:.2f}s): Window for Copy (Palm->Fist) & Paste (Fist->Palm)", (20, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"C_TIME ({val_c_time:.2f}s): Circle Speed Limit.", (20, y_txt+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"C_SIZE ({val_c_size:.3f}): Circle Size Limit.", (20, y_txt+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"VOL    ({val_vol}): Volume Sensitivity.", (20, y_txt+90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.putText(frame, "TEST 1: Palm -> Fist (COPY) | Fist -> Palm (PASTE)", (20, y_txt+130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "TEST 2: Circle with PALM (SELECT ALL)", (20, y_txt+155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        curr_time = time.time()

        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            label, conf = brain.predict(lms.landmark)
            rx, ry = int(lms.landmark[8].x * w), int(lms.landmark[8].y * h)

            # --- LOGIC ---
            
            # 1. PALM LOGIC (Start Copy, Start Circle, Finish Paste)
            if label == "PALM":
                # Check Paste (Did we just fist?)
                if (curr_time - last_fist) < val_win:
                    msg = "PASTE DETECTED 📋"
                    msg_color = (0, 255, 0) # Green
                    msg_timer = curr_time + 1.0
                    last_fist = 0

                last_palm = curr_time
                
                # Circle Logic
                if prev_gesture != "PALM":
                    palm_path = []
                    palm_start = curr_time
                palm_path.append((rx, ry))
                
                # Check Circle
                if (curr_time - palm_start) < val_c_time:
                     # Draw Path
                    if len(palm_path) > 1:
                        pts = np.array(palm_path, np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], False, (0, 255, 255), 2)

                    if len(palm_path) > 10:
                        min_x, max_x = min(p[0] for p in palm_path), max(p[0] for p in palm_path)
                        width = (max_x - min_x) / w 
                        if width > val_c_size:
                            msg = "SELECT ALL DETECTED ⭕"
                            msg_color = (255, 0, 255) # Pink
                            msg_timer = curr_time + 1.0
                            palm_path = [] # Reset

            # 2. FIST LOGIC (Finish Copy, Start Paste)
            elif label == "FIST":
                # Check Copy (Did we just Palm?)
                if (curr_time - last_palm) < val_win:
                    msg = "COPY DETECTED 📑"
                    msg_color = (0, 165, 255) # Orange
                    msg_timer = curr_time + 1.0
                    last_palm = 0
                
                last_fist = curr_time
                palm_path = [] # Break circle

            # --- VOLUME TEST ---
            if label == "VOLUME":
                if prev_gesture != "VOLUME":
                    vol_anchor = lms.landmark[8].y
                
                delta = lms.landmark[8].y - vol_anchor
                if abs(delta) > CONFIG.get("VOLUME_DEADZONE", 0.04):
                    change = -delta * val_vol
                    vol_sim = np.clip(vol_sim + change, 0, 100)
                
                # Visuals
                cv2.rectangle(frame, (w-50, int(h - (vol_sim/100)*h)), (w-20, h), (255, 0, 255), -1)
                cv2.rectangle(frame, (w-50, 0), (w-20, h), (255, 255, 255), 1)

            prev_gesture = label
            
            # Draw Hand
            cv2.putText(frame, f"GESTURE: {label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw Message
            if curr_time < msg_timer:
                cv2.putText(frame, msg, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, msg_color, 3)

            # Timer Bar for Combo (Visual feedback for 'Window')
            # If recently palmed, show bar counting down for Copy
            if (curr_time - last_palm) < val_win:
                pct = 1.0 - ((curr_time - last_palm) / val_win)
                cv2.rectangle(frame, (20, 120), (20 + int(200*pct), 140), (0, 165, 255), -1)
                cv2.putText(frame, "COPY WINDOW", (230, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

            # If recently fisted, show bar counting down for Paste
            elif (curr_time - last_fist) < val_win:
                pct = 1.0 - ((curr_time - last_fist) / val_win)
                cv2.rectangle(frame, (20, 120), (20 + int(200*pct), 140), (0, 255, 0), -1)
                cv2.putText(frame, "PASTE WINDOW", (230, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        cv2.imshow("GestureGym", frame)
        k = cv2.waitKey(1)
        if k == 27: break
        if k == ord('s'):
            print("\n" + "="*40)
            print("👇 COPY TO CONFIG (GESTURES) 👇")
            print(f'    "COPY_WINDOW": {val_win:.2f},')
            print(f'    "SELECT_ALL_WINDOW": {val_c_time:.2f},')
            print(f'    "SELECT_ALL_DIAMETER": {val_c_size:.3f},')
            print(f'    "VOLUME_SENSITIVITY": {val_vol:.1f},')
            print("="*40 + "\n")

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_gesture_gym()