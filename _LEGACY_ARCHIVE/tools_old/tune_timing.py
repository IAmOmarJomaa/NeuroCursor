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

def run_timing_dojo():
    print("⏱️ TIMING DOJO: CLICK & DRAG LOGIC")
    print("   [S] Save & Print  |  [ESC] Exit")
    
    cv2.namedWindow("TimingDojo")
    cv2.resizeWindow("TimingDojo", 600, 700) # Taller for legend

    # Load Config (defaults from your file)
    def_dead = int(CONFIG.get("CLICK_DEADZONE", 35))
    def_delay = int(CONFIG.get("DRAG_START_DELAY", 0.45) * 1000)
    def_dbl = int(CONFIG.get("DOUBLE_CLICK_GAP", 0.40) * 1000)
    def_buff = int(CONFIG.get("DRAG_BUFFER", 0.15) * 1000)

    cv2.createTrackbar("DEADZONE", "TimingDojo", def_dead, 100, nothing)
    cv2.createTrackbar("DELAY", "TimingDojo", def_delay, 1000, nothing)
    cv2.createTrackbar("DBL_GAP", "TimingDojo", def_dbl, 1000, nothing)
    cv2.createTrackbar("BUFFER", "TimingDojo", def_buff, 500, nothing)

    brain = NeuroCursorBrain()
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, model_complexity=0)
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Sim State
    is_pinching = False
    click_start_time = 0
    anchor = (0,0)
    state = "IDLE" # IDLE, WAITING, DRAGGING
    last_click_event = 0

    while True:
        # Values
        val_dead = cv2.getTrackbarPos("DEADZONE", "TimingDojo")
        val_delay = cv2.getTrackbarPos("DELAY", "TimingDojo") / 1000.0
        val_dbl = cv2.getTrackbarPos("DBL_GAP", "TimingDojo") / 1000.0
        val_buff = cv2.getTrackbarPos("BUFFER", "TimingDojo") / 1000.0
        
        ret, frame = cam.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # --- DRAW LEGEND PANEL ---
        panel_h = 220
        cv2.rectangle(frame, (0, h-panel_h), (w, h), (30, 30, 30), -1)
        y_txt = h - panel_h + 30
        
        cv2.putText(frame, f"DEADZONE ({val_dead}px): Radius to freeze cursor (Anti-Jitter).", (20, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"DELAY    ({val_delay:.2f}s): Hold this long to start DRAGGING.", (20, y_txt+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"DBL_GAP  ({val_dbl:.2f}s): Max time between taps for Double Click.", (20, y_txt+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"BUFFER   ({val_buff:.2f}s): Safety time to hold drag if you slip.", (20, y_txt+90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Instructions
        cv2.putText(frame, "TEST: Tap fast (Click) vs Hold (Drag)", (20, y_txt+140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            rx, ry = int(lms.landmark[8].x * w), int(lms.landmark[8].y * h)
            
            # Physics Check (Simple version for sim)
            # We use hardcoded thresholds just to trigger the TIMING logic simulation
            norm_dist = math.hypot(lms.landmark[8].x - lms.landmark[4].x, lms.landmark[8].y - lms.landmark[4].y)
            is_pinching = norm_dist < (CONFIG.get("PINCH_START", 0.05) if not is_pinching else CONFIG.get("PINCH_STOP", 0.10))

            # --- LOGIC SIMULATION ---
            curr_time = time.time()
            
            if is_pinching:
                if click_start_time == 0:
                    click_start_time = curr_time
                    anchor = (rx, ry)
                    state = "WAITING"
                
                # Drag Check
                held_time = curr_time - click_start_time
                move_dist = math.hypot(rx - anchor[0], ry - anchor[1])
                
                if state == "WAITING":
                    if held_time > val_delay:
                        state = "DRAGGING (Time)"
                    elif move_dist > val_dead:
                        state = "DRAGGING (Move)"
                        
                # VISUALS
                cv2.circle(frame, anchor, val_dead, (255, 0, 0), 2) # Deadzone
                cv2.line(frame, anchor, (rx, ry), (0, 255, 255), 1) # Tether
                
                # Progress Bar for Time Drag
                if state == "WAITING":
                    progress = min(held_time / val_delay, 1.0)
                    # Bar Background
                    cv2.rectangle(frame, (anchor[0]-30, anchor[1]-40), (anchor[0]+30, anchor[1]-30), (50,50,50), -1)
                    # Bar Fill
                    cv2.rectangle(frame, (anchor[0]-30, anchor[1]-40), (anchor[0]-30 + int(60*progress), anchor[1]-30), (0,255,255), -1)
                    cv2.putText(frame, "HOLD...", (anchor[0]-20, anchor[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            else:
                if click_start_time != 0:
                    # Released
                    if "DRAGGING" in state:
                        state = "DROP"
                    else:
                        # Click
                        if (curr_time - last_click_event) < val_dbl:
                            state = "DOUBLE CLICK 🔥"
                            last_click_event = 0
                        else:
                            state = "SINGLE CLICK ✅"
                            last_click_event = curr_time
                    click_start_time = 0

            # Draw Cursor
            col = (0, 255, 0) if "DRAGGING" in state else (0, 255, 255)
            cv2.circle(frame, (rx, ry), 8, col, -1)
            cv2.putText(frame, f"{state}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)
            
        cv2.imshow("TimingDojo", frame)
        k = cv2.waitKey(1)
        if k == 27: break
        if k == ord('s'):
            print("\n" + "="*40)
            print("👇 COPY TO CONFIG (TIMING) 👇")
            print(f'    "CLICK_DEADZONE": {val_dead},')
            print(f'    "DRAG_START_DELAY": {val_delay:.2f},')
            print(f'    "DOUBLE_CLICK_GAP": {val_dbl:.2f},')
            print(f'    "DRAG_BUFFER": {val_buff:.2f},')
            print("="*40 + "\n")

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_timing_dojo()