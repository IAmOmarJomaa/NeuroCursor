import cv2
import mediapipe as mp
import sys
import os
import time
import math
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import CONFIG
from src.core.kinematics import KinematicsEngine

def run_lab():
    print("⏱️ TIMING LAB (Layer 4)")
    print("   -> Tune Double Click & Drag Delays.")
    print("   -> PINCH to start tests.")
    
    cv2.namedWindow("Timing Lab", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Timing Lab", 800, 600)
    
    def nothing(x): pass
    
    cv2.createTrackbar("DRAG DELAY (ms)", "Timing Lab", int(CONFIG["DRAG_START_DELAY"]*1000), 1000, nothing)
    cv2.createTrackbar("DBL CLICK (ms)", "Timing Lab", int(CONFIG["DOUBLE_CLICK_GAP"]*1000), 1000, nothing)

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, model_complexity=1)
    
    physics = KinematicsEngine()
    
    # State
    pinch_start_time = 0
    last_click_time = 0
    is_pinching = False
    
    msg_log = []
    
    while True:
        # Config Update
        CONFIG["DRAG_START_DELAY"] = cv2.getTrackbarPos("DRAG DELAY (ms)", "Timing Lab") / 1000.0
        CONFIG["DOUBLE_CLICK_GAP"] = cv2.getTrackbarPos("DBL CLICK (ms)", "Timing Lab") / 1000.0
        
        ret, frame = cam.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            
            # Physics
            dist = physics.get_pinch_distance(lms)
            
            # Logic Simulation
            if dist < CONFIG["PINCH_START"]:
                if not is_pinching:
                    # START PINCH
                    is_pinching = True
                    pinch_start_time = time.time()
                
                # HELD STATE
                duration = time.time() - pinch_start_time
                
                # Draw Progress Bar for Drag
                progress = min(duration / CONFIG["DRAG_START_DELAY"], 1.0)
                
                # Visuals
                bar_w = 300
                fill = int(progress * bar_w)
                col = (0, 255, 255)
                if progress >= 1.0: col = (0, 255, 0)
                
                cx, cy = int(lms.landmark[8].x * w), int(lms.landmark[8].y * h)
                cv2.rectangle(frame, (cx-150, cy-60), (cx-150+bar_w, cy-40), (50,50,50), -1)
                cv2.rectangle(frame, (cx-150, cy-60), (cx-150+fill, cy-40), col, -1)
                
                label = "HOLDING..." if progress < 1.0 else "DRAG READY!"
                cv2.putText(frame, label, (cx-50, cy-70), 1, 0.8, col, 2)
                
            else:
                # RELEASE
                if is_pinching:
                    is_pinching = False
                    duration = time.time() - pinch_start_time
                    
                    if duration < CONFIG["DRAG_START_DELAY"]:
                        # IT WAS A CLICK
                        now = time.time()
                        if (now - last_click_time) < CONFIG["DOUBLE_CLICK_GAP"]:
                            msg_log.append(f"DOUBLE CLICK! ({now - last_click_time:.2f}s gap)")
                            last_click_time = 0 # Consume
                        else:
                            msg_log.append("SINGLE CLICK")
                            last_click_time = now
                    else:
                        msg_log.append("DRAG ENDED")
        
        # Draw Log
        y = 50
        cv2.putText(frame, "EVENT LOG:", (20, 30), 1, 1, (200,200,200), 2)
        for msg in msg_log[-5:]:
            col = (0, 255, 0) if "DOUBLE" in msg else ((0, 255, 255) if "DRAG" in msg else (255, 255, 255))
            cv2.putText(frame, msg, (20, y), 1, 0.7, col, 1)
            y += 30

        cv2.imshow("Timing Lab", frame)
        k = cv2.waitKey(1)
        if k == 27: break
        if k == ord('s'):
            print("\n" + "="*40)
            print("💾 CONFIG VALUES (TIMING):")
            print(f'    "DRAG_START_DELAY": {CONFIG["DRAG_START_DELAY"]:.2f},')
            print(f'    "DOUBLE_CLICK_GAP": {CONFIG["DOUBLE_CLICK_GAP"]:.2f},')
            print("="*40 + "\n")

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_lab()