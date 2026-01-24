import cv2
import mediapipe as mp
import sys
import os
import time
import math
import pandas as pd
from datetime import datetime

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.gesture_engine import NeuroCursorBrain

def run_profiler():
    print("🕵️ DRAG PROFILER: THE BLACK BOX RECORDER")
    print("---------------------------------------")
    print("TASK:   Grab the Red Box and hold it.")
    print("GOAL:   Hold for 5 seconds.")
    print("RESULT: When it drops, I will tell you WHY.")
    print("---------------------------------------")
    
    # 1. Init Systems
    brain = NeuroCursorBrain()
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, model_complexity=0)
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # 2. State Recording
    history = []
    is_dragging = False
    drag_start_time = 0
    box_pos = [300, 300]
    box_size = 60
    
    # Deadzone Logic (Simulated)
    anchor = None
    deadzone = 30 
    
    while True:
        frame_start = time.time()
        ret, frame = cam.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture = "NONE"
        conf = 0.0
        
        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            
            # RAW DATA COLLECTION
            gesture, conf = brain.predict(lms.landmark)
            
            # Physics Calculation
            raw_x = int(lms.landmark[8].x * w)
            raw_y = int(lms.landmark[8].y * h)
            
            # --- THE LOGIC LOOP ---
            if gesture == "THE_CLICK":
                if not is_dragging:
                    # START ATTEMPT
                    if anchor is None: anchor = (raw_x, raw_y)
                    dist = math.hypot(raw_x - anchor[0], raw_y - anchor[1])
                    
                    cv2.circle(frame, anchor, deadzone, (255, 0, 0), 2)
                    
                    if dist > deadzone:
                        print("\n🟢 DRAG STARTED!")
                        is_dragging = True
                        drag_start_time = time.time()
                        history = [] # Clear history for this run
                
                else:
                    # MAINTAIN DRAG
                    cv2.line(frame, anchor, (raw_x, raw_y), (0, 255, 0), 2)
                    duration = time.time() - drag_start_time
                    
                    # Log Health
                    history.append({
                        "time": duration,
                        "label": gesture,
                        "conf": conf,
                        "status": "HEALTHY"
                    })
                    
                    cv2.putText(frame, f"HOLDING: {duration:.2f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            else:
                # GESTURE IS NOT 'THE_CLICK'
                if is_dragging:
                    # !!! DEATH DETECTED !!!
                    print("🔴 DRAG DROPPED!")
                    duration = time.time() - drag_start_time
                    
                    # Log Death
                    history.append({
                        "time": duration,
                        "label": gesture,
                        "conf": conf,
                        "status": "DEATH"
                    })
                    
                    # --- GENERATE AUTOPSY REPORT ---
                    print("\n" + "="*40)
                    print(f"💀 AUTOPSY REPORT (Duration: {duration:.2f}s)")
                    print("="*40)
                    
                    # Look at the last 5 frames
                    print("Frame History (Last 5 frames before death):")
                    recent = history[-5:]
                    for entry in recent:
                        print(f"   T+{entry['time']:.2f}s | {entry['label']} ({int(entry['conf']*100)}%)")
                        
                    print("-" * 40)
                    print(f"CAUSE OF DEATH: Switched to '{gesture}' ({int(conf*100)}%)")
                    print("="*40 + "\n")
                    
                    is_dragging = False
                    anchor = None
                    
                    # Wait so user can read console
                    cv2.putText(frame, "DRAG DIED! CHECK CONSOLE", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.imshow("Drag Profiler", frame)
                    cv2.waitKey(2000) # Freeze for 2s
                
                else:
                    anchor = None

            # Visualize Cursor
            col = (0, 255, 0) if is_dragging else (0, 255, 255)
            cv2.circle(frame, (raw_x, raw_y), 8, col, -1)
            cv2.putText(frame, f"{gesture} {int(conf*100)}%", (raw_x+15, raw_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        # FPS Calc
        fps = 1.0 / (time.time() - frame_start)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Drag Profiler", frame)
        if cv2.waitKey(1) == 27: break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_profiler()