import cv2
import mediapipe as mp
import sys
import os
import time
import math
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import CONFIG
from src.gesture_engine import NeuroCursorBrain
from src.core.kinematics import KinematicsEngine

def run_lab():
    print("⚡ SHORTCUT LAB (Real Logic V2)")
    print("   -> Includes 'Dirty Flag' fix (No Copy->Paste loop).")
    print("   -> 'S' to save config.")
    
    cv2.namedWindow("Shortcut Lab", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Shortcut Lab", 1000, 700)
    
    def nothing(x): pass
    
    # Sliders
    cv2.createTrackbar("COOLDOWN (x100)", "Shortcut Lab", int(CONFIG["SHORTCUT_COOLDOWN"]*100), 500, nothing)
    cv2.createTrackbar("COPY WIN (ms)", "Shortcut Lab", int(CONFIG["COPY_WINDOW"]*1000), 2000, nothing)
    cv2.createTrackbar("CIRCLE DIA (x100)", "Shortcut Lab", int(CONFIG["SELECT_ALL_DIAMETER"]*1000), 500, nothing)
    cv2.createTrackbar("CIRCLE TIME (ms)", "Shortcut Lab", int(CONFIG["SELECT_ALL_WINDOW"]*1000), 3000, nothing)

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, model_complexity=1)
    
    brain = NeuroCursorBrain()
    physics = KinematicsEngine()
    
    # Logic State
    last_fist_time = 0
    palm_start_time = 0
    palm_path = []
    
    # [NEW] The Dirty Flag (Mirrors main.py)
    fist_handled = False 
    
    # Cooldown Timers
    last_copy_time = 0
    last_paste_time = 0
    last_select_time = 0
    
    msg_log = []
    
    while True:
        # Live Config Update
        CONFIG["SHORTCUT_COOLDOWN"] = cv2.getTrackbarPos("COOLDOWN (x100)", "Shortcut Lab") / 100.0
        CONFIG["COPY_WINDOW"] = cv2.getTrackbarPos("COPY WIN (ms)", "Shortcut Lab") / 1000.0
        CONFIG["SELECT_ALL_DIAMETER"] = cv2.getTrackbarPos("CIRCLE DIA (x100)", "Shortcut Lab") / 1000.0
        CONFIG["SELECT_ALL_WINDOW"] = cv2.getTrackbarPos("CIRCLE TIME (ms)", "Shortcut Lab") / 1000.0
        
        ret, frame = cam.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        curr_time = time.time()
        
        # --- DRAW COOLDOWN STATUS ---
        def draw_timer(label, last_time, y_pos):
            elapsed = curr_time - last_time
            cooldown = CONFIG["SHORTCUT_COOLDOWN"]
            remaining = max(0, cooldown - elapsed)
            
            # Color: Red if cooling down, Green if ready
            col = (0, 0, 255) if remaining > 0 else (0, 255, 0)
            status = f"{remaining:.1f}s" if remaining > 0 else "READY"
            
            cv2.putText(frame, f"{label}: {status}", (w-250, y_pos), 1, 1, col, 2)
            
            # Bar
            if remaining > 0:
                bar_w = 200
                fill = int((remaining / cooldown) * bar_w)
                cv2.rectangle(frame, (w-250, y_pos+10), (w-250+fill, y_pos+20), col, -1)

        draw_timer("COPY", last_copy_time, 50)
        draw_timer("PASTE", last_paste_time, 100)
        draw_timer("SELECT", last_select_time, 150)

        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            label, _ = brain.predict(lms.landmark)
            
            raw_x = lms.landmark[9].x
            raw_y = lms.landmark[9].y
            
            # --- LOGIC SIMULATION ---
            
            # 1. Reset Flag on New Fist Entry
            # In simulation, we check if label changed to FIST
            if "FIST" in label:
                if last_fist_time == 0: # Simulating "Just Entered"
                     pass 
            
            if "FIST" in label:
                # If this is a new fist event (simplification for lab)
                if (curr_time - last_fist_time) > 0.5: # Debounce for lab purposes
                    fist_handled = False 

                last_fist_time = curr_time
                palm_path = []
                
                # SIMULATE COPY (PALM -> FIST)
                # Check Window
                if (curr_time - palm_start_time) < CONFIG["COPY_WINDOW"]: 
                     # [FIX] Check Flag
                     if not fist_handled:
                         # Check Cooldown
                         if (curr_time - last_copy_time) > CONFIG["SHORTCUT_COOLDOWN"]:
                             msg_log.append("✅ COPY!")
                             last_copy_time = curr_time
                             fist_handled = True # [FIX] Mark as Used
                         else:
                             msg_log.append("❄️ COPY BLOCKED (Cooldown)")

            elif "PALM" in label:
                palm_start_time = curr_time
                
                # SIMULATE PASTE (FIST -> PALM)
                if (curr_time - last_fist_time) < CONFIG["PASTE_WINDOW"]:
                    # [FIX] Check Flag - Only Paste if Fist wasn't used for Copy
                    if not fist_handled:
                        if (curr_time - last_paste_time) > CONFIG["SHORTCUT_COOLDOWN"]:
                            msg_log.append("📋 PASTE!")
                            last_paste_time = curr_time
                            fist_handled = True # Mark as Used
                        else:
                            msg_log.append("❄️ PASTE BLOCKED (Cooldown)")
                    else:
                        # Optional: Log that it was blocked by logic
                        # msg_log.append("🚫 PASTE IGNORED (Loop Fix)")
                        pass
                
                # SIMULATE SELECT ALL (Circle)
                palm_path.append((raw_x, raw_y))
                if len(palm_path) > 60: palm_path.pop(0)
                
                # Visualize Path
                path_col = (0, 255, 255) # Yellow = Drawing
                
                is_circle = physics.check_full_circle(palm_path)
                
                if is_circle:
                    path_col = (0, 255, 0) # Green = Valid
                    if (curr_time - last_select_time) > CONFIG["SHORTCUT_COOLDOWN"]:
                        msg_log.append("⭕ SELECT ALL!")
                        last_select_time = curr_time
                        palm_path = [] # Consume
                    else:
                         path_col = (0, 0, 255) # Red = Cooldown
                
                for i in range(1, len(palm_path)):
                    p1 = (int(palm_path[i-1][0]*w), int(palm_path[i-1][1]*h))
                    p2 = (int(palm_path[i][0]*w), int(palm_path[i][1]*h))
                    cv2.line(frame, p1, p2, path_col, 2)

            else:
                 pass

            # HUD
            wrist = lms.landmark[0]
            cx, cy = int(wrist.x * w), int(wrist.y * h)
            cv2.putText(frame, label, (cx, cy-30), 1, 1, (255, 255, 0), 2)
            
            # Show Flag State
            flag_status = "USED" if fist_handled else "FRESH"
            flag_col = (0,0,255) if fist_handled else (0,255,0)
            cv2.putText(frame, f"FIST: {flag_status}", (20, 150), 1, 1, flag_col, 2)

        # Log
        y = h - 150
        cv2.putText(frame, "EVENT LOG:", (20, y-30), 1, 1, (200,200,200), 2)
        for msg in msg_log[-5:]:
            col = (255, 255, 255)
            if "BLOCKED" in msg: col = (100, 100, 255)
            if "✅" in msg or "📋" in msg or "⭕" in msg: col = (0, 255, 0)
            cv2.putText(frame, msg, (20, y), 1, 0.7, col, 1)
            y += 25

        cv2.imshow("Shortcut Lab", frame)
        k = cv2.waitKey(1)
        if k == 27: break
        if k == ord('s'):
            print("\n" + "="*40)
            print("💾 CONFIG VALUES (SHORTCUTS):")
            print(f'    "SHORTCUT_COOLDOWN": {CONFIG["SHORTCUT_COOLDOWN"]:.2f},')
            print(f'    "COPY_WINDOW": {CONFIG["COPY_WINDOW"]:.2f},')
            print(f'    "SELECT_ALL_DIAMETER": {CONFIG["SELECT_ALL_DIAMETER"]:.3f},')
            print(f'    "SELECT_ALL_WINDOW": {CONFIG["SELECT_ALL_WINDOW"]:.2f},')
            print("="*40 + "\n")

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_lab()