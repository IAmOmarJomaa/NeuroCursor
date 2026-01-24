"""
Dynamic Deadzone Lab V2.
Upgraded to use Kinematics V2 (Squared Math & Shared Logic).
"""
import cv2
import mediapipe as mp
import time
import math
import sys
import os
import numpy as np

# Setup Path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import CONFIG
from src.core.kinematics import KinematicsEngine, Point2D

def run_lab():
    print("🧪 DYNAMIC CLICK LAB V2")
    print("   -> Now uses Real Physics Engine logic.")
    print("   -> Green Ring = Dynamic Deadzone (The Bunker).")
    
    # 1. Setup Vision
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, model_complexity=1)
    
    # 2. Setup Real Engine
    physics = KinematicsEngine()
    
    # 3. Sliders
    window = "Dynamic Deadzone Lab"
    cv2.namedWindow(window)
    cv2.resizeWindow(window, 1000, 700)
    
    def nothing(x): pass
    
    # --- DYNAMIC DEADZONE SLIDERS ---
    # Max: Size when hand is still (e.g., 75px)
    cv2.createTrackbar("DZ MAX (px)", window, int(CONFIG["DEADZONE_MAX"]), 150, nothing)
    # Min: Size when hand is fast (e.g., 15px)
    cv2.createTrackbar("DZ MIN (px)", window, int(CONFIG["DEADZONE_MIN"]), 150, nothing)
    # Decay: How fast it shrinks (x1000 for precision)
    cv2.createTrackbar("DECAY (x1000)", window, int(CONFIG["DEADZONE_DECAY_VELOCITY"]*1000), 100, nothing)
    
    # Click Thresholds
    cv2.createTrackbar("PINCH START", window, int(CONFIG["PINCH_START"]*1000), 100, nothing)

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # Simulation State
    anchor = Point2D(0, 0)       
    virtual_cursor = Point2D(0, 0) 
    is_pinching = False
    is_dragging = False
    prev_x, prev_y = 0, 0
    
    while True:
        # A. Live Update Config
        CONFIG["DEADZONE_MAX"] = cv2.getTrackbarPos("DZ MAX (px)", window)
        CONFIG["DEADZONE_MIN"] = cv2.getTrackbarPos("DZ MIN (px)", window)
        decay_raw = cv2.getTrackbarPos("DECAY (x1000)", window)
        CONFIG["DEADZONE_DECAY_VELOCITY"] = decay_raw / 1000.0 if decay_raw > 0 else 0.001
        
        CONFIG["PINCH_START"] = cv2.getTrackbarPos("PINCH START", window) / 1000.0
        
        # B. Vision
        ret, frame = cam.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            
            # 1. Physics Inputs
            raw_x = lms.landmark[8].x
            raw_y = lms.landmark[8].y
            screen_x, screen_y = int(raw_x * w), int(raw_y * h)
            
            # Velocity Calculation
            curr_speed = math.hypot(raw_x - prev_x, raw_y - prev_y)
            prev_x, prev_y = raw_x, raw_y
            
            # [FIX] Use Squared Dist from Engine
            dist_sq = physics.get_pinch_sq_dist(lms)
            
            # 2. GET REAL DYNAMIC DEADZONE
            # This calls the exact code used in production
            dz_norm = physics.get_dynamic_deadzone(curr_speed)
            current_dz_px = int(dz_norm * 1920) # Lab visualization scale
            
            # 3. State Machine (Simplified for Lab)
            # We use square root here just for the boolean check in the lab UI
            # (In prod, we check dist_sq directly)
            dist_real = math.sqrt(dist_sq)
            
            if is_pinching: is_pinching = dist_real < (CONFIG["PINCH_START"] + 0.02)
            else: is_pinching = dist_real < CONFIG["PINCH_START"]
            
            # Reset anchor on new pinch
            if is_pinching and anchor.x == 0:
                anchor = Point2D(screen_x, screen_y)
                is_dragging = False
            
            if is_pinching:
                if not is_dragging:
                    # Check against DYNAMIC Deadzone
                    dist_from_anchor = math.hypot(screen_x - anchor.x, screen_y - anchor.y)
                    
                    if dist_from_anchor > current_dz_px:
                        is_dragging = True
                        virtual_cursor = Point2D(screen_x, screen_y)
                    else:
                        virtual_cursor = anchor 
                else:
                    virtual_cursor = Point2D(screen_x, screen_y)
            else:
                anchor = Point2D(0, 0)
                is_dragging = False
                virtual_cursor = Point2D(screen_x, screen_y)

            # --- VISUALIZATION ---
            
            # Hand
            cv2.circle(frame, (screen_x, screen_y), 5, (0, 255, 0), -1)
            
            # Speed Bar
            bar_w = 200
            fill = int((curr_speed / CONFIG["DEADZONE_DECAY_VELOCITY"]) * bar_w)
            fill = min(fill, bar_w)
            cv2.rectangle(frame, (20, 50), (20+bar_w, 70), (50,50,50), -1)
            cv2.rectangle(frame, (20, 50), (20+fill, 70), (0, 255, 255), -1)
            cv2.putText(frame, "SPEED -> SHRINK", (20, 45), 1, 1, (255,255,255), 1)

            # Dynamic Deadzone
            if anchor.x != 0:
                # Draw the Current Dynamic Size
                cv2.circle(frame, (int(anchor.x), int(anchor.y)), current_dz_px, (255, 0, 255), 2)
                
                # Draw Min/Max reference rings
                cv2.circle(frame, (int(anchor.x), int(anchor.y)), int(CONFIG["DEADZONE_MIN"]), (50, 50, 50), 1)
                cv2.circle(frame, (int(anchor.x), int(anchor.y)), int(CONFIG["DEADZONE_MAX"]), (50, 50, 50), 1)
                
                cv2.putText(frame, f"DZ: {current_dz_px}px", (int(anchor.x)+10, int(anchor.y)+current_dz_px+20), 
                           1, 1, (255, 0, 255), 1)

            # Virtual Cursor (Red Dot)
            col = (0, 0, 255) if is_pinching and not is_dragging else (0, 255, 0)
            cv2.circle(frame, (int(virtual_cursor.x), int(virtual_cursor.y)), 8, col, -1)

        cv2.imshow(window, frame)
        k = cv2.waitKey(1)
        if k == 27: break
        if k == ord('s'):
            print("\n" + "="*40)
            print("💾 CONFIG VALUES:")
            print(f'    "DEADZONE_MAX": {CONFIG["DEADZONE_MAX"]},')
            print(f'    "DEADZONE_MIN": {CONFIG["DEADZONE_MIN"]},')
            print(f'    "DEADZONE_DECAY_VELOCITY": {CONFIG["DEADZONE_DECAY_VELOCITY"]:.3f},')
            print("="*40 + "\n")

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_lab()