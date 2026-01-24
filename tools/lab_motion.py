import cv2
import mediapipe as mp
import sys
import os
import math
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import CONFIG
from src.control.cursor_engine import CursorEngine

def run_lab():
    print("🌊 MOTION LAB (Layer 2)")
    print("   -> Tune 'Liquid Friction' logic.")
    print("   -> Adjust sliders to find the balance between Precision and Speed.")
    
    cv2.namedWindow("Motion Lab", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Motion Lab", 1000, 700)
    
    def nothing(x): pass
    
    # Sliders
    cv2.createTrackbar("GATE (x1000)", "Motion Lab", int(CONFIG["VELOCITY_GATE"]*1000), 20, nothing)
    cv2.createTrackbar("FRICTION LO (%)", "Motion Lab", int(CONFIG["FRICTION_LOW"]*100), 99, nothing)
    cv2.createTrackbar("FRICTION HI (%)", "Motion Lab", int(CONFIG["FRICTION_HIGH"]*100), 99, nothing)
    cv2.createTrackbar("BREAKOUT (x100)", "Motion Lab", int(CONFIG["BREAKOUT_VELOCITY"]*100), 10, nothing)

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, model_complexity=1)
    
    engine = CursorEngine()
    engine.screen_w = 1000 # Override for window size
    engine.screen_h = 700
    
    prev_raw_x, prev_raw_y = 0, 0
    
    while True:
        # Live Update Config
        CONFIG["VELOCITY_GATE"] = cv2.getTrackbarPos("GATE (x1000)", "Motion Lab") / 1000.0
        CONFIG["FRICTION_LOW"] = cv2.getTrackbarPos("FRICTION LO (%)", "Motion Lab") / 100.0
        CONFIG["FRICTION_HIGH"] = cv2.getTrackbarPos("FRICTION HI (%)", "Motion Lab") / 100.0
        CONFIG["BREAKOUT_VELOCITY"] = cv2.getTrackbarPos("BREAKOUT (x100)", "Motion Lab") / 100.0

        ret, frame = cam.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # Draw Workspace Box
        bx, by, bw, bh = engine.box_cx, engine.box_cy, engine.box_w, engine.box_h
        x1, y1 = int((bx - bw/2)*w), int((by - bh/2)*h)
        x2, y2 = int((bx + bw/2)*w), int((by + bh/2)*h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 50), 2)
        
        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            
            # Physics
            raw_x = lms.landmark[8].x
            raw_y = lms.landmark[8].y
            speed = math.hypot(raw_x - prev_raw_x, raw_y - prev_raw_y)
            prev_raw_x, prev_raw_y = raw_x, raw_y
            
            # Run Engine
            # Note: We simulate the screen mapping on the cv2 window itself
            tx, ty = engine.get_screen_coordinates(raw_x, raw_y)
            # Re-map back to window coords for visualization
            vis_tx = int((tx / 1920) * 1000) if engine.screen_w != 1000 else tx
            vis_ty = int((ty / 1080) * 700) if engine.screen_h != 700 else ty
            
            # Apply Smoothing
            sx, sy = engine.apply_smoothing(tx, ty, current_speed=speed)
            
            # Visualize
            # White X = Target (Raw Hand)
            cv2.line(frame, (tx-10, ty-10), (tx+10, ty+10), (255,255,255), 1)
            cv2.line(frame, (tx+10, ty-10), (tx-10, ty+10), (255,255,255), 1)
            
            # Red Circle = Virtual Cursor (Smoothed)
            cv2.circle(frame, (sx, sy), 8, (0, 0, 255), -1)
            cv2.line(frame, (tx, ty), (sx, sy), (100, 100, 100), 1) # Lag Line
            
            # Speedometer
            bar_w = 200
            fill = int((speed / 0.05) * bar_w)
            cv2.rectangle(frame, (20, 50), (20+bar_w, 70), (0,0,0), -1)
            cv2.rectangle(frame, (20, 50), (20+fill, 70), (0, 255, 255), -1)
            
            # Threshold Markers on Speedometer
            gate_x = 20 + int((CONFIG["VELOCITY_GATE"] / 0.05) * bar_w)
            break_x = 20 + int((CONFIG["BREAKOUT_VELOCITY"] / 0.05) * bar_w)
            
            cv2.line(frame, (gate_x, 45), (gate_x, 75), (0, 0, 255), 2) # Deadzone
            cv2.line(frame, (break_x, 45), (break_x, 75), (0, 255, 0), 2) # Breakout

            cv2.putText(frame, f"SPEED: {speed:.4f}", (20, 40), 1, 1, (255,255,255), 1)

        cv2.imshow("Motion Lab", frame)
        k = cv2.waitKey(1)
        if k == 27: break
        if k == ord('s'):
            print("\n" + "="*40)
            print("💾 CONFIG VALUES (MOTION):")
            print(f'    "VELOCITY_GATE": {CONFIG["VELOCITY_GATE"]:.3f},')
            print(f'    "BREAKOUT_VELOCITY": {CONFIG["BREAKOUT_VELOCITY"]:.3f},')
            print(f'    "FRICTION_LOW": {CONFIG["FRICTION_LOW"]:.2f},')
            print(f'    "FRICTION_HIGH": {CONFIG["FRICTION_HIGH"]:.2f},')
            print("="*40 + "\n")

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_lab()