import cv2, mediapipe as mp, math, numpy as np, sys, os
from collections import deque
import win32api, win32con # To get real screen size

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.gesture_engine import NeuroCursorBrain
from src.control.cursor_engine import CursorEngine
from src.config import CONFIG

def nothing(x): pass

def run_stabilizer_dojo():
    print("🥋 STABILIZER DOJO V3: Real-Scale Tuning")
    print("   [S] Save Config | [ESC] Exit")
    
    # Get Real Screen Resolution for accurate simulation
    SCREEN_W = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
    SCREEN_H = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
    
    cv2.namedWindow("StabilizerDojo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("StabilizerDojo", SCREEN_W, SCREEN_H) 
    # Try to set fullscreen to match real cursor feel
    cv2.setWindowProperty("StabilizerDojo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # --- THE KNOBS (Explained) ---
    # 1. BRAIN: How many frames to average?
    # Low (1-3) = Fast but jittery. High (5-10) = Smooth but laggy.
    cv2.createTrackbar("STABILITY (Buffer)", "StabilizerDojo", CONFIG.get("SKELETON_BUFFER_SIZE", 5), 10, nothing)
    
    # 2. FILTER: Ignore sudden huge jumps?
    # Low (10px) = Strict safety. High (100px) = Loose.
    cv2.createTrackbar("GLITCH FILTER", "StabilizerDojo", CONFIG.get("OUTLIER_REJECTION_PX", 20), 100, nothing)
    
    # 3. MAGNET: How still must you be to freeze?
    # High = Cursor sticks to screen. Low = Cursor floats.
    cv2.createTrackbar("MAGNET", "StabilizerDojo", int(CONFIG.get("VELOCITY_GATE", 0.008)*1000), 50, nothing)
    
    # 4. LIQUID: How heavy is the cursor moving slow?
    # High (99) = Moving through honey. Low (50) = Moving through water.
    cv2.createTrackbar("LIQUID", "StabilizerDojo", int(CONFIG.get("FRICTION_LOW", 0.97)*100), 100, nothing)

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    brain = NeuroCursorBrain()
    cursor_eng = CursorEngine()
    
    # Force cursor engine to match this window size exactly
    cursor_eng.screen_w = SCREEN_W 
    cursor_eng.screen_h = SCREEN_H
    
    hands = mp.solutions.hands.Hands(max_num_hands=1, model_complexity=0)

    while True:
        ret, frame = cam.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        # Resize frame to fill screen for realism (optional, but helps visualization)
        frame = cv2.resize(frame, (SCREEN_W, SCREEN_H))
        
        # 1. Update Config Live
        new_buf = max(1, cv2.getTrackbarPos("STABILITY (Buffer)", "StabilizerDojo"))
        if new_buf != brain.buffer_size:
            brain.buffer_size = new_buf
            brain.skeleton_buffer = deque(maxlen=new_buf)

        CONFIG["SKELETON_BUFFER_SIZE"] = new_buf
        CONFIG["OUTLIER_REJECTION_PX"] = cv2.getTrackbarPos("GLITCH FILTER", "StabilizerDojo")
        CONFIG["VELOCITY_GATE"] = cv2.getTrackbarPos("MAGNET", "StabilizerDojo") / 1000.0
        CONFIG["FRICTION_LOW"] = cv2.getTrackbarPos("LIQUID", "StabilizerDojo") / 100.0

        # Process
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Instructions
        cv2.putText(frame, "TASK: Hold the Green Dot inside the Red Circle", (SCREEN_W//2 - 200, 100), 1, 2, (0,0,255), 2)

        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            
            # 1. Get STABLE Skeleton (Green)
            label, conf = brain.predict(lms.landmark, SCREEN_W, SCREEN_H)
            
            # 2. Calculate Real Cursor Pos
            raw_x, raw_y = int(lms.landmark[8].x * SCREEN_W), int(lms.landmark[8].y * SCREEN_H)
            
            # Calculate Speed
            prev_pos = cursor_eng.current_pos if cursor_eng.current_pos else (raw_x, raw_y)
            tx, ty = cursor_eng.get_screen_coordinates(raw_x, raw_y)
            dist = math.hypot(tx - prev_pos[0], ty - prev_pos[1])
            speed = dist / SCREEN_W # Normalized speed
            
            # Apply Smoothing
            sx, sy = cursor_eng.apply_smoothing(tx, ty, current_speed=speed)
            
            # --- VISUALIZATION ---
            # Blue Line: Raw Hand Input (Jittery)
            cv2.line(frame, (int(prev_pos[0]), int(prev_pos[1])), (tx, ty), (255, 0, 0), 1)
            cv2.circle(frame, (tx, ty), 5, (255, 0, 0), -1) # Raw
            
            # Green Dot: Stabilized Cursor
            cv2.circle(frame, (int(sx), int(sy)), 10, (0, 255, 0), -1) 
            
            # Draw Target Circle (Fixed center)
            target = (SCREEN_W//2, SCREEN_H//2)
            cv2.circle(frame, target, 20, (0, 0, 255), 2)
            
            # LAG METER
            lag = math.hypot(sx - tx, sy - ty)
            cv2.putText(frame, f"LAG: {int(lag)}px", (int(sx)+20, int(sy)), 1, 1, (0, 255, 255), 1)

        # UI Overlay
        ui_y = SCREEN_H - 150
        cv2.putText(frame, f"BUFFER: {CONFIG['SKELETON_BUFFER_SIZE']} frames (Avg over time)", (50, ui_y), 1, 1.5, (255,255,255), 2)
        cv2.putText(frame, f"MAGNET: {CONFIG['VELOCITY_GATE']:.3f} (Zero-movement zone)", (50, ui_y+40), 1, 1.5, (255,255,255), 2)
        cv2.putText(frame, f"LIQUID: {CONFIG['FRICTION_LOW']:.2f} (Smoothness factor)", (50, ui_y+80), 1, 1.5, (255,255,255), 2)

        cv2.imshow("StabilizerDojo", frame)
        k = cv2.waitKey(1)
        if k == 27: break
        if k == ord('s'):
            print("\n" + "="*40)
            print("💾 FINAL STABILITY SETTINGS:")
            print(f'    "SKELETON_BUFFER_SIZE": {CONFIG["SKELETON_BUFFER_SIZE"]},')
            print(f'    "OUTLIER_REJECTION_PX": {CONFIG["OUTLIER_REJECTION_PX"]},')
            print(f'    "VELOCITY_GATE": {CONFIG["VELOCITY_GATE"]:.4f},')
            print(f'    "FRICTION_LOW": {CONFIG["FRICTION_LOW"]:.2f},')
            print("="*40 + "\n")

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_stabilizer_dojo()