import cv2
import mediapipe as mp
import sys
import os
import math
import time
import numpy as np
import ctypes

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.vision.camera import FastCamera
from src.vision.logic_brain import LogicBrain
from src.config import CONFIG

def is_admin():
    try: return ctypes.windll.shell32.IsUserAnAdmin()
    except: return False

def nothing(x): pass

def run_tuner():
    print("🎛️ ULTIMATE TUNER V3: Config-Aware")
    print("   -> Loaded initial values from src/config.py")
    
    cv2.namedWindow("Ultimate Tuner")
    cv2.resizeWindow("Ultimate Tuner", 600, 550)

    # LOAD DEFAULTS
    def_start = int(CONFIG.get("PINCH_START", 0.10) * 100)
    def_stop  = int(CONFIG.get("PINCH_STOP", 0.15) * 100)
    def_dead  = int(CONFIG.get("CLICK_DEADZONE", 35))
    def_smooth= int(CONFIG.get("SMOOTHING", 5.0) * 10)
    def_fps   = int(CONFIG.get("TARGET_FPS", 20))
    def_off_x = int((CONFIG.get("X_OFFSET", 0.0) + 0.5) * 100)
    def_off_y = int((CONFIG.get("Y_OFFSET", 0.0) + 0.5) * 100)

    # SLIDERS
    cv2.createTrackbar("START (Click)", "Ultimate Tuner", def_start, 20, nothing)
    cv2.createTrackbar("STOP (Release)", "Ultimate Tuner", def_stop, 30, nothing)
    cv2.createTrackbar("Deadzone (px)", "Ultimate Tuner", def_dead, 100, nothing)
    cv2.createTrackbar("Smoothing", "Ultimate Tuner", def_smooth, 100, nothing)
    cv2.createTrackbar("FPS Limit", "Ultimate Tuner", def_fps, 60, nothing)
    cv2.createTrackbar("Offset X", "Ultimate Tuner", def_off_x, 100, nothing)
    cv2.createTrackbar("Offset Y", "Ultimate Tuner", def_off_y, 100, nothing)

    cam = FastCamera()
    brain = LogicBrain()
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, model_complexity=0, min_detection_confidence=0.5)

    prev_time = 0
    click_state = "IDLE"
    anchor_x, anchor_y = 0, 0
    curr_x, curr_y = 0, 0
    
    while True:
        # READ SLIDERS
        p_start = cv2.getTrackbarPos("START (Click)", "Ultimate Tuner") / 100.0
        p_stop = cv2.getTrackbarPos("STOP (Release)", "Ultimate Tuner") / 100.0
        deadzone = cv2.getTrackbarPos("Deadzone (px)", "Ultimate Tuner")
        smooth_val = cv2.getTrackbarPos("Smoothing", "Ultimate Tuner") / 10.0
        target_fps = cv2.getTrackbarPos("FPS Limit", "Ultimate Tuner")
        off_x = (cv2.getTrackbarPos("Offset X", "Ultimate Tuner") - 50) / 100.0
        off_y = (cv2.getTrackbarPos("Offset Y", "Ultimate Tuner") - 50) / 100.0

        if p_stop < p_start: p_stop = p_start
        if target_fps < 1: target_fps = 1
        if smooth_val < 1: smooth_val = 1.0

        # UPDATE CONFIG LIVE IN MEMORY ONLY
        CONFIG["PINCH_START"] = p_start
        CONFIG["PINCH_STOP"] = p_stop
        CONFIG["CLICK_DEADZONE"] = deadzone
        CONFIG["SMOOTHING"] = smooth_val
        CONFIG["X_OFFSET"] = off_x
        CONFIG["Y_OFFSET"] = off_y
        
        # FPS LIMIT
        frame_duration = 1.0 / target_fps
        now = time.time()
        if (now - prev_time) < frame_duration:
            time.sleep(0.001); continue
        prev_time = now

        frame = cam.read()
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            gesture, debug = brain.predict(lms.landmark)
            pinch = debug.get("pinch_dist", 0)
            
            # Use updated offsets for visualization
            center_x = 0.5 + off_x
            center_y = 0.5 + off_y
            box_w = 1.0 / CONFIG["DPI_SENSITIVITY"]
            box_h = 1.0 / CONFIG["DPI_SENSITIVITY"]
            x1 = center_x - (box_w / 2)
            y1 = center_y - (box_h / 2)
            
            raw_hx = lms.landmark[8].x
            raw_hy = lms.landmark[8].y
            norm_x = np.clip((raw_hx - x1) / box_w, 0, 1)
            norm_y = np.clip((raw_hy - y1) / box_h, 0, 1)
            target_x = norm_x * w
            target_y = norm_y * h
            
            # Physics Sim
            if gesture == "CLICK":
                color = (0, 255, 0); state_text = "CLICKED"
                if click_state == "IDLE":
                    click_state = "ANCHORED"; anchor_x, anchor_y = curr_x, curr_y
                
                dist = math.hypot(target_x - anchor_x, target_y - anchor_y)
                cv2.circle(frame, (int(anchor_x), int(anchor_y)), deadzone, (255, 0, 0), 2)
                
                if dist > deadzone:
                    click_state = "DRAGGING"
                    curr_x += (target_x - curr_x) / smooth_val
                    curr_y += (target_y - curr_y) / smooth_val
                    cv2.line(frame, (int(anchor_x), int(anchor_y)), (int(curr_x), int(curr_y)), (255,255,255), 2)
            elif pinch < p_stop:
                color = (0, 255, 255); state_text = "HOLDING"
                curr_x += (target_x - curr_x) / smooth_val
                curr_y += (target_y - curr_y) / smooth_val
            else:
                color = (0, 0, 255); state_text = "RELEASED"
                click_state = "IDLE"
                curr_x += (target_x - curr_x) / smooth_val
                curr_y += (target_y - curr_y) / smooth_val

            cv2.circle(frame, (int(curr_x), int(curr_y)), 8, color, -1)
            p4 = (int(lms.landmark[4].x * w), int(lms.landmark[4].y * h))
            p8 = (int(lms.landmark[8].x * w), int(lms.landmark[8].y * h))
            cv2.line(frame, p4, p8, color, 2)
            cv2.putText(frame, f"STATE: {state_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Ultimate Tuner", frame)
        key = cv2.waitKey(1)
        if key == 27: break
        if key == ord('s'):
            print("\n" + "="*50)
            print("✅ COPY TO src/config.py:")
            print("-" * 50)
            print(f'    "PINCH_START": {p_start},')
            print(f'    "PINCH_STOP": {p_stop},')
            print(f'    "CLICK_DEADZONE": {deadzone},')
            print(f'    "SMOOTHING": {smooth_val},')
            print(f'    "TARGET_FPS": {target_fps},')
            print(f'    "X_OFFSET": {off_x},')
            print(f'    "Y_OFFSET": {off_y},')
            print("="*50 + "\n")

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tuner()