import cv2, mediapipe as mp, math, numpy as np, sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.gesture_engine import NeuroCursorBrain
from src.config import CONFIG

def nothing(x): pass

def run_click_dojo():
    print("🖱️ CLICK DOJO: Fix Index Dip & Drag Drops")
    print("   [S] Save Config | [ESC] Exit")
    
    cv2.namedWindow("ClickDojo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ClickDojo", 1000, 800)

    # CLICK & LOGIC PARAMETERS
    cv2.createTrackbar("DEADZONE", "ClickDojo", CONFIG["CLICK_DEADZONE"], 150, nothing)
    cv2.createTrackbar("FREEZE_S", "ClickDojo", int(CONFIG["CLICK_FREEZE_SENSITIVITY"]*1000), 100, nothing)
    cv2.createTrackbar("PINCH_STOP", "ClickDojo", int(CONFIG["PINCH_STOP"]*1000), 150, nothing)
    cv2.createTrackbar("STICKY_S", "ClickDojo", int(CONFIG["PINCH_DYNAMIC_SCALE"]*100), 200, nothing)

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    hands = mp.solutions.hands.Hands(max_num_hands=1, model_complexity=0)

    # State
    anchor = None
    is_pinching = False
    freeze_active = False

    while True:
        ret, frame = cam.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Live Config
        CONFIG["CLICK_DEADZONE"] = cv2.getTrackbarPos("DEADZONE", "ClickDojo")
        CONFIG["CLICK_FREEZE_SENSITIVITY"] = cv2.getTrackbarPos("FREEZE_S", "ClickDojo") / 1000.0
        p_stop_base = cv2.getTrackbarPos("PINCH_STOP", "ClickDojo") / 1000.0
        sticky_scale = cv2.getTrackbarPos("STICKY_S", "ClickDojo") / 100.0

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            ix, iy = int(lms.landmark[8].x * w), int(lms.landmark[8].y * h)
            tx, ty = int(lms.landmark[4].x * w), int(lms.landmark[4].y * h)
            
            # Physics Math
            dist = math.hypot(lms.landmark[8].x - lms.landmark[4].x, lms.landmark[8].y - lms.landmark[4].y)
            p_start = CONFIG.get("PINCH_START", 0.034)
            
            # 1. VISUALIZE FREEZE ZONE (The Anti-Dip)
            # If distance is approaching Click Start, show the Freeze Warning
            if not is_pinching and dist < (p_start + CONFIG["CLICK_FREEZE_SENSITIVITY"]):
                cv2.circle(frame, (ix, iy), 20, (0, 165, 255), 2)
                cv2.putText(frame, "PRE-CLICK FREEZE", (ix+30, iy), 1, 1, (0, 165, 255), 2)
            
            # 2. CLICK LOGIC
            if dist < p_start and not is_pinching:
                is_pinching = True
                anchor = (ix, iy) # Deadzone anchors here
            
            # Dynamic Release (Sticky Logic Simulation)
            # We assume a fake speed for sim
            sim_speed = 0.02 # Moderate speed
            dynamic_boost = sim_speed * sticky_scale
            curr_stop = min(p_stop_base + dynamic_boost, CONFIG["PINCH_STOP_MAX"])
            
            if dist > curr_stop:
                is_pinching = False
                anchor = None

            # 3. DRAW DEADZONE
            if anchor:
                cv2.circle(frame, anchor, CONFIG["CLICK_DEADZONE"], (255, 0, 0), 2)
                cv2.line(frame, anchor, (ix, iy), (0, 255, 255), 1)
                
                # Check if we broke the deadzone (Drag Start)
                dip_dist = math.hypot(ix - anchor[0], iy - anchor[1])
                if dip_dist > CONFIG["CLICK_DEADZONE"]:
                    cv2.putText(frame, "DRAGGING", (anchor[0], anchor[1]-20), 1, 2, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "CLICK HELD (FROZEN)", (anchor[0], anchor[1]-20), 1, 1, (255, 0, 0), 2)

            # Info
            cv2.rectangle(frame, (20, h-120), (400, h), (0,0,0), -1)
            cv2.putText(frame, f"PINCH DIST: {dist:.4f}", (30, h-90), 1, 1, (255,255,255), 1)
            cv2.putText(frame, f"RELEASE THRESH: {curr_stop:.4f}", (30, h-60), 1, 1, (255,255,255), 1)
            cv2.putText(frame, f"DEADZONE: {CONFIG['CLICK_DEADZONE']}px", (30, h-30), 1, 1, (255,255,255), 1)

        cv2.imshow("ClickDojo", frame)
        k = cv2.waitKey(1)
        if k == 27: break
        if k == ord('s'):
            print("\n" + "="*40)
            print("💾 COPY TO CONFIG (CLICKING):")
            print(f'    "CLICK_DEADZONE": {CONFIG["CLICK_DEADZONE"]},')
            print(f'    "CLICK_FREEZE_SENSITIVITY": {CONFIG["CLICK_FREEZE_SENSITIVITY"]:.4f},')
            print(f'    "PINCH_STOP": {p_stop_base:.3f},')
            print(f'    "PINCH_DYNAMIC_SCALE": {sticky_scale:.2f},')
            print("="*40 + "\n")

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_click_dojo()