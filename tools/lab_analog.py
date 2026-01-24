import cv2
import mediapipe as mp
import sys
import os
import math
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import CONFIG
from src.gesture_engine import NeuroCursorBrain

def run_lab():
    print("🎚️ ANALOG LAB (V3: The Safety Brake)")
    print("   -> Scroll: Lower 'SCR DEADZONE' if it's not moving.")
    print("   -> Zoom: Uses 'ZOOM BRAKE' to ignore fast hand drops.")
    
    cv2.namedWindow("Analog Lab", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Analog Lab", 1000, 700)
    
    def nothing(x): pass
    
    # Sliders
    # Volume
    cv2.createTrackbar("VOL D-ZONE", "Analog Lab", int(CONFIG["VOLUME_DEADZONE"]*1000), 100, nothing)
    cv2.createTrackbar("VOL SENS", "Analog Lab", int(CONFIG["VOLUME_SENSITIVITY"]), 100, nothing)
    
    # Scroll (Now with Deadzone Slider)
    cv2.createTrackbar("SCR DEADZONE", "Analog Lab", 15, 100, nothing) # Default 0.015
    cv2.createTrackbar("SCR BASE", "Analog Lab", int(CONFIG["SCROLL_SPEED_BASE"]), 20, nothing)
    cv2.createTrackbar("SCR ACC", "Analog Lab", int(CONFIG["SCROLL_ACCELERATION"]*10), 100, nothing)
    
    # Zoom
    cv2.createTrackbar("ZOOM STEP", "Analog Lab", int(CONFIG["ZOOM_STEP"]*1000), 200, nothing)
    # New: The Safety Brake (Max speed allowed for zoom)
    cv2.createTrackbar("ZOOM BRAKE", "Analog Lab", 80, 200, nothing) 

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, model_complexity=1)
    
    brain = NeuroCursorBrain()
    
    # State
    vol_anchor = None
    scroll_anchor = None
    zoom_dist_prev = None
    
    vol_accumulator = 0.0
    virtual_scroll_y = 350
    zoom_level = 1.0
    
    while True:
        # Live Update
        CONFIG["VOLUME_DEADZONE"] = cv2.getTrackbarPos("VOL D-ZONE", "Analog Lab") / 1000.0
        CONFIG["VOLUME_SENSITIVITY"] = cv2.getTrackbarPos("VOL SENS", "Analog Lab")
        
        scroll_deadzone = cv2.getTrackbarPos("SCR DEADZONE", "Analog Lab") / 1000.0
        CONFIG["SCROLL_SPEED_BASE"] = cv2.getTrackbarPos("SCR BASE", "Analog Lab")
        CONFIG["SCROLL_ACCELERATION"] = cv2.getTrackbarPos("SCR ACC", "Analog Lab") / 10.0
        
        CONFIG["ZOOM_STEP"] = cv2.getTrackbarPos("ZOOM STEP", "Analog Lab") / 1000.0
        zoom_brake_thresh = cv2.getTrackbarPos("ZOOM BRAKE", "Analog Lab") / 1000.0
        
        ret, frame = cam.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # --- UI DRAWING ---
        # Scroll Bar
        cv2.putText(frame, "SCROLL", (50, 40), 1, 1, (255, 255, 255), 1)
        cv2.rectangle(frame, (50, 60), (100, 600), (40,40,40), -1)
        thumb_y = int(np.clip(virtual_scroll_y, 60, 560))
        cv2.rectangle(frame, (50, thumb_y), (100, thumb_y+40), (0, 255, 255), -1)

        # Volume Bar
        cv2.putText(frame, "VOLUME", (w-300, 40), 1, 1, (255, 255, 255), 1)
        cv2.rectangle(frame, (w-300, 60), (w-50, 150), (40,40,40), -1)
        vol_fill = int(np.clip(vol_accumulator + 125, 0, 250))
        cv2.rectangle(frame, (w-300, 60), (w-300+vol_fill, 150), (0, 0, 255), -1)
        
        # Zoom Circle
        cv2.circle(frame, (w//2, h//2), int(50 * zoom_level), (255, 0, 255), 2)
        cv2.putText(frame, f"{zoom_level:.2f}x", (w//2-40, h//2+10), 1, 1, (255,255,255), 2)

        active_action = "NONE"
        status_msg = ""

        if results.multi_hand_landmarks:
            # --- ZOOM LOGIC (Priority) ---
            if len(results.multi_hand_landmarks) == 2:
                lms1 = results.multi_hand_landmarks[0]
                lms2 = results.multi_hand_landmarks[1]
                
                # Check Labels
                lbl1, _ = brain.predict(lms1.landmark)
                lbl2, _ = brain.predict(lms2.landmark)
                
                # Flexible Zoom Trigger: ZOOM gesture OR just pointing with both hands
                if "ZOOM" in lbl1 or "ZOOM" in lbl2 or ("POINT" in lbl1 and "POINT" in lbl2):
                    active_action = "ZOOMING"
                    
                    x1, y1 = lms1.landmark[8].x, lms1.landmark[8].y
                    x2, y2 = lms2.landmark[8].x, lms2.landmark[8].y
                    dist = math.hypot(x2-x1, y2-y1)
                    
                    if zoom_dist_prev is None:
                        zoom_dist_prev = dist
                    else:
                        delta = dist - zoom_dist_prev
                        
                        # THE SAFETY BRAKE
                        # If movement is faster than 'ZOOM BRAKE', we ignore it (Dropping hands)
                        if abs(delta) > zoom_brake_thresh:
                            status_msg = "BRAKE ACTIVE (Too Fast)"
                            zoom_dist_prev = dist # Reset anchor so we don't jump later
                        
                        # Normal Zoom Step
                        elif abs(delta) > CONFIG["ZOOM_STEP"]:
                            direction = 1 if delta > 0 else -1
                            zoom_level += (direction * 0.1)
                            zoom_level = max(0.5, min(zoom_level, 3.0))
                            zoom_dist_prev = dist
                            status_msg = "ZOOMING..."
            else:
                zoom_dist_prev = None
                
                # --- SINGLE HAND LOGIC ---
                lms = results.multi_hand_landmarks[0]
                label, _ = brain.predict(lms.landmark)
                raw_y = lms.landmark[8].y
                
                # Scroll
                if "SCROLL" in label:
                    active_action = "SCROLLING"
                    if scroll_anchor is None: scroll_anchor = raw_y
                    
                    delta = raw_y - scroll_anchor
                    
                    # Debug Info
                    cv2.putText(frame, f"Delta: {abs(delta):.4f}", (120, 80), 1, 1, (200,200,200), 1)
                    cv2.putText(frame, f"Thresh: {scroll_deadzone:.4f}", (120, 110), 1, 1, (200,200,200), 1)

                    if abs(delta) > scroll_deadzone:
                        speed = delta * CONFIG["SCROLL_SPEED_BASE"] * CONFIG["SCROLL_ACCELERATION"] * 10
                        virtual_scroll_y += speed
                        virtual_scroll_y = max(60, min(virtual_scroll_y, 560))
                        status_msg = "MOVING"
                    else:
                        status_msg = "DEADZONE"
                else:
                    scroll_anchor = None
                
                # Volume
                if "VOLUME" in label:
                    active_action = "VOLUME"
                    if vol_anchor is None: vol_anchor = raw_y
                    
                    vol_delta = raw_y - vol_anchor
                    if abs(vol_delta) > CONFIG["VOLUME_DEADZONE"]:
                        speed = -vol_delta * CONFIG["VOLUME_SENSITIVITY"] * 0.5
                        vol_accumulator += speed
                        if abs(vol_accumulator) > 50: vol_accumulator = 0
                else:
                    vol_anchor = None
                    
                # Draw Label
                wrist = lms.landmark[0]
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(frame, label, (cx, cy-30), 1, 1, (0,255,0), 2)

        else:
            scroll_anchor = None
            vol_anchor = None
            zoom_dist_prev = None

        # HUD
        cv2.putText(frame, f"ACTION: {active_action}", (w//2 - 100, 30), 1, 1.5, (0, 255, 255), 2)
        if status_msg:
            col = (0, 0, 255) if "BRAKE" in status_msg else (0, 255, 0)
            cv2.putText(frame, status_msg, (w//2 - 100, 60), 1, 1, col, 2)
        
        cv2.imshow("Analog Lab", frame)
        k = cv2.waitKey(1)
        if k == 27: break
        if k == ord('s'):
            print("\n" + "="*40)
            print("💾 CONFIG VALUES (ANALOG):")
            print(f'    "VOLUME_DEADZONE": {CONFIG["VOLUME_DEADZONE"]:.3f},')
            print(f'    "SCROLL_DEADZONE": {scroll_deadzone:.3f},  # NEW PARAMETER')
            print(f'    "SCROLL_SPEED_BASE": {CONFIG["SCROLL_SPEED_BASE"]},')
            print(f'    "SCROLL_ACCELERATION": {CONFIG["SCROLL_ACCELERATION"]:.1f},')
            print(f'    "ZOOM_STEP": {CONFIG["ZOOM_STEP"]:.3f},')
            print(f'    "ZOOM_BRAKE": {zoom_brake_thresh:.3f},  # NEW PARAMETER')
            print("="*40 + "\n")

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_lab()