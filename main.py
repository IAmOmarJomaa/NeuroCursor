import cv2
import mediapipe as mp
import time
import math
import pyautogui
import numpy as np
from config import PATHS
from utils.camera_thread import CameraStream
from features.gestures import GestureEngine
from features.mouse import MouseController
from app.visualizer import Visualizer 

# --- CONFIGURATION ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2, # ENABLE 2 HANDS FOR ZOOM
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def calculate_hand_distance(hand1, hand2):
    """Calculates distance between the Index Tips of two hands."""
    x1, y1 = hand1.landmark[8].x, hand1.landmark[8].y
    x2, y2 = hand2.landmark[8].x, hand2.landmark[8].y
    return math.hypot(x2 - x1, y2 - y1)

def run_neurocursor():
    # 1. INITIALIZATION
    cam = CameraStream().start()
    time.sleep(1.0) # Warmup
    
    engine = GestureEngine()
    mouse = MouseController()
    viz = Visualizer()
    
    # 2. STATE VARIABLES
    SYSTEM_LOCKED = True    # Starts LOCKED (Safety)
    lock_timer = 0          # For "Hold to Unlock"
    wave_timer = 0          # For "Wave to Exit"
    last_wrist_x = 0
    
    # Zoom State
    prev_zoom_dist = None
    
    print("🚀 NEUROCURSOR V3 LIVE.")
    print("   - HOLD 'GEN Z HEART' (3s) to Unlock")
    print("   - SHOW 2 HANDS to Zoom")
    print("   - WAVE FAST to Exit")

    while True:
        ret, frame = cam.read()
        if not ret: break
        
        # Mirror for intuition
        # Note: CameraStream might already mirror, but double check visuals
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        h, w, _ = frame.shape
        
        # Draw Status Overlay
        status_text = "LOCKED (Hold Heart)" if SYSTEM_LOCKED else "ACTIVE"
        viz.draw_hud(frame, "IDLE" if SYSTEM_LOCKED else "READY", status_text)
        
        if not SYSTEM_LOCKED:
            viz.draw_roi(frame, mouse.mapper.roi)

        if results.multi_hand_landmarks:
            
            # =================================================
            # A. TWO HANDS DETECTED -> ZOOM MODE
            # =================================================
            if len(results.multi_hand_landmarks) == 2 and not SYSTEM_LOCKED:
                h1, h2 = results.multi_hand_landmarks
                
                # Visual Feedback
                p1 = (int(h1.landmark[8].x * w), int(h1.landmark[8].y * h))
                p2 = (int(h2.landmark[8].x * w), int(h2.landmark[8].y * h))
                cv2.line(frame, p1, p2, (255, 0, 255), 3)
                cv2.putText(frame, "ZOOMING", ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                
                current_dist = calculate_hand_distance(h1, h2)
                
                if prev_zoom_dist:
                    delta = current_dist - prev_zoom_dist
                    # Threshold to ignore jitter
                    if abs(delta) > 0.01: 
                        scroll_amount = int(delta * 800) # Sensitivity
                        if scroll_amount != 0:
                            # Windows Zoom Shortcut: Ctrl + Scroll
                            pyautogui.keyDown('ctrl')
                            pyautogui.scroll(scroll_amount)
                            pyautogui.keyUp('ctrl')
                
                prev_zoom_dist = current_dist
            
            # =================================================
            # B. ONE HAND DETECTED -> MOUSE / GESTURE MODE
            # =================================================
            else:
                prev_zoom_dist = None # Reset Zoom
                hand_lms = results.multi_hand_landmarks[0]
                
                # 1. Detect Signature
                state = engine.detect_state(hand_lms)
                
                # 2. Master Lock Logic (GEN_Z_HEART)
                if state == "GEN_Z_HEART":
                    lock_timer += 1
                    # Draw Filling Circle
                    radius = int((lock_timer / 45) * 50)
                    cv2.circle(frame, (w//2, h//2), 60, (100, 100, 100), 2)
                    cv2.circle(frame, (w//2, h//2), radius, (255, 0, 255), -1)
                    
                    if lock_timer > 45: # ~3 Seconds
                        SYSTEM_LOCKED = not SYSTEM_LOCKED
                        lock_timer = 0
                else:
                    lock_timer = 0

                # 3. Wave to Exit Logic (Fast Wrist Movement)
                wrist_x = hand_lms.landmark[0].x
                speed = abs(wrist_x - last_wrist_x) * 100
                if state == "PALM" and speed > 8: # High speed threshold
                    wave_timer += 1
                    cv2.putText(frame, "EXITING...", (w//2, h//2), 1, 3, (0, 0, 255), 3)
                    if wave_timer > 30: # ~2 Seconds
                        print("👋 GOODBYE.")
                        break
                else:
                    wave_timer = max(0, wave_timer - 1)
                last_wrist_x = wrist_x

                # 4. Mouse Control (If Unlocked)
                if not SYSTEM_LOCKED:
                    # Pointer Move
                    if state in ["POINTER", "PALM"]:
                        mouse.process_hand(hand_lms, state)
                    
                    # Click Triggers (Using recorded Pinch threshold or default)
                    thresh = engine.thresholds.get("pinch_dist", 30.0)
                    mouse.check_clicks(hand_lms, thresh)
                    
                    # Scroll Mode
                    if state == "SCROLL_StANDBY":
                        mouse.scroll(hand_lms)
                        cv2.putText(frame, "SCROLL MODE", (w//2, 100), 1, 2, (0, 255, 255), 2)
                
                # Debug Info
                viz.draw_hud(frame, state, "LOCKED" if SYSTEM_LOCKED else "ACTIVE")
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("NeuroCursor V3 Final", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_neurocursor()