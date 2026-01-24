import cv2
import mediapipe as mp
import time
import math
import sys
import os

# Add the project root to path so we can import features
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.gestures import GestureEngine

def draw_hud(frame, state, metrics, hand_lms, mp_draw, mp_hands):
    h, w, _ = frame.shape
    
    # 1. COLOR CODING
    color = (200, 200, 200) # Gray (Unknown)
    if state == "POINTER": color = (0, 255, 0)       # Green
    elif state == "GEN_Z_HEART": color = (255, 0, 255) # Magenta
    elif state == "PINCH": color = (0, 255, 255)     # Yellow
    elif state == "FIST": color = (0, 0, 255)        # Red
    elif state == "SCROLL_SPIDEY": color = (255, 165, 0) # Orange
    elif state == "NOISE_RANDOM": color = (100, 100, 100) # Dim Gray
    
    # 2. SKELETON
    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                           mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=2),
                           mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1))
    
    # 3. HUD PANEL
    cv2.rectangle(frame, (0, 0), (350, 160), (30, 30, 30), -1)
    cv2.rectangle(frame, (0, 0), (350, 160), color, 2)
    
    # 4. BIG LABEL
    cv2.putText(frame, f"STATE: {state}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    # 5. LIVE METRICS (The "Why")
    # Helper to color code good/bad values
    def val_col(val, thresh, mode=">"): 
        good = (val > thresh) if mode == ">" else (val < thresh)
        return (0, 255, 0) if good else (100, 100, 255)

    # A. Heart Logic
    th_val = metrics['thumb_height']
    th_col = val_col(th_val, 0.12, mode=">") # Green if Pointer (>0.12), Red if Heart
    cv2.putText(frame, f"Thumb Ht (Heart<0.12): {th_val:.3f}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, th_col, 1)

    # B. Noise Logic
    stab_val = metrics['noise_sep']
    stab_col = val_col(stab_val, 0.60, mode=">") # Green if Stable (>0.60)
    cv2.putText(frame, f"Stabilty (Noise<0.60): {stab_val:.3f}", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, stab_col, 1)

    # C. Click Logic
    click_val = metrics['pinch_dist']
    click_col = val_col(click_val, 0.25, mode="<") # Green if Clicking (<0.25)
    cv2.putText(frame, f"Click Dist (Click<0.25): {click_val:.3f}", (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, click_col, 1)

def run_debug():
    cap = cv2.VideoCapture(0)
    
    # Setup
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils
    
    engine = GestureEngine()
    
    print("🐞 LIVE DEBUGGER LAUNCHED")
    print("Press [ESC] to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. FLIP (Mirroring Fixed)
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            for lms in results.multi_hand_landmarks:
                # RUN THE ENGINE
                state = engine.detect_state(lms)
                
                # RE-CALCULATE METRICS FOR DISPLAY (Visualization Only)
                # (We do this here to visualize exactly what the engine saw)
                wrist = lms.landmark[0]
                idx_mcp = lms.landmark[5]
                scale = math.hypot(wrist.x - idx_mcp.x, wrist.y - idx_mcp.y) or 1.0
                
                thumb_tip = lms.landmark[4]
                idx_tip = lms.landmark[8]
                idx_dip = lms.landmark[7]
                mid_dip = lms.landmark[11]

                metrics = {
                    'pinch_dist': math.hypot(thumb_tip.x - idx_tip.x, thumb_tip.y - idx_tip.y) / scale,
                    'thumb_height': (thumb_tip.y - idx_dip.y) / scale, # DY_TM_TIP_IDX_DIP
                    'noise_sep': math.hypot(idx_tip.x - mid_dip.x, idx_tip.y - mid_dip.y) / scale
                }
                
                draw_hud(frame, state, metrics, lms, mp_draw, mp_hands)

        cv2.imshow("Gesture Debugger", frame)
        if cv2.waitKey(1) == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_debug()