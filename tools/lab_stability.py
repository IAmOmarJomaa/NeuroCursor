import cv2
import mediapipe as mp
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import CONFIG
from src.core.stabilizer import SkeletonStabilizer

def run_lab():
    print("⚓ STABILIZER LAB (Layer 1)")
    print("   -> Tune 'The Anchor' logic.")
    print("   -> Ghost Hand = Raw Input | Green Hand = Stabilized Output")
    
    cv2.namedWindow("Stabilizer Lab", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Stabilizer Lab", 1000, 700)
    
    def nothing(x): pass
    
    # Sliders
    # Threshold: 0.001 to 0.010 (1px to 10px approx)
    cv2.createTrackbar("THRESHOLD (x1000)", "Stabilizer Lab", int(CONFIG["STABILIZER_THRESHOLD"]*1000), 15, nothing)
    # Alpha: 1 to 100 (0.01 to 1.00)
    cv2.createTrackbar("ALPHA (%)", "Stabilizer Lab", int(CONFIG["STABILIZER_ALPHA"]*100), 100, nothing)
    # Buffer: 1 to 10
    cv2.createTrackbar("BUFFER SIZE", "Stabilizer Lab", CONFIG["SKELETON_BUFFER_SIZE"], 10, nothing)

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, model_complexity=1)
    mp_draw = mp.solutions.drawing_utils
    
    # Initialize Stabilizer
    stabilizer = SkeletonStabilizer()
    
    while True:
        # Live Updates
        thresh_val = cv2.getTrackbarPos("THRESHOLD (x1000)", "Stabilizer Lab") / 1000.0
        alpha_val = cv2.getTrackbarPos("ALPHA (%)", "Stabilizer Lab") / 100.0
        if alpha_val == 0: alpha_val = 0.01 # Prevent div by zero logic if any
        
        stabilizer.thresh = thresh_val
        stabilizer.alpha = alpha_val
        
        ret, frame = cam.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            raw_lms = results.multi_hand_landmarks[0]
            
            # 1. Draw GHOST (Raw Input) in GREY
            mp_draw.draw_landmarks(
                frame, raw_lms, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(80,80,80), thickness=1, circle_radius=1),
                mp_draw.DrawingSpec(color=(80,80,80), thickness=1, circle_radius=1)
            )
            
            # 2. Process Stabilizer
            stable_lms_obj = stabilizer.process(raw_lms)
            
            # 3. Draw STABLE Output in GREEN (or BLUE if Frozen)
            col = (0, 255, 0)
            status = "LIVE"
            if stabilizer.locked: 
                col = (255, 200, 0) # Blue-ish for Frozen
                status = "ANCHORED (FROZEN)"
                
            # We have to extract the list from our Mock Object to draw it
            # MediaPipe draw utils expect a specific format, so we manually draw specifically for visual clarity
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in stable_lms_obj.landmark]
            
            # Draw Connections
            for pair in mp_hands.HAND_CONNECTIONS:
                cv2.line(frame, pts[pair[0]], pts[pair[1]], col, 2)
            
            # Draw Joints
            for p in pts:
                cv2.circle(frame, p, 3, col, -1)
                
            # HUD
            cv2.rectangle(frame, (20, h-120), (400, h-20), (0,0,0), -1)
            cv2.putText(frame, f"STATUS: {status}", (30, h-90), 1, 1.5, col, 2)
            cv2.putText(frame, f"THRESH: {thresh_val:.4f}", (30, h-60), 1, 1, (255,255,255), 1)
            cv2.putText(frame, f"ALPHA: {alpha_val:.2f} (Lag vs Smooth)", (30, h-30), 1, 1, (255,255,255), 1)

        cv2.imshow("Stabilizer Lab", frame)
        k = cv2.waitKey(1)
        if k == 27: break
        if k == ord('s'):
            print("\n" + "="*40)
            print("💾 CONFIG VALUES (STABILIZER):")
            print(f'    "STABILIZER_THRESHOLD": {thresh_val:.3f},')
            print(f'    "STABILIZER_ALPHA": {alpha_val:.2f},')
            print(f'    "SKELETON_BUFFER_SIZE": {cv2.getTrackbarPos("BUFFER SIZE", "Stabilizer Lab")},')
            print("="*40 + "\n")

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_lab()