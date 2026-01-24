import cv2
import mediapipe as mp
import sys
import os
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.camera import FastCamera
from src.vision.logic_brain import LogicBrain

# MAPPING RULES TO REAL-TIME VALUES
# This tells the HUD which live sensor value compares to which rule
RULE_MAP = {
    "CTRL_PINKY_X": ("pinky_x_off", ">"),
    "CTRL_DIST": ("idx_pinky_k", "<"),
    "CUT_PINKY_EXT": ("pinky_ext", ">"),
    "FIST_THUMB_MID": ("thumb_mid_tip", "<"),
    "DELETE_THUMB": ("thumb_idx_k", "<"),
    "PALM_THUMB_X": ("thumb_x_off", ">"),
    "SCROLL_SPREAD": ("mid_ring_tip", "<"),
    "RIGHT_CLICK_LOCK": ("wrist_ring_k", "<"),
    "ZOOM_CLAW": ("thumb_mid_k", ">"),
    "PINCH_START": ("pinch", "<"),
}

def run_tuner():
    cam = FastCamera()
    brain = LogicBrain()
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, model_complexity=0, min_detection_confidence=0.5)

    # UI State
    selected_idx = 0
    rule_keys = list(brain.rules.keys())
    
    print("🎹 NEURO-HUD STARTED")
    print("   [W/S]: Select Rule")
    print("   [A/D]: Tweak Threshold")
    print("   [SPACE]: Save")
    
    while True:
        frame = cam.read()
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # 1. GET DATA
        gesture = "NOISE"
        debug = {}
        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            gesture, debug = brain.predict(lms.landmark)
            mp.solutions.drawing_utils.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

        # 2. DRAW HUD (The Parallel Window)
        hud_w = 500
        hud = np.zeros((h, hud_w, 3), dtype=np.uint8)
        
        # Header
        cv2.putText(hud, f"DETECTED: {gesture}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.line(hud, (0, 60), (hud_w, 60), (100, 100, 100), 2)
        
        # List Rules
        y = 90
        for i, key in enumerate(rule_keys):
            thresh = brain.rules[key]
            
            # Selector
            prefix = "  "
            color = (200, 200, 200)
            if i == selected_idx:
                prefix = "> "
                color = (0, 255, 255)
            
            # Show Threshold
            rule_str = f"{prefix}{key}: {thresh:.2f}"
            cv2.putText(hud, rule_str, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            
            # Show LIVE Value vs Threshold
            if key in RULE_MAP:
                metric_name, operator = RULE_MAP[key]
                live_val = debug.get(metric_name, 0.0)
                
                # Check pass/fail logic visually
                passed = False
                if operator == "<" and live_val <= thresh: passed = True
                if operator == ">" and live_val > thresh: passed = True
                
                val_color = (0, 255, 0) if passed else (0, 0, 255)
                
                # Draw Bar visualization
                bar_len = int(min(live_val, 2.0) * 100)
                cv2.rectangle(hud, (280, y-10), (280 + bar_len, y), val_color, -1)
                cv2.rectangle(hud, (280, y-10), (480, y), (50, 50, 50), 1)
                
                # Draw Threshold Line on Bar
                thresh_x = 280 + int(min(thresh, 2.0) * 100)
                cv2.line(hud, (thresh_x, y-15), (thresh_x, y+5), (255, 255, 0), 2)
                
                cv2.putText(hud, f"{live_val:.2f}", (220, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, val_color, 1)

            y += 30

        # Combine Windows side-by-side
        combined = np.hstack((frame, hud))
        cv2.imshow("Neuro-HUD", combined)
        
        # 3. CONTROLS
        k = cv2.waitKey(1)
        if k == 27: break
        
        if k == ord('w'): selected_idx = max(0, selected_idx - 1)
        if k == ord('s'): selected_idx = min(len(rule_keys)-1, selected_idx + 1)
        if k == ord('a'): brain.rules[rule_keys[selected_idx]] -= 0.01
        if k == ord('d'): brain.rules[rule_keys[selected_idx]] += 0.01
        
        if k == 32: # Space to Save
            print("\n💾 SAVING RULES...")
            with open("src/logic_rules.py", "w") as f:
                f.write("# UPDATED LIVE\nRULES = " + json.dumps(brain.rules, indent=4))
            print("✅ Saved to src/logic_rules.py")

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tuner()