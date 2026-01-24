import cv2
import mediapipe as mp
import sys
import os
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.camera import FastCamera
from src.vision.logic_brain import LogicBrain

# Context Mapping
CONTEXTS = {
    "POINTER_GROUP": ["TIGHT_FIST_RING", "ZOOM_THUMB"],
    "V_SHAPE_GROUP": ["SCROLL_GAP", "WIN_GAP", "RIGHT_CLICK_WRIST"],
    "PALM_GROUP": ["PALM_THUMB_X"],
    "FIST_GROUP": ["CUT_PINKY", "SIDE_GESTURE_X", "CTRL_SQUEEZE", "PREV_SQUEEZE"]
}

def run_inspector():
    # 1. Start Camera
    cam = FastCamera()
    brain = LogicBrain()
    
    mp_hands = mp.solutions.hands
    # Model Complexity 0 is fastest
    hands = mp_hands.Hands(max_num_hands=1, model_complexity=0, min_detection_confidence=0.5)

    print("⚡ FAST INSPECTOR STARTED")
    print("   [A/D]: Adjust Value")
    print("   [W/S]: Select Rule")
    print("   [SPACE]: Save")
    
    selected_idx = 0
    
    while True:
        raw_frame = cam.read()
        if raw_frame is None: continue

        # 2. FORCE LOW RES (The Lag Fix)
        # We process on a small image, but we can display it bigger if needed.
        frame = cv2.resize(raw_frame, (640, 480))
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        gesture = "NOISE"
        debug = {}
        
        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            gesture, debug = brain.predict(lms.landmark)
            
            # Draw Skeleton (Simple)
            mp.solutions.drawing_utils.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)
            
            # --- CONTEXT SWITCHING ---
            active_group = "FIST_GROUP"
            idx_ext = brain.get_dist(lms.landmark, 8, 0) / debug.get("scale", 1)
            mid_ext = brain.get_dist(lms.landmark, 12, 0) / debug.get("scale", 1)
            
            if idx_ext > 1.0 and mid_ext < 1.0: active_group = "POINTER_GROUP"
            elif idx_ext > 1.0 and mid_ext > 1.0: 
                if brain.get_dist(lms.landmark, 16, 0)/debug.get("scale", 1) > 1.0: active_group = "PALM_GROUP"
                else: active_group = "V_SHAPE_GROUP"
            
            # --- DRAW UI (No Transparency/Blending) ---
            rules_to_show = CONTEXTS.get(active_group, [])
            
            # Draw Black Box for Text (Faster than alpha blending)
            cv2.rectangle(frame, (0, 0), (300, 480), (0, 0, 0), -1)
            
            cv2.putText(frame, f"GROUP: {active_group}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            # Color based on match
            res_col = (0, 255, 0) if gesture != "NOISE" else (0, 0, 255)
            cv2.putText(frame, f"RES: {gesture}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, res_col, 2)
            
            y = 100
            for i, rule_key in enumerate(rules_to_show):
                thresh = brain.rules[rule_key]
                
                # Retrieve Value manually
                val = 0.0
                if rule_key == "TIGHT_FIST_RING": val = debug.get("ring_curl", 0)
                elif rule_key == "ZOOM_THUMB": val = debug.get("thumb_knuckle", 0)
                elif rule_key == "SCROLL_GAP": val = debug.get("mid_ring_gap", 0)
                elif rule_key == "WIN_GAP": val = debug.get("ring_pinky_gap", 0)
                elif rule_key == "RIGHT_CLICK_WRIST": val = debug.get("wrist_ring_k", 0)
                elif rule_key == "PALM_THUMB_X": val = debug.get("thumb_x", 0)
                elif rule_key == "CUT_PINKY": val = debug.get("pinky_ext", 0)
                elif rule_key == "SIDE_GESTURE_X": val = debug.get("pinky_x", 0)
                elif rule_key == "CTRL_SQUEEZE": val = debug.get("idx_pinky_k", 0)
                elif rule_key == "PREV_SQUEEZE": val = debug.get("idx_mid_tip", 0)

                # Cursor
                prefix = "   "
                color = (200, 200, 200)
                if i == selected_idx:
                    prefix = "-> "
                    color = (0, 255, 255)
                
                # Determine Pass/Fail (Rough Logic for Viz)
                # This is just visual; the brain decides the real result
                pass_color = (100, 100, 100) # Gray if neutral
                
                text = f"{prefix}{rule_key}: {thresh:.2f} [{val:.2f}]"
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y += 25
                
            # Selection Clamping
            if selected_idx >= len(rules_to_show): selected_idx = 0

            # --- KEYBOARD ---
            k = cv2.waitKey(1)
            if k == 27: break
            
            if k == ord('w'): selected_idx = max(0, selected_idx - 1)
            if k == ord('s'): selected_idx = min(len(rules_to_show)-1, selected_idx + 1)
            
            if k == ord('a') or k == ord('d'):
                key_to_mod = rules_to_show[selected_idx]
                delta = 0.01 if k == ord('d') else -0.01
                brain.rules[key_to_mod] += delta
                
            if k == 32: # SPACE
                print("💾 Saving...")
                with open("src/logic_rules.py", "w") as f:
                    f.write("# LIVE TUNED\nRULES = " + json.dumps(brain.rules, indent=4))

        cv2.imshow("Fast Inspector", frame)
        if cv2.waitKey(1) == 27: break
            
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inspector()