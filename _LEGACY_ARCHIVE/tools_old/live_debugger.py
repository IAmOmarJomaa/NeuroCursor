import cv2
import mediapipe as mp
import sys
import os
import time
import numpy as np

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the ACTUAL brain to ensure we debug the real logic
from src.vision.brain import NeuroBrain
from src.vision.camera import FastCamera

def run_debugger():
    print("🐞 NEUROCURSOR DEBUGGER (PASSIVE MODE)")
    print("   -> Mouse disconnected.")
    print("   -> Showing raw AI predictions + Logic overrides.")
    
    # 1. Load Systems
    cam = FastCamera()
    brain = NeuroBrain()
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2, 
        model_complexity=0, 
        min_detection_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    print("✅ DEBUGGER LIVE.")

    while True:
        frame = cam.read()
        if frame is None: continue
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(rgb)
        
        # UI Background
        cv2.rectangle(frame, (0, 0), (w, 120), (20, 20, 20), -1)
        
        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            
            # --- DEBUG LOGIC ---
            
            # 1. Get Raw Predictions for ALL hands
            hand_data = []
            for i, lms in enumerate(results.multi_hand_landmarks):
                # Get raw prediction
                gesture = brain.predict(lms.landmark)
                
                # Get confidence (Probability of that class)
                # Note: predict_proba expects 2D array, we wrap the single sample
                feats = brain._extract_features_internal(lms.landmark) # Helper to get raw feats if needed
                # Re-predict using proba to get confidence score
                probs = brain.model.predict_proba(feats)[0]
                confidence = np.max(probs) * 100
                
                hand_data.append({
                    "id": i,
                    "gesture": gesture,
                    "conf": confidence,
                    "lms": lms
                })

            # 2. Simulate the "Smart Selector" Logic
            active_hand_idx = -1
            final_decision = "NOISE (Idle)"
            
            # Logic Step A: Filter Candidates
            candidates = [h for h in hand_data if h["gesture"] != "NOISE"]
            
            # Logic Step B: Decision
            if len(candidates) > 0:
                best_hand = candidates[0] # Pick first valid
                active_hand_idx = best_hand["id"]
                final_decision = f"ACT: {best_hand['gesture']}"
                logic_reason = "Valid Gesture Found"
            else:
                # Logic Step C: Fallback
                # If 1 hand only, Force Pointer
                if hand_count == 1:
                    active_hand_idx = 0
                    final_decision = "FORCE: POINTER"
                    logic_reason = "1 Hand Override (Anti-Noise)"
                else:
                    final_decision = "IGNORE (All Noise)"
                    logic_reason = "2 Hands + Both Resting"

            # --- DRAWING ---
            
            # Draw Status Header
            cv2.putText(frame, f"HANDS: {hand_count} | DECISION: {final_decision}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"REASON: {logic_reason}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Draw Hands with Debug Info
            for i, data in enumerate(hand_data):
                lms = data["lms"]
                is_active = (i == active_hand_idx)
                
                # Color: Green = Selected, Red = Ignored
                color = (0, 255, 0) if is_active else (0, 0, 255)
                
                # Draw Skeleton
                mp_draw.draw_landmarks(
                    frame, lms, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=2)
                )
                
                # Draw Label ON THE HAND
                wrist = lms.landmark[0]
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                
                label_text = f"{data['gesture']} ({data['conf']:.0f}%)"
                cv2.putText(frame, label_text, (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                if is_active:
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1) # Yellow dot on active wrist

        else:
            cv2.putText(frame, "NO HANDS DETECTED", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("NeuroCursor Debugger", frame)
        
        if cv2.waitKey(1) & 0xFF == 27: break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Small patch to allow accessing internal features for confidence score
    # We monkey-patch the brain class temporarily for this debug script
    def _extract_features_internal(self, lms):
        import pandas as pd
        import numpy as np
        # Copy-paste the extraction logic purely to get the DataFrame for predict_proba
        coords = np.array([[lm.x, lm.y, lm.z] for lm in lms])
        def get_ang(i1, i2, i3):
            v1 = coords[i2] - coords[i1]; v2 = coords[i3] - coords[i2]
            dot = np.dot(v1, v2); norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            return np.degrees(np.arccos(np.clip(dot/(norm+1e-6), -1.0, 1.0)))
        def get_dist(i1, i2, n): return np.linalg.norm(coords[i1]-coords[i2])/n
        palm = np.linalg.norm(coords[9]-coords[0])+1e-6
        data = {
            'thumb_bend': get_ang(1, 2, 3), 'index_bend': get_ang(5, 6, 7),
            'mid_bend': get_ang(9, 10, 11), 'ring_bend': get_ang(13, 14, 15),
            'pinky_bend': get_ang(17, 18, 19), 'pinch_dist': get_dist(4, 8, palm),
            'thumb_spread': get_dist(4, 5, palm), 'mid_palm_dist': get_dist(12, 0, palm),
            'orientation_y': coords[9][1] - coords[0][1], 'orientation_x': coords[9][0] - coords[0][0]
        }
        from src.vision.brain import FEATURE_COLUMNS
        return pd.DataFrame([data], columns=FEATURE_COLUMNS)
    
    NeuroBrain._extract_features_internal = _extract_features_internal
    
    run_debugger()