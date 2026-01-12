import cv2
import mediapipe as mp
import numpy as np
import json
import time
import sys
import os
from pathlib import Path

# --- 1. PRO PATH NAVIGATION ---
# Adds C:\NeuroCursor_Final to python path so we can import 'core'
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.geometry import HandGeometry

class CalibrationLab:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # We will record these specific "Canonical Poses"
        self.tasks = ["OPEN_PALM", "FIST", "PINCH_INDEX", "PINCH_MIDDLE"]
        self.task_idx = 0
        
        self.is_recording = False
        self.buffer = [] # Stores 300 frames of angle data
        self.rules_db = {} # The final calculated thresholds

    def draw_live_metrics(self, frame, angles):
        """Draws the live angles on the screen so you can SEE the math."""
        h, w, _ = frame.shape
        
        # Background Panel
        cv2.rectangle(frame, (0, 0), (250, h), (20, 20, 20), -1)
        
        y = 40
        cv2.putText(frame, "LIVE ANGLES:", (10, y), 1, 1.2, (0, 255, 255), 2)
        y += 40
        
        for finger, angle in angles.items():
            # Green if straight (>150), Red if curled (<90), Yellow otherwise
            color = (0, 255, 0)
            if angle < 100: color = (0, 0, 255)
            elif angle < 150: color = (0, 255, 255)
            
            text = f"{finger.upper()}: {int(angle)}deg"
            cv2.putText(frame, text, (10, y), 1, 1.0, color, 1)
            y += 35

    def analyze_buffer(self, task_name):
        """The Data Science Part: Calculate Mean, Min, Max for the gesture."""
        print(f"📊 Analyzing {len(self.buffer)} frames for {task_name}...")
        
        # Structure: { "thumb": [150, 151, 149...], "index": ... }
        aggregated = {f: [] for f in ["thumb", "index", "middle", "ring", "pinky"]}
        
        for frame_data in self.buffer:
            for f, angle in frame_data.items():
                aggregated[f].append(angle)
        
        stats = {}
        for f, vals in aggregated.items():
            stats[f] = {
                "min": np.min(vals),
                "max": np.max(vals),
                "mean": np.mean(vals),
                "std": np.std(vals)
            }
            
        self.rules_db[task_name] = stats
        print(f"✅ Statistics captured for {task_name}")

    def save_rules(self):
        """Compiles the raw stats into simple IF/ELSE thresholds."""
        # We create a 'derived_rules' object that the runtime engine can use fast.
        final_output = {
            "raw_stats": self.rules_db,
            "thresholds": {}
        }
        
        # Example Logic Extraction:
        # The "Fist Threshold" for Index finger should be halfway between 
        # the Mean Index Angle of OPEN_PALM and the Mean Index Angle of FIST.
        
        if "OPEN_PALM" in self.rules_db and "FIST" in self.rules_db:
            palm_idx = self.rules_db["OPEN_PALM"]["index"]["mean"]
            fist_idx = self.rules_db["FIST"]["index"]["mean"]
            
            # The "Cutoff" line
            idx_cutoff = (palm_idx + fist_idx) / 2
            final_output["thresholds"]["fist_curl_threshold"] = idx_cutoff
            print(f"📐 CALCULATED FIST THRESHOLD: {idx_cutoff:.1f} degrees")

        save_path = ROOT_DIR / "data" / "heuristic_rules.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, "w") as f:
            json.dump(final_output, f, indent=4)
        print(f"💾 RULES SAVED TO: {save_path}")

    def run(self):
        cap = cv2.VideoCapture(0)
        
        print("🎥 CAMERA ACTIVE.")
        print("   [SPACE] Start/Stop Recording")
        print("   [n] Next Task")
        print("   [s] Save & Exit")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            current_task = self.tasks[self.task_idx]
            
            # Draw HUD
            self.draw_live_metrics(frame, {}) # Blank if no hand
            
            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                
                # 1. GET THE MATH
                angles = HandGeometry.get_all_finger_states(hand_lms)
                self.draw_live_metrics(frame, angles)
                
                # 2. RECORD DATA
                if self.is_recording:
                    self.buffer.append(angles)
                    cv2.circle(frame, (w-50, 50), 20, (0, 0, 255), -1) # Red Dot
                    cv2.putText(frame, f"{len(self.buffer)}", (w-80, 55), 1, 1, (255, 255, 255), 2)
            
            # UI Instructions
            status_color = (0, 0, 255) if self.is_recording else (0, 255, 0)
            cv2.putText(frame, f"TASK: {current_task}", (270, 50), 1, 2, (255, 255, 255), 4)
            cv2.putText(frame, f"TASK: {current_task}", (270, 50), 1, 2, status_color, 2)
            
            if not self.is_recording:
                cv2.putText(frame, "PRESS [SPACE] TO RECORD", (270, 100), 1, 1, (200, 200, 200), 1)

            cv2.imshow("Calibration Lab", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break # ESC
            
            if key == ord(' '): # SPACE
                if not self.is_recording:
                    self.is_recording = True
                    self.buffer = []
                    print(f"🔴 Recording {current_task}...")
                else:
                    self.is_recording = False
                    self.analyze_buffer(current_task)
            
            if key == ord('n'): # NEXT
                self.task_idx = (self.task_idx + 1) % len(self.tasks)
                self.buffer = []
                self.is_recording = False
            
            if key == ord('s'): # SAVE
                self.save_rules()
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    CalibrationLab().run()