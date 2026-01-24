import cv2
import mediapipe as mp
import numpy as np
import json
import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

class VarianceRecorder:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # We only care about the problem children right now
        self.tasks = ["POINTER", "GEN_Z_HEART", "PINCH"] 
        self.task_idx = 0
        self.is_recording = False
        
        # Structure: { "POINTER": { "f1": [0.1, 0.12...], "f2": [...] } }
        self.dataset = {task: {f"f{i}": [] for i in range(1, 6)} for task in self.tasks}
        self.frame_count = 0

    def get_distance(self, lm_list, idx1, idx2, scale):
        p1 = np.array([lm_list[idx1].x, lm_list[idx1].y])
        p2 = np.array([lm_list[idx2].x, lm_list[idx2].y])
        # Return Normalized Distance
        return np.linalg.norm(p1 - p2) / scale

    def process_frame(self, hand_lms):
        # 1. Calculate Scale (Wrist to Middle Base)
        # This makes the math work even if you move your hand closer/further
        wrist = np.array([hand_lms.landmark[0].x, hand_lms.landmark[0].y])
        middle_base = np.array([hand_lms.landmark[9].x, hand_lms.landmark[9].y])
        scale = np.linalg.norm(wrist - middle_base)
        if scale == 0: scale = 1.0

        lm = hand_lms.landmark
        
        # 2. EXTRACT CANDIDATE FEATURES (The "Probes")
        # F1: Thumb Tip to Middle Base (Cross check)
        f1 = self.get_distance(lm, 4, 9, scale)
        # F2: Thumb Tip to Index Base (Tuck check)
        f2 = self.get_distance(lm, 4, 5, scale)
        # F3: Thumb Tip to Index Tip (Pinch check)
        f3 = self.get_distance(lm, 4, 8, scale)
        # F4: Thumb Tip to Pinky Base (Extreme Cross)
        f4 = self.get_distance(lm, 4, 17, scale)
        # F5: Index Tip to Middle Tip (Finger separation)
        f5 = self.get_distance(lm, 8, 12, scale)
        
        return {"f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5}

    def save_data(self):
        path = ROOT_DIR / "data" / "variance_data.json"
        with open(path, "w") as f:
            json.dump(self.dataset, f, indent=4)
        print(f"💾 RAW DATA DUMPED TO: {path}")

    def run(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            task = self.tasks[self.task_idx]
            status_color = (0, 0, 255) if self.is_recording else (0, 255, 0)
            
            if results.multi_hand_landmarks:
                lms = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, lms, self.mp_hands.HAND_CONNECTIONS)
                
                if self.is_recording:
                    feats = self.process_frame(lms)
                    # Append data to the specific task buffers
                    for key, val in feats.items():
                        self.dataset[task][key].append(val)
                    self.frame_count += 1
            
            # UI
            cv2.rectangle(frame, (0,0), (w, 80), (30,30,30), -1)
            cv2.putText(frame, f"TASK: {task}", (20, 50), 1, 2, (255,255,255), 2)
            
            if self.is_recording:
                cv2.circle(frame, (w-50, 40), 20, (0,0,255), -1)
                cv2.putText(frame, str(self.frame_count), (w-120, 50), 1, 2, (0,0,255), 2)
            else:
                cv2.putText(frame, "[SPACE] Start/Stop | [N] Next | [S] Save & Quit", (20, h-20), 1, 1, (200,200,200), 1)

            cv2.imshow("Variance Lab", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break
            if key == ord(' '):
                self.is_recording = not self.is_recording
                if self.is_recording: self.frame_count = 0
            if key == ord('n'):
                self.task_idx = (self.task_idx + 1) % len(self.tasks)
                self.is_recording = False
            if key == ord('s'):
                self.save_data()
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    VarianceRecorder().run()