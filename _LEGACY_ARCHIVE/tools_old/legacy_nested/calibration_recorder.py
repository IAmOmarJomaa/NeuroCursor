import cv2
import mediapipe as mp
import numpy as np
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

class DataScienceRecorder:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # RECORDING LIST - EXACTLY WHAT WE NEED TO DISTINGUISH
        self.tasks = [
            "POINTER",      # Index UP, Thumb TUCKED.
            "GEN_Z_HEART",  # Index UP, Thumb CROSSED.
            "PINCH",        # Index + Thumb TOUCHING.
            "SCROLL_POSE",  # 3 Fingers UP.
            "FIST",         # All Closed.
            "OPEN_PALM"     # All Open.

        ]
        
        self.task_idx = 0
        self.is_recording = False
        self.buffer = [] 
        self.dataset = {} 

    def get_normalized_skeleton(self, hand_lms):
        """
        Converts the hand to a list of 63 numbers (x,y,z per joint),
        Invariant to position on screen (centered on wrist).
        """
        # 1. Center on Wrist
        wrist = hand_lms.landmark[0]
        wx, wy, wz = wrist.x, wrist.y, wrist.z
        
        # 2. Scale Normalization (Size of hand shouldn't matter)
        # We use distance from Wrist(0) to Middle MPC(9) as scale unit
        middle_mcp = hand_lms.landmark[9]
        scale = np.linalg.norm([middle_mcp.x-wx, middle_mcp.y-wy, middle_mcp.z-wz])
        if scale == 0: scale = 1.0 # Safety
        
        skeleton = []
        for lm in hand_lms.landmark:
            # Relative to wrist, scaled by hand size
            skeleton.append((lm.x - wx) / scale)
            skeleton.append((lm.y - wy) / scale)
            skeleton.append((lm.z - wz) / scale)
            
        return skeleton # List of 63 floats

    def draw_ui(self, frame, skeleton):
        h, w, _ = frame.shape
        task = self.tasks[self.task_idx]
        
        # Status Panel
        cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)
        
        # Instructions
        color = (0, 255, 0) if not self.is_recording else (0, 0, 255)
        text = f"TASK: {task}"
        if self.is_recording: text += f" (REC: {len(self.buffer)})"
        
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Prompt
        if not self.is_recording:
             cv2.putText(frame, "PRESS [SPACE] TO RECORD", (50, 120), 1, 1, (200, 200, 200), 2)

    def analyze_and_save(self, task):
        print(f"📊 Processing {task}...")
        
        # DATA SCIENCE: We calculate the MEAN VECTOR (The 'Centroid' of the cluster)
        data_matrix = np.array(self.buffer) # Shape: (N, 63)
        centroid = np.mean(data_matrix, axis=0) # Shape: (63,)
        
        self.dataset[task] = centroid.tolist()
        print(f"✅ Saved Centroid for {task}")

    def save_to_disk(self):
        # We save ONLY the centroids. This is our "Model".
        save_path = ROOT_DIR / "data" / "knn_dataset.json"
        
        final_data = {
            "centroids": self.dataset,
            # We also save a default distance threshold for clicking
            "meta": {"created_at": "v3_ds_pivot"}
        }
        
        with open(save_path, "w") as f:
            json.dump(final_data, f, indent=4)
        print(f"💾 DATABASE SAVED: {save_path}")

    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            skeleton = []
            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                
                # GET RAW DATA
                skeleton = self.get_normalized_skeleton(hand_lms)
                
                if self.is_recording:
                    self.buffer.append(skeleton)
            
            self.draw_ui(frame, skeleton)
            cv2.imshow("Data Science Recorder", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break
            if key == ord(' '):
                if not self.is_recording:
                    self.is_recording = True
                    self.buffer = []
                else:
                    self.is_recording = False
                    if len(self.buffer) > 0:
                        self.analyze_and_save(self.tasks[self.task_idx])
            if key == ord('n'):
                self.task_idx = (self.task_idx + 1) % len(self.tasks)
                self.buffer = []
                self.is_recording = False
            if key == ord('s'):
                self.save_to_disk()
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    DataScienceRecorder().run()