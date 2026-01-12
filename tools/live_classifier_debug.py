import cv2
import mediapipe as mp
import numpy as np
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

class LiveDebugger:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1)
        
        # Load Dataset
        path = ROOT_DIR / "data" / "knn_dataset.json"
        if not path.exists():
            print("❌ NO DATA. Record first.")
            sys.exit()
            
        with open(path, "r") as f:
            self.centroids = json.load(f)["centroids"]
            
    def get_skeleton(self, hand_lms):
        # COPY OF LOGIC FROM RECORDER
        wrist = hand_lms.landmark[0]
        wx, wy, wz = wrist.x, wrist.y, wrist.z
        middle_mcp = hand_lms.landmark[9]
        scale = np.linalg.norm([middle_mcp.x-wx, middle_mcp.y-wy, middle_mcp.z-wz]) or 1.0
        
        skel = []
        for lm in hand_lms.landmark:
            skel.append((lm.x - wx) / scale)
            skel.append((lm.y - wy) / scale)
            skel.append((lm.z - wz) / scale)
        return np.array(skel)

    def run(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            h, w, _ = frame.shape
            
            if results.multi_hand_landmarks:
                lms = results.multi_hand_landmarks[0]
                mp.solutions.drawing_utils.draw_landmarks(frame, lms, self.mp_hands.HAND_CONNECTIONS)
                
                # 1. GET LIVE SKELETON
                live_vec = self.get_skeleton(lms)
                
                # 2. CALCULATE DISTANCES
                distances = {}
                for name, center in self.centroids.items():
                    center_vec = np.array(center)
                    # Euclidean Distance
                    dist = np.linalg.norm(live_vec - center_vec)
                    distances[name] = dist
                
                # 3. VISUALIZE SCORES
                y = 50
                sorted_dists = sorted(distances.items(), key=lambda x: x[1])
                winner = sorted_dists[0][0]
                
                for name, dist in sorted_dists:
                    # Bar length (shorter distance = better match)
                    # We invert it for visual: 1.0 = Perfect, 0.0 = Far
                    score = max(0, 2.0 - dist) / 2.0 
                    bar_w = int(score * 200)
                    
                    color = (0, 255, 0) if name == winner else (0, 0, 255)
                    cv2.rectangle(frame, (150, y-15), (150+bar_w, y), color, -1)
                    cv2.putText(frame, f"{name}: {dist:.2f}", (10, y), 1, 1, (255,255,255), 1)
                    y += 30
                    
                cv2.putText(frame, f"WINNER: {winner}", (10, h-50), 1, 3, (0, 255, 0), 3)

            cv2.imshow("KNN Debugger", frame)
            if cv2.waitKey(1) == 27: break

if __name__ == "__main__":
    LiveDebugger().run()