import cv2
import mediapipe as mp
import math
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.gesture_engine import NeuroCursorBrain

def run_measure():
    print("📏 DISTANCE MEASURER")
    print("   -> Open Hand (Point) vs Closed Hand (Click)")
    print("   -> Find the number that separates them.")
    
    brain = NeuroCursorBrain()
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, model_complexity=0)
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cam.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            
            # 1. Get AI Prediction
            label, conf = brain.predict(lms.landmark)
            
            # 2. Calculate Geometry (Thumb Tip #4 to Index Tip #8)
            x1, y1 = lms.landmark[4].x, lms.landmark[4].y
            x2, y2 = lms.landmark[8].x, lms.landmark[8].y
            
            # Distance in "Normalized Space" (0.0 to 1.0)
            # This is robust regardless of screen resolution
            dist = math.hypot(x2 - x1, y2 - y1)
            
            # Visuals
            p4 = (int(x1 * w), int(y1 * h))
            p8 = (int(x2 * w), int(y2 * h))
            
            col = (0, 255, 0) # Green
            if label == "THE_POINT": col = (0, 0, 255) # Red
            
            cv2.line(frame, p4, p8, (255, 255, 255), 2)
            cv2.circle(frame, p4, 5, (0, 255, 255), -1)
            cv2.circle(frame, p8, 5, (0, 255, 255), -1)
            
            # Draw Data
            cv2.putText(frame, f"AI SAYS: {label} ({int(conf*100)}%)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)
            cv2.putText(frame, f"DIST: {dist:.3f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        cv2.imshow("Measure Pinch", frame)
        if cv2.waitKey(1) == 27: break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_measure()