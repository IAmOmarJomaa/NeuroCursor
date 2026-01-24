import cv2, mediapipe as mp, time, math, numpy as np, sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.gesture_engine import NeuroCursorBrain
from src.config import CONFIG

def run_diagnostic():
    print("🥋 DIAGNOSTIC DOJO: SYSTEM TELEMETRY")
    print("Commands: [Space] Start 5s Drill | [ESC] Exit")
    
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    brain = NeuroCursorBrain()
    hands = mp.solutions.hands.Hands(max_num_hands=1, model_complexity=0)
    
    recording = False
    start_time = 0
    telemetry = []

    while True:
        ret, frame = cam.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        now = time.time()
        
        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            label, conf = brain.predict(lms.landmark)
            
            # 1. Raw Data
            ix, iy = lms.landmark[8].x, lms.landmark[8].y
            tx, ty = lms.landmark[4].x, lms.landmark[4].y
            dist = math.hypot(ix - tx, iy - ty)
            
            # 2. Physics Check (Internal Logic Simulation)
            p_start = CONFIG.get("PINCH_START", 0.034)
            is_p = dist < p_start

            if recording:
                # Optimized for token efficiency: [Time, Label, Dist, RawX, RawY]
                telemetry.append([round(now-start_time, 3), label[:4], round(dist, 4), round(ix, 4), round(iy, 4)])
                cv2.putText(frame, f"RECORDING... {5 - int(now-start_time)}s", (20, 50), 1, 2, (0,0,255), 2)
                if now - start_time > 5:
                    recording = False
                    process_telemetry(telemetry)

            cv2.circle(frame, (int(ix*w), int(iy*h)), 5, (0,255,0), -1)

        cv2.imshow("Dojo", frame)
        k = cv2.waitKey(1)
        if k == 27: break
        if k == ord(' ') and not recording:
            print("🚀 DRILL STARTING: Perform the problematic action (e.g. Select Text) NOW.")
            telemetry = []
            start_time = time.time()
            recording = True

    cam.release()
    cv2.destroyAllWindows()

def process_telemetry(data):
    print("\n📊 --- TELEMETRY DUMP (Paste this) ---")
    # Headers: T(Time), L(Label), D(Dist), X, Y
    print("T,L,D,X,Y")
    # Sample every 3rd frame to save tokens while keeping resolution
    for row in data[::3]:
        print(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}")
    print("--- END DUMP ---")

if __name__ == "__main__":
    run_diagnostic()