import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import os
import sys

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import CONFIG

def main():
    print("🩺 MODEL DOCTOR: DIAGNOSTIC TOOL")
    print("=================================")
    print(" [M] TOGGLE MODE (Absolute vs Relative)")
    print(" [ESC] EXIT")

    # 1. LOAD MODEL
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_dir = os.path.join(project_root, "models")
    
    model_path = os.path.join(model_dir, "neurocursor_model.keras")
    label_path = os.path.join(model_dir, "label_map.pkl")
    
    print(f"Loading: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        le = joblib.load(label_path)
        print("✅ Model Loaded Successfully")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        return

    # 2. SETUP CAMERA
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, model_complexity=0)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # 3. MODES
    mode = 0 # 0 = Absolute, 1 = Relative
    mode_names = ["ABSOLUTE (Raw)", "RELATIVE (Wrist=0)"]

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # UI
        cv2.rectangle(frame, (0, 0), (640, 80), (30, 30, 30), -1)
        cv2.putText(frame, f"MODE: {mode_names[mode]} (Press 'M')", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            mp.solutions.drawing_utils.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

            # --- DATA PREPROCESSING ---
            data_row = []
            
            if mode == 0:
                # ABSOLUTE (What we have now)
                for lm in lms.landmark:
                    data_row.extend([lm.x, lm.y, lm.z])
                    
            elif mode == 1:
                # RELATIVE (Wrist at 0,0,0)
                base_x, base_y, base_z = lms.landmark[0].x, lms.landmark[0].y, lms.landmark[0].z
                for lm in lms.landmark:
                    data_row.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])

            # --- PREDICT ---
            input_data = np.array([data_row], dtype=np.float32)
            
            try:
                prediction = model.predict(input_data, verbose=0)
                class_idx = np.argmax(prediction)
                conf = np.max(prediction)
                
                # DECODE LABEL
                if isinstance(le, dict):
                    label = le.get(class_idx, "UNKNOWN")
                else:
                    label = le.inverse_transform([class_idx])[0]
                
                # DISPLAY
                color = (0, 255, 0) if conf > 0.7 else (0, 0, 255)
                cv2.putText(frame, f"PREDICTION: {label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"CONF: {conf:.4f}", (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
                
                # Print raw array to console if confused
                # print(f"Raw: {prediction}")

            except Exception as e:
                print(f"Prediction Error: {e}")

        cv2.imshow("Model Doctor", frame)
        
        k = cv2.waitKey(1)
        if k == 27: break
        if k == ord('m'): 
            mode = (mode + 1) % 2
            print(f"🔄 Switched to {mode_names[mode]}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()