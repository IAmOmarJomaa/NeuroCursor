import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import PATHS
from src.hand_utils import pre_process_landmark # <--- UPDATED IMPORT

MODEL_PATH = str(PATHS["MODELS_DIR"]) + "/neurocursor_model.keras"
LABEL_MAP_PATH = str(PATHS["MODELS_DIR"]) + "/label_map.pkl"

def run_live_test():
    print("🧠 LOADING NEURO-CURSOR BRAIN...")
    if not os.path.exists(MODEL_PATH): return
    
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_MAP_PATH, 'rb') as f:
        label_map = pickle.load(f)
    
    print("✅ READY.")
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    while True:
        ret, frame = cam.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            mp.solutions.drawing_utils.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)
            
            # --- THE MAGIC FIX ---
            input_vec = pre_process_landmark(lms.landmark)
            input_tensor = np.expand_dims(input_vec, axis=0)
            
            predictions = model(input_tensor, training=False).numpy()[0]
            best_idx = np.argmax(predictions)
            label_text = label_map[best_idx]
            conf = predictions[best_idx]
            
            col = (0, 255, 0) if conf > 0.8 else (0, 255, 255)
            if conf < 0.5: col = (0, 0, 255)
            
            cv2.rectangle(frame, (50, 50), (350, 100), (20, 20, 20), -1)
            cv2.putText(frame, f"{label_text}", (60, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)
            cv2.putText(frame, f"{conf*100:.0f}%", (280, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            sorted_idxs = np.argsort(predictions)[::-1]
            if predictions[sorted_idxs[1]] > 0.05:
                sec_lbl = label_map[sorted_idxs[1]]
                cv2.putText(frame, f"vs {sec_lbl}", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

        cv2.imshow("NeuroCursor Debug", frame)
        if cv2.waitKey(1) == 27: break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_test()