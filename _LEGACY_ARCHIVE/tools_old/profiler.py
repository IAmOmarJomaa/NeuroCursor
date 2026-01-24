import cv2
import mediapipe as mp
import time
import pickle
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import PATHS

# --- LOAD RESOURCES ---
print("⏳ LOADING RESOURCES...")
try:
    with open(PATHS["MODELS_DIR"] / "neurocursor_model.pkl", 'rb') as f:
        MODEL = pickle.load(f)
except:
    sys.exit("❌ MODEL MISSING")

FEATURE_COLUMNS = [
    'thumb_bend', 'index_bend', 'mid_bend', 'ring_bend', 'pinky_bend',
    'pinch_dist', 'thumb_spread', 'mid_palm_dist', 
    'orientation_y', 'orientation_x'
]

# --- MATH ENGINE (The suspected bottleneck) ---
class FeatureExtractor:
    @staticmethod
    def extract(lms):
        coords = np.array([[lm.x, lm.y, lm.z] for lm in lms])
        
        def get_ang(i1, i2, i3):
            v1 = coords[i2] - coords[i1]
            v2 = coords[i3] - coords[i2]
            dot = np.dot(v1, v2)
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            return np.degrees(np.arccos(np.clip(dot/(norm+1e-6), -1.0, 1.0)))

        def get_dist(i1, i2, norm_factor):
            return np.linalg.norm(coords[i1] - coords[i2]) / norm_factor

        palm = np.linalg.norm(coords[9] - coords[0]) + 1e-6
        
        data = {
            'thumb_bend': get_ang(1, 2, 3),
            'index_bend': get_ang(5, 6, 7),
            'mid_bend':   get_ang(9, 10, 11),
            'ring_bend':  get_ang(13, 14, 15),
            'pinky_bend': get_ang(17, 18, 19),
            'pinch_dist': get_dist(4, 8, palm),
            'thumb_spread': get_dist(4, 5, palm),
            'mid_palm_dist': get_dist(12, 0, palm),
            'orientation_y': coords[9][1] - coords[0][1],
            'orientation_x': coords[9][0] - coords[0][0]
        }
        # CRITICAL: Pandas creation might be slow
        return pd.DataFrame([data], columns=FEATURE_COLUMNS)

def run_profiler():
    cap = cv2.VideoCapture(0)
    hands = mp.solutions.hands.Hands(max_num_hands=1, model_complexity=0)
    
    print("\n🚀 STARTING PROFILER (Running for 100 Frames)...")
    print("   -> Move your hand in front of the camera now.")
    
    # TIMING STORAGE
    t_cam = []
    t_mp = []
    t_math = []
    t_pred = []
    t_total = []
    
    frame_count = 0
    
    while frame_count < 100:
        start_loop = time.perf_counter()
        
        # 1. CAMERA
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret: continue
        t_cam.append(time.perf_counter() - t0)
        
        # 2. MEDIAPIPE
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        t_mp.append(time.perf_counter() - t0)
        
        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            
            # 3. MATH (Feature Extraction)
            t0 = time.perf_counter()
            feats = FeatureExtractor.extract(lms.landmark)
            t_math.append(time.perf_counter() - t0)
            
            # 4. PREDICTION (The Model)
            t0 = time.perf_counter()
            _ = MODEL.predict(feats)[0]
            t_pred.append(time.perf_counter() - t0)
            
            frame_count += 1
            sys.stdout.write(f"\rProgress: {frame_count}/100 frames")
            sys.stdout.flush()
        
        t_total.append(time.perf_counter() - start_loop)

    cap.release()
    
    # --- REPORT GENERATION ---
    avg_cam = np.mean(t_cam) * 1000
    avg_mp = np.mean(t_mp) * 1000
    avg_math = np.mean(t_math) * 1000
    avg_pred = np.mean(t_pred) * 1000
    avg_total = np.mean(t_total) * 1000
    
    fps = 1000 / avg_total
    
    print("\n\n📊 --- LATENCY REPORT (Lower is Better) ---")
    print(f"📷 Camera Read:      {avg_cam:.2f} ms")
    print(f"🖐️ MediaPipe:       {avg_mp:.2f} ms")
    print(f"📐 Math (Extract):  {avg_math:.2f} ms  <-- Check This")
    print(f"🧠 Brain (Predict): {avg_pred:.2f} ms  <-- Check This")
    print(f"----------------------------------------")
    print(f"⏱️ TOTAL FRAME TIME: {avg_total:.2f} ms")
    print(f"🚀 ACTUAL FPS:       {fps:.2f} FPS")
    print("\nVERDICT:")
    
    if fps < 15:
        print("❌ UNUSABLE. Major bottleneck detected.")
    elif fps < 30:
        print("⚠️ LAGGY. Optimization required.")
    else:
        print("✅ SMOOTH. The lag might be in the Mouse Thread (PyAutoGUI).")

if __name__ == "__main__":
    run_profiler()