"""
NeuroCursor Dataset Recorder (The Studio).

This tool captures raw hand landmarks for training the Neural Network.
It aligns strictly with the runtime configuration (FPS, Resolution) to ensure
data consistency.

Usage:
    Run this script, select a gesture label with [W/S], and hold [SPACE] to record.
"""

import cv2
import mediapipe as mp
import pandas as pd
import sys
import os
import csv
import numpy as np

# --- PATH SETUP ---
# Add project root to path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import PATHS, GESTURE_LABELS, CONFIG

# CONFIG
OUTPUT_FILE = str(PATHS["RAW_DATA"])
LABELS = GESTURE_LABELS

# SETUP
mp_hands = mp.solutions.hands
# Note: We use higher confidence (0.7) for recording to ensure high-quality ground truth
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def is_hand_safe(lms):
    """
    Safety Guard: Refuses to record if hand is touching the edge of the screen.
    Prevents 'half-hand' data which confuses the AI.
    """
    margin = 0.02 # 2% margin
    for lm in lms.landmark:
        if lm.x < margin or lm.x > (1-margin) or lm.y < margin or lm.y > (1-margin):
            return False
    return True

def load_existing_data():
    """Loads current dataset to allow appending/resuming session."""
    if not os.path.exists(OUTPUT_FILE):
        return []
    try:
        df = pd.read_csv(OUTPUT_FILE)
        print(f"🔄 Loaded {len(df)} samples.")
        return df.to_dict('records')
    except Exception as e:
        print(f"⚠️ Error loading CSV: {e}")
        return []

def save_full_csv(data_list):
    """Atomic save of the entire dataset."""
    if not data_list:
        if os.path.exists(OUTPUT_FILE): os.remove(OUTPUT_FILE)
        return
    
    keys = data_list[0].keys()
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data_list)
    print(f"💾 Saved {len(data_list)} total samples.")

def draw_interface(frame, current_idx, recording, count, last_msg, counts, safe_status):
    """Renders the Studio UI overlay."""
    h, w, _ = frame.shape
    
    # 1. Status Header
    # Red = Recording, Dark = Standby, Orange = Unsafe Hand
    if recording:
        bar_col = (0, 0, 255) if safe_status else (0, 165, 255) 
        status_txt = f"REC: {LABELS[current_idx].split('. ')[1]}"
        if not safe_status: status_txt = "⚠️ HAND OFF SCREEN"
    else:
        bar_col = (50, 50, 50)
        status_txt = "STANDBY (Select Gesture)"

    cv2.rectangle(frame, (0, 0), (w, 60), bar_col, -1)
    cv2.putText(frame, status_txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"TOTAL: {count}", (w-200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # 2. Sidebar Menu
    panel_w = 300
    cv2.rectangle(frame, (0, 60), (panel_w, h), (30, 30, 30), -1)
    
    # Scroll Logic (Keep selection centered)
    max_items = (h - 100) // 30
    start_view = max(0, current_idx - 5)
    end_view = min(len(LABELS), start_view + max_items)
    
    y = 100
    for i in range(start_view, end_view):
        clean_lbl = LABELS[i].split(". ")[1]
        is_selected = (i == current_idx)
        n_samples = counts.get(clean_lbl, 0)
        
        # Highlight Selection
        if is_selected:
            cv2.rectangle(frame, (0, y-20), (panel_w, y+10), (80, 80, 80), -1)
            color = (0, 255, 0)
            prefix = ">> "
        else:
            color = (180, 180, 180)
            prefix = "   "
            
        cv2.putText(frame, f"{prefix}{clean_lbl} ({n_samples})", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 30

    # 3. Footer Controls
    cv2.line(frame, (0, h-70), (panel_w, h-70), (100, 100, 100), 1)
    cv2.putText(frame, f"MSG: {last_msg}", (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, "[W/S] Nav  [SPACE] Rec  [R] Redo  [X] Wipe", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    if recording and safe_status:
        cv2.rectangle(frame, (0,0), (w,h), (0,0,255), 4)

def run_recorder():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # [CRITICAL] SYNC FPS WITH RUNTIME CONFIG
    # This ensures motion blur matches what the AI will see in production.
    cap.set(cv2.CAP_PROP_FPS, CONFIG["TARGET_FPS"])
    
    # Load Data
    full_data = load_existing_data()
    
    current_idx = 0
    recording = False
    batch_buffer = [] 
    last_msg = "Ready."
    
    print(f"🚀 STUDIO RECORDER: Online (Targeting {CONFIG['TARGET_FPS']} FPS)")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        hand_is_safe = False
        
        if results.multi_hand_landmarks:
            for lms in results.multi_hand_landmarks:
                # 1. Safety Check
                if is_hand_safe(lms):
                    hand_is_safe = True
                    # Green Skeleton = Good
                    mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS, 
                                           mp_draw.DrawingSpec(color=(0,255,0))) 
                    
                    if recording:
                        clean_lbl = LABELS[current_idx].split(". ")[1]
                        row = {"label": clean_lbl}
                        for i, lm in enumerate(lms.landmark):
                            row[f"x{i}"], row[f"y{i}"], row[f"z{i}"] = lm.x, lm.y, lm.z
                        batch_buffer.append(row)
                else:
                    # Red Skeleton = Bad (Edge)
                    mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS,
                                           mp_draw.DrawingSpec(color=(0,0,255)))

        # Stats Helper
        counts = {}
        # Count existing data
        for r in full_data: 
            counts[r['label']] = counts.get(r['label'], 0) + 1
        # Add current batch
        if recording:
            lbl = LABELS[current_idx].split(". ")[1]
            counts[lbl] = counts.get(lbl, 0) + len(batch_buffer)

        draw_interface(frame, current_idx, recording, len(full_data) + len(batch_buffer), last_msg, counts, hand_is_safe)
        cv2.imshow("Studio Recorder", frame)
        
        k = cv2.waitKey(1)
        if k == 27: # ESC
            if batch_buffer: full_data.extend(batch_buffer)
            save_full_csv(full_data)
            break
            
        # --- NAVIGATION ---
        elif k == ord('w'): current_idx = (current_idx - 1) % len(LABELS)
        elif k == ord('s'): current_idx = (current_idx + 1) % len(LABELS)
        
        # --- RECORDING ---
        elif k == 32: # SPACE
            if not recording:
                recording = True
                batch_buffer = []
                last_msg = "Recording..."
            else:
                recording = False
                if batch_buffer:
                    full_data.extend(batch_buffer)
                    save_full_csv(full_data) # Auto-save on stop
                    last_msg = f"Saved {len(batch_buffer)} frames."
                batch_buffer = []

        # --- DELETION ---
        elif k == ord('r') or k == ord('R'):
            # Redo CURRENT label
            target = LABELS[current_idx].split(". ")[1]
            before = len(full_data)
            full_data = [r for r in full_data if r['label'] != target]
            after = len(full_data)
            save_full_csv(full_data)
            last_msg = f"Deleted {before - after} samples of {target}."

        elif k == ord('x') or k == ord('X'):
            # Wipe ALL
            full_data = []
            save_full_csv([])
            last_msg = "⚠️ DATASET WIPED."

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_recorder()