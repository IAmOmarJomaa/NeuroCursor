import cv2
import mediapipe as mp
import math
import time
import pandas as pd
import numpy as np

# CONFIG
OUTPUT_FILE = "gesture_data_dump.csv"

# SETUP
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ORDERED LABELS FOR CYCLING
LABEL_LIST = [
    "POINTER",
    "PINCH",
    "SCROLL_SPIDEY",
    "VOL_SHAKA",
    "PALM",
    "FIST",
    "HEART_GEN_Z",
    "ZOOM_READY",
    "ZOOM_ACTIVE"
]

def get_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def extract_metrics(hand_lms, label, second_hand=None):
    """
    Extracts geometric data for the table.
    """
    wrist = hand_lms.landmark[0]
    idx_mcp = hand_lms.landmark[5]
    
    # 1. SCALE (Normalization)
    scale = get_dist(wrist, idx_mcp)
    if scale < 0.01: scale = 1.0

    data = {
        "label": label,
        "timestamp": time.time()
    }

    # 2. RATIOS (Open/Closed)
    fingers = [(8,5,"idx"), (12,9,"mid"), (16,13,"ring"), (20,17,"pinky")]
    for tip, base, name in fingers:
        d_tip = get_dist(hand_lms.landmark[tip], wrist)
        d_base = get_dist(hand_lms.landmark[base], wrist)
        data[f"{name}_open_ratio"] = round(d_tip / (d_base + 0.001), 3)

    # 3. CRITICAL DISTANCES
    data["pinch_dist"] = round(get_dist(hand_lms.landmark[4], hand_lms.landmark[8]) / scale, 3)
    data["thumb_ext"] = round(get_dist(hand_lms.landmark[4], hand_lms.landmark[5]) / scale, 3)
    data["heart_cross"] = round(get_dist(hand_lms.landmark[4], hand_lms.landmark[9]) / scale, 3)
    data["shaka_dist"] = round(get_dist(hand_lms.landmark[4], hand_lms.landmark[20]) / scale, 3)

    # 4. DUAL HAND (Zoom)
    if second_hand:
        p1 = hand_lms.landmark[8]
        p2 = second_hand.landmark[4]
        data["dual_hand_dist"] = round(math.hypot(p1.x - p2.x, p1.y - p2.y), 3)
    else:
        data["dual_hand_dist"] = 0.0

    return data

def run_snapshot_tool():
    cap = cv2.VideoCapture(0)
    
    # APP STATE
    label_idx = 0
    snapshots = []
    last_action = ""
    
    print("📸 VISUAL SNAPSHOT TOOL V2")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. FLIP FIRST (Fixes the "Ghost Hand" Bug)
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # 2. PROCESS FLIPPED FRAME
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # 3. DRAW UI
        # Background Panel
        cv2.rectangle(frame, (0,0), (320, h), (40, 40, 40), -1)
        cv2.rectangle(frame, (320,0), (322, h), (0, 255, 0), -1) # Divider
        
        # Instructions
        y_cursor = 40
        def draw_text(txt, color=(200,200,200), size=0.6):
            nonlocal y_cursor
            cv2.putText(frame, txt, (15, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1)
            y_cursor += 30

        draw_text("CONTROLS:", (0, 255, 255))
        draw_text("[TAB]   Next Gesture")
        draw_text("[SPACE] Capture Snap")
        draw_text("[Z]     Undo Last")
        draw_text("[X]     Clear Current")
        draw_text("[S]     Save & Analyze")
        draw_text("[ESC]   Quit")
        y_cursor += 20
        
        # Gesture Menu
        draw_text("GESTURE LIST:", (0, 255, 255))
        current_label = LABEL_LIST[label_idx]
        
        counts = {lbl: 0 for lbl in LABEL_LIST}
        for s in snapshots: counts[s['label']] += 1

        for i, lbl in enumerate(LABEL_LIST):
            color = (100, 100, 100) # Dim
            prefix = "   "
            if i == label_idx:
                color = (0, 255, 0) # Active Green
                prefix = "-> "
            
            count = counts[lbl]
            count_str = f"[{count}]" if count > 0 else ""
            draw_text(f"{prefix}{lbl} {count_str}", color)

        # Status Message
        cv2.putText(frame, f"STATUS: {last_action}", (340, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Draw Skeletons
        if results.multi_hand_landmarks:
            for lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Visual Snapshot V2", frame)
        
        key = cv2.waitKey(1)
        
        # --- LOGIC ---
        
        if key == 9: # TAB
            label_idx = (label_idx + 1) % len(LABEL_LIST)
            last_action = f"Switched to {LABEL_LIST[label_idx]}"

        elif key == 32: # SPACE
            if results.multi_hand_landmarks:
                h1 = results.multi_hand_landmarks[0]
                h2 = results.multi_hand_landmarks[1] if len(results.multi_hand_landmarks) > 1 else None
                
                data_row = extract_metrics(h1, current_label, h2)
                snapshots.append(data_row)
                last_action = f"Captured {current_label} (#{counts[current_label]+1})"
            else:
                last_action = "ERROR: No Hand Visible!"

        elif key == ord('z') or key == ord('Z'): # UNDO
            if snapshots:
                removed = snapshots.pop()
                last_action = f"Removed last {removed['label']}"
            else:
                last_action = "Nothing to Undo."

        elif key == ord('x') or key == ord('X'): # CLEAR CURRENT
            new_snaps = [s for s in snapshots if s['label'] != current_label]
            diff = len(snapshots) - len(new_snaps)
            snapshots = new_snaps
            last_action = f"Cleared {diff} from {current_label}"

        elif key == ord('s') or key == ord('S'): # SAVE
            break
        elif key == 27: # ESC
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    # --- ANALYSIS REPORT ---
    if not snapshots:
        print("No data captured.")
        return

    df = pd.DataFrame(snapshots)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Data Saved to: {OUTPUT_FILE}")

    print("\n📊 COMPARATIVE ANALYSIS TABLE")
    print("="*120)
    
    # We group by Label and show the Mean of all metrics
    summary = df.groupby("label")[
        ['pinch_dist', 'thumb_ext', 'heart_cross', 'idx_open_ratio', 'mid_open_ratio', 'pinky_open_ratio', 'dual_hand_dist']
    ].mean()
    
    # Clean print format
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(summary)
    print("="*120)
    print("Use these values to calibrate 'gestures.py'!")

if __name__ == "__main__":
    run_snapshot_tool()