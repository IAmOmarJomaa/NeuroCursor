import cv2
import mediapipe as mp
import sys
import os
import math
import numpy as np
import time
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import CONFIG
from src.gesture_engine import NeuroCursorBrain

def calculate_geometry_metrics(lms):
    """
    Calculates the hidden numbers that differentiate similar gestures.
    Returns a dictionary of 'Discriminator Values'.
    """
    # 1. Palm Scale (Wrist to Middle MCP) - normalization factor
    palm_size = math.hypot(lms[9].x - lms[0].x, lms[9].y - lms[0].y)
    if palm_size == 0: palm_size = 1.0

    # 2. Thumb Tuckedness (Thumb Tip to Index MCP)
    # LOW = Tucked (Delete), HIGH = Out (Palm)
    thumb_tip = lms[4]
    index_mcp = lms[5]
    thumb_dist = math.hypot(thumb_tip.x - index_mcp.x, thumb_tip.y - index_mcp.y)
    thumb_score = thumb_dist / palm_size

    # 3. Pinky/Ring Curl (Tip to Wrist)
    # LOW = Curled (Tab), HIGH = Extended (Delete/Palm)
    ring_tip = lms[16]
    pinky_tip = lms[20]
    wrist = lms[0]
    
    ring_dist = math.hypot(ring_tip.x - wrist.x, ring_tip.y - wrist.y) / palm_size
    pinky_dist = math.hypot(pinky_tip.x - wrist.x, pinky_tip.y - wrist.y) / palm_size
    
    curl_score = (ring_dist + pinky_dist) / 2.0

    # 4. Index/Middle Separation (Scissors/Peace sign)
    # Used for Tab vs others
    index_tip = lms[8]
    mid_tip = lms[12]
    finger_spread = math.hypot(index_tip.x - mid_tip.x, index_tip.y - mid_tip.y) / palm_size

    return {
        "Thumb Tucked": thumb_score,
        "Ring/Pinky Ext": curl_score,
        "Index/Mid Spread": finger_spread
    }

def run_lab():
    print("🔬 CONFUSION LAB")
    print("   -> Identify exactly why gestures get mixed up.")
    print("   -> Watch the 'METRICS' panel to find the cutoff numbers.")
    print("   -> Press 'R' to reset recording stats.")
    
    cv2.namedWindow("Confusion Lab", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Confusion Lab", 1200, 700)
    
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, model_complexity=1)
    
    brain = NeuroCursorBrain()
    
    # Statistics Container
    history_log = []
    
    while True:
        ret, frame = cam.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # UI Layout
        # Left: Camera, Right: Data
        ui_w = 400
        canvas = np.zeros((h, w + ui_w, 3), dtype=np.uint8)
        canvas[:h, :w] = frame
        
        # Draw UI Background
        cv2.rectangle(canvas, (w, 0), (w+ui_w, h), (30, 30, 30), -1)
        cv2.putText(canvas, "GEOMETRY METRICS", (w+20, 40), 1, 1.2, (0, 255, 255), 2)
        cv2.putText(canvas, "(Use these for Rules)", (w+20, 70), 1, 0.7, (150, 150, 150), 1)

        if results.multi_hand_landmarks:
            for i, lms in enumerate(results.multi_hand_landmarks):
                # 1. Get Prediction
                label, conf = brain.predict(lms.landmark)
                history_log.append(label)
                if len(history_log) > 50: history_log.pop(0)
                
                # 2. Get Geometry Math
                metrics = calculate_geometry_metrics(lms.landmark)
                
                # 3. Draw on Camera
                mp.solutions.drawing_utils.draw_landmarks(
                    canvas[:h, :w], lms, mp_hands.HAND_CONNECTIONS)
                
                wrist = lms.landmark[0]
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(canvas, f"H{i}: {label}", (cx, cy-20), 1, 1, (0, 255, 0), 2)

                # 4. Draw Stats Panel
                y = 120 + (i * 250)
                
                # Header
                cv2.putText(canvas, f"HAND {i+1} PREDICTION:", (w+20, y), 1, 1, (255, 255, 255), 2)
                cv2.putText(canvas, f"{label} ({conf:.2f})", (w+20, y+35), 1, 1.5, (0, 255, 0), 2)
                
                # Divider
                cv2.line(canvas, (w+20, y+50), (w+ui_w-20, y+50), (100,100,100), 1)
                
                # Metrics (The G-Spot for Rules)
                
                # A. THUMB TUCKED (Palm vs Delete)
                # Value < 0.35 usually means Tucked
                val = metrics["Thumb Tucked"]
                col = (0, 0, 255) if val < 0.35 else (0, 255, 0)
                state = "TUCKED" if val < 0.35 else "OUT"
                cv2.putText(canvas, f"Thumb: {val:.3f} [{state}]", (w+20, y+80), 1, 0.9, col, 1)
                
                # B. RING/PINKY EXT (Tab vs Delete)
                # Value < 1.0 usually means Curled
                val = metrics["Ring/Pinky Ext"]
                col = (0, 0, 255) if val < 1.0 else (0, 255, 0)
                state = "CURLED" if val < 1.0 else "OPEN"
                cv2.putText(canvas, f"Pinky: {val:.3f} [{state}]", (w+20, y+110), 1, 0.9, col, 1)

                # C. SPREAD (Tab vs Peace)
                val = metrics["Index/Mid Spread"]
                cv2.putText(canvas, f"Spread: {val:.3f}", (w+20, y+140), 1, 0.9, (200,200,200), 1)

        # 5. History Distribution (Bottom Panel)
        y_hist = h - 150
        cv2.putText(canvas, "RECENT HISTORY (50 frames):", (w+20, y_hist), 1, 0.8, (255,255,255), 1)
        
        counts = Counter(history_log)
        top_3 = counts.most_common(3)
        
        for idx, (lbl, count) in enumerate(top_3):
            bar_w = int((count / 50) * 300)
            cv2.rectangle(canvas, (w+20, y_hist + 20 + (idx*30)), (w+20+bar_w, y_hist + 40 + (idx*30)), (255, 100, 0), -1)
            cv2.putText(canvas, f"{lbl}: {count}", (w+30, y_hist + 38 + (idx*30)), 1, 0.8, (255,255,255), 1)

        cv2.imshow("Confusion Lab", canvas)
        
        k = cv2.waitKey(1)
        if k == 27: break
        if k == ord('r'): history_log = []

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_lab()