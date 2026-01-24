import cv2
import mediapipe as mp
import math
import sys
import numpy as np

# SETUP
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def get_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def print_separator(title=""):
    print(f"\n{title}".ljust(60, "="))

def analyze_frame(lms):
    # 1. NORMALIZATION SCALE (Wrist to Index MCP)
    wrist = lms.landmark[0]
    idx_mcp = lms.landmark[5]
    scale = get_dist(wrist, idx_mcp)
    if scale < 0.01: scale = 1.0

    print_separator(f" 📸 SNAPSHOT ANALYSIS (Scale Factor: {scale:.4f}) ")

    # --- PART A: FINGER OPENNESS (The Ratios) ---
    # Logic: Dist(Tip, Wrist) / Dist(Base, Wrist)
    # > 1.25 usually means OPEN. < 1.0 usually means CLOSED.
    print(f"\n{'FINGER':<12} | {'TIP-WRIST':<10} | {'BASE-WRIST':<10} | {'RATIO (Open?)':<15}")
    print("-" * 55)
    
    fingers = [
        ("Thumb", 4, 2),   # Thumb behaves differently, but let's see raw data
        ("Index", 8, 5),
        ("Middle", 12, 9),
        ("Ring", 16, 13),
        ("Pinky", 20, 17)
    ]
    
    for name, tip_idx, base_idx in fingers:
        d_tip = get_dist(lms.landmark[tip_idx], wrist)
        d_base = get_dist(lms.landmark[base_idx], wrist)
        ratio = d_tip / (d_base + 0.001)
        status = "OPEN  ✅" if ratio > 1.20 else "CLOSED ❌"
        print(f"{name:<12} | {d_tip:.3f}      | {d_base:.3f}       | {ratio:.3f} ({status})")

    # --- PART B: THE THUMB RADAR (Finding the Heart) ---
    # This shows the distance from THUMB TIP (4) to CRITICAL BASES.
    # For Gen Z Heart, look at "Thumb -> Mid Base (9)".
    print_separator(" 📡 THUMB RADAR (Distance from Thumb Tip #4) ")
    print(f"{'TARGET JOINT':<25} | {'LANDMARK #':<10} | {'NORMALIZED DIST':<15}")
    print("-" * 55)

    targets = [
        ("Index Base (MCP)", 5),
        ("Middle Base (MCP)", 9),  # <--- WATCH THIS FOR HEART
        ("Ring Base (MCP)", 13),
        ("Pinky Base (MCP)", 17),
        ("Index Tip", 8),          # Pinch check
        ("Middle Tip", 12),
        ("Ring Tip", 16),
        ("Pinky Tip", 20)          # Shaka check
    ]

    for name, idx in targets:
        d = get_dist(lms.landmark[4], lms.landmark[idx]) / scale
        mark = ""
        if idx == 9: mark = "  <-- HEART CHECK ❤️"
        if idx == 8: mark = "  <-- PINCH CHECK 👌"
        print(f"{name:<25} | {idx:<10} | {d:.4f}{mark}")

    print("\n" + "="*60 + "\n")

def run_inspector():
    cap = cv2.VideoCapture(0)
    print("🕵️ GESTURE INSPECTOR LAUNCHED")
    print("Position hand -> Press [SPACE] to Freeze & Analyze -> Press [SPACE] again to Resume.")
    
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret: break
            
            # Standard View
            frame = cv2.flip(frame, 1) 
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            
            if res.multi_hand_landmarks:
                for lms in res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)
                    
                    # Draw visual for Heart Target (Thumb 4 -> Middle Base 9)
                    p1 = lms.landmark[4]
                    p2 = lms.landmark[9]
                    h, w, _ = frame.shape
                    cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (0, 0, 255), 2)

            cv2.putText(frame, "PRESS [SPACE] TO SNAPSHOT DATA", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Gesture Inspector", frame)

        key = cv2.waitKey(1)
        if key == 32: # SPACE
            if not paused:
                # Capture and Analyze
                if res.multi_hand_landmarks:
                    analyze_frame(res.multi_hand_landmarks[0])
                    print("❄️  FRAME FROZEN. CHECK CONSOLE. PRESS [SPACE] TO RESUME.")
                    paused = True
            else:
                print("▶️  RESUMING...")
                paused = False
        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inspector()