import cv2
import mediapipe as mp
import math
import numpy as np
import time
import sys

# CONFIGURATION
FRAMES_PER_GESTURE = 200

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) 
mp_draw = mp.solutions.drawing_utils

def get_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def draw_hud(frame, text_lines, color=(0, 255, 0)):
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)
    y = 30
    for line in text_lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 30

def calculate_metrics(lms, wrist, scale, metrics_list):
    """
    Calculates requested metrics for a single frame.
    Returns a dictionary {metric_name: value}.
    """
    data = {}
    
    # Helper for Open/Closed Ratio
    def get_open_score(tip_idx, base_idx):
        d_tip = get_dist(lms.landmark[tip_idx], wrist)
        d_base = get_dist(lms.landmark[base_idx], wrist)
        return d_tip / (d_base + 0.001)

    # 1. PINCH (Thumb Tip to Index Tip)
    if "pinch_dist" in metrics_list:
        data["pinch_dist"] = get_dist(lms.landmark[4], lms.landmark[8]) / scale

    # 2. FINGER OPEN SCORES
    if "index_open_score" in metrics_list:
        data["index_open_score"] = get_open_score(8, 5)
    if "middle_open_score" in metrics_list:
        data["middle_open_score"] = get_open_score(12, 9)
    if "pinky_open_score" in metrics_list:
        data["pinky_open_score"] = get_open_score(20, 17)

    # 3. THUMB EXTENSION (Thumb Tip to Index Base)
    if "thumb_out_dist" in metrics_list:
        data["thumb_out_dist"] = get_dist(lms.landmark[4], lms.landmark[5]) / scale

    # 4. GEN Z HEART CROSS (Thumb Tip to Middle Finger Base)
    if "thumb_cross_dist" in metrics_list:
        # This is the critical metric for the Heart gesture
        data["thumb_cross_dist"] = get_dist(lms.landmark[4], lms.landmark[9]) / scale

    return data

def capture_session(name, metrics, cap):
    samples = {m: [] for m in metrics}
    valid_frames = 0
    state = "WAITING"
    countdown_start = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        
        # --- STATE: WAITING ---
        if state == "WAITING":
            draw_hud(frame, [
                f"GESTURE: {name}", 
                "Press [SPACE] to start 3s Countdown.",
                "(Two hands supported)"
            ], (0, 255, 255))
            
            if res.multi_hand_landmarks:
                for lms in res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

            cv2.imshow("Gesture Tuner V7", frame)
            key = cv2.waitKey(1)
            if key == 32: # SPACE
                state = "COUNTDOWN"
                countdown_start = time.time()
            elif key == 27:
                sys.exit()

        # --- STATE: COUNTDOWN ---
        elif state == "COUNTDOWN":
            elapsed = time.time() - countdown_start
            remaining = 3.0 - elapsed
            
            if res.multi_hand_landmarks:
                for lms in res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)
            
            if remaining <= 0:
                state = "RECORDING"
                valid_frames = 0
                samples = {m: [] for m in metrics}
            else:
                center_x, center_y = w // 2, h // 2
                cv2.putText(frame, f"{int(remaining)+1}", (center_x-50, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
                draw_hud(frame, [f"PREPARE: {name}", "Position your hands now!"], (0, 0, 255))

            cv2.imshow("Gesture Tuner V7", frame)
            cv2.waitKey(1)

        # --- STATE: RECORDING ---
        elif state == "RECORDING":
            if res.multi_hand_landmarks:
                target_hand = res.multi_hand_landmarks[0]
                
                # Intelligent Selection (same as before)
                if len(res.multi_hand_landmarks) > 1:
                    if "thumb_out_dist" in metrics:
                        best_val = 0
                        for hand in res.multi_hand_landmarks:
                            wrist = hand.landmark[0]
                            idx_mcp = hand.landmark[5]
                            scale = get_dist(wrist, idx_mcp) or 1.0
                            val = get_dist(hand.landmark[4], hand.landmark[5]) / scale
                            if val > best_val:
                                best_val = val
                                target_hand = hand
                
                # Visuals
                for lms in res.multi_hand_landmarks:
                    color_spec = mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
                    if lms == target_hand:
                        color_spec = mp_draw.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=4)
                    mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS, color_spec)

                # Capture
                lms = target_hand
                wrist = lms.landmark[0]
                idx_mcp = lms.landmark[5]
                scale = get_dist(wrist, idx_mcp)
                if scale < 0.01: scale = 1.0

                frame_data = calculate_metrics(lms, wrist, scale, metrics)
                for k, v in frame_data.items():
                    samples[k].append(v)

                valid_frames += 1
                
                bar_w = int((valid_frames / FRAMES_PER_GESTURE) * w)
                cv2.rectangle(frame, (0, h-20), (bar_w, h), (0, 255, 0), -1)
                
                # Show live value of first metric for feedback
                first_metric = metrics[0]
                live_val = frame_data[first_metric]
                draw_hud(frame, [
                    f"RECORDING... {valid_frames}/{FRAMES_PER_GESTURE}", 
                    f"Live {first_metric}: {live_val:.2f}"
                ], (0, 255, 0))
                
                if valid_frames >= FRAMES_PER_GESTURE:
                    state = "REVIEW"
            else:
                draw_hud(frame, ["NO HAND FOUND", "Check camera."], (0, 0, 255))

            cv2.imshow("Gesture Tuner V7", frame)
            cv2.waitKey(1)

        # --- STATE: REVIEW ---
        elif state == "REVIEW":
            stats_text = []
            # Calculate quick stats for display
            for m in metrics:
                vals = samples[m]
                avg = sum(vals)/len(vals)
                stats_text.append(f"{m}: {avg:.2f}")

            draw_hud(frame, [
                f"DONE! Captured {valid_frames} frames.", 
                f"Avgs: {', '.join(stats_text)}",
                "Press [S] to SAVE, [R] to REDO."
            ], (255, 0, 255))
            
            cv2.imshow("Gesture Tuner V7", frame)
            key = cv2.waitKey(0)
            
            if key == ord('s') or key == ord('S'):
                return samples
            elif key == ord('r') or key == ord('R'):
                state = "WAITING"
            elif key == 27:
                sys.exit()

def run_tuner():
    cap = cv2.VideoCapture(0)
    print(f"🧪 INTERACTIVE TUNER V7 (Data Table Mode)")
    
    # Define what to capture for each gesture
    targets = [
        ("1. PINCH (Index touches Thumb)", ["pinch_dist"]),
        ("2. POINTER (Index Up)", ["index_open_score", "middle_open_score", "thumb_cross_dist"]), # Added thumb_cross to see baseline
        ("3. ZOOM HAND (Thumb OUT)", ["thumb_out_dist"]),
        ("4. FIST (All Closed)", ["index_open_score", "pinky_open_score", "thumb_cross_dist"]), # Added thumb_cross to see baseline
        ("5. GEN Z HEART (Index Up + Thumb Cross)", ["thumb_cross_dist", "index_open_score", "middle_open_score"])
    ]
    
    report_data = {}

    for name, metrics in targets:
        data = capture_session(name, metrics, cap)
        report_data[name] = data

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*80)
    print("🔬 FINAL CALIBRATION ANALYSIS TABLE")
    print("="*80)
    
    # Print Header
    print(f"{'GESTURE':<35} | {'METRIC':<20} | {'MIN':<6} | {'MAX':<6} | {'AVG':<6} | {'STD':<6}")
    print("-" * 80)

    for name, data_dict in report_data.items():
        short_name = name.split("(")[0].strip()
        for metric, values in data_dict.items():
            v_min = min(values)
            v_max = max(values)
            v_avg = sum(values) / len(values)
            v_std = np.std(values)
            
            print(f"{short_name:<35} | {metric:<20} | {v_min:.3f}  | {v_max:.3f}  | {v_avg:.3f}  | {v_std:.3f}")
        print("-" * 80)

    print("\n🧠 ANALYSIS GUIDE:")
    print("1. GEN Z HEART THRESHOLD:")
    print("   Look at 'thumb_cross_dist' for Gesture 5.")
    print("   Compare it to 'thumb_cross_dist' for Gesture 2 (Pointer) and 4 (Fist).")
    print("   Your threshold should be halfway between the Heart MAX and the Fist/Pointer MIN.")
    print("   (Lower value = Thumb is closer to middle base).")
    
    print("\n2. COPY-PASTE TO gestures.py:")
    
    # Auto-calculate recommendation for Heart
    try:
        heart_cross = report_data["5. GEN Z HEART (Index Up + Thumb Cross)"]["thumb_cross_dist"]
        # Use the MAX of the heart gesture (worst case valid gesture)
        # Add a small buffer (e.g., 0.05) to ensure reliability
        rec_thresh = max(heart_cross) + 0.05
        print(f"   self.heart_cross_thresh = {rec_thresh:.2f}  # (Derived from your max {max(heart_cross):.2f})")
    except:
        print("   (Could not calculate Heart threshold - missing data)")

    print("="*80)

if __name__ == "__main__":
    run_tuner()