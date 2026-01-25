"""
NeuroCursor Gesture Photo Booth.
===============================
A utility to rapidly capture reference images for the README.
It iterates through your config's GESTURE_LABELS list one by one.

Usage:
    1. Run script.
    2. Strike the pose shown on screen.
    3. Press 'SPACE' to capture and move to the next.
    4. Press 'S' to skip a gesture.
    5. Press 'ESC' to exit.
"""

import cv2
import os
import sys
import time

# Add root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import GESTURE_LABELS

def run_photobooth():
    # Setup Paths
    SAVE_DIR = os.path.join("assets", "gestures")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Setup Camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print(f"📸 STARTING PHOTO BOOTH")
    print(f"📂 Saving images to: {SAVE_DIR}")
    print(f"📝 Total Gestures: {len(GESTURE_LABELS)}\n")

    current_idx = 0
    
    while current_idx < len(GESTURE_LABELS):
        target_label = GESTURE_LABELS[current_idx]
        file_name = f"{target_label}.png"
        save_path = os.path.join(SAVE_DIR, file_name)
        
        # Check if already exists
        status_msg = "READY TO CAPTURE"
        color = (0, 255, 255) # Yellow
        if os.path.exists(save_path):
            status_msg = "EXISTS (Space to Overwrite)"
            color = (0, 255, 0) # Green

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            display = frame.copy()
            h, w, _ = frame.shape
            
            # Draw UI
            # Dark Header
            cv2.rectangle(display, (0, 0), (w, 100), (20, 20, 20), -1)
            
            # Progress Bar
            progress = int((current_idx / len(GESTURE_LABELS)) * w)
            cv2.line(display, (0, 98), (progress, 98), (0, 255, 0), 4)

            # Text Info
            cv2.putText(display, f"POSE {current_idx + 1}/{len(GESTURE_LABELS)}: {target_label}", 
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            cv2.putText(display, status_msg, (30, 85), 
                        cv2.FONT_HERSHEY_PLAIN, 1.2, color, 1)

            cv2.putText(display, "[SPACE] Snap  [S] Skip  [ESC] Exit", (w-450, 85), 
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (200, 200, 200), 1)

            # Center Crosshair (Helper)
            cx, cy = w//2, h//2
            cv2.line(display, (cx-20, cy), (cx+20, cy), (0, 255, 0), 1)
            cv2.line(display, (cx, cy-20), (cx, cy+20), (0, 255, 0), 1)

            cv2.imshow("Gesture Photo Booth", display)
            
            k = cv2.waitKey(1)
            
            if k == 27: # ESC
                cap.release()
                cv2.destroyAllWindows()
                return
            
            elif k == 32: # SPACE (Capture)
                # Flash Effect
                white = cv2.addWeighted(frame, 1.5, frame, 0, 0)
                cv2.imshow("Gesture Photo Booth", white)
                cv2.waitKey(50)
                
                # Save Raw Frame (Clean, no text)
                cv2.imwrite(save_path, frame)
                print(f"✅ Saved: {file_name}")
                current_idx += 1
                break
            
            elif k == ord('s'): # Skip
                print(f"⏭️ Skipped: {target_label}")
                current_idx += 1
                break

    cap.release()
    cv2.destroyAllWindows()
    print("\n🎉 SESSION COMPLETE. Check 'assets/gestures/' folder.")

if __name__ == "__main__":
    run_photobooth()