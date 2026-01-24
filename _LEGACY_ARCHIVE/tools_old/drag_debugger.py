import cv2
import mediapipe as mp
import sys
import os
import time
import json
import math

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.camera import FastCamera
from src.vision.logic_brain import LogicBrain
from src.config import CONFIG

class DragDebugger:
    def __init__(self):
        self.cam = FastCamera()
        self.brain = LogicBrain()
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, model_complexity=0, min_detection_confidence=0.5)
        
        self.is_dragging = False
        self.frame_buffer = [] 
        self.drop_count = 0

    def get_hand_metrics(self, lms, w, h):
        scale = self.brain.get_dist(lms, 0, 9)
        pinch = self.brain.get_dist(lms, 4, 8) / scale
        wrist_y = lms[0].y
        wrist_visible = wrist_y < 0.95
        
        return {
            "timestamp": time.time(),
            "pinch_dist": round(pinch, 3),
            "wrist_y": round(wrist_y, 2),
            "wrist_safe": wrist_visible,
            "gesture": "",
            "raw_x": round(lms[8].x, 2),
            "raw_y": round(lms[8].y, 2)
        }

    def run(self):
        print("🕵️ DRAG DEBUGGER INITIALIZED")
        print("   -> Perform a drag operation.")
        print("   -> When it fails, check the console/file immediately.")
        
        try:
            while True:
                frame = self.cam.read()
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)
                
                status_text = "IDLE - WAITING FOR CLICK"
                status_color = (100, 100, 100)

                if results.multi_hand_landmarks:
                    lms = results.multi_hand_landmarks[0]
                    
                    # 1. Get Logic
                    gesture, debug = self.brain.predict(lms.landmark)
                    
                    # 2. Capture Physics
                    metrics = self.get_hand_metrics(lms.landmark, w, h)
                    metrics["gesture"] = gesture
                    
                    # --- CRASH RECORDER LOGIC ---
                    
                    if gesture == "CLICK":
                        if not self.is_dragging:
                            print("\n🟢 DRAG STARTED")
                            self.is_dragging = True
                            self.frame_buffer = [] 
                        
                        # Record healthy frame
                        self.frame_buffer.append(metrics)
                        status_text = f"DRAGGING... (Pinch: {metrics['pinch_dist']})"
                        status_color = (0, 255, 0)
                        
                        # Draw Green Line
                        p4 = (int(lms.landmark[4].x * w), int(lms.landmark[4].y * h))
                        p8 = (int(lms.landmark[8].x * w), int(lms.landmark[8].y * h))
                        cv2.line(frame, p4, p8, (0, 255, 0), 2)

                    elif self.is_dragging:
                        # !!! DRAG DROPPED !!!
                        print(f"🔴 DRAG DROPPED! New State: {gesture}")
                        
                        # Analyze WHY
                        self.drop_count += 1
                        filename = f"drag_fail_{self.drop_count}.json"
                        
                        failure_report = {
                            "event": "DRAG_DROP",
                            "final_state": gesture,
                            "config_snapshot": CONFIG,
                            "frame_history": self.frame_buffer[-20:] + [metrics] 
                        }
                        
                        with open(filename, "w") as f:
                            json.dump(failure_report, f, indent=4)
                        
                        print(f"   -> CRASH REPORT SAVED: {filename}")
                        print(f"   -> Pinch at fail: {metrics['pinch_dist']}")
                        print(f"   -> Wrist Y at fail: {metrics['wrist_y']}")
                        
                        self.is_dragging = False
                        status_text = "DROP DETECTED! (Saved JSON)"
                        status_color = (0, 0, 255)

                    else:
                        # Just hovering
                        status_text = f"HOVER: {gesture} (Pinch: {metrics['pinch_dist']})"
                        if metrics["wrist_y"] > 0.95:
                            status_text += " [WARNING: WRIST OFF SCREEN]"
                            status_color = (0, 165, 255)

                # Draw UI
                cv2.rectangle(frame, (0, h-50), (w, h), (20, 20, 20), -1)
                cv2.putText(frame, status_text, (20, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                
                # Safe Zone Line
                cv2.line(frame, (0, int(h*0.95)), (w, int(h*0.95)), (0, 0, 255), 1)
                cv2.putText(frame, "DANGER ZONE", (10, int(h*0.95)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                cv2.imshow("Drag Debugger", frame)
                if cv2.waitKey(1) == 27: break
        
        finally:
            self.cam.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = DragDebugger()
    app.run()