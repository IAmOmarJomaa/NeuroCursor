"""
NeuroCursor OS - Main Entry Point.
=================================

This module serves as the central bootloader for the NeuroCursor system.
It orchestrates the "Layer Cake" architecture by:
1. Initializing the Perception Layer (MediaPipe + Camera Thread).
2. Spawning the Cognition Engines (NeuroCursorBrain).
3. Linking the Control Nervous System (NeuroController).
4. Rendering the Feedback Loop (HUD).

Usage:
    Run directly to start the application:
    $ python main.py
"""
import numpy as np
import cv2
import mediapipe as mp
import time
import threading
from typing import Optional, Tuple, Any

# Internal Modules
from src.control.controller import NeuroController
from src.gesture_engine import NeuroCursorBrain
from src.config import CONFIG, init_environment
from src.core.stabilizer import SkeletonStabilizer
from src.ui.hud import HUD 

class ThreadedCamera:
    """
    High-Performance Camera Reader.
    
    Why this exists:
    Standard cv2.VideoCapture.read() is blocking. If the vision model takes 
    30ms to infer, the camera buffer fills up, leading to "lag" where the 
    cursor movement happens seconds after the hand moves.
    
    This class runs the camera I/O in a separate daemon thread, ensuring 
    the main loop always gets the *freshest* frame available (O(1) access).
    """
    def __init__(self, src: int = 0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        # Force a high target FPS to minimize hardware buffering latency
        self.cap.set(cv2.CAP_PROP_FPS, CONFIG.get("TARGET_FPS", 30))
        
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        
        # Start the I/O thread
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        """Background thread loop for frame grabbing."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret: 
                self.running = False
                break
            # Lock ensures we don't read a half-written frame
            with self.lock: 
                self.ret, self.frame = ret, frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Returns the most recent frame.
        Non-blocking.
        """
        with self.lock: 
            return self.ret, self.frame.copy() if self.frame is not None else None

    def release(self):
        """Safely stops the thread and releases hardware."""
        self.running = False
        self.cap.release()

def main():
    """
    Main Event Loop.
    """
    # 1. Boot Sequence
    init_environment()
    print("🚀 NEURO-CURSOR OS: ONLINE")
    print("   -> Press 'ESC' to Exit")
    print("   -> Press 'B' to Toggle Workspace Box")
    print("   -> Press 'V' to Toggle Visuals")
    
    # 2. Initialize Subsystems
    hud = HUD() 
    window_name = "NeuroCursor OS"
    cv2.namedWindow(window_name) # Setup OpenGL context if available

    # Dual-Brain Architecture (Support for Two Hands)
    brain_1 = NeuroCursorBrain()
    brain_2 = NeuroCursorBrain()
    
    # The Pilot (Central Nervous System)
    pilot = NeuroController()
    
    # The Vision Input
    cam = ThreadedCamera(CONFIG["CAMERA_INDEX"])
    
    # Stabilization Layer (Jitter Filter)
    stab_1 = SkeletonStabilizer()
    stab_2 = SkeletonStabilizer()
    
    # MediaPipe Setup
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5,
        model_complexity=1 # 0=Fast, 1=Balanced
    )

    prev_time = 0
    show_visuals = True
    
    # Warmup time for auto-exposure cameras
    time.sleep(1.0) 

    try:
        while True:
            # --- 1. PERCEPTION ---
            ret, frame = cam.read()
            if not ret or frame is None: continue
            
            # MediaPipe requires RGB; OpenCV uses BGR
            # Flip horizontal for mirror effect (intuitive interaction)
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            hand_1_data = None
            hand_2_data = None

            if results.multi_hand_landmarks:
                # --- 2. COGNITION (Primary Hand) ---
                raw_lms1 = results.multi_hand_landmarks[0]
                # Stabilize raw jitter
                lms1 = stab_1.process(raw_lms1)
                # Infer Gesture
                label1, conf1 = brain_1.predict(lms1.landmark)
                hand_1_data = (label1, conf1, lms1)
                
                # --- 2b. COGNITION (Secondary Hand) ---
                if len(results.multi_hand_landmarks) > 1:
                    raw_lms2 = results.multi_hand_landmarks[1]
                    lms2 = stab_2.process(raw_lms2)
                    label2, conf2 = brain_2.predict(lms2.landmark)
                    hand_2_data = (label2, conf2, lms2)

                # --- 3. CONTROL (The Nervous System) ---
                # Passes data to Physics Engine -> Action Dispatcher
                pilot.process(hand_1_data, hand_2_data)

                # --- 4. FEEDBACK (The HUD) ---
                if show_visuals:
                    hud.render(frame, pilot, hand_1_data, stab_1)

            # Performance Monitoring
            curr = time.time()
            fps = 1/(curr-prev_time) if (curr-prev_time)>0 else 0
            prev_time = curr
            hud.draw_fps(frame, fps)

            # Display
            cv2.imshow(window_name, frame)
            
            # Input Handling
            k = cv2.waitKey(1)
            if k == 27: break # ESC
            elif k == ord('b'): pilot.toggle_box()
            elif k == ord('v'): show_visuals = not show_visuals

    finally:
        # Graceful Shutdown
        cam.release()
        cv2.destroyAllWindows()
        print("🔴 SYSTEM OFFLINE")

if __name__ == "__main__":
    main()