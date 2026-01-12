import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        pass

    def draw_roi(self, frame, roi_config):
        """Draws the active 'Mouse Pad' area on screen."""
        h, w, _ = frame.shape
        rx, ry, rw, rh = roi_config
        
        # Convert relative (0.0-1.0) to pixels
        x = int(rx * w)
        y = int(ry * h)
        width = int(rw * w)
        height = int(rh * h)
        
        # Draw the "Glass" Box
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
        cv2.putText(frame, "ACTIVE ZONE", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def draw_hud(self, frame, state, mode_text=""):
        h, w, _ = frame.shape
        
        # 1. State Pill (Top Left)
        color = (0, 255, 0) # Green for active
        if state == "FIST": color = (0, 0, 255) # Red for stop
        elif state == "GEN_Z_HEART": color = (255, 0, 255) # Purple for Menu
        elif state == "PINCH": color = (0, 165, 255) # Orange for Click
        
        # Background
        cv2.rectangle(frame, (10, 10), (250, 60), (30, 30, 30), -1)
        cv2.putText(frame, f"STATE: {state}", (20, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 2. Mode Text (e.g. "SCROLLING")
        if mode_text:
             cv2.putText(frame, mode_text, (w//2 - 100, h - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)