"""
NeuroCursor Coordinate Engine.
=============================

This module handles the translation of Normalized Hand Coordinates (0.0 - 1.0)
into Absolute Screen Pixels.

Key Concept: "Liquid Friction"
To solve the jitter vs. latency trade-off, we implement a dynamic smoothing filter:
1. **Low Speed:** High Friction (High smoothing). Acts like moving through honey.
   This allows pixel-perfect text selection.
2. **High Speed:** Low Friction (Low smoothing). Acts like moving through water.
   This allows instant flick shots across dual monitors.
"""

import win32api, win32con
import numpy as np
from typing import Tuple, Optional
from src.config import CONFIG

class CursorEngine:
    """
    Manages cursor positioning, coordinate transformation, and motion smoothing.
    """
    def __init__(self):
        # Get System Screen Resolution
        self.screen_w = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        self.screen_h = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
        
        # State
        self.current_pos: Optional[Tuple[float, float]] = None 
        
        # Load Workspace Configuration (The "Active Box")
        # This box defines the area of the camera frame that maps to the screen.
        self.box_w = CONFIG["BOX_WIDTH"]
        self.box_h = CONFIG["BOX_HEIGHT"]
        self.box_cx = CONFIG["BOX_CENTER_X"]
        self.box_cy = CONFIG["BOX_CENTER_Y"]
        
        self.show_box = True

    def get_screen_coordinates(self, raw_x: float, raw_y: float) -> Tuple[int, int]:
        """
        Maps normalized camera coordinates to screen pixels.
        Includes clipping to ensure the cursor stays within bounds.
        """
        # Calculate Box Boundaries
        x_min = self.box_cx - (self.box_w / 2)
        x_max = self.box_cx + (self.box_w / 2)
        y_min = self.box_cy - (self.box_h / 2)
        y_max = self.box_cy + (self.box_h / 2)
        
        # Normalize relative to Box (0.0 to 1.0 inside the box)
        norm_x = (raw_x - x_min) / (x_max - x_min)
        norm_y = (raw_y - y_min) / (y_max - y_min)
        
        # Map to Screen and Clip
        target_x = int(np.clip(norm_x, 0, 1) * self.screen_w)
        target_y = int(np.clip(norm_y, 0, 1) * self.screen_h)
        
        return target_x, target_y

    def apply_smoothing(self, target_x: int, target_y: int, current_speed: float = 0) -> Tuple[int, int]:
        """
        Applies 'Liquid Friction' Physics (Dynamic Exponential Moving Average).
        
        Args:
            target_x, target_y: The raw, noisy target pixel.
            current_speed: The normalized velocity of the hand.
            
        Returns:
            (x, y): The smoothed cursor coordinates.
        """
        # 1. VELOCITY GATE (Magnetism)
        # If speed is negligible, lock the cursor to prevent micro-jitter.
        if current_speed < CONFIG["VELOCITY_GATE"]:
            if self.current_pos:
                return int(self.current_pos[0]), int(self.current_pos[1])
        
        # 2. DYNAMIC FRICTION CALCULATION
        v_min = CONFIG["VELOCITY_GATE"]
        v_max = CONFIG["BREAKOUT_VELOCITY"]
        
        # Calculate Alpha (Smoothing Factor)
        # Alpha 1.0 = No Smoothing (Raw Input), Alpha 0.0 = Frozen
        low_alpha = 1.0 - CONFIG["FRICTION_LOW"]   # High Friction state
        high_alpha = 1.0 - CONFIG["FRICTION_HIGH"] # Low Friction state
        
        # Interpolate Alpha based on speed (Lerp)
        # t goes from 0.0 (Slow) to 1.0 (Fast)
        t = np.clip((current_speed - v_min) / (v_max - v_min), 0, 1)
        dynamic_alpha = low_alpha + (t * (high_alpha - low_alpha))

        # 3. APPLY FILTER
        if self.current_pos is None:
            self.current_pos = (float(target_x), float(target_y))
            return int(target_x), int(target_y)
            
        prev_x, prev_y = self.current_pos
        curr_x = (dynamic_alpha * target_x) + ((1 - dynamic_alpha) * prev_x)
        curr_y = (dynamic_alpha * target_y) + ((1 - dynamic_alpha) * prev_y)
        
        self.current_pos = (curr_x, curr_y)
        return int(round(curr_x)), int(round(curr_y))

    def move_mouse(self, raw_x: float, raw_y: float, current_speed: float = 0):
        """Main entry point to move the OS cursor."""
        tx, ty = self.get_screen_coordinates(raw_x, raw_y)
        sx, sy = self.apply_smoothing(tx, ty, current_speed=current_speed)
        win32api.SetCursorPos((int(sx), int(sy)))

    def reset_smoothing(self): 
        """Resets the smoothing filter (e.g., after a teleport or track loss)."""
        self.current_pos = None

    def toggle_visuals(self): 
        self.show_box = not self.show_box

    # --- Runtime Workspace Adjustments (Used by Calibration Tools) ---
    def adjust_width(self, d): self.box_w = np.clip(self.box_w + d, 0.1, 1.0)
    def adjust_height(self, d): self.box_h = np.clip(self.box_h + d, 0.1, 1.0)
    def adjust_cx(self, d): self.box_cx = np.clip(self.box_cx + d, 0.0, 1.0)
    def adjust_cy(self, d): self.box_cy = np.clip(self.box_cy + d, 0.0, 1.0)