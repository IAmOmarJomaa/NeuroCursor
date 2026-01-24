"""
NeuroCursor State Management.
Refactored for Type Safety (Enums) and Encapsulation.
"""
from src.core.types import Gesture

class StateManager:
    def __init__(self):
        # --- GESTURE HISTORY (Now Typed) ---
        self.prev_gesture = Gesture.RESTING
        self.curr_gesture = Gesture.RESTING
        
        # Shortcut State
        self.last_palm_time = 0
        self.last_fist_time = 0         
        self.palm_path = []             
        self.palm_start_time = 0
        
        # --- LOGIC FLAGS ---
        self.is_locked = True
        self.is_pinching = False
        self.is_dragging = False
        
        # --- TIMERS & ANCHORS ---
        self.lock_hold_time = 0
        self.click_start_time = 0
        self.last_click_time = 0
        self.click_anchor = (0, 0)
        self.scroll_anchor_y = None
        self.vol_anchor_y = None

    def update_gesture(self, gesture: Gesture):
        """Updates history with strictly typed Enums."""
        self.prev_gesture = self.curr_gesture
        self.curr_gesture = gesture

    def reset_click_state(self):
        self.click_start_time = 0
        self.click_anchor = (0, 0)