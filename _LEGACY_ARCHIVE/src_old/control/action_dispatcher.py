from src.control.cursor_engine import CursorEngine
from src.control.sequence_engine import SequenceEngine
from src.control.system_engine import SystemEngine

class ActionDispatcher:
    def __init__(self, state, mouse):
        self.state = state
        self.cursor = CursorEngine(state, mouse)
        self.sequence = SequenceEngine(state)
        self.system = SystemEngine(state)

    def dispatch(self, gesture, primary_lms, secondary_lms, hand_count):
        
        # --- LAW 0: THE "NO NOISE" OVERRIDE ---
        # You asked for this: If 1 Hand is visible, NOISE is IMPOSSIBLE.
        # We instantly force it to POINTER.
        if hand_count == 1 and gesture == "NOISE":
            gesture = "POINTER"

        # --- LAW 1: THE ZONE GUARD ---
        # Volume/Scroll separation based on screen position
        hand_x = primary_lms.landmark[9].x
        if gesture == "VOLUME" and hand_x < 0.8: 
            gesture = "SCROLL" if gesture == "SCROLL" else "POINTER"

        # --- LAW 2: THE HAND COUNT GUARD ---
        # Zoom requires 2 hands. If 1 hand, it's noise/pointer.
        if gesture == "ZOOM" and hand_count < 2:
            gesture = "POINTER" # Fallback to pointer, not noise

        # 1. Update System (Locking, Volume, Wave Exit)
        self.system.update(gesture, primary_lms.landmark)
        
        # If Locked, stop here
        if self.state.locked and gesture != "LOCK":
            return

        # 2. Update Sequences (Delete, Nav, Copy/Paste)
        # We pass landmarks now so it can track the "Select All" circle
        self.sequence.process(gesture, primary_lms.landmark)
        
        # 3. Update Cursor (Pointer, Scroll, Zoom)
        self.cursor.update(gesture, primary_lms.landmark, secondary_lms)