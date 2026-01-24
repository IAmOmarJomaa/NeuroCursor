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
        
        # --- LAW 1: THE ZONE GUARD (Fixes Volume vs Scroll) ---
        # Volume is ONLY allowed on the right 20% of the camera view (x > 0.8)
        # Remember: Camera is mirrored, so x > 0.8 is the RIGHT side of screen.
        hand_x = primary_lms.landmark[9].x
        
        if gesture == "VOLUME":
            if hand_x < 0.8: 
                # Hand is in center/left -> It cannot be Volume.
                # It's likely Scroll or Pointer confused.
                gesture = "SCROLL" if gesture == "SCROLL" else "POINTER"

        # --- LAW 2: THE HAND COUNT GUARD (Fixes Zoom vs Lock) ---
        if gesture == "ZOOM":
            if hand_count < 2:
                # Impossible to zoom with 1 hand. Must be noise or Lock.
                gesture = "NOISE"

        # --- LAW 3: HYSTERESIS (Fixes Jitter) ---
        # (This logic is implicit in the state stability below)

        # 1. Update System (Locking, Volume)
        self.system.update(gesture, primary_lms.landmark)
        
        if self.state.locked and gesture != "LOCK":
            return

        # 2. Update Sequences
        self.sequence.process(gesture)
        
        # 3. Update Cursor
        self.cursor.update(gesture, primary_lms.landmark, secondary_lms)