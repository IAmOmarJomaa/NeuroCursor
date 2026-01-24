import time

class SystemState:
    def __init__(self):
        # System Flags
        self.running = True
        self.locked = True
        self.visuals = True
        self.is_admin = False # <--- NEW FLAG
        
        # Physics State (Mouse Position)
        self.curr_x = 0
        self.curr_y = 0
        self.lock_progress = 0
        self.zoom_anchor = None

        # Logic Counters
        self.hold_timer = 0
        self.scroll_anchor = None
        self.wave_energy = 0
        self.last_x = 0
        self.hand_centroid_x = 0
        
        # Sequence State (For Copy/Paste chains)
        self.sequence_state = "IDLE"
        self.sequence_timer = 0
        
        # Active Data
        self.active_gesture = "NOISE"