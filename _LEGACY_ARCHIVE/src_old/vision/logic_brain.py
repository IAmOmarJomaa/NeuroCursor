import math
import numpy as np
from src.config import CONFIG
from src.logic_rules import RULES

class LogicBrain:
    def __init__(self):
        print("📐 LOGIC BRAIN V14 [HUNTER ENFORCED]")
        self.last_state = "POINTER"
        self.rules = RULES.copy()

    def get_dist(self, lms, i1, i2):
        return math.hypot(lms[i1].x - lms[i2].x, lms[i1].y - lms[i2].y)

    def get_x_offset(self, lms, i_tip, i_wrist, scale):
        return abs(lms[i_tip].x - lms[i_wrist].x) / scale

    def predict(self, lms):
        scale = self.get_dist(lms, 0, 9)
        if scale == 0: return "NOISE", {}

        # --- 1. SENSORS (Calculated exactly as Hunter did) ---
        pinch = self.get_dist(lms, 4, 8) / scale
        
        # Extensions
        idx_ext = self.get_dist(lms, 8, 0) / scale
        mid_ext = self.get_dist(lms, 12, 0) / scale
        ring_ext = self.get_dist(lms, 16, 0) / scale
        pinky_ext = self.get_dist(lms, 20, 0) / scale
        
        # War Zone Metrics
        ring_curl = self.get_dist(lms, 16, 13) / scale       # For Point/Zoom
        thumb_knuckle = self.get_dist(lms, 4, 9) / scale     # For Zoom
        mid_ring_gap = self.get_dist(lms, 12, 16) / scale    # For Scroll
        ring_pinky_gap = self.get_dist(lms, 16, 20) / scale  # For Win
        wrist_ring_k = self.get_dist(lms, 0, 13) / scale     # For Right Click
        thumb_x = self.get_x_offset(lms, 4, 0, scale)        # For Palm
        pinky_x = self.get_x_offset(lms, 20, 0, scale)       # For Fist/Ctrl
        idx_pinky_k = self.get_dist(lms, 8, 17) / scale      # For Ctrl
        idx_mid_tip = self.get_dist(lms, 8, 12) / scale      # For Previous

        # Debug Packet
        d = {
            "scale": scale, "pinch": pinch, 
            "ring_curl": ring_curl, "thumb_knuckle": thumb_knuckle,
            "mid_ring_gap": mid_ring_gap, "ring_pinky_gap": ring_pinky_gap,
            "wrist_ring_k": wrist_ring_k, "thumb_x": thumb_x,
            "pinky_x": pinky_x, "pinky_ext": pinky_ext,
            "idx_pinky_k": idx_pinky_k, "idx_mid_tip": idx_mid_tip
        }
        
        r = self.rules

        # --- 2. THE TREE ---
        
        # PRIORITY: CLICK
        if self.last_state == "CLICK":
            if pinch < r["PINCH_STOP"]: return "CLICK", d
            else: self.last_state = "POINTER"; return "THE_POINT", d

        # BRANCH 1: INDEX UP, MIDDLE DOWN (Pointer Group)
        if idx_ext > r["OPEN_EXT"] and mid_ext < r["OPEN_EXT"]:
            if ring_curl <= r["TIGHT_FIST_RING"]: # 0.54
                if thumb_knuckle <= r["ZOOM_THUMB"]: # 0.81
                    # Check for click start
                    if pinch < r["PINCH_START"]:
                        self.last_state = "CLICK"; return "CLICK", d
                    return "THE_POINT", d
                else:
                    return "ZOOM", d
            else:
                return "THE_SHHH", d

        # BRANCH 2: INDEX UP, MIDDLE UP (V-Shape Group)
        if idx_ext > r["OPEN_EXT"] and mid_ext > r["OPEN_EXT"]:
            if mid_ring_gap <= r["SCROLL_GAP"]: return "SCROLL", d # 0.55
            
            # If gap is wide, check Ring/Pinky
            if ring_pinky_gap > r["WIN_GAP"]: return "WIN", d # 0.36
            
            # If Ring/Pinky closed, check Wrist Angle
            if wrist_ring_k <= r["RIGHT_CLICK_WRIST"]: return "RIGHT_CLICK", d # 0.98
            return "TAB", d

        # BRANCH 3: ALL OPEN (Palm Group)
        if idx_ext > 1.2 and mid_ext > 1.2 and ring_ext > 1.2:
            if thumb_x > r["PALM_THUMB_X"]: return "PALM", d # 0.25
            return "DELETE_OPENED", d

        # BRANCH 4: ALL CLOSED (Fist Group)
        # We assume if it's not the above, it's a closed gesture
        if pinky_ext > r["CUT_PINKY"]: return "CUT", d # 1.63
        
        if pinky_x <= r["SIDE_GESTURE_X"]: # 0.54 (Upright Hand)
            if idx_mid_tip <= r["PREV_SQUEEZE"]: return "PREVIOUS", d # 0.48
            
            # The Hunter said GEN_Z is > 0.48 here.
            # But remember our Logic Bug fix: Check Thumb Extension for Gen Z
            if self.get_dist(lms, 4, 0)/scale > 1.0: return "GEN_Z", d
            
            # Default upright closed hand
            # To distinguish Delete Closed vs Fist, we use the Thumb Tuck rule from Hunter
            # Hunter said DeleteClosed: Dist(Thumb-IndexKnuckle) <= 0.26
            if self.get_dist(lms, 4, 5)/scale <= 0.28: return "DELETE_CLOSED", d
            return "FIST", d
            
        else: # pinky_x > 0.54 (Sideways Hand)
            if idx_pinky_k <= r["CTRL_SQUEEZE"]: return "CTRL_PINKY", d # 0.72
            return "NEXT", d

        return "NOISE", d