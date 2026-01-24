import math
import numpy as np

class GestureEngine:
    def __init__(self):
        # ---------------------------------------------------------
        # 🧬 CALIBRATION
        # ---------------------------------------------------------
        self.heart_thumb_threshold = 0.12 
        self.pinky_open_thresh = 1.2
        self.pointer_stability_thresh = 0.60
        self.click_dist_thresh = 0.25 
        self.fingers_touching_thresh = 0.30

    def get_dist(self, p1, p2, scale):
        return math.hypot(p1.x - p2.x, p1.y - p2.y) / scale

    def get_dy(self, p1, p2, scale):
        return (p1.y - p2.y) / scale

    def detect_state(self, hand_lms):
        # 1. NORMALIZATION
        wrist = hand_lms.landmark[0]
        idx_mcp = hand_lms.landmark[5]
        scale = math.hypot(wrist.x - idx_mcp.x, wrist.y - idx_mcp.y)
        if scale < 0.01: scale = 1.0

        # 2. LANDMARKS
        thumb_tip = hand_lms.landmark[4]
        idx_tip   = hand_lms.landmark[8]
        idx_dip   = hand_lms.landmark[7]
        mid_dip   = hand_lms.landmark[11]
        mid_tip   = hand_lms.landmark[12]
        ring_tip  = hand_lms.landmark[16]
        pinky_tip = hand_lms.landmark[20]
        
        # 3. METRICS
        pinch_dist = self.get_dist(thumb_tip, idx_tip, scale)
        thumb_height_score = self.get_dy(thumb_tip, idx_dip, scale)
        idx_mid_sep = self.get_dist(idx_tip, mid_dip, scale)
        idx_mid_tip_dist = self.get_dist(idx_tip, mid_tip, scale) 
        
        # 4. FINGER STATES
        def is_open(tip, base):
            d_tip = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
            d_base = math.hypot(base.x - wrist.x, base.y - wrist.y)
            return (d_tip / (d_base + 0.001)) > 1.2

        idx_open = is_open(idx_tip, hand_lms.landmark[5])
        mid_open = is_open(mid_tip, hand_lms.landmark[9])
        ring_open = is_open(ring_tip, hand_lms.landmark[13])
        pinky_open = is_open(pinky_tip, hand_lms.landmark[17])
        thumb_out = self.get_dist(thumb_tip, hand_lms.landmark[5], scale) > 0.5

        # --- LOGIC TREE ---

        # 1. PINCH (Click)
        if pinch_dist < self.click_dist_thresh:
            return "PINCH"

        # 2. GEN Z HEART (Priority Over Everything for Locking)
        # Rule: Index Up, Thumb High, Middle Closed.
        if idx_open and thumb_height_score < self.heart_thumb_threshold and not mid_open:
            return "GEN_Z_HEART"

        # 3. PALM (Start of Copy)
        if idx_open and mid_open and ring_open and pinky_open:
            return "PALM"

        # 4. FIST (End of Copy / Start of Paste)
        if not idx_open and not mid_open and not ring_open and not pinky_open:
            return "FIST"

        # 5. ALT-TAB (Closed V)
        if idx_open and mid_open and not ring_open and not pinky_open:
            if idx_mid_tip_dist < self.fingers_touching_thresh:
                return "ALT_TAB_GLUED"

        # 6. RIGHT CLICK (Middle Only)
        if mid_open and not idx_open and not ring_open:
             return "RIGHT_CLICK_READY"

        # 7. SPIDER-MAN (Scroll)
        if idx_open and pinky_open and thumb_out and not mid_open and not ring_open:
            return "SCROLL_SPIDERMAN"

        # 8. SHAKA (Volume)
        if thumb_out and pinky_open and not idx_open and not mid_open and not ring_open:
            return "VOLUME_SHAKA"

        # 9. POINTER
        if idx_open and not mid_open and not ring_open and not pinky_open:
            if idx_mid_sep > self.pointer_stability_thresh:
                if thumb_height_score > self.heart_thumb_threshold:
                    return "POINTER"
                
        return "UNKNOWN"

    def get_role(self, hand_lms):
        state = self.detect_state(hand_lms)
        # Any specific gesture overrides Zoom role
        if state in ["POINTER", "PINCH", "GEN_Z_HEART", "SCROLL_SPIDERMAN", "RIGHT_CLICK_READY", "ALT_TAB_GLUED", "PALM", "FIST"]:
            return "POINTER"

        wrist = hand_lms.landmark[0]
        thumb_tip = hand_lms.landmark[4]
        idx_tip = hand_lms.landmark[8]
        idx_base = hand_lms.landmark[5]
        scale = math.hypot(wrist.x - idx_base.x, wrist.y - idx_base.y) or 1.0
        
        thumb_out = (math.hypot(thumb_tip.x - idx_base.x, thumb_tip.y - idx_base.y) / scale) > 0.5
        idx_open = (math.hypot(idx_tip.x - wrist.x, idx_tip.y - wrist.y) / math.hypot(idx_base.x - wrist.x, idx_base.y - wrist.y)) > 1.2
        
        if thumb_out and not idx_open:
            return "ZOOM_CONTROLLER"
            
        return "NONE"

    def is_zoom_active(self, hand_lms):
        wrist = hand_lms.landmark[0]
        pinky_tip = hand_lms.landmark[20]
        pinky_base = hand_lms.landmark[17]
        d_tip = math.hypot(pinky_tip.x - wrist.x, pinky_tip.y - wrist.y)
        d_base = math.hypot(pinky_base.x - wrist.x, pinky_base.y - wrist.y)
        return (d_tip / (d_base + 0.001)) > self.pinky_open_thresh