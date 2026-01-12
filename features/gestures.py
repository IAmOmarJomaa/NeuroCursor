import numpy as np
import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.geometry import HandGeometry

class GestureEngine:
    def __init__(self):
        # We don't even need the JSON anymore for the critical logic.
        # The data science proved the numbers are universal for your hand.
        pass

    def get_distance(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def calculate_golden_features(self, hand_lms):
        """
        Calculates the exact normalized features found by the Data Science Audit.
        Normalization ensures this works whether hand is close or far.
        """
        # Landmarks
        wrist = hand_lms.landmark[0]
        thumb_tip = hand_lms.landmark[4]
        index_tip = hand_lms.landmark[8]
        middle_mcp = hand_lms.landmark[9]  # Middle Finger Base
        
        # 1. CALCULATE SCALE UNIT (Wrist to Middle Base)
        # This is the "Ruler" we measure everything against.
        scale = self.get_distance(wrist, middle_mcp)
        if scale == 0: scale = 1.0
        
        # 2. CALCULATE RAW DISTANCES
        raw_thumb_to_middle = self.get_distance(thumb_tip, middle_mcp)
        raw_thumb_to_index = self.get_distance(thumb_tip, index_tip)
        
        # 3. NORMALIZE (The "Magic Number")
        # This matches the numbers from your Audit Report exactly.
        feat_thumb_middle_base = raw_thumb_to_middle / scale
        feat_pinch_dist = raw_thumb_to_index / scale
        
        return feat_thumb_middle_base, feat_pinch_dist

    def detect_state(self, hand_lms):
        # 1. Get Basic Geometry (Up/Down)
        angles = HandGeometry.get_all_finger_states(hand_lms)
        
        # Basic Finger States (Using 90 degree fold threshold)
        index_up = angles["index"] > 100
        middle_up = angles["middle"] > 100
        ring_up = angles["ring"] > 100
        pinky_up = angles["pinky"] > 100

        # 2. Get The "Golden Features" (The Data Science Math)
        thumb_middle_dist, pinch_dist = self.calculate_golden_features(hand_lms)
        
        # --- LOGIC TREE BASED ON DATA ---

        # 1. SCROLL MODE (3 Fingers Up)
        if index_up and middle_up and ring_up and not pinky_up:
            return "SCROLL_StANDBY"

        # 2. OPEN PALM (All Up)
        if index_up and middle_up and ring_up and pinky_up:
            return "PALM"

        # 3. FIST (All Down)
        if not index_up and not middle_up and not ring_up and not pinky_up:
            return "FIST"

        # 4. THE CRITICAL DECISION: POINTER vs HEART vs PINCH
        # Condition: Index is UP, others are DOWN.
        if index_up and not middle_up and not ring_up and not pinky_up:
            
            # CHECK A: IS IT A HEART?
            # Audit Logic: If Thumb_Tip_to_Middle_Base > 0.95 -> HEART
            # (Your data said Heart is ~1.6, Pointer is ~0.2)
            if thumb_middle_dist > 0.95:
                return "GEN_Z_HEART"
            
            # CHECK B: IS IT A PINCH?
            # We use a tight threshold for the actual click trigger
            # 0.3 is roughly "Touching" in this normalized scale
            if pinch_dist < 0.3: 
                # It's a Pinch (Click), but system sees it as Pointer w/ Click
                return "POINTER"
            
            # OTHERWISE: IT IS A SOLID POINTER
            return "POINTER"

        return "UNKNOWN"