import json
import numpy as np
import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.geometry import HandGeometry
from config import PATHS

class GestureEngine:
    def __init__(self):
        self.signatures = {}
        self.thresholds = {"pinch_dist": 30.0, "fist_curl": 90.0} # Default prevents crash
        self.load_data()

    def load_data(self):
        try:
            with open(PATHS["RULES"], "r") as f:
                data = json.load(f)
                self.signatures = data.get("signatures", {})
                # FORCE LOAD THRESHOLDS
                if "thresholds" in data:
                    self.thresholds = data["thresholds"]
        except:
            print("⚠️ DATA ERROR. Using Defaults.")

    def get_similarity(self, live_angles, target_name):
        if target_name not in self.signatures:
            return 9999.0 
        target = self.signatures[target_name]
        diff = 0
        for f in ["thumb", "index", "middle", "ring", "pinky"]:
            d = live_angles[f] - target[f]
            diff += d * d 
        return np.sqrt(diff)

    def verify_heart_geometry(self, hand_lms):
        """
        CRITICAL GUARD: Even if the angle looks like a heart, 
        is the thumb actually touching the index base?
        """
        # Landmark 4 = Thumb Tip
        # Landmark 5 = Index MCP (Base knuckle)
        t_tip = hand_lms.landmark[4]
        i_base = hand_lms.landmark[5]
        
        # Calculate raw distance between Thumb Tip and Index Base
        # We ignore Z for this 2D check
        dist = math.hypot(t_tip.x - i_base.x, t_tip.y - i_base.y)
        
        # Threshold: 0.05 is roughly "Touching" in normalized coordinates
        # If distance is LARGE (> 0.05), the thumb is floating/tucked -> POINTER
        return dist < 0.06

    def detect_state(self, hand_lms):
        angles = HandGeometry.get_all_finger_states(hand_lms)
        
        # 1. SCROLL OVERRIDE (3 fingers straight)
        curl_limit = self.thresholds.get("fist_curl", 90)
        if (angles["index"] > curl_limit and 
            angles["middle"] > curl_limit and 
            angles["ring"] > curl_limit and 
            angles["pinky"] < curl_limit):
            return "SCROLL_StANDBY"

        # 2. SIGNATURE MATCHING
        candidates = ["POINTER", "GEN_Z_HEART", "PINCH", "OPEN_PALM", "FIST"]
        best_match = "UNKNOWN"
        lowest_error = 9999.0
        
        for cand in candidates:
            error = self.get_similarity(angles, cand)
            if error < lowest_error:
                lowest_error = error
                best_match = cand
        
        # 3. THE EMERGENCY GUARD (Fixing your issue)
        # If AI thinks it's a HEART, we double-check with Geometry.
        if best_match == "GEN_Z_HEART":
            is_touching = self.verify_heart_geometry(hand_lms)
            if not is_touching:
                # "AI says Heart, but Thumb is too far away. Forcing Pointer."
                return "POINTER"

        # 4. MAPPING
        if best_match == "OPEN_PALM": return "PALM"
        if best_match == "PINCH": return "POINTER" 
        
        return best_match