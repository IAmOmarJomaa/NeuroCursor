"""
NeuroCursor Kinematics V2 (God Tier).
Strictly Typed, Optimized, and Enum-Driven.
"""
import math
import time
from typing import List, Tuple
from src.config import CONFIG
from src.core.types import DragTrigger, Point2D

class KinematicsEngine:
    SCREEN_REF_WIDTH = 1920.0 
    
    # Circle Detection Constants
    MIN_POINTS_FOR_CIRCLE = 15
    MIN_CIRCLE_ANGLE = 5.0

    def get_pinch_sq_dist(self, lms) -> float:
        """
        Returns Squared Euclidean distance between Thumb(4) and Index(8).
        SQRT is expensive. We avoid it for high-frequency checks.
        """
        # [OPTIMIZATION] Direct access to coordinates
        x1, y1 = lms.landmark[4].x, lms.landmark[4].y
        x2, y2 = lms.landmark[8].x, lms.landmark[8].y
        return (x2 - x1)**2 + (y2 - y1)**2

    def check_pinch_hysteresis(self, current_sq_dist: float, is_pinching: bool, velocity: float) -> bool:
        """
        Schmitt Trigger implementation using Squared Distances.
        """
        # Pre-calc squared threshold
        start_thresh_sq = CONFIG["PINCH_START"] ** 2
        
        if is_pinching:
            # Dynamic release threshold logic (Sticky Drag)
            base_stop = CONFIG["PINCH_STOP"]
            max_stop = CONFIG["PINCH_STOP_MAX"]
            scale = CONFIG["PINCH_DYNAMIC_SCALE"]
            
            # If moving fast, increase the release threshold (harder to drop item)
            dynamic_boost = velocity * scale if velocity > 0.01 else 0
            stop_thresh = min(base_stop + dynamic_boost, max_stop)
            
            return current_sq_dist < (stop_thresh ** 2)
        else:
            return current_sq_dist < start_thresh_sq

    def analyze_drag_intent(self, 
                          current: Point2D, 
                          anchor: Point2D, 
                          start_time: float, 
                          velocity: float) -> DragTrigger:
        """
        Determines drag state using Enums and Dynamic Deadzones.
        """
        # 1. Time Check
        if (time.time() - start_time) > CONFIG["DRAG_START_DELAY"]:
            return DragTrigger.TIMED

        # 2. Distance Check (Deadzone)
        dist_sq = (current.x - anchor.x)**2 + (current.y - anchor.y)**2
        
        # Calculate dynamic radius (normalized 0.0-1.0)
        deadzone_norm = self.get_dynamic_deadzone(velocity)
        
        # Check squared distance against squared deadzone
        if dist_sq > (deadzone_norm ** 2):
            return DragTrigger.MOVED
            
        return DragTrigger.NONE

    def get_dynamic_deadzone(self, velocity: float) -> float:
        """
        Calculates velocity-dependent deadzone radius.
        Public API: Used by Labs and HUD.
        """
        max_d = CONFIG["DEADZONE_MAX"]
        min_d = CONFIG["DEADZONE_MIN"]
        decay_v = CONFIG["DEADZONE_DECAY_VELOCITY"]
        
        # Interpolation Factor 't' (0.0 to 1.0)
        t = min(velocity / decay_v, 1.0)
        
        # Linear Interpolation (Lerp)
        pixel_size = max_d - (t * (max_d - min_d))
        
        return pixel_size / self.SCREEN_REF_WIDTH

    def check_full_circle(self, path: List[Tuple[float, float]]) -> bool:
        """
        Robust Circle Detection (Winding Number).
        Kept for ShortcutHandler compatibility.
        """
        if len(path) < self.MIN_POINTS_FOR_CIRCLE: return False 
        
        min_x = min(p[0] for p in path)
        max_x = max(p[0] for p in path)
        min_y = min(p[1] for p in path)
        max_y = max(p[1] for p in path)
        
        width = max_x - min_x
        height = max_y - min_y
        avg_dia = (width + height) / 2
        
        if avg_dia < CONFIG["SELECT_ALL_DIAMETER"]: return False
            
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        total_angle = 0
        for i in range(1, len(path)):
            va_x = path[i-1][0] - center_x
            va_y = path[i-1][1] - center_y
            vb_x = path[i][0] - center_x
            vb_y = path[i][1] - center_y
            
            angle_a = math.atan2(va_y, va_x)
            angle_b = math.atan2(vb_y, vb_x)
            
            diff = angle_b - angle_a
            if diff > math.pi: diff -= 2*math.pi
            if diff < -math.pi: diff += 2*math.pi
            total_angle += diff
            
        return abs(total_angle) > self.MIN_CIRCLE_ANGLE