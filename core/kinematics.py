import math
import numpy as np

class ScreenMapper:
    def __init__(self):
        # ROI: We only use the center 50% of the camera to avoid stretching arms
        # (x_start, y_start, width, height) in 0.0-1.0 scale
        self.roi = (0.2, 0.2, 0.6, 0.5) 
        
    def sigmoid_gain(self, velocity):
        """
        Dynamic Gain: 
        - Low speed -> Low gain (Pixel precision)
        - High speed -> High gain (Cross screen instantly)
        """
        min_gain = 1.0
        max_gain = 3.5
        slope = 10.0
        inflection = 0.02
        
        gain = min_gain + (max_gain - min_gain) / (1 + math.exp(-slope * (velocity - inflection)))
        return gain

    def map(self, x, y, screen_w, screen_h):
        # 1. Normalize coords within ROI
        roi_x, roi_y, roi_w, roi_h = self.roi
        
        # Clamp to ROI edges
        x = max(roi_x, min(x, roi_x + roi_w))
        y = max(roi_y, min(y, roi_y + roi_h))
        
        # Remap to 0.0-1.0 relative to ROI
        norm_x = (x - roi_x) / roi_w
        norm_y = (y - roi_y) / roi_h
        
        # 2. Scale to Screen
        out_x = norm_x * screen_w
        out_y = norm_y * screen_h
        
        return int(out_x), int(out_y)