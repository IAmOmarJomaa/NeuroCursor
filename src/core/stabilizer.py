"""
NeuroCursor Stabilization Layer (The Anchor).
Optimized for high-frequency calling.
"""

import numpy as np
from src.config import CONFIG

# --- OPTIMIZATION: Defined at module level to prevent re-allocation ---
class MockLandmark:
    """Mimics mediapipe.framework.formats.landmark_pb2.NormalizedLandmark"""
    __slots__ = ['x', 'y', 'z', 'visibility', 'presence'] 
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z
        self.visibility = 1.0 
        self.presence = 1.0   

    def HasField(self, field_name):
        return field_name in ['visibility', 'presence']

class MockHand:
    """Mimics the top-level NamedTuple from MediaPipe Results"""
    __slots__ = ['landmark']
    def __init__(self, lms):
        self.landmark = lms

class SkeletonStabilizer:
    def __init__(self, jitter_threshold=None, smoothing_factor=None):
        self.thresh = jitter_threshold or CONFIG["STABILIZER_THRESHOLD"]
        self.alpha = smoothing_factor or CONFIG["STABILIZER_ALPHA"]
        self.prev_lms = None
        self.locked = False 

    def process(self, raw_lms):
        if raw_lms is None:
            self.prev_lms = None
            return None

        # Convert to numpy for vectorized math
        curr = np.array([[lm.x, lm.y, lm.z] for lm in raw_lms.landmark], dtype=np.float32)

        if self.prev_lms is None:
            self.prev_lms = curr
            return raw_lms

        # Vectorized Euclidean Distance
        deltas = np.linalg.norm(curr - self.prev_lms, axis=1)
        avg_movement = np.mean(deltas)

        if avg_movement < self.thresh:
            self.locked = True
            return self._pack_landmarks(self.prev_lms)
        
        self.locked = False
        smoothed = (self.alpha * curr) + ((1 - self.alpha) * self.prev_lms)
        self.prev_lms = smoothed
        
        return self._pack_landmarks(smoothed)

    def _pack_landmarks(self, coords):
        # List comprehension is faster than appending in a loop
        return MockHand([MockLandmark(row[0], row[1], row[2]) for row in coords])