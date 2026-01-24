"""
NeuroCursor Landmark Processing Utilities.
========================================

Handles the spatial normalization of hand data.
Raw camera coordinates are useless for ML because they depend on:
1. Where the hand is in the frame (Translation).
2. How close the hand is to the camera (Scale).

This module makes the data Invariant to Translation and Scale.
"""

import numpy as np
from typing import List, Union, Any

def pre_process_landmark(landmark_list: Any, flip_x: bool = False) -> List[float]:
    """
    Transforms raw landmarks into a normalized 63-float feature vector.
    
    Steps:
    1. Convert to NumPy.
    2. Mirror X if Left Hand mode.
    3. Translate: Make Wrist (Point 0) the origin (0,0,0).
    4. Normalize: Scale max absolute value to 1.0.
    5. Flatten: 21x3 Matrix -> 63x1 Vector.
    """
    # 1. Data Structuring: Convert MediaPipe/List input to NumPy matrix
    if hasattr(landmark_list[0], 'x'):
        # MediaPipe Object -> Numpy
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmark_list])
    else:
        # Raw List/CSV Input -> Numpy
        coords = np.array(landmark_list).reshape(-1, 3)

    # 2. Mirroring (Handedness Normalization)
    if flip_x:
        coords[:, 0] *= -1 

    # 3. Relativization (Translation Invariance)
    # Set the Wrist (Index 0) as the Origin (0,0,0)
    wrist = coords[0].copy()
    coords -= wrist

    # 4. Normalization (Scale Invariance)
    # Scale all points so the maximum reach is 1.0. 
    # This ensures a hand close to the camera looks the same as one far away.
    max_value = np.max(np.abs(coords))
    if max_value == 0: 
        max_value = 1.0 
    
    coords /= max_value

    # 5. Flattening -> Returns 63 items (21 points * 3 dims)
    return coords.flatten().tolist()