import pickle
import numpy as np
import pandas as pd  # <--- ADDED
import os
import sys
from src.config import PATHS

# MUST MATCH TRAINING EXACTLY
FEATURE_COLUMNS = [
    'thumb_bend', 'index_bend', 'mid_bend', 'ring_bend', 'pinky_bend',
    'pinch_dist', 'thumb_spread', 'mid_palm_dist', 
    'orientation_y', 'orientation_x'
]

class NeuroBrain:
    def __init__(self):
        print("🧠 LOADING MODEL...")
        try:
            with open(PATHS["MODELS_DIR"] / "neurocursor_model.pkl", 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            sys.exit(f"❌ MODEL ERROR: {e}")

    def predict(self, lms):
        """Converts landmarks to labeled features and returns prediction."""
        coords = np.array([[lm.x, lm.y, lm.z] for lm in lms])
        
        def get_ang(i1, i2, i3):
            v1 = coords[i2] - coords[i1]
            v2 = coords[i3] - coords[i2]
            dot = np.dot(v1, v2)
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            return np.degrees(np.arccos(np.clip(dot/(norm+1e-6), -1.0, 1.0)))

        def get_dist(i1, i2, norm):
            return np.linalg.norm(coords[i1] - coords[i2]) / norm

        palm = np.linalg.norm(coords[9] - coords[0]) + 1e-6
        
        # Calculate raw values
        data = {
            'thumb_bend': get_ang(1, 2, 3),
            'index_bend': get_ang(5, 6, 7),
            'mid_bend':   get_ang(9, 10, 11),
            'ring_bend':  get_ang(13, 14, 15),
            'pinky_bend': get_ang(17, 18, 19),
            'pinch_dist': get_dist(4, 8, palm),
            'thumb_spread': get_dist(4, 5, palm),
            'mid_palm_dist': get_dist(12, 0, palm),
            'orientation_y': coords[9][1] - coords[0][1],
            'orientation_x': coords[9][0] - coords[0][0]
        }
        
        # Convert to DataFrame with names (Silences Warnings)
        df = pd.DataFrame([data], columns=FEATURE_COLUMNS)
        
        return self.model.predict(df)[0]