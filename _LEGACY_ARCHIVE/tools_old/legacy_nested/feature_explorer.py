import json
import numpy as np
import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from config import PATHS

class FeatureExplorer:
    def __init__(self):
        self.load_data()
        
    def load_data(self):
        path = ROOT_DIR / "data" / "knn_dataset.json"
        if not path.exists():
            print("❌ NO DATA. Run recorder first.")
            sys.exit()
        with open(path, "r") as f:
            self.dataset = json.load(f)["centroids"]

    def get_vec(self, skeleton, idx):
        # Extract (x,y,z) for a specific landmark index
        return np.array([skeleton[idx*3], skeleton[idx*3+1], skeleton[idx*3+2]])

    def calc_distance(self, skeleton, idx1, idx2):
        p1 = self.get_vec(skeleton, idx1)
        p2 = self.get_vec(skeleton, idx2)
        return np.linalg.norm(p1 - p2)

    def calc_angle(self, skeleton, idx1, idx2, idx3):
        # Angle at idx2
        a = self.get_vec(skeleton, idx1)
        b = self.get_vec(skeleton, idx2)
        c = self.get_vec(skeleton, idx3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cosine_angle))

    def extract_smart_features(self, skeleton):
        """
        Calculates a suite of potential 'Golden Features'.
        """
        feats = {}
        
        # 1. THUMB GEOMETRY (The usual suspect)
        feats["Thumb_Tip_to_Index_Base"] = self.calc_distance(skeleton, 4, 5)
        feats["Thumb_Tip_to_Middle_Base"] = self.calc_distance(skeleton, 4, 9)
        feats["Thumb_Tip_to_Pinky_Base"] = self.calc_distance(skeleton, 4, 17)
        feats["Thumb_Bend_Angle"] = self.calc_angle(skeleton, 2, 3, 4)
        
        # 2. INDEX GEOMETRY
        feats["Index_Bend_Angle"] = self.calc_angle(skeleton, 5, 6, 8)
        feats["Index_Tip_to_Wrist"] = self.calc_distance(skeleton, 8, 0)
        
        # 3. INTER-FINGER DISTANCES (Adduction/Abduction)
        feats["Index_Tip_to_Middle_Tip"] = self.calc_distance(skeleton, 8, 12)
        feats["Thumb_Tip_to_Index_Tip"] = self.calc_distance(skeleton, 4, 8)
        
        # 4. COMPACTNESS (Fist check)
        feats["Pinky_Tip_to_Wrist"] = self.calc_distance(skeleton, 20, 0)
        
        return feats

    def analyze_conflict(self, gesture_A, gesture_B):
        if gesture_A not in self.dataset or gesture_B not in self.dataset:
            print(f"⚠️ Cannot compare {gesture_A} vs {gesture_B} (Missing Data)")
            return

        print(f"\n⚔️ ANALYZING: {gesture_A} vs {gesture_B}")
        print("-" * 60)
        
        feats_A = self.extract_smart_features(self.dataset[gesture_A])
        feats_B = self.extract_smart_features(self.dataset[gesture_B])
        
        # Calculate Difference (Discriminative Power)
        ranking = []
        for key in feats_A:
            val_A = feats_A[key]
            val_B = feats_B[key]
            diff = abs(val_A - val_B)
            # Relative difference matters more than absolute
            avg = (val_A + val_B) / 2
            if avg == 0: avg = 0.001
            percent_diff = (diff / avg) * 100
            
            ranking.append((key, diff, percent_diff, val_A, val_B))
            
        # Sort by % Difference (Most distinguishing first)
        ranking.sort(key=lambda x: x[2], reverse=True)
        
        # Print Top 3 "Golden Features"
        for i in range(3):
            name, diff, pct, vA, vB = ranking[i]
            print(f"🏆 RANK #{i+1}: {name}")
            print(f"   - {gesture_A}: {vA:.4f}")
            print(f"   - {gesture_B}: {vB:.4f}")
            print(f"   - DELTA: {diff:.4f} ({pct:.1f}%)")
            
            # Suggest a Cutoff Logic
            midpoint = (vA + vB) / 2
            if vA > vB:
                print(f"   👉 LOGIC: If {name} > {midpoint:.4f} THEN {gesture_A}")
            else:
                print(f"   👉 LOGIC: If {name} < {midpoint:.4f} THEN {gesture_A}")
            print("")

if __name__ == "__main__":
    app = FeatureExplorer()
    app.analyze_conflict("POINTER", "GEN_Z_HEART")
    app.analyze_conflict("OPEN_PALM", "FIST")