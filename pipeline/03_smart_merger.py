"""
NeuroCursor Smart Merger (The Compiler).

This module prepares the "Golden" refined dataset for training.
It solves the "Semantic Collapse" problem:
1. It looks at the refined clusters (e.g., 'PALM_0', 'PALM_1', 'PALM_2').
2. It asks the user whether to MERGE them back into a single 'PALM' class 
   or keep them separate (e.g., if 'PALM_2' is actually 'WAVE').
3. It applies final normalization to label names.

Architecture:
    [Golden CSV] -> [Interactive Merge Strategy] -> [Label Polishing] -> [Training Ready CSV]
"""

import cv2
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import PATHS

INPUT_FILE = str(PATHS["GOLDEN_DATA"])
OUTPUT_FILE = str(PATHS["DATA_READY"]) # Final Output

# Visualization Constants
W, H = 1000, 700
CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),    # Thumb
    (0,5), (5,6), (6,7), (7,8),    # Index
    (0,9), (9,10), (10,11), (11,12), # Middle
    (0,13), (13,14), (14,15), (15,16), # Ring
    (0,17), (17,18), (18,19), (19,20)  # Pinky
]

class SmartMerger:
    """
    Manages the logic for consolidating refined micro-clusters into 
    macro-classes for the Neural Network.
    """
    def __init__(self):
        print("🧠 INITIALIZING SMART MERGER + AUTO-POLISH...")
        if not os.path.exists(INPUT_FILE):
            print("❌ Golden dataset not found! Run '02_data_refinery.py' first.")
            sys.exit()
            
        self.df = pd.read_csv(INPUT_FILE)
        self.feature_cols = [c for c in self.df.columns if c.startswith('x') or c.startswith('y') or c.startswith('z')]
        
        # --- POLISH MAP ---
        # Canonicalization rules to ensure labels match src/config.py
        self.polish_map = {
            "NEXT_open": "NEXT_OPENED",
            "NEXT_closed": "NEXT_CLOSED",
            "PREVIOUS_open": "PREVIOUS_OPENED",
            "PREVIOUS_closed": "PREVIOUS_CLOSED",
            "DELETE_open": "DELETE_OPENED",
            "DELETE_closed": "DELETE_CLOSED",
        }
        
        # Pre-calculate centroids for visualization
        self.centroids = {}
        self.counts = {}
        self.sub_labels = sorted(self.df['label'].unique())
        
        for label in self.sub_labels:
            subset = self.df[self.df['label'] == label]
            self.centroids[label] = subset[self.feature_cols].mean().values
            self.counts[label] = len(subset)

        self.final_mappings = {} 
        self.ignore_list = []    

    def draw_skeleton(self, canvas, landmarks, color, offset_x=0, label=""):
        """Draws a single skeleton on the shared canvas."""
        coords = landmarks.reshape(-1, 3)
        xs, ys = coords[:, 0], coords[:, 1]
        scale, center_x, center_y = 300, 300 + offset_x, 350
        
        for s, e in CONNECTIONS:
            pt1 = (int(xs[s] * scale + center_x), int(ys[s] * scale + center_y))
            pt2 = (int(xs[e] * scale + center_x), int(ys[e] * scale + center_y))
            cv2.line(canvas, pt1, pt2, color, 2)
            
        cv2.putText(canvas, label, (int(xs[0]*scale+center_x)-20, int(ys[0]*scale+center_y)+20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def interactive_merge(self, parent_group, sub_vars):
        """
        UI for deciding whether to Collapse or Split variations.
        
        Args:
            parent_group (str): The inferred base name (e.g. "PALM").
            sub_vars (list): The variations (e.g. ["PALM_0", "PALM_1"]).
            
        Returns:
            list[bool]: Mask of variations to KEEP merged.
            OR "SPLIT": If the user wants to keep them as distinct classes.
        """
        selected_mask = [True] * len(sub_vars)
        while True:
            canvas = np.zeros((H, W, 3), dtype=np.uint8)
            cv2.rectangle(canvas, (0,0), (W, 100), (40,40,40), -1)
            cv2.putText(canvas, f"MERGE GROUP: {parent_group}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Draw all variations side-by-side
            for i, sub in enumerate(sub_vars):
                if not selected_mask[i]: continue
                # Rotate colors for distinctness
                color = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)][i % 6]
                self.draw_skeleton(canvas, self.centroids[sub], color, offset_x=-150)
            
            # Draw Control Menu
            for i, sub in enumerate(sub_vars):
                color = (0, 255, 0) if selected_mask[i] else (50, 50, 50)
                txt = f"{i+1}. {sub} ({self.counts[sub]} rows) {'[KEEP]' if selected_mask[i] else '[DROP]'}"
                cv2.putText(canvas, txt, (600, 150 + (i*40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.imshow("Smart Merger", canvas)
            key = cv2.waitKey(0)
            
            if key == 13: return selected_mask # Enter -> Merge Selection
            elif key == ord('s') or key == ord('S'): return "SPLIT" # Split -> Keep Separate
            elif ord('1') <= key <= ord('9'):
                idx = key - ord('1')
                if idx < len(sub_vars): selected_mask[idx] = not selected_mask[idx]

    def run(self):
        # 1. GROUPING: Identify candidates for merging
        # Logic: "PALM_0", "PALM_1" -> Group "PALM"
        groups = {}
        for label in self.sub_labels:
            parent = label.rsplit('_', 1)[0] if (len(label.rsplit('_', 1)) > 1 and label.rsplit('_', 1)[1].isdigit()) else label
            if parent not in groups: groups[parent] = []
            groups[parent].append(label)

        # 2. MERGING: User Decision Loop
        for parent, subs in groups.items():
            if len(subs) == 1:
                # Auto-accept singletons
                self.final_mappings[subs[0]] = parent
                continue
                
            result = self.interactive_merge(parent, subs)
            
            if result == "SPLIT":
                # User chose to keep them separate (e.g. PALM_0 vs PALM_1)
                for sub in subs: self.final_mappings[sub] = sub
            else:
                # User chose to merge specific subsets
                for i, sub in enumerate(subs):
                    if result[i]: self.final_mappings[sub] = parent
                    else: self.ignore_list.append(sub)

        # 3. POLISHING: Final Cleanup
        clean_df = self.df[~self.df['label'].isin(self.ignore_list)].copy()
        clean_df['label'] = clean_df['label'].map(self.final_mappings)
        
        print("\n✨ APPLYING FINAL LABEL POLISH...")
        clean_df['label'] = clean_df['label'].replace(self.polish_map)
        
        # Save Final Output
        clean_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ SUCCESS. Training ready dataset saved to: {OUTPUT_FILE}")
        print(clean_df['label'].value_counts())
        print("\n✨ APPLYING FINAL LABEL POLISH...")
        clean_df['label'] = clean_df['label'].replace(self.polish_map)

if __name__ == "__main__":
    SmartMerger().run()