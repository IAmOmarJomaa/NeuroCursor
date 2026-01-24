"""
NeuroCursor Data Refinery (The Distiller).

This module implements an unsupervised learning pipeline to clean and refine
raw motion capture data. It uses a custom "Weighted K-Means" algorithm to
cluster variations of a gesture while prioritizing critical anatomical features
(Fingertips) over stable anchors (Wrist).

Algorithm:
    1. Feature Weighting: Scales fingertip coordinates by 10x to force the 
       clustering algorithm to focus on intent-defining features.
    2. Strict Silhouette Analysis: Automatically determines the optimal number 
       of variations (k) for each gesture label.
    3. Human-in-the-Loop: Provides a visual interface for the developer to 
       manually split, rename, or reject clusters.

Architecture:
    [Raw CSV] -> [Feature Scaling] -> [K-Means] -> [Manual Review] -> [Golden CSV]
"""

import cv2
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
import os
import sys
import re

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import PATHS

# CONFIGURATION
INPUT_FILE = str(PATHS["RAW_DATA"])
OUTPUT_FILE = str(PATHS["GOLDEN_DATA"])

# Visualization Constants
W, H = 600, 600
CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),    # Thumb
    (0,5), (5,6), (6,7), (7,8),    # Index
    (0,9), (9,10), (10,11), (11,12), # Middle
    (0,13), (13,14), (14,15), (15,16), # Ring
    (0,17), (17,18), (18,19), (19,20)  # Pinky
]

# --- FEATURE ENGINEERING: FINGER PRIORITY WEIGHTS ---
# We multiply coordinate values by these weights BEFORE clustering.
# This forces the Euclidean distance metric to prioritize fingertip positions
# (which define the gesture) over the wrist/palm (which are often stationary).

WEIGHT_LOW = 1.0    # Wrist & Palm Base
WEIGHT_MED = 3.0    # Lower Knuckles
WEIGHT_HIGH = 5.0   # Upper Knuckles
WEIGHT_CRITICAL = 10.0 # Fingertips (The Intent)

FEATURE_WEIGHTS = {
    # Wrist & Palm Base
    0: WEIGHT_LOW, 1: WEIGHT_LOW, 5: WEIGHT_LOW, 9: WEIGHT_LOW, 13: WEIGHT_LOW, 17: WEIGHT_LOW,
    
    # Lower Knuckles
    2: WEIGHT_MED, 6: WEIGHT_MED, 10: WEIGHT_MED, 14: WEIGHT_MED, 18: WEIGHT_MED,
    
    # Upper Knuckles
    3: WEIGHT_HIGH, 7: WEIGHT_HIGH, 11: WEIGHT_HIGH, 15: WEIGHT_HIGH, 19: WEIGHT_HIGH,
    
    # FINGERTIPS (Critical Intent Features)
    4: WEIGHT_CRITICAL,  # Thumb
    8: WEIGHT_CRITICAL,  # Index
    12: WEIGHT_CRITICAL, # Middle
    16: WEIGHT_CRITICAL, # Ring
    20: WEIGHT_CRITICAL  # Pinky
}

def apply_weights(X_raw):
    """
    Scales the input feature matrix based on anatomical importance.
    
    Args:
        X_raw: Numpy array of shape (N, 63)
        
    Returns:
        X_weighted: Scaled feature matrix optimized for clustering.
    """
    X_weighted = X_raw.copy()
    # X_raw is flattened: x0, y0, z0, x1, y1, z1...
    for i in range(21):
        w = FEATURE_WEIGHTS.get(i, 1.0)
        X_weighted[:, i*3]     *= w # Scale x
        X_weighted[:, i*3 + 1] *= w # Scale y
        X_weighted[:, i*3 + 2] *= w # Scale z
    return X_weighted

def load_data():
    """Loads raw and golden datasets, preserving lineage via 'source_label'."""
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found!")
        sys.exit()
    raw_df = pd.read_csv(INPUT_FILE)
    
    golden_df = pd.DataFrame()
    if os.path.exists(OUTPUT_FILE):
        golden_df = pd.read_csv(OUTPUT_FILE)
        # Infer source label if missing (for legacy data support)
        if 'source_label' not in golden_df.columns:
             def infer_source(row_lbl):
                base = re.sub(r'_\d+$', '', str(row_lbl))
                parts = str(row_lbl).split('_')
                if parts[0] in raw_df['label'].unique(): return parts[0]
                return base
             golden_df['source_label'] = golden_df['label'].apply(infer_source)
             
    return raw_df, golden_df

def get_strict_clusters(X_raw, max_k=8):
    """
    Determines optimal K using Silhouette Analysis on Weighted Features.
    Biased towards higher K to capture subtle variations (micro-clusters).
    """
    # Apply Physics-based Weighting
    X_focused = apply_weights(X_raw)
    
    if len(X_focused) < 20: return 1
    
    best_k = 1
    best_score = -1
    
    for k in range(2, max_k + 1):
        try:
            # n_init=20 ensures we find the global optimum, not a local one
            kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
            lbls = kmeans.fit_predict(X_focused)
            score = silhouette_score(X_focused, lbls)
            
            # Prefer structural distinction (Score > 0.25 is generally decent)
            if score > best_score:
                best_score = score
                best_k = k
        except: pass
        
    if best_score < 0.25: return 1
    return best_k

def draw_skeleton(landmarks, label_name, cid, total, manual_k=False):
    """Renders the skeletal visualization for the Review Interface."""
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    coords = landmarks.reshape(-1, 3)
    xs, ys = coords[:, 0], coords[:, 1]
    scale, ox, oy = 400, 100, 100
    
    for s, e in CONNECTIONS:
        cv2.line(canvas, (int(xs[s]*scale+ox), int(ys[s]*scale+oy)), 
                 (int(xs[e]*scale+ox), int(ys[e]*scale+oy)), (100,100,100), 2)
    
    # Highlight Fingertips for easy identification
    for i in range(21):
        rad = 6 if i in [4,8,12,16,20] else 3
        col = (0,255,255) if i in [4,8,12,16,20] else ((0,255,0) if i==0 else (0,0,255))
        cv2.circle(canvas, (int(xs[i]*scale+ox), int(ys[i]*scale+oy)), rad, col, -1)

    # UI Overlay
    cv2.rectangle(canvas, (0,0), (W, 90), (30,30,30), -1)
    cv2.putText(canvas, f"SRC: {label_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    k_mode = "MANUAL" if manual_k else "STRICT-AUTO"
    cv2.putText(canvas, f"VAR: {cid} | N: {total} | MODE: {k_mode}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    
    cv2.putText(canvas, "[ENTER] Keep   [N] Rename   [X] Delete", (10, H-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.putText(canvas, "[C] Force Split", (10, H-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,150,255), 1)
    return canvas

def run_refinery():
    print("🏭 REFINERY V6 (STRICT FINGER FOCUS)")
    print(f"   Input: {INPUT_FILE}")
    print(f"   Output: {OUTPUT_FILE}")

    while True:
        raw_df, golden_df = load_data()
        unique_raw = sorted(raw_df['label'].unique())

        print("\n" + "="*60)
        print("📊 DATASET X-RAY")
        print("="*60)
        
        todo_list = []
        for i, lbl in enumerate(unique_raw):
            is_done = False
            sub_labels = []
            if not golden_df.empty and 'source_label' in golden_df.columns:
                matches = golden_df[golden_df['source_label'] == lbl]
                if not matches.empty:
                    is_done = True
                    sub_labels = sorted(matches['label'].unique())
            
            status = "✅" if is_done else "🆕"
            sub_str = f"{sub_labels[0]} +{len(sub_labels)-1}" if len(sub_labels)>1 else (sub_labels[0] if sub_labels else "---")
            print(f" [{str(i).rjust(2)}] | {status} | {lbl.ljust(15)} | {sub_str}")
            todo_list.append(lbl)

        print("-" * 60)
        print(" Enter ID to process (or 'q'): ", end="")
        choice = input().strip().lower()
        if choice == 'q': break
        
        try:
            target_idx = int(choice)
            target_label = todo_list[target_idx]
        except:
            print("❌ Invalid.")
            continue
            
        print(f"\n🔄 LOADING RAW DATA: {target_label}...")
        label_df = raw_df[raw_df['label'] == target_label]
        feature_cols = [c for c in raw_df.columns if c.startswith('x') or c.startswith('y') or c.startswith('z')]
        X = label_df[feature_cols].values
        
        # 1. Determine optimal Cluster Count (K)
        k = get_strict_clusters(X)
        manual_override = False
        
        while True:
            # 2. Apply Weighted Clustering
            X_weighted = apply_weights(X)
            
            print(f"   ⚙️  Strict-Clustering with k={k}...")
            kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
            c_labels = kmeans.fit_predict(X_weighted) 
            
            # Find representative samples (Centroids) for visualization
            closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_weighted)
            
            new_rows = []
            restart_clustering = False
            
            for cid, rep_idx in enumerate(closest):
                if restart_clustering: break
                
                mask = (c_labels == cid)
                c_rows = label_df[mask].copy()
                
                # 3. Interactive Review Loop
                while True:
                    # Visualization uses RAW coordinates (Reality), not Weighted
                    canvas = draw_skeleton(X[rep_idx], target_label, cid+1, len(c_rows), manual_override)
                    cv2.imshow("Refinery", canvas)
                    key = cv2.waitKey(0)
                    
                    if key == ord('c') or key == ord('C'):
                        print(f"\n   ⌨️  Force Split K (Current: {k}): ", end="")
                        sys.stdout.flush()
                        try:
                            new_k = int(input())
                            if new_k > 0:
                                k = new_k
                                manual_override = True
                                restart_clustering = True
                                break 
                        except: print("Invalid.")
                    
                    elif key == 13: # Enter (Keep)
                        final_name = f"{target_label}_{cid}"
                        print(f"   ✅ Kept as {final_name}")
                        c_rows['label'] = final_name
                        c_rows['source_label'] = target_label
                        new_rows.append(c_rows)
                        break
                        
                    elif key == ord('n') or key == ord('N'): # Rename
                        print(f"   > Rename Var {cid+1}: ", end="")
                        sys.stdout.flush()
                        while True:
                            val = input().strip().upper().replace(" ", "_")
                            if val: 
                                final_name = val
                                break
                        print(f"   ✅ Kept as {final_name}")
                        c_rows['label'] = final_name
                        c_rows['source_label'] = target_label
                        new_rows.append(c_rows)
                        break
                        
                    elif key == ord('x') or key == ord('X'): # Delete
                        print("   ❌ Deleted cluster.")
                        break
            
            if restart_clustering: continue
            else: break

        cv2.destroyAllWindows()
        
        # 4. Save Changes
        print("💾 Saving...")
        if not golden_df.empty and 'source_label' in golden_df.columns:
            # Overwrite previous refined data for this label
            golden_df = golden_df[golden_df['source_label'] != target_label]
            
        if new_rows:
            golden_df = pd.concat([golden_df] + new_rows, ignore_index=True)
            
        golden_df.to_csv(OUTPUT_FILE, index=False)
        print("✨ Done.")

if __name__ == "__main__":
    run_refinery()