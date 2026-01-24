import pandas as pd
import numpy as np
import os
import sys
from itertools import combinations
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

# Setup Paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import CONFIG, GESTURE_LABELS

# --- CONFIGURATION ---
# Define the "War Zones" - groups of classes that confuse the AI.
# The script will hunt for formulas specifically to separate these groups.
CONFLICT_GROUPS = [
    {"name": "POINT vs ZOOM vs SHHH", "classes": ["1. THE_POINT", "10. ZOOM", "16. THE_SHHH"]},
    {"name": "CLICK vs POINTER", "classes": ["1. THE_POINT", "2. THE_CLICK"]},
    {"name": "FIST vs CLICK (The Drop Issue)", "classes": ["5. FIST", "2. THE_CLICK"]},
]

def load_data():
    path = os.path.join("data", "processed", "golden_dataset.csv")
    if not os.path.exists(path):
        print(f"❌ ERROR: File not found at {path}")
        return None
    
    print(f"📂 Loading {path}...")
    df = pd.read_csv(path, header=None)
    
    # Check if header exists or needs mapping
    # Assuming standard MediaPipe CSV format (label, x0, y0, z0...)
    # First column is label
    return df

def generate_brute_force_features(row):
    # Convert row to (21, 3) numpy array
    # Skip label (col 0)
    coords = row[1:].values.reshape(21, 3)
    
    feats = {}
    
    # 1. Base Scale (Wrist 0 to Middle MCP 9)
    scale = np.linalg.norm(coords[0] - coords[9])
    if scale == 0: scale = 1.0 # Avoid div/0
    
    # 2. GENERATE ALL 210 PAIRWISE DISTANCES (Normalized)
    for i, j in combinations(range(21), 2):
        dist = np.linalg.norm(coords[i] - coords[j])
        feats[f"dist_{i}_{j}"] = dist / scale
        
    # 3. GENERATE KEY 3-POINT ANGLES (Optional, computationally heavy, sticking to dists first)
    # (We can add this if distances aren't enough)

    return feats

def hunt_formulas(df):
    # Pre-compute features for ALL rows (GPU abuse would happen here in a massive scale, 
    # but CPU is fast enough for ~50k rows of geometry)
    print("⚙️  Generating 200+ geometric features per sample... (This finds the hidden relationships)")
    
    feature_list = []
    labels = []
    
    for _, row in df.iterrows():
        try:
            label_str = row[0]
            if label_str not in GESTURE_LABELS: continue # Skip junk
            
            f = generate_brute_force_features(row)
            feature_list.append(f)
            labels.append(label_str)
        except Exception as e:
            continue
            
    feat_df = pd.DataFrame(feature_list)
    feat_df["label"] = labels
    
    print(f"✅ Generated {len(feat_df.columns)-1} features for {len(feat_df)} samples.\n")

    # --- THE HUNT ---
    for group in CONFLICT_GROUPS:
        print(f"⚔️  ANALYZING GROUP: {group['name']}")
        print(f"    Target Classes: {group['classes']}")
        
        # Filter Data
        sub_df = feat_df[feat_df["label"].isin(group["classes"])]
        
        if len(sub_df) < 10:
            print("    ⚠️  Not enough data for this group. Skipping.\n")
            continue
            
        X = sub_df.drop("label", axis=1)
        y = sub_df["label"]
        
        # Train a Shallow Decision Tree (Depth 2 or 3)
        # We want simple, human-readable rules, not a black box.
        clf = DecisionTreeClassifier(max_depth=2, random_state=42)
        clf.fit(X, y)
        
        # Score
        acc = clf.score(X, y) * 100
        print(f"    🎯 Model Accuracy: {acc:.1f}%")
        
        # Extract Logic
        tree_rules = export_text(clf, feature_names=list(X.columns))
        
        print("    📜 THE MAGIC FORMULA:")
        print("    ---------------------------------------------------")
        # Parse the tree text to make it friendlier
        lines = tree_rules.split('\n')
        for line in lines:
            if not line: continue
            # Colorize the output
            if "class:" in line:
                print(f"    \033[92m{line}\033[0m") # Green for result
            else:
                print(f"    {line}")
        print("    ---------------------------------------------------")
        
        # Feature Importance (The "Why")
        importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
        top_feat = importances.index[0]
        
        # Translate Feature Name to English
        # e.g., "dist_4_8" -> "Distance between ThumbTip(4) and IndexTip(8)"
        def translate_lm(idx):
            names = {
                0:"Wrist", 1:"ThumbCMC", 2:"ThumbMCP", 3:"ThumbIP", 4:"ThumbTip",
                5:"IndexMCP", 6:"IndexPIP", 7:"IndexDIP", 8:"IndexTip",
                9:"MidMCP", 10:"MidPIP", 11:"MidDIP", 12:"MidTip",
                13:"RingMCP", 14:"RingPIP", 15:"RingDIP", 16:"RingTip",
                17:"PinkyMCP", 18:"PinkyPIP", 19:"PinkyDIP", 20:"PinkyTip"
            }
            return names.get(int(idx), str(idx))

        if "dist_" in top_feat:
            parts = top_feat.split('_')
            p1 = translate_lm(parts[1])
            p2 = translate_lm(parts[2])
            print(f"    🔑 KEY VARIABLE: Distance between {p1} and {p2}")
            print(f"       (This feature alone solves {importances.iloc[0]*100:.1f}% of the confusion)\n")
        
if __name__ == "__main__":
    df = load_data()
    if df is not None:
        hunt_formulas(df)