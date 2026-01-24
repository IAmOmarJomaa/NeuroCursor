import pandas as pd
import numpy as np
import math
import os
import sys
from sklearn.tree import DecisionTreeClassifier, export_text

INPUT_FILE = "data/training_data.csv"

def get_dist(row, i, j):
    """Normalized Distance between Point I and Point J"""
    # Scale based on Wrist(0) to IndexMCP(5)
    # We use a safe fallback if scale is too small (e.g. hand not fully visible)
    dx = row["x0"] - row["x5"]
    dy = row["y0"] - row["y5"]
    scale = math.hypot(dx, dy)
    if scale < 0.01: scale = 1.0
    
    dist = math.hypot(row[f"x{i}"] - row[f"x{j}"], row[f"y{i}"] - row[f"y{j}"])
    return dist / scale

def feature_engineer(df):
    print("⚙️  CALCULATING BIOMETRICS...")
    
    # 1. THUMB INTERACTIONS
    # Pinch: Thumb(4) to Index Tip(8)
    df["pinch_dist"] = df.apply(lambda r: get_dist(r, 4, 8), axis=1)
    # Heart Cross: Thumb(4) to Middle Base(9)
    df["heart_cross"] = df.apply(lambda r: get_dist(r, 4, 9), axis=1)
    # Thumb Ext: Thumb(4) to Index Base(5)
    df["thumb_ext"] = df.apply(lambda r: get_dist(r, 4, 5), axis=1)
    
    # 2. FINGER "TUCKS" (Tip to Base distance)
    # High value (~1.5+) = Straight/Open
    # Low value (<1.0) = Tucked/Curled
    df["idx_tuck"] = df.apply(lambda r: get_dist(r, 8, 5), axis=1)
    df["mid_tuck"] = df.apply(lambda r: get_dist(r, 12, 9), axis=1)
    df["ring_tuck"] = df.apply(lambda r: get_dist(r, 16, 13), axis=1)
    df["pinky_tuck"] = df.apply(lambda r: get_dist(r, 20, 17), axis=1)

    # 3. OPENNESS RATIOS (Tip-Wrist / Base-Wrist)
    # Good for checking if a finger is vertically "Up" relative to wrist
    for tip, base, name in [(8,5,"idx"), (12,9,"mid"), (16,13,"ring"), (20,17,"pinky")]:
        def get_ratio(r):
            d_tip = math.hypot(r[f"x{tip}"]-r["x0"], r[f"y{tip}"]-r["y0"])
            d_base = math.hypot(r[f"x{base}"]-r["x0"], r[f"y{base}"]-r["y0"])
            return d_tip / (d_base + 0.001)
        df[f"{name}_ratio"] = df.apply(get_ratio, axis=1)

    return df

def analyze_conflict(df, label_a, label_b):
    print(f"\n🥊 FIGHT: {label_a} vs {label_b}")
    print("-" * 50)
    
    # Filter data for just these two labels
    subset = df[df["label"].isin([label_a, label_b])].copy()
    
    # Check if we have data for both
    unique_labels = subset["label"].unique()
    if len(unique_labels) < 2:
        print(f"⚠️  Skipping: Missing data. Found: {unique_labels}")
        return

    # Select only our calculated features (exclude raw x,y,z and label)
    feature_cols = [c for c in subset.columns if c not in ["label"] and not c.startswith("x") and not c.startswith("y") and not c.startswith("z")]
    
    X = subset[feature_cols]
    y = subset["label"]

    # TRAIN TREE (Depth 2 = Human Readable Logic)
    clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf.fit(X, y)

    # PRINT RULES
    tree_rules = export_text(clf, feature_names=feature_cols)
    print("🧠 THE GOLDEN RULE:")
    print(tree_rules)
    
    # STATS FOR VERIFICATION
    # Find the single most important feature to show stats for
    best_idx = clf.feature_importances_.argmax()
    best_feat = feature_cols[best_idx]
    
    mean_a = subset[subset["label"]==label_a][best_feat].mean()
    mean_b = subset[subset["label"]==label_b][best_feat].mean()
    
    print(f"📊 AVG VALUES for '{best_feat}':")
    print(f"   {label_a}: {mean_a:.3f}")
    print(f"   {label_b}: {mean_b:.3f}")

def main():
    if not os.path.exists(INPUT_FILE):
        print("❌ No data found. Run 'dataset_recorder.py' first.")
        return

    print(f"📂 Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"✅ Loaded {len(df)} samples.")
    
    # Calculate all the distances/ratios
    df = feature_engineer(df)

    # --- BATTLE ARENA ---
    # This is where we ask the AI to separate the gestures
    
    # 1. The Main Event: Pointer vs Gen Z Heart
    analyze_conflict(df, "POINTER", "HEART_GEN_Z")

    # 2. The Noise Filter: Pointer vs Scratching/Resting
    analyze_conflict(df, "POINTER", "NOISE_RANDOM")

    # 3. The Click Guard: Pinch vs Fist
    analyze_conflict(df, "PINCH", "FIST")
    
    # 4. Scroll Guard: Spidey vs Pointer
    analyze_conflict(df, "SCROLL_SPIDEY", "POINTER")

if __name__ == "__main__":
    main()