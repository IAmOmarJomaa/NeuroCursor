import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os
import sys
from itertools import combinations

INPUT_FILE = "data/training_data.csv"
OUTPUT_DIR = "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# READABLE MAP
P_NAMES = {
    0:"WRIST", 1:"TM_CMC", 2:"TM_MCP", 3:"TM_IP", 4:"TM_TIP",
    5:"IDX_MCP", 6:"IDX_PIP", 7:"IDX_DIP", 8:"IDX_TIP",
    9:"MID_MCP", 10:"MID_PIP", 11:"MID_DIP", 12:"MID_TIP",
    13:"RNG_MCP", 14:"RNG_PIP", 15:"RNG_DIP", 16:"RNG_TIP",
    17:"PNK_MCP", 18:"PNK_PIP", 19:"PNK_DIP", 20:"PNK_TIP"
}

def load_and_engineer_features(df):
    print(f"⚙️  SIMULATION STARTED: Processing {len(df)} frames...")
    print("    -> Calculating Universe of Relationships (Dist, DX, DY)...")
    
    # 1. SCALE FACTOR (Wrist to Index MCP)
    x0, y0 = df["x0"].values, df["y0"].values
    x5, y5 = df["x5"].values, df["y5"].values
    scales = np.sqrt((x0-x5)**2 + (y0-y5)**2)
    scales[scales < 0.01] = 1.0 
    
    features = {}
    
    # 2. GENERATE ALL PAIRS (Brute Force)
    for i in range(21):
        for j in range(21):
            if i == j: continue
            
            # Raw coords
            xi, yi = df[f"x{i}"].values, df[f"y{i}"].values
            xj, yj = df[f"x{j}"].values, df[f"y{j}"].values
            
            # Metrics
            dx = (xi - xj) / scales # Relative X
            dy = (yi - yj) / scales # Relative Y (Up/Down)
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2) / scales # Euclidean
            
            if i < j:
                features[f"DIST_{P_NAMES[i]}_{P_NAMES[j]}"] = dist
            
            features[f"DX_{P_NAMES[i]}_{P_NAMES[j]}"] = dx
            features[f"DY_{P_NAMES[i]}_{P_NAMES[j]}"] = dy

    feat_df = pd.DataFrame(features)
    feat_df["label"] = df["label"].values
    return feat_df

def calculate_separation_power(df, feature, class_a, class_b):
    """
    Calculates Fisher Discriminant Ratio (higher is better separation)
    """
    a_vals = df[df["label"]==class_a][feature]
    b_vals = df[df["label"]==class_b][feature]
    
    if len(a_vals) < 2 or len(b_vals) < 2: return 0
    
    mu_a, mu_b = a_vals.mean(), b_vals.mean()
    var_a, var_b = a_vals.var(), b_vals.var()
    
    return ((mu_a - mu_b)**2) / (var_a + var_b + 1e-9)

def visualize_battle(df, class_a, class_b, top_feature):
    """
    Generates a scatter plot proving the separation.
    """
    plt.figure(figsize=(10, 6))
    subset = df[df["label"].isin([class_a, class_b])]
    
    sns.kdeplot(data=subset, x=top_feature, hue="label", fill=True, palette="bright")
    
    # Calculate perfect threshold
    vals_a = subset[subset["label"]==class_a][top_feature]
    vals_b = subset[subset["label"]==class_b][top_feature]
    
    # Determine split point
    mean_a, mean_b = vals_a.mean(), vals_b.mean()
    threshold = (mean_a + mean_b) / 2
    
    plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold: {threshold:.3f}")
    plt.title(f"Visual Proof: {class_a} vs {class_b}\nFeature: {top_feature}")
    plt.legend()
    
    filename = f"{OUTPUT_DIR}/VS_{class_a}_vs_{class_b}.png"
    plt.savefig(filename)
    plt.close()
    print(f"    📸 Saved visual proof to: {filename}")
    return threshold

def solve_universe(feat_df):
    labels = feat_df["label"].unique()
    pairs = list(combinations(labels, 2))
    
    print(f"\n🌌 ANALYZING {len(pairs)} GESTURE CONFLICTS...")
    
    final_rules = []
    
    for class_a, class_b in pairs:
        print(f"\n⚔️  {class_a} vs {class_b}")
        
        # Find the single best feature to separate these two
        best_score = -1
        best_feat = None
        
        # Random sampling for speed if features > 1000, else do all
        cols = feat_df.columns if len(feat_df.columns) < 2000 else np.random.choice(feat_df.columns, 2000, replace=False)
        
        for col in cols:
            if col == "label": continue
            score = calculate_separation_power(feat_df, col, class_a, class_b)
            if score > best_score:
                best_score = score
                best_feat = col
        
        # Visualize and get threshold
        thresh = visualize_battle(feat_df, class_a, class_b, best_feat)
        
        # Determine Logic (< or >)
        mean_a = feat_df[feat_df["label"]==class_a][best_feat].mean()
        op = "<" if mean_a < thresh else ">"
        
        print(f"    🏆 Winner: {best_feat} (Score: {best_score:.1f})")
        print(f"    📝 Rule: If {best_feat} {op} {thresh:.3f} -> Likely {class_a}")
        
        final_rules.append({
            "pair": f"{class_a}_vs_{class_b}",
            "feature": best_feat,
            "threshold": thresh,
            "logic_for_a": op
        })

    return final_rules

def generate_code_block(rules):
    print("\n" + "="*80)
    print("📜 THE GOD CODE (Paste this into gestures.py)")
    print("="*80)
    
    # Group rules by Primary Gesture to build specific checks
    # This is a simplified generator. You would manually refine this based on the visual proof.
    print("# AUTO-GENERATED RULES FROM OMNISCIENT ANALYZER")
    print("def check_gestures(self, m):")
    
    for r in rules:
        print(f"    # Conflict: {r['pair']}")
        print(f"    # Feature: {r['feature']}")
        print(f"    # Threshold: {r['threshold']:.3f}")
        print(f"    pass # Logic {r['logic_for_a']}\n")

def main():
    if not os.path.exists(INPUT_FILE):
        print("Run Recorder first.")
        return
        
    df = pd.read_csv(INPUT_FILE)
    feat_df = load_and_engineer_features(df)
    
    rules = solve_universe(feat_df)
    generate_code_block(rules)
    print(f"\n✅ ANALYSIS COMPLETE. CHECK '{OUTPUT_DIR}' FOR VISUAL PROOFS.")

if __name__ == "__main__":
    main()