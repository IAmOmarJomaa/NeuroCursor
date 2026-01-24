import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from config import PATHS

def calculate_distance(sig1, sig2):
    """Calculates Euclidean Distance between two 5D finger vectors."""
    diff = 0
    for f in ["thumb", "index", "middle", "ring", "pinky"]:
        d = sig1[f] - sig2[f]
        diff += d * d
    return np.sqrt(diff)

def audit_signatures():
    if not PATHS["RULES"].exists():
        print("❌ NO DATA. Run recorder first.")
        return

    with open(PATHS["RULES"], "r") as f:
        data = json.load(f)
        signatures = data.get("signatures", {})

    keys = list(signatures.keys())
    if not keys:
        print("❌ NO SIGNATURES FOUND.")
        return

    # Build Distance Matrix
    n = len(keys)
    matrix = np.zeros((n, n))

    print("\n📊 --- GESTURE DISTANCE MATRIX ---")
    print("(Higher number = Better separation. < 10 is DANGEROUS)")
    print(f"{'':<15} | " + " | ".join([f"{k[:6]:<6}" for k in keys]))
    print("-" * (15 + n * 9))

    for i in range(n):
        row_str = f"{keys[i]:<15} | "
        for j in range(n):
            dist = calculate_distance(signatures[keys[i]], signatures[keys[j]])
            matrix[i][j] = dist
            
            # Color code text output
            val_str = f"{dist:.1f}"
            if i != j and dist < 10: val_str += "⚠️"
            row_str += f"{val_str:<6} | "
        print(row_str)

    # VISUAL HEATMAP
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, xticklabels=keys, yticklabels=keys, cmap="viridis", fmt=".1f")
    plt.title("Gesture Euclidean Distances (The 'Data Science' View)")
    plt.show()

if __name__ == "__main__":
    audit_signatures()