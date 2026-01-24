import pandas as pd
import numpy as np
import math
import sys
import os

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import PATHS

def get_dist(row, i1, i2):
    """Euclidean distance between two landmarks in a dataframe row."""
    return math.hypot(row[f'x{i1}'] - row[f'x{i2}'], row[f'y{i1}'] - row[f'y{i2}'])

def get_finger_state(row, tip, pip, wrist, scale):
    """
    Returns a 'Extension Ratio'.
    > 1.0 means fully extended (Tip is far from wrist).
    < 0.5 means fully curled (Tip is close to wrist).
    """
    # Distance from Wrist to Tip
    dist_tip = get_dist(row, 0, tip)
    # Distance from Wrist to Knuckle (PIP)
    dist_pip = get_dist(row, 0, pip)
    
    # We normalize by the hand scale to make it size-invariant
    # But a simpler robust metric is just Tip_Dist / Scale_Reference
    # Let's use: (Tip_Dist - PIP_Dist) relative to Scale
    # If Tip is further than PIP, value is Positive (Open).
    # If Tip is closer than PIP, value is Negative/Low (Closed).
    
    return (dist_tip - dist_pip) / scale

def analyze_dna():
    input_file = PATHS["GOLDEN_DATA"]
    if not os.path.exists(input_file):
        print("❌ Golden Data not found.")
        return

    print(f"🧬 SEQUENCING GESTURE DNA from: {input_file}...")
    df = pd.read_csv(input_file)
    
    # 1. CALCULATE CORE METRICS FOR EVERY ROW
    print("   -> Calculating physics for all samples...")
    
    # Hand Scale (Wrist to Middle Base) - Critical for normalization
    df['scale'] = df.apply(lambda r: get_dist(r, 0, 9), axis=1)
    
    # Pinch Distance (Thumb Tip to Index Tip) - Normalized
    df['pinch_dist'] = df.apply(lambda r: get_dist(r, 4, 8), axis=1) / df['scale']
    
    # Finger States (Extension Ratio)
    # High Value (>0.2) = OPEN
    # Low Value (<0.1) = CLOSED
    df['index_ext'] = df.apply(lambda r: get_finger_state(r, 8, 6, 0, r['scale']), axis=1)
    df['mid_ext']   = df.apply(lambda r: get_finger_state(r, 12, 10, 0, r['scale']), axis=1)
    df['ring_ext']  = df.apply(lambda r: get_finger_state(r, 16, 14, 0, r['scale']), axis=1)
    df['pinky_ext'] = df.apply(lambda r: get_finger_state(r, 20, 18, 0, r['scale']), axis=1)
    
    # Thumb State (Distance from Pinky Base #17) - Normalized
    # Far = Extended, Close = Tucked
    df['thumb_ext'] = df.apply(lambda r: get_dist(r, 4, 17), axis=1) / df['scale']

    # 2. GROUP BY LABEL AND EXTRACT RANGES
    labels = sorted(df['label'].unique())
    
    print("\n" + "="*100)
    print(f"{'LABEL':<20} | {'INDEX':<12} | {'MIDDLE':<12} | {'RING':<12} | {'PINKY':<12} | {'THUMB':<12} | {'PINCH':<12}")
    print("="*100)
    print(" Format: Mean (Min -> Max)")

    for label in labels:
        subset = df[df['label'] == label]
        
        # Helper to format stats
        def fmt(col):
            mean = subset[col].mean()
            # We take 10th and 90th percentile to ignore outliers/bad data
            low = subset[col].quantile(0.10)
            high = subset[col].quantile(0.90)
            
            # Simple Text Classification for easy reading
            state = "?"
            if "ext" in col:
                state = "OPEN" if mean > 0.2 else "CLOSED"
            if "pinch" in col:
                state = f"{mean:.2f}"
            
            return f"{state} ({low:.2f})"

        print(f"{label:<20} | "
              f"{fmt('index_ext'):<12} | "
              f"{fmt('mid_ext'):<12} | "
              f"{fmt('ring_ext'):<12} | "
              f"{fmt('pinky_ext'):<12} | "
              f"{subset['thumb_ext'].mean():.2f} | "
              f"{subset['pinch_dist'].mean():.2f}")

    print("="*100)
    print("\n✅ ANALYSIS COMPLETE.")
    print("   -> Copy the table above and paste it to the engineer.")
    print("   -> 'OPEN' usually means > 0.20")
    print("   -> 'CLOSED' usually means < 0.10")
    print("   -> 'PINCH' is the critical click threshold.")

if __name__ == "__main__":
    analyze_dna()