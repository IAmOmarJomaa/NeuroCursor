import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import PATHS

# CONFIG: Robust Pathing
# We derived the path from the GOLDEN_DATA path to be safe
PROCESSED_DIR = Path(PATHS["GOLDEN_DATA"]).parent
INPUT_FILE = PROCESSED_DIR / "training_ready.csv"
OUTPUT_FILE = PROCESSED_DIR / "final_features.csv"

def get_vector(p1, p2):
    return np.array([p2['x'] - p1['x'], p2['y'] - p1['y'], p2['z'] - p1['z']])

def get_angle(v1, v2):
    """Returns angle in degrees between two 3D vectors."""
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0: return 0.0
    cos_theta = dot / norm
    # Clip to handle floating point errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def extract_physics_features(row):
    """
    The Golden Formula: Converts 63 raw coords into high-IQ features.
    """
    # 1. Parse Landmarks
    lms = {}
    for i in range(21):
        lms[i] = {'x': row[f'x{i}'], 'y': row[f'y{i}'], 'z': row[f'z{i}']}

    features = {}

    # --- A. FINGER CURL ANGLES (0 = Straight, 180 = Closed) ---
    # THUMB: Angle between 1-2 and 2-3
    v_thumb_1 = get_vector(lms[1], lms[2])
    v_thumb_2 = get_vector(lms[2], lms[3])
    features['thumb_bend'] = get_angle(v_thumb_1, v_thumb_2)

    # FINGERS: Angle between Meta(5) -> Prox(6) and Prox(6) -> Dist(7)
    
    # INDEX (5-6-7-8)
    v_index_meta = get_vector(lms[5], lms[6])
    v_index_dist = get_vector(lms[6], lms[7])
    features['index_bend'] = get_angle(v_index_meta, v_index_dist)
    
    # MIDDLE (9-10-11-12)
    v_mid_meta = get_vector(lms[9], lms[10])
    v_mid_dist = get_vector(lms[10], lms[11])
    features['mid_bend'] = get_angle(v_mid_meta, v_mid_dist)

    # RING (13-14-15-16)
    v_ring_meta = get_vector(lms[13], lms[14])
    v_ring_dist = get_vector(lms[14], lms[15])
    features['ring_bend'] = get_angle(v_ring_meta, v_ring_dist)

    # PINKY (17-18-19-20)
    v_pinky_meta = get_vector(lms[17], lms[18])
    v_pinky_dist = get_vector(lms[18], lms[19])
    features['pinky_bend'] = get_angle(v_pinky_meta, v_pinky_dist)

    # --- B. CRITICAL DISTANCES (Normalized by Palm Size) ---
    # Palm Size = Distance from Wrist(0) to Middle Finger Base(9)
    palm_size = np.linalg.norm(get_vector(lms[0], lms[9]))
    if palm_size == 0: palm_size = 1.0 # Safety

    # PINCH DISTANCE (Thumb Tip 4 to Index Tip 8)
    pinch_dist = np.linalg.norm(get_vector(lms[4], lms[8]))
    features['pinch_dist'] = pinch_dist / palm_size

    # THUMB ABDUCTION (How far thumb is from index base)
    thumb_spread = np.linalg.norm(get_vector(lms[4], lms[5]))
    features['thumb_spread'] = thumb_spread / palm_size
    
    # SPIDER-MAN DISTANCE (Middle Tip 12 to Palm Center 0) -> Middle curled?
    mid_curl_dist = np.linalg.norm(get_vector(lms[12], lms[0]))
    features['mid_palm_dist'] = mid_curl_dist / palm_size

    # --- C. ORIENTATION (Is Hand Inverted?) ---
    # Vector from Wrist(0) to Middle Base(9)
    # If Y component is positive, hand points DOWN (screen coords). Negative = UP.
    features['orientation_y'] = lms[9]['y'] - lms[0]['y'] 
    
    # X orientation (Left/Right tilt)
    features['orientation_x'] = lms[9]['x'] - lms[0]['x']

    return features

def run_feature_engineering():
    print("🧪 STARTING FEATURE EXTRACTION...")
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input not found: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"   Input Rows: {len(df)}")
    
    extracted_data = []
    
    for _, row in df.iterrows():
        feats = extract_physics_features(row)
        feats['label'] = row['label'] # Keep the label!
        extracted_data.append(feats)

    # Convert to DataFrame
    feat_df = pd.DataFrame(extracted_data)
    
    # Move 'label' to last column for cleanliness
    cols = [c for c in feat_df.columns if c != 'label'] + ['label']
    feat_df = feat_df[cols]
    
    feat_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ SUCCESS. Extracted {len(feat_df.columns)-1} Physics Features.")
    print(f"   Saved to: {OUTPUT_FILE}")
    print("   -> These 'Smart Features' make the model size invariant.")

if __name__ == "__main__":
    run_feature_engineering()