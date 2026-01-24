import pandas as pd
import numpy as np
import os
import sys
import re
from itertools import combinations
from sklearn.tree import DecisionTreeClassifier, export_text
import warnings

# Setup
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- WAR ZONES (Specific Confusions) ---
WAR_ZONES = [
    {
        "name": "⚔️ THE FIST WARS (Fist vs Ctrl vs Cut vs Del_Closed)",
        "targets": ["FIST", "CTRL", "CUT", "DELETE_CLOSED", "GEN_Z", "NEXT_CLOSED", "PREVIOUS_CLOSED"]
    },
    {
        "name": "⚔️ THE PALM WARS (Palm vs Delete_Open)",
        "targets": ["PALM", "DELETE_OPEN"]
    },
    {
        "name": "⚔️ THE V-SHAPE WARS (Right Click vs Win vs Scroll vs Tab)",
        "targets": ["RIGHT_CLICK", "WIN", "SCROLL", "TAB"]
    },
    {
        "name": "⚔️ THE POINTER WARS (Point vs Shhh vs Zoom)",
        "targets": ["THE_POINT", "THE_SHHH", "ZOOM"]
    },
    {
        "name": "⚔️ THE ANNOYING TRIO (Volume vs Scroll vs Right Click)",
        "targets": ["VOLUME", "SCROLL", "RIGHT_CLICK"]
    },
]

def load_golden_data():
    path = os.path.join("data", "processed", "golden_dataset.csv")
    if not os.path.exists(path):
        print(f"❌ CRITICAL: {path} not found.")
        sys.exit(1)
    
    print(f"📂 Loading Golden Data from {path}...")
    
    # 1. READ WITH HEADER (Since debug showed headers exist)
    try:
        df = pd.read_csv(path, header=0, low_memory=False)
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        sys.exit(1)
        
    # 2. RENAME COLUMNS to ensure consistency
    # We expect Label + 63 coords
    if df.shape[1] < 64:
        print(f"❌ Data has {df.shape[1]} columns. Expected 64+.")
        sys.exit(1)
        
    # Force column names to be safe
    new_cols = ["label"] + [f"v{i}" for i in range(df.shape[1]-1)]
    df.columns = new_cols
    
    # 3. CLEAN NUMERICS
    # Coerce all coordinate columns to numbers
    print("   -> Cleaning numeric data...")
    coord_cols = df.columns[1:64] # First 63 coord cols
    df[coord_cols] = df[coord_cols].apply(pd.to_numeric, errors='coerce')
    
    # 4. DROP BAD ROWS
    before = len(df)
    df.dropna(subset=coord_cols, inplace=True)
    print(f"   -> Data Cleaned. Valid Rows: {len(df)} (Dropped {before - len(df)})")
    
    return df

def get_broad_label(label):
    # Removes _0, _1, _2 suffixes
    s = str(label)
    return re.sub(r'_\d+$', '', s)

def calculate_geometry(row_values):
    # row_values: numpy array of the 63 coordinates
    try:
        # Force float type explicitly
        coords = np.array(row_values, dtype=float).reshape(21, 3)
    except Exception as e:
        return None

    feats = {}
    
    # 1. BASE SCALE
    scale = np.linalg.norm(coords[0] - coords[9]) + 1e-6
    
    # 2. KEY DISTANCES
    interesting_points = [0, 4, 8, 12, 16, 20, 5, 9, 13, 17] 
    for i, j in combinations(interesting_points, 2):
        dist = np.linalg.norm(coords[i] - coords[j])
        feats[f"dist_{i}_{j}"] = dist / scale

    # 3. OFFSETS (X-Axis spread)
    feats["thumb_x_offset"] = abs(coords[4][0] - coords[0][0]) / scale
    feats["index_x_offset"] = abs(coords[8][0] - coords[0][0]) / scale
    feats["pinky_x_offset"] = abs(coords[20][0] - coords[0][0]) / scale
    
    # 4. EXTENSIONS (Tip to Wrist)
    feats["thumb_ext"] = np.linalg.norm(coords[4] - coords[0]) / scale
    feats["index_ext"] = np.linalg.norm(coords[8] - coords[0]) / scale
    feats["middle_ext"] = np.linalg.norm(coords[12] - coords[0]) / scale
    feats["ring_ext"] = np.linalg.norm(coords[16] - coords[0]) / scale
    feats["pinky_ext"] = np.linalg.norm(coords[20] - coords[0]) / scale

    return feats

def translate_feature_name(name):
    map_pts = {
        '0':'WRIST', '4':'THUMB_TIP', '8':'INDEX_TIP', '12':'MID_TIP', '16':'RING_TIP', '20':'PINKY_TIP',
        '5':'INDEX_KNUCKLE', '9':'MID_KNUCKLE', '13':'RING_KNUCKLE', '17':'PINKY_KNUCKLE'
    }
    if "dist_" in name:
        parts = name.split('_')
        p1 = map_pts.get(parts[1], parts[1])
        p2 = map_pts.get(parts[2], parts[2])
        return f"Dist({p1}-{p2})"
    return name

def analyze_one_vs_all(df):
    print("\n" + "="*80)
    print("🌍 PART 1: GLOBAL IDENTITY REPORT (One vs All)")
    print("="*80)

    df['broad'] = df['label'].apply(get_broad_label)
    
    print("⚙️  Calculating features...")
    all_X = []
    all_y = []
    
    # Process rows
    # df columns: label, v0, v1, ... v62
    # We pass values of v0...v62 (indices 1 to 64)
    data_matrix = df.iloc[:, 1:64].values
    labels = df['broad'].values
    
    for i in range(len(data_matrix)):
        f = calculate_geometry(data_matrix[i])
        if f:
            all_X.append(f)
            all_y.append(labels[i])
            
    if not all_X:
        print("❌ Error: Feature calculation failed for all rows.")
        return

    X_df = pd.DataFrame(all_X)
    y_series = pd.Series(all_y)
    unique_labels = sorted(y_series.unique())

    for target in unique_labels:
        if "NOISE" in target: continue
        
        y_binary = (y_series == target).astype(int)
        if y_binary.sum() < 5: continue

        clf = DecisionTreeClassifier(max_depth=2, class_weight='balanced')
        clf.fit(X_df, y_binary)
        
        importances = pd.Series(clf.feature_importances_, index=X_df.columns).sort_values(ascending=False)
        top_feat = importances.index[0]
        readable = translate_feature_name(top_feat)
        
        tree_text = export_text(clf, feature_names=list(X_df.columns))
        threshold = "Unknown"
        for line in tree_text.split('\n'):
            if top_feat in line:
                threshold = line.split(top_feat)[1].strip()
                break

        print(f"🔍 {target.ljust(20)} | Key: {readable.ljust(35)} | Rule: {threshold}")

def analyze_war_zones(df):
    print("\n" + "="*80)
    print("⚔️ PART 2: WAR ZONE REPORT (Close Combat)")
    print("="*80)

    for zone in WAR_ZONES:
        print(f"\n{zone['name']}")
        print("-" * 60)
        
        mask = df['label'].apply(lambda x: any(t in str(x) for t in zone['targets']))
        zone_df = df[mask].copy()
        
        if len(zone_df) < 5:
            print("   ⚠️ Not enough data.")
            continue
            
        zone_df['broad_label'] = zone_df['label'].apply(get_broad_label)
        
        X_data = []
        y_data = []
        
        data_matrix = zone_df.iloc[:, 1:64].values
        labels = zone_df['broad_label'].values
        
        for i in range(len(data_matrix)):
            f = calculate_geometry(data_matrix[i])
            if f:
                X_data.append(f)
                y_data.append(labels[i])
        
        if not X_data:
            print("   ⚠️ Error: No valid geometry calculated.")
            continue

        X = pd.DataFrame(X_data)
        clf = DecisionTreeClassifier(max_depth=3, random_state=42, class_weight='balanced')
        clf.fit(X, y_data)
        
        rules = export_text(clf, feature_names=list(X.columns))
        
        print("   📜 SEPARATION LOGIC:")
        for line in rules.split('\n'):
            if not line: continue
            for feat in X.columns:
                if feat in line:
                    line = line.replace(feat, translate_feature_name(feat))
            if "class:" in line:
                print(f"      \033[92m{line}\033[0m")
            else:
                print(f"      {line}")

if __name__ == "__main__":
    df = load_golden_data()
    analyze_one_vs_all(df)
    analyze_war_zones(df)