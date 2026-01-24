import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
import os

# CONFIG
INPUT_FILE = "data/training_data.csv"
OUTPUT_DIR = "data/cluster_visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SKELETON DEFINITION (Mediapipe Standard)
CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),    # Thumb
    (0,5), (5,6), (6,7), (7,8),    # Index
    (0,9), (9,10), (10,11), (11,12), # Middle
    (0,13), (13,14), (14,15), (15,16), # Ring
    (0,17), (17,18), (18,19), (19,20)  # Pinky
]

def load_data():
    if not os.path.exists(INPUT_FILE):
        print("❌ No data file found! Record some gestures first.")
        return None
    df = pd.read_csv(INPUT_FILE)
    print(f"✅ Loaded {len(df)} samples.")
    return df

def get_optimal_clusters(data, max_k=5):
    """
    Auto-detects how many variations exist (1 to 5).
    """
    if len(data) < 15: return 1
    
    best_k = 1
    best_score = -1
    
    # Try 2 to max_k clusters
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(data)
        
        # Silhouette Score: How distinct are the groups?
        score = silhouette_score(data, labels)
        
        # If score is high, these clusters are very distinct
        if score > best_score:
            best_score = score
            best_k = k
            
    # Threshold: If split isn't clean (< 0.25), assume it's just 1 variation
    if best_score < 0.25: 
        return 1
        
    return best_k

def draw_skeleton(ax, landmarks, color='#00ff00', title=""):
    """
    Draws a professional schematic of the hand.
    """
    # Landmarks = [x0, y0, z0, x1, y1, z1, ...]
    coords = landmarks.reshape(-1, 3)
    
    # Invert Y so it looks like the screen (Top-Left origin)
    xs = coords[:, 0]
    ys = -coords[:, 1] 
    
    # Draw Joints
    ax.scatter(xs, ys, s=30, c='red', zorder=2)
    
    # Draw Bones
    for start, end in CONNECTIONS:
        ax.plot([xs[start], xs[end]], [ys[start], ys[end]], c='black', linewidth=2, zorder=1)
        
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=10, fontweight='bold')

def analyze_and_visualize():
    df = load_data()
    if df is None: return

    # Get labels sorted alphabetically
    labels = sorted(df['label'].unique())
    
    print(f"🔍 Analyzing {len(labels)} gesture classes...")

    for label in labels:
        print(f"   > Processing: {label}...")
        
        # 1. Get raw data (X, Y, Z columns only)
        label_df = df[df['label'] == label]
        feature_cols = [c for c in df.columns if c.startswith('x') or c.startswith('y') or c.startswith('z')]
        X = label_df[feature_cols].values
        
        # 2. Auto-Detect Clusters
        n_clusters = get_optimal_clusters(X)
        
        # 3. Fit K-Means
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(X)
        
        # 4. FIND REPRESENTATIVES (The "Medoids")
        # Find the index of the point closest to each cluster center
        closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
        
        # 5. DRAW
        fig, axes = plt.subplots(1, n_clusters, figsize=(3 * n_clusters, 3.5))
        if n_clusters == 1: axes = [axes]
        
        fig.suptitle(f"Gesture: {label} ({n_clusters} variations)", fontsize=14)
        
        for i, idx in enumerate(closest_indices):
            # Get the ACTUAL recorded frame (not the math average)
            representative_sample = X[idx]
            count = np.sum(kmeans.labels_ == i)
            percentage = int((count / len(X)) * 100)
            
            draw_skeleton(axes[i], representative_sample, title=f"Var {i+1}\n({percentage}% freq)")
            
        # Save
        filename = f"{OUTPUT_DIR}/{label}_analysis.png"
        plt.savefig(filename, dpi=100)
        plt.close()
        
    print(f"\n✅ DONE. Images saved to: {OUTPUT_DIR}")
    print("👉 Open that folder to verify your gestures!")

if __name__ == "__main__":
    analyze_and_visualize()