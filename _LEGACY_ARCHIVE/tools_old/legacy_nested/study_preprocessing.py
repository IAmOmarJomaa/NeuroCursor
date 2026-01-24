import json
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

def study_normalization():
    path = ROOT_DIR / "data" / "training_data.json"
    if not path.exists():
        print("❌ NO DATA. Run Recorder first.")
        return

    with open(path, "r") as f:
        data = json.load(f)

    # We will study the "POINTER" class as an example
    target_class = "POINTER"
    samples = [d['data'] for d in data if d['label'] == target_class]
    
    if not samples:
        print(f"❌ No samples found for {target_class}. Record some data first.")
        return

    print(f"🔬 STUDYING {len(samples)} SAMPLES OF '{target_class}'...")

    # --- VISUALIZATION SETUP ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Preprocessing Study: {target_class} Variance", fontsize=16)

    # 1. THE "IDEALIZED" PLOT (Normalized)
    # Since we saved normalized data, we can plot it directly.
    # We plot the Index Finger Tip (Landmark 8) to see stability.
    
    ax_norm = axes[0]
    ax_norm.set_title("NORMALIZED DATA (What the AI Sees)")
    ax_norm.set_xlabel("Normalized X")
    ax_norm.set_ylabel("Normalized Y")
    ax_norm.grid(True)
    ax_norm.axhline(0, color='black', linewidth=0.5)
    ax_norm.axvline(0, color='black', linewidth=0.5)

    # Collect X, Y points for Index Tip (Index 8 -> indices 24, 25 in flat list)
    norm_xs = []
    norm_ys = []
    
    for row in samples:
        # Landmarks are stored as [x0, y0, z0, x1, y1, z1...]
        # Index Tip is landmark #8 -> 8*3 = 24
        x = row[24]
        y = row[25] # Note: In our recorder, we already normalized these!
        norm_xs.append(x)
        norm_ys.append(y)
        
        # Draw the skeleton for the first few samples to visualize shape
        if len(norm_xs) < 5: 
            # Reconstruct (x,y) pairs
            pts = []
            for i in range(21):
                pts.append((row[i*3], row[i*3+1]))
            pts = np.array(pts)
            ax_norm.plot(pts[:, 0], pts[:, 1], 'o-', color='lightgreen', alpha=0.5, markersize=3)

    # Plot the cluster of Index Tips
    ax_norm.scatter(norm_xs, norm_ys, color='green', alpha=0.6, label="Index Tip Cluster")
    
    # Calculate Variance
    var_x = np.var(norm_xs)
    var_y = np.var(norm_ys)
    ax_norm.text(0.05, 0.95, f"Variance X: {var_x:.4f}\nVariance Y: {var_y:.4f}", 
                 transform=ax_norm.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))


    # 2. THE FEATURE DISTRIBUTION (Histogram)
    # Let's verify a specific feature distribution (e.g., Thumb-Middle Distance)
    # If normalized correctly, this should be a sharp Bell Curve (Gaussian).
    
    ax_hist = axes[1]
    ax_hist.set_title("Feature Stability: Thumb-Middle Distance")
    
    distances = []
    for row in samples:
        # Thumb Tip (4) -> indices 12, 13, 14
        t_vec = np.array([row[12], row[13], row[14]])
        # Middle Base (9) -> indices 27, 28, 29
        m_vec = np.array([row[27], row[28], row[29]])
        
        d = np.linalg.norm(t_vec - m_vec)
        distances.append(d)
        
    ax_hist.hist(distances, bins=30, color='skyblue', edgecolor='black')
    ax_hist.set_xlabel("Normalized Distance")
    ax_hist.set_ylabel("Frequency")
    
    # Stats
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    ax_hist.axvline(mean_dist, color='red', linestyle='--', linewidth=2, label=f"Mean: {mean_dist:.2f}")
    ax_hist.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    study_normalization()