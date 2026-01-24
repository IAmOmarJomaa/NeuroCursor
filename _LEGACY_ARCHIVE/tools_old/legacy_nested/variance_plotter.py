import json
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

def plot_variance():
    path = ROOT_DIR / "data" / "variance_data.json"
    if not path.exists():
        print("❌ No data found.")
        return

    with open(path, "r") as f:
        data = json.load(f)

    # Features we recorded
    feature_names = {
        "f1": "Thumb Tip -> Middle Base",
        "f2": "Thumb Tip -> Index Base",
        "f3": "Thumb Tip -> Index Tip (Pinch)",
        "f4": "Thumb Tip -> Pinky Base",
        "f5": "Index Tip -> Middle Tip"
    }

    # Setup Plot (1 row per feature)
    fig, axes = plt.subplots(len(feature_names), 1, figsize=(10, 15))
    fig.suptitle("Feature Variance Analysis (The 'Jitter' Test)", fontsize=16)
    
    tasks = ["POINTER", "GEN_Z_HEART", "PINCH"]
    colors = ['lightblue', 'lightgreen', 'salmon']

    for idx, (f_key, f_name) in enumerate(feature_names.items()):
        ax = axes[idx]
        plot_data = []
        
        # Collect data for this feature across all tasks
        for task in tasks:
            # Get the list of numbers recorded
            values = data[task].get(f_key, [])
            plot_data.append(values)

        # Draw Box Plot
        bplot = ax.boxplot(plot_data, patch_artist=True, vert=False, labels=tasks)
        
        # Color code
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_title(f"Feature: {f_name}")
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    plot_variance()