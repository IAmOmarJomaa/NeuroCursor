import pandas as pd
import sys
import os

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import PATHS

def inspect_data():
    input_file = PATHS["GOLDEN_DATA"]
    
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return

    print(f"📂 Reading: {input_file}...")
    df = pd.read_csv(input_file)
    
    # Get unique labels sorted alphabetically
    labels = sorted(df['label'].unique())
    
    print(f"\n🔎 FOUND {len(labels)} UNIQUE LABELS. SHOWING 3 SAMPLES EACH:\n")
    print("="*80)

    for label in labels:
        # Filter for this label
        subset = df[df['label'] == label].head(3)
        
        print(f"🏷️  LABEL: {label} (Total Rows: {len(df[df['label'] == label])})")
        
        # Print the rows nicely
        # We drop the label column in the view since we know what it is
        print(subset.drop(columns=['label']).to_string(index=False))
        print("-" * 80)

if __name__ == "__main__":
    inspect_data()