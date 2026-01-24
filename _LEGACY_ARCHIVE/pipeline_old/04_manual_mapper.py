import pandas as pd
import os
import sys
import json
from pathlib import Path

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import PATHS

# --- CONFIG ---
# We derive the folder from 'DATA_READY' which is guaranteed to exist in your config
PROCESSED_FOLDER = Path(PATHS["DATA_READY"]).parent

INPUT_FILE = str(PATHS["GOLDEN_DATA"])
OUTPUT_FILE = str(PATHS["DATA_READY"])
MAPPING_FILE = str(PROCESSED_FOLDER / "label_mapping.json")

def get_prefix(label):
    parts = str(label).split('_')
    if parts[-1].isdigit():
        return "_".join(parts[:-1])
    return label

def run_mapper():
    print("🔀 MANUAL MAPPER: Define Your Model Classes")
    print(f"   Input: {INPUT_FILE}")
    print(f"   Output: {OUTPUT_FILE}")
    print(f"   Map:   {MAPPING_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        print("❌ Golden Dataset not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    unique_labels = sorted(df['label'].unique())
    
    # 1. Group by Parent
    groups = {}
    for lbl in unique_labels:
        if 'source_label' in df.columns:
            # Use the tracking column if available
            try:
                parent = df[df['label'] == lbl]['source_label'].iloc[0]
            except:
                parent = lbl.split('_')[0]
        else:
            parent = lbl.split('_')[0]
            
        if parent not in groups: groups[parent] = []
        groups[parent].append(lbl)

    # 2. Load existing map
    final_map = {}
    if os.path.exists(MAPPING_FILE):
        try:
            with open(MAPPING_FILE, 'r') as f:
                final_map = json.load(f)
            print("   -> Loaded existing mapping configuration.")
        except:
            print("   -> Existing mapping file corrupt, starting fresh.")

    # 3. Interactive Wizard
    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print(" - Press [ENTER] to merge all variations into the Group Name.")
    print(" - Type a NEW NAME to rename the whole group.")
    print(" - Type 'SPLIT' to map each variation individually.")
    print("="*60)

    sorted_parents = sorted(groups.keys())
    
    for parent in sorted_parents:
        variations = groups[parent]
        
        # Skip if already fully mapped
        if all(v in final_map for v in variations):
            targets = set(final_map[v] for v in variations)
            print(f"✅ {parent.ljust(15)} -> {', '.join(targets)}")
            continue

        print(f"\n📂 GROUP: {parent}")
        print(f"   Variations: {', '.join(variations)}")
        
        print(f"   👉 Target Label [Default: {parent}]: ", end="")
        choice = input().strip()
        
        if choice == "":
            for v in variations:
                final_map[v] = parent
            print(f"      -> All mapped to '{parent}'")
            
        elif choice.upper() == "SPLIT":
            print("      ⚡ Entering Split Mode:")
            for v in variations:
                default_v = get_prefix(v)
                print(f"         Map '{v}' to [{default_v}]: ", end="")
                sub_choice = input().strip()
                if sub_choice == "":
                    final_map[v] = default_v
                else:
                    final_map[v] = sub_choice
                    
        else:
            for v in variations:
                final_map[v] = choice
            print(f"      -> All mapped to '{choice}'")

    # 4. Save & Apply
    print("\n💾 Saving Mapping...")
    with open(MAPPING_FILE, 'w') as f:
        json.dump(final_map, f, indent=4)

    print("🔄 Applying to Dataset...")
    df['label'] = df['label'].map(final_map)
    
    # Save training ready file
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n📊 FINAL CLASS DISTRIBUTION:")
    print(df['label'].value_counts())
    print(f"\n✅ Ready for Training: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_mapper()