import pandas as pd
import os
import sys
import numpy as np

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import PATHS

def run_audit():
    print("🕵️  NEURO-CURSOR DATA AUDIT")
    print("="*60)
    
    # 1. Load Files
    raw_path = str(PATHS["RAW_DATA"])
    golden_path = str(PATHS["GOLDEN_DATA"])
    
    if not os.path.exists(raw_path):
        print(f"❌ CRITICAL: Raw data not found at {raw_path}")
        return
    if not os.path.exists(golden_path):
        print(f"❌ CRITICAL: Golden data not found at {golden_path}")
        return

    print(f"📂 Loading Raw:    {raw_path}")
    print(f"📂 Loading Golden: {golden_path}")
    
    try:
        raw_df = pd.read_csv(raw_path)
        gold_df = pd.read_csv(golden_path)
    except Exception as e:
        print(f"❌ Error reading CSVs: {e}")
        return

    print(f"   -> Raw Samples:    {len(raw_df)}")
    print(f"   -> Golden Samples: {len(gold_df)}")
    print("-" * 60)

    # 2. Analyze Mismatches
    raw_labels = sorted(raw_df['label'].unique())
    
    print(f"{'SOURCE LABEL'.ljust(20)} | {'RAW'.rjust(6)} | {'GOLDEN'.rjust(6)} | {'LOSS %'.rjust(7)} | {'SUB-LABELS'}")
    print("-" * 80)
    
    # Ensure source_label column exists (Backwards compatibility)
    if 'source_label' not in gold_df.columns:
        print("⚠️  'source_label' column missing in Golden Data. Audit may be inaccurate.")
        # Attempt minimal fallback or just rely on partial matching?
        # Let's assume the refinery added it as per your script.
    
    for lbl in raw_labels:
        # Count Raw
        n_raw = len(raw_df[raw_df['label'] == lbl])
        
        # Count Golden (matched by source_label)
        if 'source_label' in gold_df.columns:
            gold_subset = gold_df[gold_df['source_label'] == lbl]
        else:
            # Fallback: Find labels starting with the name (Less accurate)
            gold_subset = gold_df[gold_df['label'].str.startswith(lbl)]
            
        n_gold = len(gold_subset)
        
        # Calculate Loss (How much data did you throw away?)
        loss_pct = 0.0
        if n_raw > 0:
            loss_pct = ((n_raw - n_gold) / n_raw) * 100
        
        # Status Logic
        loss_str = f"{loss_pct:.1f}%"
        if loss_pct > 50: loss_str = f"❗{loss_str}" # High data loss warning
        elif loss_pct < 0: loss_str = f"❓{loss_str}" # Golden has MORE than raw? (Duplicate issue)
        
        # Sub-labels found
        subs = gold_subset['label'].unique() if not gold_subset.empty else ["MISSING"]
        sub_str = ", ".join(subs)
        if len(sub_str) > 30: sub_str = sub_str[:27] + "..."
        
        print(f"{lbl.ljust(20)} | {str(n_raw).rjust(6)} | {str(n_gold).rjust(6)} | {loss_str.rjust(7)} | {sub_str}")

    print("-" * 80)
    
    # 3. Check for Orphans (Golden labels with no Raw parent)
    if 'source_label' in gold_df.columns:
        orphans = gold_df[~gold_df['source_label'].isin(raw_labels)]['source_label'].unique()
        if len(orphans) > 0:
            print("\n👻 GHOST DATA DETECTED (Exists in Golden, but deleted from Raw):")
            for o in orphans:
                print(f"   - {o}")
                
    print("\n✅ Audit Complete.")

if __name__ == "__main__":
    run_audit()