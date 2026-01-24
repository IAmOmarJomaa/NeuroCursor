import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split
import sys
import os

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import PATHS

# CONFIG
INPUT_FILE = str(PATHS["FINAL_FEATURES"])
OUTPUT_PYTHON_FILE = "src/core/gesture_rules.py"

def clean_label(label):
    # Ensure label is a valid python variable name if needed
    return label.upper()

def tree_to_code(tree, feature_names):
    """
    Converts a Scikit-Learn Decision Tree into a Python function string.
    """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    code_lines = []
    code_lines.append("def classify_gesture(features):")
    code_lines.append("    # Unpack features for speed (and readability)")
    
    # Unpack dictionary to local variables
    for f in feature_names:
        code_lines.append(f"    {f} = features['{f}']")
    
    code_lines.append("")

    def recurse(node, depth):
        indent = "    " * (depth + 1)
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            # Write the IF
            code_lines.append(f"{indent}if {name} <= {threshold:.4f}:")
            recurse(tree_.children_left[node], depth + 1)
            
            # Write the ELSE
            code_lines.append(f"{indent}else:  # {name} > {threshold:.4f}")
            recurse(tree_.children_right[node], depth + 1)
        else:
            # Leaf Node (The Answer)
            # Find the class with the highest probability in this leaf
            value = tree_.value[node][0]
            class_idx = np.argmax(value)
            class_name = CLASSES[class_idx]
            confidence = value[class_idx] / np.sum(value)
            
            # We can add a confidence threshold here if we want
            code_lines.append(f"{indent}return '{class_name}'")

    # We need the class names list from the model
    # (Global var hack for the recursive function)
    recurse(0, 0)
    
    # Fallback
    code_lines.append("    return 'UNKNOWN'")
    
    return "\n".join(code_lines)

def generate_rules():
    print("⛏️ STARTING RULE MINER...")
    
    if not os.path.exists(INPUT_FILE):
        print("❌ Feature file missing. Run pipeline/03_feature_extractor.py first.")
        return

    # 1. Load Data
    df = pd.read_csv(INPUT_FILE)
    
    # MAPPING: Map sub-labels to Master Labels (Using the logic we agreed on)
    # Ideally, we should use the same map as before. 
    # For now, let's build a simple map dictionary inside here or load it.
    # Let's auto-generate a map that strips numbers (THE_POINT_1 -> THE_POINT)
    # This keeps it granular enough to be accurate but grouped enough to be useful.
    
    # OPTIONAL: You can customize this map to group things tighter if you want
    # For now, let's train on the SUB-LABELS so the rules are super precise.
    # We will let the Decision Tree find the pattern.
    
    X = df.drop(columns=['label'])
    y = df['label']
    feature_names = X.columns.tolist()
    
    global CLASSES
    CLASSES = sorted(y.unique())

    print(f"   > Mining rules from {len(df)} samples...")
    print(f"   > Features: {feature_names}")

    # 2. Train Decision Tree
    # max_depth=8: Keeps rules readable but accurate.
    # min_samples_leaf=5: Prevents overly specific "glitch" rules.
    clf = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42)
    clf.fit(X, y)
    
    score = clf.score(X, y)
    print(f"   > Rule Accuracy: {score*100:.2f}%")

    # 3. Generate Code
    print("   > Transpiling Tree to Python Code...")
    python_code = tree_to_code(clf, feature_names)
    
    # 4. Save to File
    os.makedirs(os.path.dirname(OUTPUT_PYTHON_FILE), exist_ok=True)
    with open(OUTPUT_PYTHON_FILE, "w") as f:
        f.write("# AUTO-GENERATED RULES BY NEUROCURSOR RULE MINER\n")
        f.write("# DO NOT EDIT MANUALLY UNLESS YOU KNOW WHAT YOU ARE DOING\n\n")
        f.write(python_code)
        
    print(f"✅ SUCCESS! Logic saved to: {OUTPUT_PYTHON_FILE}")
    print("   -> Open that file to see your Golden Formulas.")

if __name__ == "__main__":
    generate_rules()