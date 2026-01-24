"""
NeuroCursor Model Trainer (The Academy).

This script trains the TFLite model using the refined dataset.
It applies Data Augmentation to simulate real-world jitter and rotation.

Outputs:
    1. neurocursor_model.keras (Resume training)
    2. neurocursor_model.tflite (Float32 for Runtime)
    3. label_map.pkl (Label decoder)
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import pickle
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import PATHS
from src.hand_utils import pre_process_landmark

INPUT_FILE = str(PATHS["DATA_READY"])
# [CRITICAL] Convert Path object to string for safe OS operations
MODEL_PATH = str(PATHS["MODELS_DIR"])

def augment_data(X, y):
    """
    Artificial Intelligence Steroids:
    Creates variations of your data to make the model robust against
    rotation, scale changes, and sensor noise.
    """
    print("   🧪 Applying Data Augmentation (Noise, Rotation, Scale)...")
    X_aug, y_aug = [], []
    
    # Rotation Matrix Helper
    def rotate(coords, angle_deg):
        theta = np.radians(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1))) # Rotate around Z
        return np.dot(coords, R)

    for i, row in enumerate(X):
        # Reshape to (21, 3) for geometric manipulation
        skeleton = row.reshape(21, 3)
        label = y[i]
        
        # 1. Original
        X_aug.append(row)
        y_aug.append(label)
        
        # 2. Add Noise (Jitter Simulation)
        noise = np.random.normal(0, 0.005, skeleton.shape) # +/- 5mm approx
        X_aug.append((skeleton + noise).flatten())
        y_aug.append(label)
        
        # 3. Rotate Left (-10 deg)
        X_aug.append(rotate(skeleton, -10).flatten())
        y_aug.append(label)
        
        # 4. Rotate Right (+10 deg)
        X_aug.append(rotate(skeleton, 10).flatten())
        y_aug.append(label)
        
        # 5. Scale Up (Different Hand Sizes)
        X_aug.append((skeleton * 1.1).flatten())
        y_aug.append(label)

    return np.array(X_aug), np.array(y_aug)

def train_neural_network():
    print("🧠 STARTING ACCURACY-FIRST TRAINING (V3)...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        print("   Run '03_smart_merger.py' to generate training data.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"   -> Raw Samples: {len(df)}")

    # 1. Normalization (Using Master Hand Utils)
    print("   -> Normalizing Data...")
    feature_cols = [c for c in df.columns if c.startswith(('x', 'y', 'z'))]
    X_raw = df[feature_cols].values
    
    X_processed = []
    for row in X_raw:
        # Calls src.hand_utils.pre_process_landmark
        X_processed.append(pre_process_landmark(row))
    X = np.array(X_processed)
    y_raw = df['label'].values

    # 2. Encode Labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    
    # Save Label Map for Runtime
    label_map = dict(zip(le.classes_, le.transform(le.classes_)))
    inv_map = {v: k for k, v in label_map.items()}
    
    with open(os.path.join(MODEL_PATH, "label_map.pkl"), 'wb') as f:
        pickle.dump(inv_map, f)

    # 3. Split & Augment
    # Stratify ensures we have equal representation of rare gestures in the test set
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    # Augment ONLY Training data (Keep test data pure/real)
    X_train, y_train = augment_data(X_train_raw, y_train_raw)
    print(f"   -> Training Samples after Augmentation: {len(X_train)}")

    # 4. Class Weights (Handle Imbalance)
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    weights_dict = dict(enumerate(class_weights))

    # 5. Architecture (Deep & Regularized)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(63,)),
        
        # Dense Block 1
        tf.keras.layers.Dense(256),
        tf.keras.layers.BatchNormalization(), # Stabilizes learning
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.3),         # Prevents overfitting
        
        # Dense Block 2
        tf.keras.layers.Dense(128),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.3),
        
        # Dense Block 3 (Refinement)
        tf.keras.layers.Dense(64, activation='relu'),
        
        # Output
        tf.keras.layers.Dense(len(le.classes_), activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    # 6. Train
    print("\n⚡ Training...")
    history = model.fit(
        X_train, y_train, 
        epochs=60, 
        batch_size=64, # Larger batch for stable gradients
        validation_data=(X_test, y_test), 
        class_weight=weights_dict,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ],
        verbose=1
    )

    # 7. Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n🏆 TEST SET ACCURACY: {acc*100:.2f}%")

    # 8. Export (Dual Format)
    print("\n💾 EXPORTING MODELS...")
    
    # A. Keras (Backup)
    keras_path = os.path.join(MODEL_PATH, "neurocursor_model.keras")
    model.save(keras_path)
    print(f"   -> Saved Keras: {keras_path}")
    
    # B. TFLite (FLOAT32 - High Accuracy, No Quantization)
    tflite_path = os.path.join(MODEL_PATH, "neurocursor_model.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Note: We do NOT set optimizations. This keeps it Float32.
    tflite_model = converter.convert()
    
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"   -> Saved TFLite (Float32): {tflite_path}")
    print("      (This model is fast AND accurate. Use this one.)")

    # Diagnostics
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\n" + classification_report(y_test, y_pred, target_names=le.classes_))
    
    try:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f"Accuracy: {acc*100:.2f}%")
        plt.show()
    except: pass

if __name__ == "__main__":
    train_neural_network()