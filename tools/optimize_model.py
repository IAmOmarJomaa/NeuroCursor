import tensorflow as tf
import os
import sys

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import PATHS

def convert_to_tflite():
    print("📉 STARTING MODEL COMPRESSION...")
    
    keras_path = str(PATHS["MODELS_DIR"]) + "/neurocursor_model.keras"
    tflite_path = str(PATHS["MODELS_DIR"]) + "/neurocursor_model.tflite"
    
    if not os.path.exists(keras_path):
        print(f"❌ Error: {keras_path} not found.")
        return

    # 1. Load Keras Model
    print("   -> Loading Heavy Keras Model...")
    model = tf.keras.models.load_model(keras_path)
    
    # 2. Convert
    print("   -> Converting to TFLite (CPU Optimized)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimization flags
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # 3. Save
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
        
    print(f"✅ SUCCESS. Optimized model saved to: {tflite_path}")
    print("   -> You can now update gesture_engine.py to use this speed demon.")

if __name__ == "__main__":
    convert_to_tflite()