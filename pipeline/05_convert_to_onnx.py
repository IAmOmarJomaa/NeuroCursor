import os
import tensorflow as tf
import tf2onnx
import onnx
from src.config import PATHS

def convert():
    keras_path = str(PATHS["MODELS_DIR"] / "neurocursor_model.keras")
    onnx_path = str(PATHS["MODELS_DIR"] / "neurocursor_model.onnx")

    if not os.path.exists(keras_path):
        print(f"❌ Keras model not found at: {keras_path}")
        print("   Run '04_train_model.py' first!")
        return

    print(f"🔄 Loading Keras model from: {keras_path}")
    model = tf.keras.models.load_model(keras_path)

    # Convert to ONNX
    # opset 13 is a safe default for modern runtimes
    print("⚙️ Converting to ONNX...")
    spec = (tf.TensorSpec((None, 63), tf.float32, name="input"),)
    output_path = onnx_path

    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    
    # Save
    with open(output_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    
    print(f"✅ ONNX Model saved to: {output_path}")
    print("   You can now run 'python run.py'!")

if __name__ == "__main__":
    convert()