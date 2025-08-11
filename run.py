import os
import cv2
import time
import numpy as np
import psutil
import tensorflow as tf
from tensorflow import keras

# Enable XLA (Accelerated Linear Algebra) for performance boost
tf.config.optimizer.set_jit(True)

# Load the trained model once globally
model = keras.models.load_model("speaking_detection_model.keras")

# Get model input shape
input_shape = model.input_shape
if len(input_shape) == 4:
    IMG_HEIGHT, IMG_WIDTH = input_shape[1], input_shape[2]
    IS_FLATTENED = False
elif len(input_shape) == 2:
    IMG_HEIGHT, IMG_WIDTH = int(np.sqrt(input_shape[1] // 3)), int(np.sqrt(input_shape[1] // 3))
    IS_FLATTENED = True
else:
    raise ValueError("Unsupported model input shape!")

# Convert model to TensorFlow function for speed
@tf.function
def predict_speaking_tensor(img):
    return model(img, training=False)

def detect_speaking(image_path: str):
    """
    Detects if a person is speaking in the given image.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        dict: Prediction result and performance stats.
    """
    process = psutil.Process(os.getpid())

    # Start measuring performance
    start_time = time.time()
    cpu_before = process.cpu_percent(interval=None)
    memory_before = process.memory_info().rss / (1024 * 1024)  # MB

    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0

    if IS_FLATTENED:
        img = img.reshape(1, -1)  # (1, features)
    else:
        img = np.expand_dims(img, axis=0)  # (1, H, W, C)

    # Run prediction
    prediction = predict_speaking_tensor(img)

    # Measure after prediction
    cpu_after = process.cpu_percent(interval=0.1)
    memory_after = process.memory_info().rss / (1024 * 1024)  # MB
    end_time = time.time()

    # Format result
    result = "Speaking" if prediction.numpy()[0][0] > 0.5 else "Not Speaking"

    return {
        "prediction": result,
        "execution_time_sec": round(end_time - start_time, 4),
        "memory_used_mb": round(memory_after - memory_before, 2),
        "cpu_used_percent": round(cpu_after - cpu_before, 2)
    }

# Example usage
if __name__ == "__main__":
    file_path = r"C:\Users\bishw\Downloads\test3.jpg"
    output = detect_speaking(file_path)
    print(f"🗣️ Prediction: {output['prediction']}")
    print(f"⏳ Execution Time: {output['execution_time_sec']} sec")
    print(f"💾 Memory Usage: {output['memory_used_mb']} MB")
    print(f"⚙️ CPU Usage: {output['cpu_used_percent']}%")
