import tensorflow as tf
from tensorflow import keras

# Load the trained model
model = keras.models.load_model("speaking_detection_model.keras")

# Convert to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optimize for size & speed
converter.target_spec.supported_types = [tf.float16]  # Use FP16 for faster inference

tflite_model = converter.convert()

# Save the optimized TFLite model
with open("speaking_detection_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite Model Saved: speaking_detection_model.tflite")
