import os
import cv2
import time
import numpy as np
import psutil
import tensorflow.lite as tflite


def detect_speaking(image_path: str, model_path: str = "speaking_detection_model.tflite"):
    """
    Detects whether the person in the image is speaking using a TFLite model.

    Args:
        image_path (str): Path to the image file.
        model_path (str): Path to the TFLite model file.

    Returns:
        dict: Results containing prediction, execution time, memory usage, and CPU usage.
    """
    # Initialize process monitoring
    process = psutil.Process(os.getpid())

    # Load the TFLite model
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input & output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Start measuring execution time, CPU & memory
    start_time = time.time()
    cpu_before = process.cpu_percent(interval=None)
    memory_before = process.memory_info().rss / (1024 * 1024)  # MB

    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Get expected input size from model
    input_shape = input_details[0]['shape']
    target_height, target_width = input_shape[1], input_shape[2]

    img = cv2.resize(img, (target_width, target_height))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run inference
    interpreter.invoke()

    # Get prediction result
    prediction = interpreter.get_tensor(output_details[0]['index'])

    # Measure final CPU & memory usage
    cpu_after = process.cpu_percent(interval=0.05)  # More accurate CPU usage
    memory_after = process.memory_info().rss / (1024 * 1024)  # MB
    end_time = time.time()

    # Prepare result
    result_label = "Speaking" if prediction[0][0] > 0.5 else "Not Speaking"
    results = {
        "prediction": result_label,
        "execution_time_sec": round(end_time - start_time, 4),
        "memory_usage_mb": round(memory_after - memory_before, 2),
        "cpu_usage_percent": round(cpu_after - cpu_before, 2)
    }

    return results


# Example usage
if __name__ == "__main__":
    output = detect_speaking(r"C:\Users\bishw\Downloads\test3.jpg")
    print(f"🗣️ Prediction: {output['prediction']}")
    print(f"⏳ Execution Time: {output['execution_time_sec']} sec")
    print(f"💾 Memory Usage: {output['memory_usage_mb']} MB")
    print(f"⚙️ CPU Usage: {output['cpu_usage_percent']}%")
