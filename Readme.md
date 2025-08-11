# 🗣️ Speaker State Detection

A simple pipeline to detect whether a person is **Speaking** or **Not Speaking** from an image.  
Supports both **full Keras model** and **lightweight TensorFlow Lite model** for faster inference.

---

## 📦 Requirements

Install dependencies using **Pipenv**:

```bash
pip install pipenv
pipenv shell
pipenv install
```

---

## 📂 Project Workflow

### 1️⃣ Image Conversion & Preprocessing
Prepare images for model training.

```bash
python imageConvert.py
```
- Takes images from your input folder
- Processes & saves them into `output/` folder
- You will see **two subfolders** inside `output/` for labeled data

---

### 2️⃣ Model Training
Train the **Speaking Detection Model**.

```bash
python modelGenerate.py
```
- After training, you will get:
  ```
  speaking_detection_model.keras
  ```

---

### 3️⃣ Convert Heavy Model → Lite Model (Optional)
Make the model lightweight for faster execution.

```bash
python heavy_to_light_model.py
```
- Outputs:
  ```
  speaking_detection_model.tflite
  ```

---

### 4️⃣ Run Inference

#### Using **Keras** model:
```bash
python run.py
```

#### Using **TensorFlow Lite** model:
```bash
python run_lite.py
```

---

## 📁 Folder Structure

```
project/
│── imageConvert.py         # Preprocess and prepare dataset
│── modelGenerate.py        # Train the full model
│── heavy_to_light_model.py # Convert to TensorFlow Lite
│── run.py                  # Test using .keras model
│── run_lite.py             # Test using .tflite model
│── output/                 # Processed images
│── speaking_detection_model.keras
│── speaking_detection_model.tflite
```

---

## ⚡ Performance
| Model Type      | Size      | Speed        | Use Case |
|-----------------|-----------|--------------|----------|
| `.keras`        | Large     | Slower       | High accuracy offline |
| `.tflite`       | Small     | Very Fast    | Edge devices, mobile apps |

---

## 📝 Notes
- Ensure your dataset has both **Speaking** and **Not Speaking** images.
- For best results, use clear, face-visible images.
