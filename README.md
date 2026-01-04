## 📘 README.md

# 🚢 CNN Model–Based Ship Detection from High-Resolution Aerial Images

This project presents a **deep learning–based ship detection system** using **YOLOv8**, designed to detect ships and vessels from **high-resolution aerial and satellite images**.  
The system supports **model training**, **confidence-aware inference**, and an **interactive Streamlit web application** for visualization and analysis.

---

## 📌 Project Overview

Maritime monitoring and surveillance require accurate detection of ships from aerial imagery.  
This project leverages **Convolutional Neural Networks (CNNs)** through the YOLOv8 architecture to automatically detect ships and provide **confidence-based predictions**.

The final system allows users to:
- Upload an aerial image
- Detect ships with bounding boxes
- View per-ship confidence scores
- Analyze detection confidence using a dynamic bar graph
- Visualize detection heatmaps

---

## 🧠 Model & Technology Stack

- **Model:** YOLOv8 (Ultralytics)
- **Architecture:** CNN-based object detection
- **Programming Language:** Python
- **Frameworks & Libraries:**
  - Ultralytics YOLOv8
  - OpenCV
  - Streamlit
  - NumPy
  - Pandas
  - Altair

---

## 📊 Dataset Description

### Dataset Type
- **High-resolution aerial / satellite images**
- Images contain ships and vessels captured from a top-down perspective

### Annotation Format
- YOLO format (`.txt` labels)
- Each label contains:
```

class_id  x_center  y_center  width  height

```

### Dataset Structure
```

ships-aerial-images/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml

````

### data.yaml (example)
```yaml
train: ships-aerial-images/train/images
val: ships-aerial-images/valid/images
test: ships-aerial-images/test/images

nc: 1
names: ["ship"]
````

> ⚠️ **Note:**
> The dataset is **not included in this repository** due to size constraints.
> Users must download or prepare the dataset locally and update `data.yaml` accordingly.

---

## 🏋️ Model Training

The YOLOv8 model was trained using the Ultralytics framework.

### Training Script

```bash
python yolov8_train.py
```

### Example CLI Training Command

```bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
```

### Output

* Best-performing model weights are saved as:

  ```
  best.pt
  ```

---

## 🌐 Streamlit Web Application

An interactive **Streamlit-based UI** is provided for inference and analysis.

### Key Features

* Image upload for inference
* Adjustable confidence threshold
* Bounding box visualization with confidence (%)
* **Dynamic bar graph** (Ship vs Confidence)
* Average and maximum confidence metrics
* Heatmap visualization of detections
* Downloadable detection result image

### Run the App

```bash
streamlit run app.py
```

---

## 📈 Confidence Visualization

* Each detected ship is assigned a confidence score (probability)
* Confidence values are:

  * Internally handled as probabilities (0–1)
  * Displayed as percentages (0–100%) for clarity
* A **bar graph** dynamically shows:

  * Ship 1, Ship 2, … vs Confidence (%)

This allows easy comparison between detections.

---

## 📂 Repository Structure

```
ship-detection-yolov8/
├── app.py                 # Streamlit inference application
├── yolov8_train.py        # Model training script
├── data.yaml              # Dataset configuration
├── requirements.txt       # Python dependencies
└── README.md
```

> ✅ The trained model (`best.pt`) is included for **direct inference and demo purposes**.

---

## 🛠 Installation & Setup

### 1️⃣ Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the application

```bash
streamlit run app.py
```

---

## 📌 Applications

* Maritime surveillance
* Coastal and port monitoring
* Defense and security analysis
* Remote sensing research
* Academic and educational projects

---

## 👨‍💻 Author

**Gangereddy Mayur**
Final Year Graduation Project
Domain: Computer Vision | Deep Learning | Remote Sensing

---
