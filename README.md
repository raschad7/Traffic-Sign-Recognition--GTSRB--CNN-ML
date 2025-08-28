
# 🚦 Traffic Sign Recognition (GTSRB) with CNN

## 📌 Project Overview

This project implements a **Convolutional Neural Network (CNN)** to classify German Traffic Signs using the **GTSRB dataset**.  
We built the pipeline from scratch, handled data imbalance, trained a CNN, and achieved **~99.9% accuracy** on the test set.

---

## 🛠️ Tech Stack

- **Language:** Python (Google Colab, GPU runtime)
- **Deep Learning:** TensorFlow / Keras
- **Data Handling:** Pandas, NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Image Processing:** OpenCV

---

## 📂 Dataset

- **Source:** [GTSRB German Traffic Sign Dataset (Kaggle)](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- **Classes:** 43 traffic sign categories
- **Images:** ~39k total (varied sizes, avg ~50×50 pixels)

---

## 🔎 Exploratory Data Analysis (EDA)

- **Class Imbalance:** Some classes had >2000 images, others only ~200.
- **Image Sizes:** Avg 50×50 px, aspect ratio ~1.0 → resized all to **64×64**.
- **Pixel Stats:** Mean ≈ 0.33, Std ≈ 0.17 → normalization to `[0,1]` works well.
- **No corrupt images** found.

---

## 📑 Data Preparation

- **Split:** Stratified 80% train / 10% validation / 10% test.
- **Structure:**
```

/gtsrb_folders/
train/0, train/1, … train/42
val/0, val/1, … val/42
test/0, test/1, … test/42

```
- **Preprocessing:**
- Resize → `(64×64)`
- Normalize → `[0,1]`
- One-hot encoded labels

---

## 🧮 Handling Class Imbalance
Computed **class weights** using `sklearn.utils.class_weight`.
- Rare classes given higher weight.
- Used in `model.fit(..., class_weight=class_weight)`.

---

## 🏗️ Model Architecture (Baseline CNN)

```

Input (64×64×3)
├─ Conv2D(32) + BN + ReLU
├─ Conv2D(32) + BN + ReLU
├─ MaxPooling + Dropout(0.25)
├─ Conv2D(64) + BN + ReLU
├─ Conv2D(64) + BN + ReLU
├─ MaxPooling + Dropout(0.25)
├─ Conv2D(128) + BN + ReLU
├─ MaxPooling + Dropout(0.3)
├─ GlobalAveragePooling2D
├─ Dense(256, ReLU) + Dropout(0.4)
└─ Dense(43, Softmax)

````

- **Optimizer:** Adam (lr=1e-3)
- **Loss:** Categorical Crossentropy
- **Metrics:** Accuracy

---

## 🎯 Training Setup
- **Batch size:** 64
- **Epochs:** up to 50 (EarlyStopping stopped ~30)
- **Callbacks:** EarlyStopping, ReduceLROnPlateau
- **GPU:** T4 on Google Colab

---

## 📊 Results
- **Validation Accuracy:** ~99.8%
- **Test Accuracy:** ~99.9%
- **Classification Report:** Precision, Recall, F1 ≈ **1.00** across almost all classes
- **Confusion Matrix:** Nearly perfect diagonal (model almost never confuses signs)

---

## 📈 Visualizations
- Normalized Confusion Matrix
- Per-class Accuracy bar chart



---

## 💾 Model Export
- **Recommended format:**
  ```python
  model.save("gtsrb_cnn_baseline.keras")


- **Legacy format (still works):**

  ```python
  model.save("gtsrb_cnn_baseline.h5")
  ```

---

## 🚀 Future Work

- **Data Augmentation:** rotations, brightness changes, blur for robustness
- **Transfer Learning:** MobileNetV2 for faster convergence + smaller size
- **Deployment:** TFLite (Raspberry Pi / mobile) or ONNX for cross-platform

---

## ✅ Key Takeaways

- CNNs work extremely well for traffic sign recognition.
- Proper preprocessing + class weighting ensures fairness across rare classes.
- Achieved near **state-of-the-art performance** with a custom CNN.

---

## 👨‍💻 Author

Developed as part of an ML internship project.

```

```
