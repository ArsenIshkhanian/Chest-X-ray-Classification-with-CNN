# 🩺 Chest X-ray Classification with CNN  

A deep learning project to classify chest X-ray images into **Pneumonia** or **Normal** categories using **Convolutional Neural Networks (CNNs)** in TensorFlow/Keras.  

This project was inspired by a Kaggle competition and achieved promising accuracy on the test set.

---

## 📦 Dataset

I used the **Chest X-ray Images for Pneumonia Detection with Deep Learning** dataset, created by Tolga Dincer. It contains:

- **5,856** validated chest X-ray images
- Organized into **train** and **test** folders
- Two categories: **Normal** vs **Pneumonia**

**Note:**  
For this project, I used **only the training data** to train my model.  
For quick testing, I selected a **single random image** instead of using the full test set.

**Download it here:**  
[Chest X-ray Images for Pneumonia Detection with Deep Learning – Kaggle](https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images)

---

## 🚀 Features
- Preprocessing: Grayscale conversion, resizing to **128×128**, normalization  
- CNN Architecture with multiple convolutional & dense layers  
- **Binary Classification** with `sigmoid` activation  
- Training & validation accuracy tracking  

---

## 🧠 Model Architecture

Input (128x128x1)
→ Conv2D(32) + ReLU → MaxPooling
→ Conv2D(64) + ReLU → MaxPooling
→ Flatten
→ Dense(128) → Dropout(0.5)
→ Dense(64) → Dropout(0.4)
→ Dense(32) → Dropout(0.3)
→ Dense(16)
→ Dense(1) → Sigmoid

---

**Loss Function:** Binary Crossentropy  
**Optimizer:** Adam (lr=0.001)  
**Metrics:** Accuracy  
