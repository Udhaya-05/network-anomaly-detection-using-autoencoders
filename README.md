# 🚀 Network Anomaly Detection using Autoencoders

## 📌 Overview
This project focuses on detecting anomalies in network traffic using deep learning techniques, specifically **Autoencoders**. The goal is to identify malicious or abnormal network behavior by learning patterns from normal data and flagging deviations.

---

## 🎯 Objectives
- Build an anomaly detection system using deep learning
- Improve detection of rare network attacks
- Reduce false negatives in intrusion detection
- Experiment with hybrid models for better performance

---

## 📂 Dataset
- Used: **NSL-KDD Dataset** (improved version of KDD Cup 1999)
- Reason:
  - Original dataset was too large and caused memory issues
  - NSL-KDD provides a balanced and manageable dataset for experimentation

---

## 🧠 Model Development Process

### 1️⃣ Initial Approach
- Started with two ML model ideas
- Finalized on **Autoencoder-based anomaly detection**

### 2️⃣ Basic Model
- Simple 3-layer autoencoder
- Observed issues:
  - Low recall for anomalies
  - Overfitting

### 3️⃣ Improvements Applied
- Adjusted anomaly detection threshold:
  - 95% → 90% → IQR-based threshold
- Added noise to training data (denoising autoencoder concept)
- Balanced dataset using undersampling

### 4️⃣ Advanced Optimizations
- Increased model depth and number of neurons
- Used advanced activation functions:
  - `LeakyReLU`
  - `SELU`
- Tuned hyperparameters:
  - Batch size
  - Epochs
- Introduced **Isolation Forest** for hybrid anomaly detection

---

## ⚙️ Technologies Used
- Python 🐍
- TensorFlow / Keras
- Scikit-learn
- Pandas & NumPy
- Matplotlib / Seaborn

---

## 📊 Results

| Metric | Initial Model | Final Model |
|------|--------|--------|
| Accuracy | Low | **60.97%** |
| Recall (Anomalies) | 18% | **39%** |
| False Negatives | High | **Reduced significantly** |

✅ Final model shows clear improvement in anomaly detection capability.

---

## 📁 Project Structure
