# Pneumonia Detection using Deep Learning

This project focuses on building a deep learning model to classify chest X-ray images into **NORMAL** and **PNEUMONIA** cases using transfer learning with ResNet18.

---

## 📌 Project Overview

- Built a **custom dataset loader from scratch**
- Applied **image transformations**
- Used **transfer learning** with ResNet18
- Achieved **93% test accuracy**
- Deployed the model using **Gradio** on Hugging Face Spaces

---

## 📂 Dataset Handling

Instead of using built-in PyTorch datasets, a custom dataset class was implemented.

### 🔧 Key Steps:

- Read image paths using `os.listdir`
- Assign labels manually:
  - `NORMAL → 0`
  - `PNEUMONIA → 1`
- Load images using `PIL`
- Return `(image, label)` in `__getitem__`

This approach gives full control over dataset structure and preprocessing.

---

## 🔄 Data transformations

Applied transformations using `torchvision.transforms`:

- Resize images
- Normalize pixel values

---

## 🧠 Model Architecture

- Model: **ResNet18**
- Technique: **Transfer Learning**

### ⚙️ Training Strategy:

- Loaded pretrained weights
- Replaced the final fully connected layer for binary classification
- Fine-tuned the model on the dataset

---

## 📊 Results

| Metric           | Value |
|-----------------|------|
| Training Accuracy | ~95% |
| Test Accuracy     | **93%** |

---

## 🚀 Deployment

The model is deployed using **Gradio** on Hugging Face Spaces.

🔗 **Live Demo:**  
https://huggingface.co/spaces/Abdelrhman82/chest_Xray

