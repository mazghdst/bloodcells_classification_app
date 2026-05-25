# Blood Cell Classification — ML and DL 

## 📌 Overview

Interactive Streamlit app presenting a complete pipeline for automatic blood cell classification on the **PBC dataset** (17 095 images, 8 classes, 360×363 px, JPG), comparing two approaches:

- ⚙️ **Machine Learning** — handcrafted feature extraction + classical classifiers
- 🧠 **Deep Learning** — transfer learning with fine-tuning

---

## 🗂️ App Structure

| Section | Description |
|---|---|
| 📊 Data | PBC dataset description, class distribution and image examples |
| ⚙️ Preprocessing | Feature extraction pipeline and image transformations |
| 🧠 Models | Model performances, metrics and interpretability |
| 🔍 Demo | Interactive classification, Grad-CAM and error analysis |
| 💡 Conclusion | Summary, limitations and perspectives |

---

## 🧮 Machine Learning

**Feature extraction:**
- K-Means clustering in LAB color space
- Intra-cluster statistics
- GLCM, Sobel
- Dimensionality reduction

**Models:** SVM · XGBoost · Voting Classifier (SVM + XGBoost, soft vote)

**Best model:** Voting Classifier — Accuracy **98.24%** · F1 macro **98.37%**

5-fold CV: Accuracy **98.10% ± 0.31** · F1 macro **98.23% ± 0.32**

---

## 🧠 Deep Learning

**Method:** Transfer learning + fine-tuning

**Architectures:** EfficientNetV2S · EfficientNetV2M · DenseNet121 · ResNet50V2 · VGG19 · Xception

**Best models:** Ensemble (EfficientNetV2S + VGG19 + Xception) — Accuracy **99.18%** · F1 macro **99.20%**

---

## ⚖️ Comparison

| | ML (CPU) | DL (GPU) |
|---|---|---|
| Training | ~1 min | 1h 30 – 3h |
| Inference / image | 4.2–4.6 ms | ~16–25 ms (batch 32) |
| Best accuracy | 98.24% | 99.18% |

---

## 🚀 Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 👤 Auteur

Océane Saincir