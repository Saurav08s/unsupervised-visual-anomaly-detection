# Unsupervised Visual Anomaly Detection for Industrial Inspection (PatchCore)

## 📌 Overview
This project implements an **unsupervised visual anomaly detection system** for industrial inspection under **severe label scarcity**.  
Instead of relying on labeled defect samples, the system learns representations of **normal visual patterns** and flags anomalies as deviations in deep feature space.

The approach is inspired by **PatchCore**, making it suitable for real-world manufacturing environments where defect types are rare, diverse, and constantly evolving.

---

## ❓ Problem Motivation
In industrial quality inspection:
- Defect samples are **rare or unavailable**
- Defect types **change frequently**
- Supervised models require costly and repetitive labeling

**Key question:**  
> *How can we detect unseen defects when no labeled defect data is available?*

This project addresses that question by **modeling normality instead of defects**.

---

## 🧠 Core Idea
- Learn what **normal images look like** in deep feature space
- Treat defects as **outliers** in that space
- Perform detection **without training a classifier**

This reframes defect detection as a **one-class / unsupervised learning problem**.

---

## 🏗️ Methodology

### 1️⃣ Feature Extraction
- Used a **pretrained ResNet-50** backbone
- Extracted intermediate convolutional feature maps (spatial tensors)
- Backbone weights were **frozen** to prevent overfitting

### 2️⃣ Patch-Level Representation
- Feature maps were converted into **patch embeddings**
- Each patch represents local visual structure
- Preserving spatial information enables localization

### 3️⃣ Memory Bank Construction
- Patch embeddings from **normal images only** were stored
- This memory bank represents the distribution of normality

### 4️⃣ Memory Optimization (Coreset)
- Applied **coreset sampling** to reduce redundancy
- Used **approximate farthest-point selection**
- Reduced memory and computation by **~40×**

### 5️⃣ Anomaly Scoring
- Test patches were compared to the normal memory bank
- **Nearest-neighbor distance** used as anomaly score
- Image score = maximum patch anomaly score

### 6️⃣ Localization
- Patch-level scores were upsampled
- Generated **pixel-level anomaly heatmaps**
- Enables explainable defect localization

---

## 📊 Dataset
- **VisA (Visual Anomaly Detection Dataset)**
- Category used: **Candle**
- Training: Normal images only
- Testing: Normal + anomalous images
- Dataset is **not included** due to licensing and size constraints

---

## 📈 Results

| Metric | Value |
|------|------|
| ROC-AUC | **0.87** |
| Training images | ~1,000 |
| Test images | ~100 |
| Memory reduction | **~40×** |

- Clear separation between normal and anomalous samples
- Effective localization of defect regions
- Stable performance under label scarcity

---

## 🔍 Failure Case Analysis
Despite strong performance, some limitations were observed:

### ❌ Small / Subtle Defects
- Very small defects may be weakly localized
- Caused by downsampling to **14×14** feature resolution
- Fine-grained details can be averaged out

### ❌ Illumination Variations
- Lighting changes shift feature distributions
- Can lead to increased false positives

These failure cases were explicitly analyzed and documented to ensure realistic evaluation.

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Deep Learning:** PyTorch, Torchvision  
- **Computer Vision:** CNNs, ResNet-50  
- **Anomaly Detection:** PatchCore (patch-based nearest-neighbor)  
- **Evaluation:** NumPy, scikit-learn  
- **Visualization:** Matplotlib  
- **Environment:** Google Colab  
- **Dataset:** VisA  

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
Download the VisA dataset

Update dataset paths in the notebook

Run:

notebooks/anomaly_detection_patchcore.ipynb
