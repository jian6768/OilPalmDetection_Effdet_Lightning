# Oil Palm Fresh Fruit Bunch (FFB) Maturity Detection  
Using EfficientDet (Ross Wightman's `effdet`)

## 📌 Overview
This project builds an object detection model to detect **oil palm fresh fruit bunches (FFB)** and classify their **maturity level** using a modified EfficientDet architecture implemented in [Ross Wightman's `effdet`](https://github.com/rwightman/efficientdet-pytorch).

Maturity detection of oil palm fruit is essential for:
- Optimizing harvest timing  
- Improving yield quality  
- Reducing losses from overripe or underripe harvesting  
- Automating palm plantation operations

This repository contains:
- Data preprocessing scripts  
- PyTorch + EffDet training pipeline  
- Evaluation and visualization tools  
- A demo notebook for inference  

---

## 🧠 Model Architecture
This project uses:

**EfficientDet (D0–D7 variants available)**  
via the `effdet` PyTorch implementation by Ross Wightman.

Features:
- BiFPN feature pyramid  
- EfficientNet backbone  
- Weighted feature fusion  
- RetinaNet-style detection head  
- High detection speed + good accuracy for real-time applications  

The final model outputs:
- Bounding boxes around oil palm fruit  
- Predicted maturity levels (e.g., unripe, ripe, overripe)  
- Confidence scores  

---

## 📁 Dataset Structure
You should create a dataset folder like:
datasets/
├── train/
│ ├── images/
│ └── labels/
├── val/
│ ├── images/
│ └── labels/

## 🛠️ Installation

### 1. Clone this repo
```bash
git clone https://github.com/jian6768/OilPalmDetection_Effdet_Lightning.git
cd <your-repo>

### 2. Create Virtual Environment
python3 -m venv .venv
source .venv/bin/activate   # Mac/Linux

### 3. Install Dependencies
pip install -r requirements.txt


## 📁 Acknowledgements
This project uses the EfficientDet PyTorch implementation by Ross Wightman:
👉 https://github.com/rwightman/efficientdet-pytorch

Licensed under Apache 2.0.

We gratefully acknowledge Ross Wightman's contributions to the open-source vision community.

