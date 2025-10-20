# Diabetic Retinopathy Detection with Self-Supervised Learning

**Team:** [Kamola Kholmirzaeva-220713], [Aziza Ergasheva-220903]  
**Course:** Computer Vision  
**Duration:** 8 Weeks

## Project Overview
Implement self-supervised learning for diabetic retinopathy detection to reduce labeled data requirements.

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Download APTOS 2019 dataset and place in data/ folder
# Run training
python train_ssl.py
dr-detection/
├── models/          # Model architectures
├── data/           # Data loading and preprocessing
├── training/       # Training scripts
├── evaluation/     # Evaluation metrics
├── demo/          # Web interface
└── configs/       # Configuration files
