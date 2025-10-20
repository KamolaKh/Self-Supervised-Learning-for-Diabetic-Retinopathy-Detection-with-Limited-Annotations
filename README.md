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

### File 2: `requirements.txt`
```txt
torch>=1.12.0
torchvision>=0.13.0
opencv-python>=4.6.0
scikit-learn>=1.0.0
pandas>=1.4.0
numpy>=1.21.0
matplotlib>=3.5.0
streamlit>=1.12.0
wandb>=0.13.0
Pillow>=9.0.0
