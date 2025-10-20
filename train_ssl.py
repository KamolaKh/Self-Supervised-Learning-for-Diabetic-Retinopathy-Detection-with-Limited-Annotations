"""
Main training script for self-supervised DR detection
"""

import torch
import torch.nn as nn
from models.moco import MoCo
from data.loader import get_dr_dataloaders
from config import Config

def main():
    config = Config()
    
    # Get data loaders
    ssl_loader, train_loader, test_loader = get_dr_dataloaders(config)
    
    # Initialize model
    model = MoCo(config)
    
    # Phase 1: Self-supervised pretraining
    print("Starting SSL pretraining...")
    model.pretrain(ssl_loader)
    
    # Phase 2: Supervised fine-tuning
    print("Starting fine-tuning...")
    model.finetune(train_loader, test_loader)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
