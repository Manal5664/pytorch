# PyTorch Breast Cancer Training Pipeline

This repository contains a complete **PyTorch training pipeline** for breast cancer classification.  
It demonstrates how to create a **custom dataset**, **data loader**, and **training loop** using PyTorch.

---

##  Project Overview

This project implements an end-to-end pipeline for a **supervised learning task** using the Breast Cancer dataset.  
The workflow includes:
- Data preprocessing  
- Encoding categorical values  
- Creating PyTorch tensors  
- Custom Dataset and DataLoader  
- Model definition  
- Training and evaluation pipeline  

---

##  Steps Implemented

### 1. Dataset Preparation
- Used the **Breast Cancer dataset**.
- Dropped unnecessary columns:  
  `Unnamed` and `ID`.
- Encoded labels using **LabelEncoder**.
- Converted NumPy arrays to **PyTorch tensors**.

### 2. Custom Dataset and DataLoader
- Implemented a custom dataset class `CustomDataset` with:
  - `__init__` → Initialize data  
  - `__len__` → Return dataset length  
  - `__getitem__` → Return each sample  

- Created:
  - `train_dataset` → CustomDataset  
  - `test_dataset` → CustomDataset  
  - `train_loader` → DataLoader(train_dataset)  
  - `test_loader` → DataLoader(test_dataset)

### 3. Model Definition
- Defined a simple **Neural Network** using `torch.nn.Module`.
- Included essential parameters such as:
  - Input layer  
  - Hidden layers  
  - Output layer  

### 4. Optimization and Loss
- Optimizer: **SGD** (`torch.optim.SGD`)
- Loss Function: **BinaryCrossEntropyLoss**

### 5. Training Pipeline
- Two nested loops:
  - Outer loop → Epochs  
  - Inner loop → Batch features and batch labels  

- Steps:
  1. Forward pass  
  2. Compute loss  
  3. Zero gradients  
  4. Backward propagation  
  5. Update parameters  

### 6. Model Evaluation
- Used `torch.no_grad()` for evaluation to disable gradient computation.  
- Evaluated performance on the test dataset.

---
