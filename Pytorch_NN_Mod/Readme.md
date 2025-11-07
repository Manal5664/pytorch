# PyTorch Neural Network Training Pipeline — Breast Cancer Classification

This project demonstrates a **complete neural network training pipeline** using **PyTorch** and **Scikit-Learn**, applied to the **Breast Cancer dataset** (from Kaggle Hub).

---

## 🚀 Overview

The main goal of this project is to build and train a simple neural network model using **PyTorch** for binary classification (Breast Cancer detection).  
The project covers **data preprocessing, model design, training, and evaluation** — all in one streamlined pipeline.

---

##  Key Steps

### 1️⃣ Importing Libraries
The following libraries were used:
- **Pandas** — data loading and manipulation  
- **NumPy** — numerical operations  
- **Scikit-learn** — model selection and preprocessing (`StandardScaler`, `LabelEncoder`, `train_test_split`)  
- **Torch** — building and training the neural network  

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
```
---


### 2️⃣ Dataset Information
The dataset used is the **Breast Cancer dataset** imported via the Kaggle API.  
Unnecessary columns such as **“id”** and **“Ummaned32”** were dropped to clean the dataset before model training.---

### 3️⃣ Data Preprocessing
The preprocessing pipeline includes:
- Label encoding for categorical target values  
- Standard scaling for feature normalization  
- Splitting the dataset into training and testing subsets  
- Converting NumPy arrays to PyTorch tensors for model input

---

### 4️⃣ Model Architecture
The neural network is built by defining a custom class that inherits from **`torch.nn.Module`**.  
The model includes multiple layers, activation functions, and a sigmoid output for binary classification.  
The `__init__` and `forward()` functions define the structure and data flow within the network.

---

### 5️⃣ Hyperparameters and Configuration
The following parameters are initialized during training:
- Learning rate  
- Number of epochs  
- Loss function (Binary Cross Entropy Loss)  
- Optimizer (Stochastic Gradient Descent)

---

### 6️⃣ Training Pipeline
The training process follows a standard supervised learning loop:
1. Forward pass  
2. Loss calculation  
3. Clearing previous gradients  
4. Backward pass using Autograd  
5. Parameter update via optimizer step  

This pipeline ensures efficient weight adjustment and model learning over multiple epochs.

---

## 💡 Key Concepts Demonstrated
- Data preprocessing using Scikit-learn  
- Conversion of NumPy arrays to PyTorch tensors  
- Neural network design using `torch.nn.Module`  
- Loss and optimizer configuration  
- Gradient computation using Autograd  
- Full model training workflow

---

## 🧑‍💻 Author
**Manal Atif**  
MS Data Science — Muhammad Ali Jinnah University (MAJU)  
Focused on Machine Learning, Deep Learning, and Data Efficiency in Urdu Corpus Construction.  

---
