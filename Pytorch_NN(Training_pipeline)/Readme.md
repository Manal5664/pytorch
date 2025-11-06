##  PyTorch Neural Network Training Pipeline (Framework)

This repository contains my **PyTorch practice notebook** created in **Google Colab**, focused on building and understanding a **Neural Network Training Pipeline** from scratch using **PyTorch**.

---

### 📘 Overview

In this notebook, I implemented a complete neural network workflow starting from **data preprocessing** to **model training**.  
The project uses the **Breast Cancer dataset** to demonstrate how to build and train a simple feed-forward neural network using PyTorch.

---

### 🧩 Steps Covered

#### 🔹 1. Data Loading & Preprocessing
- Loaded the **Breast Cancer dataset** into a **Pandas DataFrame**  
- Scaled the numerical features using **StandardScaler** for normalization  
- Encoded the categorical **diagnosis** column using **LabelEncoder**  
- Converted **NumPy arrays** into **PyTorch tensors** for model compatibility  

#### 🔹 2. Model Definition
- Defined a **custom neural network class** `MySimpleNN` using PyTorch’s `nn.Module`  
- Implemented:
  - `__init__()` — to initialize layers and activation functions  
  - `forward()` — to define the forward pass  
  - `loss_function()` — to calculate the model’s loss  

#### 🔹 3. Training Pipeline
The training pipeline includes:
- Model creation  
- Defining the loss function and optimizer  
- Looping through **epochs**  
- **Forward pass** → **Loss calculation** → **Backward pass (backpropagation)**  
- Updating model parameters using `optimizer.step()`  
- Clearing gradients with `optimizer.zero_grad()`  

During training, I used `torch.no_grad()` to disable gradient tracking when evaluating the model, and ensured gradients were reset using `optimizer.zero_grad()` to avoid accumulation.

---

### ⚙️ Environment

- **Platform:** Google Colab  
- **Framework:** PyTorch  
- **Libraries:** Pandas, NumPy, Scikit-learn,torch
- **Language:** Python  

---

### 🎯 Key Learnings

Through this project, I learned:
- How to convert NumPy data into PyTorch tensors  
- How to define a neural network from scratch  
- How the **forward**, **backward**, and **parameter update** steps work in PyTorch  
- How to build a **training loop** manually  

---

### 🧩 Dataset Information

- **Dataset:** Breast Cancer Wisconsin (Diagnostic)  
- **Rows:** 569  
- **Columns:** 33  
- **Target:** `diagnosis` (Malignant or Benign)  
- **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

---

### 🚀 How to Run

1. Open the notebook in Google Colab  
2. Upload the dataset (Excel or CSV)  
3. Run all cells step-by-step  
4. Observe training loss and accuracy per epoch  

---

### 💡 Conclusion

This notebook helped me understand the **core working of neural network training** in PyTorch — from data handling to gradient computation and parameter updates — without relying on high-level APIs. It demonstrates the **foundation of deep learning model training**.
