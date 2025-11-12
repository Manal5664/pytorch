# Fashion MNIST Classification using PyTorch ANN (Overfitting Reduction)

## Overview
This project implements an **Artificial Neural Network (ANN)** using **PyTorch** for the **Fashion MNIST** dataset.  
It focuses on reducing **overfitting** using three key techniques:
- **Batch Normalization**
- **Dropout**
- **Weight Decay (L2 Regularization)**  

The model is trained and evaluated on **GPU**, achieving around **83% accuracy** with improved stability.

---

## Workflow

### 1. Libraries Used
- `torch`, `torch.nn`, `torch.optim`, `torch.utils.data`
- `torchvision.datasets` and `transforms`
- `numpy`
- **Kaggle API** for dataset download

---

### 2. Dataset
- Dataset: **Fashion MNIST** (10 clothing categories, 28x28 grayscale images)
- Downloaded via **Kaggle API**
- Custom dataset class implemented with:
  - `__init__`, `__len__`, `__getitem__`
- Train/Test split handled using **DataLoader**

---

### 3. Model Architecture
- **Input layer → Linear (128) → BatchNorm → ReLU → Dropout**
- **Linear (64) → BatchNorm → ReLU → Dropout**
- **Output layer (10 classes)**
- Defined using `nn.Sequential` and `forward()`  

---

### 4. GPU Integration
- Checked GPU availability using `torch.cuda.is_available()`
- Moved model and tensors to GPU using `.to(device)`
- Optimized training loop for GPU computation

---

### 5. Training Setup
- Optimizer: **SGD** with `weight_decay=1e-4`
- Loss: **CrossEntropyLoss**
- Epochs: **2** (can be increased)
- Learning rate: **0.01**

---

### 6. Results
- **Test Accuracy:** ~89%
- **Effect:** Reduced overfitting and improved generalization
- **Model Saved As:** `fashion_mnist_overfit_reduced.pth`

---

### 7. Key Techniques
| Technique | Purpose |
|------------|----------|
| Batch Normalization | Stabilizes and speeds up training |
| Dropout | Prevents co-adaptation of neurons |
| Weight Decay | Penalizes large weights (L2 regularization) |

---

### 8. References
- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
- [Fashion MNIST on Kaggle](https://www.kaggle.com/zalando-research/fashionmnist)
