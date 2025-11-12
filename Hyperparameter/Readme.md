# Fashion MNIST Classification using PyTorch ANN (Hyperparameter Tuning with Optuna)

## Overview
This project extends the previous **ANN Fashion MNIST** model by integrating **Optuna** for **hyperparameter tuning**.  
The goal is to automatically find the best combination of learning rate, dropout rate, optimizer, and other parameters to maximize model performance.  

With Optuna, the model achieved an improved **accuracy of ~88%** on the test dataset.

---

## Workflow

### 1. Libraries Used
- `torch`, `torch.nn`, `torch.optim`, `torch.utils.data`
- `torchvision.datasets` and `transforms`
- `optuna` (for hyperparameter tuning)
- `numpy`
- **Kaggle API** (for dataset download)

---

### 2. Dataset
- **Fashion MNIST** dataset (28x28 grayscale images, 10 categories)
- Downloaded using **Kaggle API**
- Custom dataset class created with:
  - `__init__`, `__len__`, `__getitem__`
- Used **DataLoader** for training and testing batches

---

### 3. Neural Network Architecture
- **Input → Linear(128) → ReLU → Dropout → Linear(64) → ReLU → Output(10)**
- Implemented using `nn.Sequential` in PyTorch
- Supports **dynamic hyperparameters** (e.g., dropout rate, learning rate)

---

### 4. Hyperparameter Tuning (Optuna)
- Defined an **objective function (trial)** to tune:
  - Learning Rate  
  - Dropout Rate  
  - Batch Size  
  - Optimizer Type (SGD, Adam, RMSProp)  
  - Weight Decay  
  - Hidden Layer Size  
- Used `study.optimize(objective, n_trials=10)` for searching best parameters.
- Evaluated each trial based on **validation accuracy**.

---

### 5. GPU Integration
- Checked GPU availability via `torch.cuda.is_available()`
- Moved model and data to GPU using `.to(device)`

---

### 6. Results
| Metric | Value |
|--------|--------|
| Accuracy | **~88%** |
| Optimizer | Adam |
| Dropout | Tuned via Optuna |
| Learning Rate | Automatically selected |
| Trials | 10 |

---

### 7. Key Advantages
- Automated hyperparameter optimization  
- Reduced manual tuning effort  
- Better model generalization and performance  

---

### 8. References
- [Optuna Documentation](https://optuna.org/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [Fashion MNIST on Kaggle](https://www.kaggle.com/zalando-research/fashionmnist)
