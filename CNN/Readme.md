#  MNIST Classification using PyTorch CNN

## Overview
This project implements a **Convolutional Neural Network (CNN)** in **PyTorch** for classifying handwritten digits from the **MNIST dataset**.  
The model was trained using **GPU acceleration** and achieves strong accuracy without overfitting.

- **Training Accuracy:** 86.0%  
- **Test Accuracy:** 85.8%  

---

## Workflow

### 1. Libraries Used
- `torch`, `torch.nn`, `torch.optim`, `torch.utils.data`
- `torchvision.datasets`, `transforms`
- `numpy`
- **Kaggle API** (for dataset download)

---

### 2. Dataset
- **MNIST Dataset** (28x28 grayscale digits, 10 classes)
- Downloaded using **Kaggle API**
- Custom dataset class implemented with:
  - `__init__`, `__len__`, `__getitem__`
- Used **DataLoader** for batching and shuffling

---

### 3. CNN Architecture
- Defined using `nn.Sequential` inside a `NeuralNetwork` class  
- Layers used:
  1. **Conv2D layer** for feature extraction  
  2. **ReLU activation**  
  3. **Flatten layer**  
  4. **Fully Connected (Linear) layer** for classification  
- Loss Function: **CrossEntropyLoss**

---

### 4. Training Setup
- **Batch Size:** same as previous projects  
- **Epochs:** standard short training  
- **Optimizer:** default PyTorch SGD (no tuning library used)  
- **Device:** model and tensors moved to **GPU** if available

---

### 5. Results
| Metric | Accuracy |
|--------|-----------|
| Training | 86.0% |
| Testing | 85.8% |
| Overfitting | Not observed |

---

### 6. Highlights
- Simple and efficient **CNN** for MNIST digit recognition  
- **Clean training curve** — no overfitting  
- **Fast convergence** on GPU  

---

### 7. References
- [PyTorch CNN Documentation](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [MNIST Dataset (Kaggle)](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
