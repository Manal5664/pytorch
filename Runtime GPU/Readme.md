# Fashion MNIST Classification using PyTorch ANN (GPU Version)

## Project Overview
This project implements an **Artificial Neural Network (ANN)** using **PyTorch** to classify images from the **Fashion MNIST** dataset, utilizing **GPU acceleration** for faster training.  

The dataset consists of 28x28 grayscale images of 10 fashion categories. The model achieves approximately **88% accuracy** on the test dataset.

---

## Project Workflow

### 1. Libraries Imported
- PyTorch (`torch`, `torch.nn`, `torch.optim`, `torch.utils.data`)
- NumPy
- torchvision (for datasets and transforms)
- Kaggle API (to fetch Fashion MNIST dataset directly)

### 2. Dataset
- Used **Kaggle API** to download Fashion MNIST dataset.
- Implemented a **custom dataset class** with:
  - `__init__`
  - `__len__`
  - `__getitem__`
- Created **train** and **test dataset objects**.
- Used **DataLoader** for batching train and test data.

### 3. Neural Network
- Custom ANN class using `torch.nn.Module`.
- **Architecture**:
  1. Linear layer with 128 features
  2. ReLU activation
  3. Linear layer with 64 features
  4. ReLU activation
  5. Linear layer with 10 output classes
- Defined **forward pass**.

### 4. GPU Integration
- **Checked GPU availability** using `torch.cuda.is_available()`.
- **Moved model to GPU** with `.to(device)`.
- **Modified training loop** to move data and labels to GPU.
- Optimized data and computation for GPU usage.

### 5. Training Setup
- Hyperparameters:
  - Learning rate
  - Number of epochs
- Loss function: **Cross-Entropy**
- Optimizer: **SGD**

### 6. Training Loop
- Trained the model for **2 epochs** (can extend to more epochs).
- For each batch:
  1. Move batch data to GPU
  2. Compute loss
  3. Zero gradients
  4. Backpropagate and update weights

### 7. Evaluation
- Evaluated model on **test dataset**.
- Predictions computed on GPU.
- Achieved approximately **83% accuracy**.

---

## Results
- Test Accuracy: **~88%**
- GPU usage significantly speeds up training compared to CPU.


---

## References
- [Fashion MNIST Dataset](https://www.kaggle.com/zalando-research/fashionmnist)
