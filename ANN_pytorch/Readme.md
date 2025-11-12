# Fashion MNIST Multi Classification using PyTorch ANN

## Project Overview
This project implements an **Artificial Neural Network (ANN)** using **PyTorch** to classify images from the **Fashion MNIST** dataset. The dataset contains 28x28 grayscale images of 10 different fashion categories (T-shirts, trousers, shoes, etc.).  

The model achieves approximately **83% accuracy** on the test dataset.

---

## Project Workflow

### 1. Libraries Imported
- PyTorch (`torch`, `torch.nn`, `torch.optim`, `torch.utils.data`)

### 2. Dataset
- Used the **Fashion MNIST dataset**.
- Implemented a **custom dataset class** using:
  - `__init__`
  - `__len__`
  - `__getitem__`
- Created **train** and **test dataset objects**.
- Used **DataLoader** for batching train and test data.

### 3. Neural Network
- Custom neural network class with `torch.nn.Module`.
- **Architecture**:
  1. Linear layer with 128 features
  2. ReLU activation
  3. Linear layer with 64 features
  4. ReLU activation
  5. Linear layer with 10 output classes
- Defined **forward pass** for computation.

### 4. Training Setup
- Hyperparameters:
  - Learning rate
  - Number of epochs
- Loss function: **Cross-Entropy**
- Optimizer: **SGD**
### 5. Training Loop
- Trained the model for **2 epochs**.
- For each batch:
  1. Compute loss
  2. Zero gradients
  3. Backpropagate and update weights

### 6. Evaluation
- Evaluated model on the **test dataset**.
- Computed **predictions** and **accuracy**.
- Achieved approximately **83% accuracy**.

---

## Results
- Test Accuracy: **~83%**
- Successfully classifies Fashion MNIST images into 10 categories.

