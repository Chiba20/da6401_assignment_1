# DA6401 Assignment 1  
## Multi-Layer Perceptron for Image Classification

## Project Links

**GitHub Repository**  
https://github.com/Chiba20/da6401_assignment_1

**Weights & Biases Report**  
https://wandb.ai/https://wandb.ai/ge26z812-iitm-india/da6401-mlp/reports/DA6401-Assignment-1-MLP-for-Image-Classification--VmlldzoxNjEzNDI2MA?accessToken=25y991ov1r5tevalopjjt0ka1jmnrn07x24ur9pv0bqer90qq6b0mk7dwygu1kqk

---

This assignment implements a **Multi-Layer Perceptron (MLP)** using **NumPy** for image classification on the **MNIST** and **Fashion-MNIST** datasets.

The implementation includes custom layers, activation functions, loss functions, and optimizers without using high-level deep learning frameworks.

All experiments are tracked and visualized using **Weights & Biases (W&B)**.

---

# Project Objectives

The goal of this assignment is to understand how neural networks work internally by implementing the core components of a neural network from scratch.

Key objectives include:

- Implementing forward propagation  
- Implementing backpropagation  
- Studying different activation functions  
- Comparing optimization algorithms  
- Analyzing weight initialization strategies  
- Performing hyperparameter tuning  

The experiments also investigate important neural network behaviors such as:

- Vanishing gradients  
- Dead neurons  
- Loss function behavior  
- Overfitting  
- Symmetry breaking in weight initialization  

---

# Repository Structure

```
da6401_assignment_1
│
├── models
│   └── .gitkeep
│
├── src
│   │
│   ├── best_model.npy
│   ├── config.json
│   ├── train.py
│   ├── inference.py
│   │
│   ├── ann
│   │   ├── activations.py
│   │   ├── neural_layer.py
│   │   ├── neural_network.py
│   │   ├── objective_functions.py
│   │   ├── optimizers.py
│   │   └── __init__.py
│   │
│   └── utils
│       ├── data_loader.py
│       └── __init__.py
│
├── requirements.txt
└── README.md
```

---

# Implemented Components

## Neural Network Architecture

The neural network supports:

- Multiple hidden layers  
- Configurable hidden layer sizes  
- Different activation functions  
- Multiple optimizers  

---

## Activation Functions

The following activation functions are implemented:

- **Sigmoid**
- **Tanh**
- **ReLU**

---

## Loss Functions

Two loss functions were implemented:

- **Cross Entropy**
- **Mean Squared Error (MSE)**

---

## Optimizers

The neural network supports the following optimizers:

- **SGD**
- **Momentum**
- **Nesterov Accelerated Gradient (NAG)**
- **RMSProp**

---

## Weight Initialization Methods

Two initialization methods were implemented:

- **Random Initialization**
- **Xavier Initialization**

Zero initialization was also used for studying **symmetry problems in neural networks**.

---

# Dataset

Two datasets were used in this assignment.

## MNIST

- 60,000 training images  
- 10,000 test images  
- 10 classes (digits 0–9)

## Fashion-MNIST

- 60,000 training images  
- 10,000 test images  
- 10 clothing categories  

All images are **28 × 28 grayscale images**, which are flattened into **784 input features** before being fed into the neural network.

---

# Training the Model

The model can be trained using the following command:

```bash
python src/train.py -d mnist -e 10 -b 64 -l cross_entropy -o rmsprop -lr 0.001 -wd 0.0 -nhl 3 -sz 128 128 128 -a relu -w_i xavier
```

---

# Training Arguments

| Argument | Description |
|--------|-------------|
| `-d` | Dataset (mnist or fashion) |
| `-e` | Number of epochs |
| `-b` | Batch size |
| `-l` | Loss function |
| `-o` | Optimizer |
| `-lr` | Learning rate |
| `-wd` | Weight decay |
| `-nhl` | Number of hidden layers |
| `-sz` | Hidden layer sizes |
| `-a` | Activation function |
| `-w_i` | Weight initialization |

---

# Hyperparameter Sweep

A **Weights & Biases sweep** was used to explore different hyperparameter combinations.

The sweep varied:

- Optimizer  
- Learning rate  
- Activation function  
- Batch size  
- Network depth  
- Hidden layer sizes  

More than **100 runs** were executed to identify the best-performing configuration.

---

# Model Evaluation

After training, the model can be evaluated using:

```bash
python src/inference.py --model_path src/best_model.npy --config_path src/config.json -d mnist
```

This script reports the following metrics:

- Accuracy  
- Precision  
- Recall  
- F1-score  

It also generates the following visualizations:

- Confusion Matrix  
- Misclassified Examples  

---

# Experiments Conducted

## 1. Data Exploration

Sample images from each class were visualized using W&B tables to understand the dataset.

---

## 2. Hyperparameter Sweep

A hyperparameter search was conducted using W&B sweeps to identify the most effective model configuration.

---

## 3. Optimizer Comparison

Different optimizers were compared:

- SGD  
- Momentum  
- NAG  
- RMSProp  

RMSProp achieved the best performance due to adaptive learning rates.

---

## 4. Vanishing Gradient Analysis

The gradient norms of the first hidden layer were monitored using different activation functions.

Sigmoid activation resulted in very small gradients, demonstrating the **vanishing gradient problem**, while ReLU maintained stronger gradient signals.

---

## 5. Dead Neuron Investigation

With a high learning rate and ReLU activation, many neurons became inactive and produced zero outputs for most inputs.

This experiment demonstrated the **dead neuron problem**.

---

## 6. Loss Function Comparison

Cross-Entropy and Mean Squared Error were compared using identical architectures.

Cross-Entropy converged faster and achieved slightly higher accuracy, making it more suitable for classification tasks.

---

## 7. Global Performance Analysis

Training and test accuracies were compared across different runs.

Some models achieved high training accuracy but lower test accuracy, indicating **overfitting**.

---

## 8. Error Analysis

A confusion matrix was used to analyze classification performance.

Misclassified examples were also visualized to understand model errors.

---

## 9. Weight Initialization & Symmetry

Two initialization strategies were compared:

- Zero Initialization  
- Xavier Initialization  

Zero initialization caused neurons to learn identical features due to symmetry, while Xavier initialization allowed the network to learn diverse features.

---

## 10. Fashion-MNIST Transfer Challenge

The best-performing configurations from MNIST experiments were tested on Fashion-MNIST.

Because Fashion-MNIST is a more complex dataset with visually similar classes, performance was slightly lower than MNIST.

---

# Best Model Configuration

The best-performing configuration used:

- **Optimizer:** RMSProp  
- **Activation:** ReLU  
- **Learning Rate:** 0.001  
- **Hidden Layers:** 3  
- **Hidden Units:** 128, 128, 128  

This configuration achieved approximately:

- **Test Accuracy (MNIST): ~98%**

---

# Experiment Tracking

All experiments were tracked using **Weights & Biases (W&B)**.

The W&B report includes:

- Training curves  
- Hyperparameter sweep visualizations  
- Gradient analysis  
- Activation statistics  
- Error analysis plots  

---

# Dependencies

Install dependencies using:

```bash
pip install -r requirements.txt
```

Required libraries include:

- numpy  
- matplotlib  
- keras  
- wandb  

---

# Conclusion

This assignment demonstrates how neural networks function internally when implemented from scratch. Through various experiments, we explored important deep learning concepts such as gradient flow, optimizer behavior, weight initialization, and model generalization.

These experiments provide insight into how architectural and optimization choices influence neural network training and performance.