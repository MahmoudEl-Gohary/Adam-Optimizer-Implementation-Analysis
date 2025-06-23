# Adam Optimizer: Implementation and Performance Analysis

A comprehensive implementation and analysis of the Adam (Adaptive Moment Estimation) optimizer, developed as part of MATH 303: Linear and Non-Linear Programming coursework at Zewail City. This project compares a custom Adam implementation with Keras' built-in version across different neural network architectures.

## 📊 Project Overview

This project provides both theoretical analysis and practical implementation of the Adam optimizer, demonstrating its effectiveness compared to traditional optimization methods through experimentation on the MNIST dataset.

## 🎯 Key Objectives

- Implement Adam optimizer from scratch, following the  mathematical definitions
- Compare custom implementation with Keras' built-in Adam optimizer
- Analyze performance across different neural network architectures
- Evaluate convergence characteristics, training efficiency, and accuracy
- Validate theoretical properties through empirical experiments

## 📁 Project Structure

```
├── notebook.ipynb                # Main implementation and analysis
├── README.md                     # Project documentation
└── report.pdf                       # Academic report and documentation
```

## 🧮 Mathematical Foundation

### Adam Update Rules
The Adam optimizer updates parameters using adaptive moment estimation:

```
mt = β1 * mt-1 + (1 - β1) * gt
vt = β2 * vt-1 + (1 - β2) * gt²
m̂t = mt / (1 - β1^t)
v̂t = vt / (1 - β2^t)
θt+1 = θt - η * m̂t / (√v̂t + ε)
```

Where:
- `η`: Learning rate
- `β1, β2`: Decay rates for moment estimates
- `ε`: Small constant for numerical stability
- `gt`: Gradient at time step t

## 🏗️ Implementation Details

### Custom Adam Optimizer Classes

1. **Standalone Implementation**: Pure NumPy implementation for educational purposes
2. **Keras-Compatible Implementation**: Custom optimizer class inheriting from `tf.keras.optimizers.Optimizer`

### Model Architectures Tested

**Simple Neural Network (SimpleNN)**:
- Flatten layer (28×28×1 → 784)
- Dense layer (128 units, ReLU)
- Dense layer (64 units, ReLU)
- Output layer (10 units, softmax)

**Convolutional Neural Network (CNN)**:
- Conv2D (32 filters, 3×3, ReLU)
- MaxPooling2D (2×2)
- Conv2D (64 filters, 3×3, ReLU)
- MaxPooling2D (2×2)
- Flatten + Dense (128 units, ReLU)
- Output layer (10 units, softmax)

## 🛠️ Technologies Used

- **Python 3.10**
- **TensorFlow/Keras** - Deep learning framework
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical plotting
- **Google Colab** - Development environment

## 📈 Key Results

### Performance Comparison

| Model & Optimizer | Training Time (s) | Test Accuracy (%) | Final Loss |
|-------------------|-------------------|-------------------|------------|
| SimpleNN (Keras Adam) | 26.7 | 97.65 | 0.0503 |
| SimpleNN (Custom Adam) | 27.1 | 97.53 | 0.0511 |
| CNN (Keras Adam) | 345.2 | **98.94** | **0.0195** |
| CNN (Custom Adam) | 348.5 | 98.88 | 0.0199 |

### Key Findings

✅ **Custom Implementation Validation**: Our custom Adam optimizer achieved performance nearly identical to Keras' implementation (difference < 3%)

✅ **Architecture Impact**: CNN models achieved significantly higher accuracy (98.94% vs 97.65%) but required 13x longer training time

✅ **Convergence Stability**: Both Adam implementations showed superior convergence characteristics with stable validation metrics

✅ **Training Efficiency**: SimpleNN models are ideal for applications prioritizing speed over maximum accuracy

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow numpy matplotlib seaborn jupyter
```

### Running the Analysis
1. Clone this repository
2. Open `notebook.ipynb` in Jupyter or Google Colab
3. Run all cells sequentially to reproduce results

## 📊 Experimental Setup

### Dataset
- **MNIST**: 60,000 training images, 10,000 test images
- **Preprocessing**: Normalized to [0,1], reshaped to (28,28,1)
- **Labels**: One-hot encoded for 10-digit classes

### Hyperparameters
- Learning rate: 0.001
- β1: 0.9 (first moment decay)
- β2: 0.999 (second moment decay)
- ε: 1e-8 (numerical stability)
- Batch size: 64
- Epochs: 5

## 📈 Visualizations

The project generates comprehensive visualizations including:
- Training vs validation loss curves
- Accuracy progression over epochs
- Performance comparison bar charts
- Convergence analysis plots

## 🔍 Analysis Highlights

### Convergence Behavior
- CNN architectures achieved significantly lower final loss (0.0195-0.0199) vs SimpleNN (0.0503-0.0511)
- Both implementations showed similar convergence patterns
- CNN models demonstrated more stable validation metrics

### Implementation Validation
- Custom Adam matched Keras Adam performance (loss difference < 0.8% for CNN, < 3% for SimpleNN)
- Training times were comparable between custom and built-in implementations
- Mathematical correctness validated through empirical results

## 🔮 Future Work

- **Dataset Expansion**: Test on CIFAR-10, ImageNet for complex scenarios
- **Hyperparameter Optimization**: Systematic exploration of learning rates and decay factors
- **Scalability Testing**: Apply to larger models (ResNet, Transformers)
- **Hybrid Approaches**: Investigate Adam combinations with other optimization techniques
- **Regularization Impact**: Study interaction with dropout, batch normalization

## 📝 Academic Context

**Course**: MATH 303 - Linear and Non-Linear Programming  
**Institution**: Zewail City  
**Authors**: Mazen Ahmed Basha, Ahmed Sameh Morsy, Mahmoud Abdelrahman, Haneen Alaa, Jilan Ismail  
**Supervision**: Prof. Ahmed Abdelsamea, TA: Hossam Fathy, Youssef Mohamed

## 📚 References

1. Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR.
2. Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. JMLR.
3. Tieleman, T., & Hinton, G. (2012). RMSProp: Divide the gradient by a running average of its recent magnitude. Coursera.

## 📄 License

This project is created for educational purposes as part of coursework requirements.

---

*This project demonstrates the mathematical foundations and practical implementation of adaptive optimization algorithms, showcasing skills in deep learning, mathematical modeling, and empirical analysis.*
