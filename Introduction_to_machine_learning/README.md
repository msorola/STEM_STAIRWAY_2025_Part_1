# Introduction to Machine Learning: Gradient Descent with MAE

This module introduces fundamental machine learning concepts through hands-on implementation of gradient descent optimization using Mean Absolute Error (MAE) for linear regression.

## 📁 File Structure

```
Introduction_to_machine_learning/
├── lesson.py          # Complete lesson with theory and implementation
└── README.md          # This file
```

## 🎯 Learning Objectives

By the end of this module, you will understand:

1. **Linear Regression Fundamentals**
   - How to model relationships between variables
   - The concept of parameters (intercept and slope)
   - Prediction using linear models

2. **Mean Absolute Error (MAE)**
   - How to measure prediction accuracy
   - Why MAE is robust to outliers
   - Mathematical formulation of MAE

3. **Gradient Descent Optimization**
   - How to iteratively update parameters
   - The role of learning rate
   - Convergence and optimization

4. **Practical Implementation**
   - Data generation and visualization
   - Training algorithms from scratch
   - Hyperparameter tuning

## 🚀 Getting Started

### Prerequisites

Install the required packages:

```bash
pip install numpy matplotlib
```

## 📚 Lesson Content

### `lesson.py` - Complete Implementation

The lesson file contains:

- **Data Generation**: Synthetic data creation with known relationships
- **MAE Calculation**: Implementation of Mean Absolute Error
- **Gradient Computation**: Manual gradient calculation for both parameters
- **Parameter Updates**: Gradient descent update rules
- **Training Loop**: Complete training algorithm
- **Visualization**: Data plots and training progress
- **Analysis**: Model evaluation and interpretation



## 🔬 Key Concepts Explained

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between predictions and true values:

```
MAE = (1/n) * Σ|y_true - y_pred|
```

**Advantages:**
- Robust to outliers
- Easy to interpret
- Scale-invariant

### Gradient Descent

Gradient descent iteratively updates parameters to minimize the loss function:

```
θ_new = θ_old - α * ∇θ
```

Where:
- `α` is the learning rate
- `∇θ` is the gradient of the loss with respect to the parameter

### Linear Regression Model

The model predicts using a linear relationship:

```
y_pred = θ₀ + θ₁ * x
```

Where:
- `θ₀` is the intercept (bias)
- `θ₁` is the slope (weight)

## 📊 Expected Outputs

When you run the lesson, you'll see:

1. **Data Visualization**: Scatter plot of generated data
2. **Training Progress**: MAE values over epochs
3. **Final Model**: Regression line fitted to data
4. **Parameter Comparison**: True vs learned parameters