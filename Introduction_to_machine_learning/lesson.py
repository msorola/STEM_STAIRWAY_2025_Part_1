"""
Introduction to Machine Learning: Gradient Descent with Mean Absolute Error (MAE)
================================================================================

This lesson covers:
1. Linear regression fundamentals
2. Mean Absolute Error (MAE) as a loss function
3. Gradient descent optimization
4. Implementation from scratch
"""

import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

def generate_sample_data(n_samples=500):
    """
    Generate synthetic data for linear regression demonstration.
    
    Parameters:
    n_samples (int): Number of data points to generate
    
    Returns:
    tuple: (X, y) where X is features and y is target values
    """
    # Generate random features
    X = 2 * np.random.rand(n_samples, 1)
    # Generate target values with some noise
    # True relationship: y = 4 + 3*X + noise
    y = 4 + 3 * X + np.random.randn(n_samples, 1)
    return X, y

def visualize_data(X, y, title="Generated Data"):
    """
    Visualize the generated data points.
    
    Parameters:
    X (array): Feature values
    y (array): Target values
    title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.6, color='blue')
    plt.xlabel('X (Features)')
    plt.ylabel('y (Target)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE).
    
    MAE = (1/n) * Σ|y_true - y_pred|
    
    Parameters:
    y_true (array): True target values
    y_pred (array): Predicted target values
    
    Returns:
    float: Mean Absolute Error
    """
    pass

def predict(X, theta_0, theta_1):
    """
    Make predictions using linear model: y = theta_0 + theta_1 * X
    
    Parameters:
    X (array): Feature values
    theta_0 (float): Intercept (bias)
    theta_1 (float): Slope (weight)
    
    Returns:
    array: Predicted values
    """
    pass

def compute_gradient_theta_0(y_i, y_hat_i):
    """
    Compute gradient for theta_0 (intercept) using MAE.
    
    ∂MAE/∂θ₀ = sign(y_i - y_hat_i)
    
    Parameters:
    y_i (float): True value
    y_hat_i (float): Predicted value
    
    Returns:
    float: Gradient for theta_0
    """
    pass

def compute_gradient_theta_1(y_i, y_hat_i, x_i):
    """
    Compute gradient for theta_1 (slope) using MAE.
    
    ∂MAE/∂θ₁ = sign(y_i - y_hat_i) * x_i
    
    Parameters:
    y_i (float): True value
    y_hat_i (float): Predicted value
    x_i (float): Feature value
    
    Returns:
    float: Gradient for theta_1
    """
    pass

def update_parameters(theta_0, theta_1, alpha, y_i, y_hat_i, x_i):
    """
    Update parameters using gradient descent.
    
    θ_new = θ_old - α * ∇θ
    
    Parameters:
    theta_0 (float): Current intercept
    theta_1 (float): Current slope
    alpha (float): Learning rate
    y_i (float): True value
    y_hat_i (float): Predicted value
    x_i (float): Feature value
    
    Returns:
    tuple: (new_theta_0, new_theta_1)
    """
    pass

def train_linear_regression_mae(X, y, epochs=100, alpha=0.01, verbose=True):
    """
    Train linear regression model using gradient descent with MAE.
    
    Parameters:
    X (array): Feature values
    y (array): Target values
    epochs (int): Number of training iterations
    alpha (float): Learning rate
    verbose (bool): Whether to print training progress
    
    Returns:
    tuple: (theta_0, theta_1, mae_history)
    """
    pass


def main():
    """
    Main function to demonstrate the complete workflow.
    """
    print("=" * 60)
    print("INTRODUCTION TO MACHINE LEARNING: GRADIENT DESCENT WITH MAE")
    print("=" * 60)


if __name__ == "__main__":
    main() 