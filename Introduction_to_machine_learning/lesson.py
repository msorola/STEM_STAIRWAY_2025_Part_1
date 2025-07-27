"""
Introduction to Machine Learning: Gradient Descent with Mean Absolute Error (MAE)
================================================================================

This lesson covers:
1. Linear regression fundamentals
2. Mean Absolute Error (MAE) as a loss function
3. Gradient descent optimization
4. Implementation from scratch

Author: AI Workshop for Teachers UTHSCSA 2025
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
    return np.mean(np.abs(y_true - y_pred))

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
    return theta_0 + theta_1 * X

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
    if y_i - y_hat_i >= 0:
        return -1
    else:
        return 1

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
    if y_i - y_hat_i >= 0:
        return -x_i
    else:
        return x_i

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
    # Compute gradients
    grad_theta_0 = compute_gradient_theta_0(y_i, y_hat_i)
    grad_theta_1 = compute_gradient_theta_1(y_i, y_hat_i, x_i)
    
    # Update parameters
    new_theta_0 = theta_0 - alpha * grad_theta_0
    new_theta_1 = theta_1 - alpha * grad_theta_1
    
    return new_theta_0, new_theta_1

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
    # Initialize parameters
    theta_0 = 0.0
    theta_1 = 0.0
    
    # Store MAE history for plotting
    mae_history = []
    
    if verbose:
        print(f"Training for {epochs} epochs with learning rate α = {alpha}")
        print("-" * 50)
    
    for epoch in range(epochs):
        # Make predictions
        y_pred = predict(X, theta_0, theta_1)
        
        # Calculate MAE
        mae = calculate_mae(y, y_pred)
        mae_history.append(mae)
        
        # Update parameters using all data points (batch gradient descent)
        for i in range(len(X)):
            y_i = y[i][0]  # Extract scalar value
            y_hat_i = y_pred[i][0]  # Extract scalar value
            x_i = X[i][0]  # Extract scalar value
            
            theta_0, theta_1 = update_parameters(theta_0, theta_1, alpha, y_i, y_hat_i, x_i)
        
        # Print progress every 10 epochs
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d}: MAE = {mae:.4f}, θ₀ = {theta_0:.4f}, θ₁ = {theta_1:.4f}")
    
    if verbose:
        print("-" * 50)
        print(f"Final parameters: θ₀ = {theta_0:.4f}, θ₁ = {theta_1:.4f}")
        print(f"Final MAE: {mae_history[-1]:.4f}")
    
    return theta_0, theta_1, mae_history

def plot_training_progress(mae_history):
    """
    Plot the training progress (MAE over epochs).
    
    Parameters:
    mae_history (list): List of MAE values for each epoch
    """
    plt.figure(figsize=(10, 6))
    plt.plot(mae_history, color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Training Progress: MAE vs Epochs')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_final_model(X, y, theta_0, theta_1):
    """
    Plot the final trained model with data points.
    
    Parameters:
    X (array): Feature values
    y (array): Target values
    theta_0 (float): Final intercept
    theta_1 (float): Final slope
    """
    # Generate predictions for plotting
    y_pred = predict(X, theta_0, theta_1)
    
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(X, y, color='blue', alpha=0.6, label='Data points')
    
    # Plot regression line
    plt.plot(X, y_pred, color='red', linewidth=2, label=f'Regression Line: y = {theta_0:.2f} + {theta_1:.2f}x')
    
    plt.xlabel('X (Features)')
    plt.ylabel('y (Target)')
    plt.title('Linear Regression using Gradient Descent with MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    """
    Main function to demonstrate the complete workflow.
    """
    print("=" * 60)
    print("INTRODUCTION TO MACHINE LEARNING: GRADIENT DESCENT WITH MAE")
    print("=" * 60)
    
    # Step 1: Generate sample data
    print("\n1. Generating sample data...")
    X, y = generate_sample_data(n_samples=500)
    print(f"Generated {len(X)} data points")
    
    # Step 2: Visualize the data
    print("\n2. Visualizing the data...")
    visualize_data(X, y, "Generated Data for Linear Regression")
    
    # Step 3: Train the model
    print("\n3. Training the model using gradient descent with MAE...")
    theta_0, theta_1, mae_history = train_linear_regression_mae(
        X, y, epochs=100, alpha=0.01, verbose=True
    )
    
    # Step 4: Plot training progress
    print("\n4. Plotting training progress...")
    plot_training_progress(mae_history)
    
    # Step 5: Plot final model
    print("\n5. Plotting final model...")
    plot_final_model(X, y, theta_0, theta_1)
    
    # Step 6: Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"True relationship: y = 4 + 3*X + noise")
    print(f"Learned relationship: y = {theta_0:.4f} + {theta_1:.4f}*X")
    print(f"Final MAE: {mae_history[-1]:.4f}")
    print(f"Parameter error - θ₀: {abs(4 - theta_0):.4f}, θ₁: {abs(3 - theta_1):.4f}")
    
    print("\nKey Concepts Learned:")
    print("- Linear regression models relationships between variables")
    print("- MAE measures prediction error using absolute differences")
    print("- Gradient descent iteratively updates parameters to minimize loss")
    print("- Learning rate controls the step size in parameter updates")

if __name__ == "__main__":
    main() 