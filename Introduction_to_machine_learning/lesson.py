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

def compute_gradient_m(y_i, y_hat_i, x_i):
    """
    This computes the gradient of the loss function with respect to m

    """
    if y_i - y_hat_i > 0:
        return -x_i
    else: return x_i

def compute_gradient_b(y_i, y_hat_i):
    """
    This computes the gradient of the loss function with respect to b
   
    """
    if y_i - y_hat_i > 0:
        return -1
    else: return 1

def compute_gradients(X, y, b, m):
    # get the number of samples we have
    N = len(y)

    # Make a prediction for a linear line
    y_hat = m * X + b

    #Initialize - we need to sum over ALL samples in X, these variables will track the sum
    gradient_b_sum = 0
    gradient_m_sum = 0

    # FOR loop to sum the gradient of each sample and add them up
    for i in range(N):
        gradient_b_sum += compute_gradient_b(y[i], y_hat[i])
        gradient_m_sum += compute_gradient_m(y[i], y_hat[i], X[i])

    # Now you can divide by the number of samples N
    gradient_m = gradient_m_sum / N
    gradient_b = gradient_b_sum / N

    return gradient_b, gradient_m

def train_model(X, y, alpha, epochs):
    # Initialize our parameters
    m = 0
    b = 0

    # Repeat the learning process
    for epoch in range(epochs):

        # Step 1: Compute the gradient
        gradient_b, gradient_m = compute_gradients(X, y, b, m)

        # Use the update equation to update the parameters
        m = m - alpha * gradient_m
        b = b - alpha * gradient_b

    return m, b




if __name__ == "__main__":
    pass
X, y = generate_sample_data()
visualize_data(X, y)

# Lets make a  generic y=mx + b line
m = 0
b = 0

# Let's train the model now to get a better version of m and b
m_better, b_better = train_model(X, y, alpha=0.01, epochs=500)

# Now let's make a prediction using our model
y_hat = m * X + b # This is the bad line
y_hat_better = m_better * X + b_better

# Make a plot
plt.scatter(X, y, alpha=0.6, color='blue')
plt.plot(X, y_hat, color='black')
plt.plot(X, y_hat_better, color='red')
plt.xlabel('X (Features)')
plt.ylabel('y (Target)')
plt.title('Linear Regression')
plt.legend()
plt.show()
