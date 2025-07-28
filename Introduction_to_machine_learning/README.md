# Introduction to Machine Learning: Linear Regression Practice

This module provides a hands-on introduction to machine learning through implementing linear regression with gradient descent optimization. Students will complete key functions to understand the fundamentals of machine learning algorithms.

## 📁 File Structure

```
Introduction_to_machine_learning/
├── lesson.py      # Core implementation with functions to complete
└── README.md      # This file
```

## 🎯 Learning Objectives

After completing this module, you will be able to:

1. **Understand Linear Regression**
   - Model relationships between variables
   - Work with parameters (intercept and slope)
   - Make predictions using linear models

2. **Implement Core ML Functions**
   - Calculate Mean Absolute Error (MAE)
   - Compute gradients for optimization
   - Update model parameters
   - Train a complete model

3. **Work with Data**
   - Generate and visualize sample data
   - Process data for training
   - Evaluate model performance

## 🔨 Functions to Implement

In `lesson.py`, you'll need to complete these key functions:

1. `calculate_mae(y_true, y_pred)`
   - Compute Mean Absolute Error
   - Formula: `MAE = (1/n) * Σ|y_true - y_pred|`

2. `predict(X, theta_0, theta_1)`
   - Make predictions using linear model
   - Formula: `y = theta_0 + theta_1 * X`

3. `compute_gradient_theta_0(y_i, y_hat_i)`
   - Calculate gradient for intercept
   - Used in parameter updates

4. `compute_gradient_theta_1(y_i, y_hat_i, x_i)`
   - Calculate gradient for slope
   - Used in parameter updates

5. `update_parameters(theta_0, theta_1, alpha, y_i, y_hat_i, x_i)`
   - Update model parameters using gradients
   - Formula: `θ_new = θ_old - α * ∇θ`

### Running the Code

1. Start with `lesson.py`
   - Implement the required functions
   - Run to test your implementation
   - Check the visualizations

2. The code will:
   - Generate sample data
   - Train the model using your implementations
   - Show learning progress
   - Visualize the final results

## 📊 Expected Results

When correctly implemented, you should see:

1. Generated data points with noise
2. Training progress with decreasing MAE
3. Final regression line fitting the data
4. Comparison between true and learned parameters

## 🔍 Implementation Tips

1. **MAE Calculation**
   - Remember to take absolute values
   - Average over all samples
   - Handle edge cases (empty arrays)

2. **Gradient Computation**
   - Consider the sign of the error
   - Include feature values for theta_1
   - Keep track of dimensions

3. **Parameter Updates**
   - Use the learning rate correctly
   - Update both parameters simultaneously
   - Maintain numerical stability