"""
Practice Exercises: Gradient Descent with Mean Absolute Error (MAE)
==================================================================

This practice file contains exercises to reinforce understanding of:
1. Gradient descent optimization
2. Mean Absolute Error (MAE) loss function
3. Parameter tuning and hyperparameter selection
4. Data visualization and analysis

Complete the exercises below to master the concepts!
"""

import numpy as np
import matplotlib.pyplot as plt
from lesson import (
    generate_sample_data, visualize_data, calculate_mae, predict,
    compute_gradient_theta_0, compute_gradient_theta_1, update_parameters,
    train_linear_regression_mae, plot_training_progress, plot_final_model
)

# Set random seed for reproducibility
np.random.seed(42)

def exercise_1_mae_calculation():
    """
    Exercise 1: Understanding Mean Absolute Error (MAE)
    
    Complete the following tasks:
    1. Calculate MAE for given data
    2. Understand how MAE differs from other loss functions
    """
    print("=" * 60)
    print("EXERCISE 1: MEAN ABSOLUTE ERROR (MAE) CALCULATION")
    print("=" * 60)
    
    # Sample data
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred_1 = np.array([1.1, 1.9, 3.1, 3.9, 5.1])  # Good predictions
    y_pred_2 = np.array([0.5, 1.5, 2.5, 3.5, 4.5])  # Poor predictions
    
    print("True values:", y_true)
    print("Predictions 1 (good):", y_pred_1)
    print("Predictions 2 (poor):", y_pred_2)
    
    # TODO: Calculate MAE for both prediction sets
    mae_1 = calculate_mae(y_true, y_pred_1)
    mae_2 = calculate_mae(y_true, y_pred_2)
    
    print(f"\nMAE for good predictions: {mae_1:.4f}")
    print(f"MAE for poor predictions: {mae_2:.4f}")
    
    # TODO: Answer these questions:
    print("\nQuestions to answer:")
    print("1. Which prediction set has lower MAE? Why?")
    print("2. How does MAE treat outliers compared to Mean Squared Error (MSE)?")
    print("3. What would be the MAE if all predictions were perfect?")
    
    return mae_1, mae_2

def exercise_2_gradient_computation():
    """
    Exercise 2: Understanding Gradient Computation
    
    Complete the following tasks:
    1. Compute gradients manually
    2. Understand the gradient descent update rule
    """
    print("\n" + "=" * 60)
    print("EXERCISE 2: GRADIENT COMPUTATION")
    print("=" * 60)
    
    # Sample data point
    x_i = 2.0
    y_i = 8.0
    y_hat_i = 6.0  # Prediction: y = 1 + 2.5*x = 1 + 2.5*2 = 6
    
    print(f"Data point: x = {x_i}, y = {y_i}")
    print(f"Prediction: y_hat = {y_hat_i}")
    print(f"Error: y - y_hat = {y_i - y_hat_i}")
    
    # TODO: Compute gradients manually
    grad_theta_0 = compute_gradient_theta_0(y_i, y_hat_i)
    grad_theta_1 = compute_gradient_theta_1(y_i, y_hat_i, x_i)
    
    print(f"\nGradient for θ₀: {grad_theta_0}")
    print(f"Gradient for θ₁: {grad_theta_1}")
    
    # TODO: Update parameters with learning rate α = 0.1
    alpha = 0.1
    theta_0_old = 1.0
    theta_1_old = 2.5
    
    theta_0_new, theta_1_new = update_parameters(
        theta_0_old, theta_1_old, alpha, y_i, y_hat_i, x_i
    )
    
    print(f"\nParameter update with α = {alpha}:")
    print(f"θ₀: {theta_0_old:.4f} → {theta_0_new:.4f}")
    print(f"θ₁: {theta_1_old:.4f} → {theta_1_new:.4f}")
    
    # TODO: Answer these questions:
    print("\nQuestions to answer:")
    print("1. Why is the gradient for θ₀ either -1 or 1?")
    print("2. How does the gradient for θ₁ depend on the feature value x_i?")
    print("3. What happens to the parameters when the prediction is too high vs too low?")
    
    return grad_theta_0, grad_theta_1, theta_0_new, theta_1_new

def exercise_3_learning_rate_experiment():
    """
    Exercise 3: Learning Rate Experiment
    
    Complete the following tasks:
    1. Train models with different learning rates
    2. Observe the effect on convergence
    3. Understand the trade-off between speed and stability
    """
    print("\n" + "=" * 60)
    print("EXERCISE 3: LEARNING RATE EXPERIMENT")
    print("=" * 60)
    
    # Generate data
    X, y = generate_sample_data(n_samples=200)
    
    # Test different learning rates
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    results = {}
    
    for alpha in learning_rates:
        print(f"\nTraining with learning rate α = {alpha}")
        print("-" * 40)
        
        theta_0, theta_1, mae_history = train_linear_regression_mae(
            X, y, epochs=50, alpha=alpha, verbose=False
        )
        
        results[alpha] = {
            'theta_0': theta_0,
            'theta_1': theta_1,
            'mae_history': mae_history,
            'final_mae': mae_history[-1]
        }
        
        print(f"Final MAE: {mae_history[-1]:.4f}")
        print(f"Final parameters: θ₀ = {theta_0:.4f}, θ₁ = {theta_1:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    for alpha in learning_rates:
        plt.plot(results[alpha]['mae_history'], 
                label=f'α = {alpha}', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Training Progress with Different Learning Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # TODO: Answer these questions:
    print("\nQuestions to answer:")
    print("1. Which learning rate converges fastest?")
    print("2. Which learning rate is most stable?")
    print("3. What happens when the learning rate is too high?")
    print("4. What happens when the learning rate is too low?")
    print("5. What would be the optimal learning rate for this problem?")
    
    return results

def exercise_4_data_size_experiment():
    """
    Exercise 4: Data Size Experiment
    
    Complete the following tasks:
    1. Train models with different data sizes
    2. Observe the effect on model performance
    3. Understand the bias-variance trade-off
    """
    print("\n" + "=" * 60)
    print("EXERCISE 4: DATA SIZE EXPERIMENT")
    print("=" * 60)
    
    data_sizes = [50, 100, 200, 500, 1000]
    results = {}
    
    for n_samples in data_sizes:
        print(f"\nTraining with {n_samples} data points")
        print("-" * 40)
        
        # Generate data
        X, y = generate_sample_data(n_samples=n_samples)
        
        # Train model
        theta_0, theta_1, mae_history = train_linear_regression_mae(
            X, y, epochs=100, alpha=0.01, verbose=False
        )
        
        results[n_samples] = {
            'theta_0': theta_0,
            'theta_1': theta_1,
            'mae_history': mae_history,
            'final_mae': mae_history[-1]
        }
        
        print(f"Final MAE: {mae_history[-1]:.4f}")
        print(f"Parameter error - θ₀: {abs(4 - theta_0):.4f}, θ₁: {abs(3 - theta_1):.4f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    for n_samples in data_sizes:
        plt.plot(results[n_samples]['mae_history'], 
                label=f'n = {n_samples}', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Training Progress with Different Data Sizes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # TODO: Answer these questions:
    print("\nQuestions to answer:")
    print("1. How does data size affect convergence speed?")
    print("2. How does data size affect final model accuracy?")
    print("3. What is the relationship between data size and overfitting?")
    print("4. Why might more data not always lead to better performance?")
    
    return results

def exercise_5_custom_experiment():
    """
    Exercise 5: Custom Experiment
    
    Design your own experiment! Choose one of the following or create your own:
    1. Test different noise levels in the data
    2. Experiment with different true parameter values
    3. Compare MAE with other loss functions (MSE, Huber)
    4. Implement mini-batch gradient descent
    """
    print("\n" + "=" * 60)
    print("EXERCISE 5: CUSTOM EXPERIMENT")
    print("=" * 60)
    
    print("Choose one of the following experiments or create your own:")
    print("1. Noise level experiment")
    print("2. Different parameter values")
    print("3. Loss function comparison")
    print("4. Mini-batch gradient descent")
    print("5. Your own idea!")
    
    # Example: Noise level experiment
    print("\nExample: Noise Level Experiment")
    print("-" * 40)
    
    noise_levels = [0.1, 0.5, 1.0, 2.0]
    results = {}
    
    for noise in noise_levels:
        print(f"\nTesting with noise level = {noise}")
        
        # Generate data with different noise levels
        np.random.seed(42)
        X = 2 * np.random.rand(200, 1)
        y = 4 + 3 * X + noise * np.random.randn(200, 1)
        
        # Train model
        theta_0, theta_1, mae_history = train_linear_regression_mae(
            X, y, epochs=100, alpha=0.01, verbose=False
        )
        
        results[noise] = {
            'theta_0': theta_0,
            'theta_1': theta_1,
            'final_mae': mae_history[-1]
        }
        
        print(f"Final MAE: {mae_history[-1]:.4f}")
        print(f"Parameter error - θ₀: {abs(4 - theta_0):.4f}, θ₁: {abs(3 - theta_1):.4f}")
    
    # TODO: Design and implement your own experiment!
    print("\nNow it's your turn! Design and implement your own experiment.")
    print("Consider what aspects of gradient descent you want to explore further.")
    
    return results

def main():
    """
    Main function to run all practice exercises.
    """
    print("PRACTICE EXERCISES: GRADIENT DESCENT WITH MAE")
    print("Complete each exercise to master the concepts!")
    
    # Run exercises
    exercise_1_mae_calculation()
    exercise_2_gradient_computation()
    exercise_3_learning_rate_experiment()
    exercise_4_data_size_experiment()
    exercise_5_custom_experiment()
    
    print("\n" + "=" * 60)
    print("PRACTICE COMPLETE!")
    print("=" * 60)
    print("You've successfully completed all exercises!")
    print("Key takeaways:")
    print("- MAE provides robust error measurement")
    print("- Learning rate is crucial for convergence")
    print("- More data generally improves model performance")
    print("- Gradient descent iteratively optimizes parameters")
    print("- Experimentation is key to understanding ML concepts")

if __name__ == "__main__":
    main() 