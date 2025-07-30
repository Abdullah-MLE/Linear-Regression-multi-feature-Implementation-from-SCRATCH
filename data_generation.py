import numpy as np

def generate_noisy_multi_feature_data(num_points, num_features, coef, intercept, noise_scale=1.0):
    """
    Generates synthetic data for a multi-feature linear regression model with noise.

    Args:
        num_points (int): The number of data samples (rows).
        num_features (int): The number of features (columns for X).
        coef (np.ndarray): A 1D array of coefficients (slopes) for each feature.
                          Its length must match num_features.
        intercept (float): The intercept (bias) value.
        noise_scale (float, optional): The scale (standard deviation) of the added noise.
                                       Defaults to 1.0.

    Returns:
        tuple: A tuple containing:
            - x (np.ndarray): A 2D array of input features.
                              Shape: (num_points, num_features).
            - y_noisy (np.ndarray): A 2D array of noisy target values.
                                   Shape: (num_points, 1).
    """
    # 1. Generate random input features (X)
    # Each feature column will have values between 0 and 10 for simplicity
    x = np.random.uniform(0, 10, size=(num_points, num_features))

    # Ensure coefficients are in the correct shape for matrix multiplication
    # If coef is (num_features,), reshape it to (num_features, 1)
    if coef.ndim == 1:
        coef = coef.reshape(-1, 1)
    elif coef.shape[1] != 1:
        raise ValueError("Coefficients array should be 1D or a column vector.")

    # 2. Calculate the true target values (y_true) without noise
    # This is a matrix multiplication: X @ coef + intercept
    y_true = x @ coef + intercept

    # 3. Generate random noise
    noise = np.random.normal(loc=0, scale=noise_scale, size=(num_points, 1))

    # 4. Add noise to the true target values
    y_noisy = y_true + noise

    return x, y_noisy

# --- Example Usage ---
if __name__ == "__main__":
    num_samples = 100
    num_input_features = 3 # Let's have 3 features (columns)

    # Define coefficients for each feature.
    # It must match num_input_features.
    true_coefficients = np.array([2.5, -1.0, 0.5])
    true_intercept = 50.0
    noise_level = 5.0 # Small noise for a clear linear relationship

    X_data, Y_data = generate_noisy_multi_feature_data(
        num_samples, num_input_features, true_coefficients, true_intercept, noise_level
    )

    print(f"Shape of X (features): {X_data.shape}") # Expected: (100, 3)
    print(f"Shape of Y (target values): {Y_data.shape}") # Expected: (100, 1)

    print("\nFirst 5 rows of X:")
    print(X_data[:5])

    print("\nFirst 5 rows of Y:")
    print(Y_data[:5])

    # You can verify the relationship for the first sample:
    # y_pred_for_first_sample = (X_data[0, 0] * true_coefficients[0] +
    #                            X_data[0, 1] * true_coefficients[1] +
    #                            X_data[0, 2] * true_coefficients[2] + true_intercept)
    # print(f"\nPredicted for first sample (without noise): {y_pred_for_first_sample}")
    # print(f"Actual noisy Y for first sample: {Y_data[0, 0]}")