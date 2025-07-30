import numpy as np
from gradient_descent_mulity_input_variable import gradient_descent
from cost_function import f, f_derivative
from data_generation import generate_noisy_multi_feature_data

if __name__ == '__main__':
    true_weights = np.array([5, 5, 5])
    true_intercept = -12
    x, y_true = generate_noisy_multi_feature_data(100, 3, true_weights, true_intercept)
    print(y_true.shape)
    print("true weights: ", true_weights)
    print("true intercept : ", true_intercept, "\n")
    ########## the code we will try to mimic is
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    model = LinearRegression()
    model.fit(x, y_true)

    model_coef_w = model.coef_
    model_intercept_w0 = model.intercept_

    y_pred = model.predict(x)
    model_mse = mean_squared_error(y_true, y_pred)
    print("model_coef_w: ", model_coef_w)
    print("model_intercept_w0: ", model_intercept_w0)
    print("model_mse: ", model_mse)

    model_my_mse = f(x,y_true, np.insert(model_coef_w, 0, model_intercept_w0))
    print('model_my_mse: ', model_my_mse * 2, '\n\n')




    ###########################################################################################
    ########## my code
    initial = np.array([-10,5,5,5], dtype='float64')
    max_iter = np.inf
    step_size = 0.00001

    final_wights = gradient_descent(f_derivative, y_true, x, initial, step_size=step_size)

    my_weights = final_wights
    my_mse = f(x, y_true, final_wights)

    print("my_weights: ", my_weights[1:])
    print("my_intercept_w0: ", my_weights[0])
    print("my_mse: ", my_mse)

    print("\nend")
    import seaborn as sns
    sns.bo




