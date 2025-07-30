import numpy as np

def f(x, y_true, weights):
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    length = x.shape[0]

    y_pred = np.dot(x, weights).reshape(-1,1)
    error = y_pred - y_true

    cost = error.T.dot(error) / (2 * length)  # dot product is WAY faster
    return cost


def f_derivative(x, y_true,  weights):
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    length = x.shape[0]

    y_pred = np.dot(x, weights).reshape(-1,1)  # (100,3) . (3) => (100)
    error = y_pred - y_true      # (100) - (100) => (100)

    gradient = x.T @ error /length  # (3,100) . (100,1) => (3,1)

    return gradient.reshape(-1)


# if __name__ == '__main__':
#     X = np.array([[0, 0],
#                   [0.2, 0.2],
#                   [0.4, 0.4],
#                   [0.8, 0.8],
#                   [1.0, 1.0]])
#     t = (X.sum(axis=1) + 5).reshape(-1, 1)
#
#     weights = np.array([1.0, 1.0, 1.0])  # starting params
#
#     print(f(X, t, weights))  # cost: 8
#
#     print(f_derivative(X, t, weights))



