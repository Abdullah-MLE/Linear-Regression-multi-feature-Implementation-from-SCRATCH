from numpy.linalg import norm
import numpy as np

def gradient_descent(f_deriv, y_true, x, initial, step_size = 0.001, precision = 0.001, max_iter = 1000):
    cur = initial
    last = np.full(cur.shape[0], np.inf)
    ls = [cur]             # let's maintain our x movements

    iter = 0
    while norm(cur - last) > precision and iter < max_iter:
        last = cur

        gradient = f_deriv(x, y_true, cur)
        cur = cur - gradient * step_size   # move in opposite direction

        ls.append(cur)            # keep copy of what we visit
        iter += 1

    return cur