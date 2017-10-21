from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np

def next_parameter_by_ei(y_min, y_mean, y_std, x_choices):
    # Calculate expecte improvement from 95% confidence interval
    expected_improvement = y_min - (y_mean - 1.96 * y_std)
    expected_improvement[expected_improvement < 0] = 0

    max_index = expected_improvement.argmax()
    # Select next choice
    next_parameter = x_choices[max_index]

    return next_parameter


def vector_2d(array):
    return np.array(array).reshape((-1, 1))

def gaussian_process(x_train, y_train, x_test):
    x_train = vector_2d(x_train)
    y_train = vector_2d(y_train)
    x_test = vector_2d(x_test)

    # Train gaussian process
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) #adjust kernel for better results?
    gp = GaussianProcessRegressor(kernel=kernel)
    gp.fit(x_train, y_train)

    # Get mean and standard deviation for each possible
    # number of hidden units
    y_mean, y_var = gp.predict(x_test, return_std=True)
    y_std = np.sqrt(vector_2d(y_var))

    return y_mean, y_std
