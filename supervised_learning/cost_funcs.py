from ..common import np

"""
MSE is applicable for regression problems (not classification - iirc does not yield convex function)

MSE v1:
- only for training data where training examples only have ONE feature
- non-vectorized
"""
def mean_squared_error_non_vectorized(x, y, w, b):
    """
    :type x: ndarray(int)
    :type y: ndarray(int)
    :type w: int
    :type b: int
    :rtype: int
    """
    m = len(x)
    mean_squared_error = 0
    for i in range(m):
        f_wb = w * x[i] + b
        mean_squared_error += (f_wb - y[i]) ** 2
    mean_squared_error /= (2 * m)
    return mean_squared_error

"""
MSE v2:
- model prediction is vectorized, handles training examples with multiple features
- not *fully* vectorized - still loops dataset for each training example
"""
def mean_squared_error_partially_vectorized(X, y, w, b):

    m = X.shape[0]
    mean_squared_error = 0
    for i in range(m):
        f_wb = np.dot(w, X[i]) + b
        mean_squared_error += (f_wb - y[i]) ** 2
    mean_squared_error /= (2 * m)
    return mean_squared_error

"""
MSE v3:
- model prediction is vectorized, handles training examples with multiple features
- not *fully* vectorized - still loops dataset for each training example
- regularization applied to circumvent possible overfitting - regularization term added to cost; added via loop, not vectorized
"""
def regularized_mse_partially_vectorized(X, y, w, b, lambda_=1):
    m,n = X.shape
    mean_squared_error = 0.
    for i in range(m):
        f_wb_i = np.dot(w, X[i]) + b
        mean_squared_error += (f_wb_i - y[i]) ** 2
    mean_squared_error /= (2 * m)
    
    reg_term = 0
    for i in range(n): 
        reg_term += w[i] ** 2
    reg_term *= (lambda_ / (2 * m))

    regularized_mse = mean_squared_error + reg_term
    return regularized_mse


def sigmoid(z):
    g_z = 1 / (1 + np.exp(-z))
    return g_z 

def regularized_sig_cost(X, y, w, b, lambda_=1):
    m,n = X.shape
    cost = 0.
    for i in range(m):
        f_wb_i = sigmoid(np.dot(w, X[i]) + b)
        loss_i = (-y[i]*np.log(f_wb_i)) - ((1 - y[i]) * (np.log(1 - f_wb_i)))
        cost += loss_i
    cost /= (2 * m)

    reg_term = 0.
    for i in range(n):
        reg_term += w[i] ** 2
    reg_term *= (lambda_ / (2 * m))
    
    regularized_log_cost = cost + reg_term
    return regularized_log_cost

