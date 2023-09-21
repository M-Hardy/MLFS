import numpy as np
import copy

def sigmoid(z):
    sig = 1 / (1 + np.exp(-z))   #g(z) = 1 / (1 + e^-z), where z = w*x = b
    return sig


"""
Logistic cost function v1 - *loss function used by cost function is ONLY thing different from
                             multi lin reg implementation (cost still mean loss over data set, 
                             compute_gradient & run_batch_gradient_descent implementations are the exact same)

*what is the formal term for the logistic regression cost function?
"""
def log_cost(X, y, w, b):
    m=X.shape[0]
    mean_cost = 0.0
    for i in range(m):
        z_i = np.dot(w, X[i]) + b
        f_wb_i = sigmoid(z_i)
        
        loss = -y[i] * np.log(f_wb_i) - (1 - y[i]) * (np.log(1 - f_wb_i))
        mean_cost += loss
    mean_cost /= m
    return mean_cost


"""
logistic compute gradient - *same as multi lin reg compute gradient
"""
def log_compute_gradient(X, y, w, b):
    m,n = X.shape
    dj_dw = np.zeroes((n,)) 
    dj_db = 0.0                 #safer to instantiate as a float

    for i in range(m):
        f_wb_i = sigmoid(np.dot(w, X[i]) + b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += err_i * X[i, j]
        dj_db += err_i
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db


"""
regularized logistic compute gradient - *same as multi lin reg compute gradient
"""
def regularized_log_compute_gradient(X, y, w, b, lambda_):
    m,n = X.shape
    dj_dw = np.zeroes((n,))
    dj_db = 0.
    for i in range(m):
        f_wb_i = sigmoid(np.dot(w, X[i]) + b)
        err = f_wb_i - y[i]
        
        #vectorized: dj_dw += err * X[i]
        for j in range(n):
            dj_dw[j] += err * X[i, j]
        
        dj_db += err
    
    dj_dw /= m
    dj_db /= m
    dj_dw += (lambda_ / m) * w
    return dj_dw, dj_db

"""
logistic batch gradient descent v1
"""
def run_logistic_batch_gradient_descent(X, y, w_in, b_in, grad_func, alpha, num_iters):

    #avoid modifying globals within function
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = grad_func(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        """
        Optional: Record J (cost) history and p history (parameter history);
                  ALSO necessary to tell if grad descent is working properly & ~ num of iterations needed to converge 
                  (to optimize num_iters and overall speed of grad descent) -> learnt via plotting learning curve (cost over iterations)
        - necessary for metadata graphing & visibility
        - need to pass a cost_func to function --> need to add a cost_func arg to function signature
        """

    return w, b #J_history

def predict(X, w, b): 
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w
    
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model

    Returns:
      p : (ndarray (m,)) The predictions for X using a threshold at 0.5
    """

    m = X.shape[0]
    p = np.zeros(m)
    
    for i in range(m):
        prediction = sigmoid(np.dot(X[i], w) + b)
        p[i] = 1 if prediction >= 0.5 else 0
    
    return p