import numpy as np
import copy

def single_prediction(w, b, X):
    pred = np.dot(w, X) + b
    return pred

"""
compute gradient v1:
- m = num training examples, n = num features per training example
- dj_dw = vector containing w parameters
- for each training example: calcuate error, compute gradient for each weight parameter (err * respective feature) and 
  sum in weight term (w[j] in dj_dw vector), then compute gradient for bias parameter and sum in bias term
- compute avg of weight & bias gradients, then return avg gradients

note: assumes linear regression model
"""
def compute_gradient(w,b,X,y):
    
    m,n = X.shape
    dj_dw, dj_db = np.zeros((n,)), 0
    for i in range(m):
        err = (np.dot(w, X[i]) + b) - y
        for j in range(n):
            dj_dw[j] += (err * X[i, j])
        dj_db += err

    dj_dw, dj_db = dj_dw / m , dj_db / m
    return dj_dw, dj_db

"""
*note with regularization terms - they are DISTINCT from gradient & loss terms in compute_gradient and cost_funcs;
                                  you just add them to the original cost/gradient terms separately (i.e. keep the 
                                  calculations of each term distinct, don't apply mean to both terms)

"""
def compute_regularized_gradient(w, b, X, y, lambda_=1):
    m,n = X.shape
    dj_dw = np.zeroes((n,))
    dj_db = 0

    for i in range(m):
        loss = np.dot(w, X[i]) + b
        for j in range(n):
            dj_dw[j] += (loss - y[i]) * X[i, j]
        dj_db += b
    
    dj_dw /= m
    dj_db /= m

    #add regularization term for each weight parameter to mean gradient w.r.t each parameter
    #*vectorized implementation - i think this is correct
    dj_dw += (lambda_ / m) * w
    return dj_dw, dj_db



def run_batch_gradient_descent(w_in, b_in, X, y, alpha, gradient_func, num_iters):

    w, b = copy.deepcopy(w_in), b_in    #have to create deep copy of array or else will assign reference and modify global within func

    for i in range(num_iters):
        dj_dw, dj_db = gradient_func(w,b,X,y)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        """
        Optional: Record J (cost) history and p history (parameter history);
                  ALSO necessary to tell if grad descent is working properly & ~ num of iterations needed to converge 
                  (to optimize num_iters and overall speed of grad descent) -> learnt via plotting learning curve (cost over iterations)
        - necessary for metadata graphing & visibility
        - need to pass a cost_func to function --> need to add a cost_func arg to function signature
        """

    #return updated w and b - add J and p history as well later*
    return w, b