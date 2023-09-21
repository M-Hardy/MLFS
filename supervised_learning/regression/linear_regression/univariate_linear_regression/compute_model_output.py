import numpy as np

"""
v1 implementation:
- assumes only one feature per training example (i.e. each training example is a single feature (is a single int))
- non-vectorized implementation
"""
def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      y (ndarray (m,)): target values
    """
    m = x.shape[0]
    f_wb = np.zeros(m)

    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb   