"""
compute gradient == compute partial derivatives; used interchangeably because gradient = vector of partial derivatives
partial derivative = rate of change of function w.r.t one variable, 
gradient = vector that encapsulates rates of change w.r.t. all variables
common that training examples of dataset contain multiple variables, so gradients are commonly computed; 
because common, terms are used interchangeably 

compute_gradient v1 implementation:
- assumes a simple linear regression model: f_wb = wx + b, only one feature per training example x[i]
"""
def compute_gradient(w,b,x,y):
    """
    :type w: int
    :type b: int
    :type x: ndarray(int)
    :type y: ndarray(int)
    :rtype: ndarray(int)
    """   

    m = x.shape(0)
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w*x[i] + b

        dj_dw_i = (f_wb - y) * x[i]
        dj_db_i = (f_wb - y)

        dj_dw += dj_dw_i
        dj_db += dj_db_i
    
    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


"""
- batch gradient descent = each step of gradient descent uses ALL the training examples
- other versions of gradient descent that look at subsets of the training data at each update step

run_batch_gradient_descent v1:
- assumes linear regression model -> gradient function computes gradients for MSE cost function (for regression problems)
"""
def run_batch_gradient_descent(x, y, w_in, b_in, alpha, cost_func, gradient_func, num_iters):

    w, b = w_in, b_in    

    for i in range(num_iters):
        dj_dw, dj_db = gradient_func(w, b, x, y)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        """
        Optional: Record J (cost) history and p history (parameter history)
        - necessary for metadata graphing & visibility
        - need to calculate cost, so can use cost_func arg in func sig
        """
        
    #return updated w and b - add J and p history as well later*
    return w, b