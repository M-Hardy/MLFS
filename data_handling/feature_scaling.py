from ..common import np

def mean_normalization():
    return

def min_max_normalization():
    return

def z_score_normalization(X):
    mu = np.mean(X, axis=0)     #(n,) vector containing feature means
    sigma = np.std(X, axis=0)   #(n,) vector containing feature STDs
    sigma[sigma == 0] = 1       # boolean mask on np.array: if sigma = 0, assign it to 1
    X_norm = (X-mu)/sigma       #(m,n) matrix containing normalized features
    return X_norm, mu, sigma    #keep mu and sigma for normalizing future new data/training examples

