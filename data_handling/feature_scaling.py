from ..common import np

def mean_normalization():
    return

"""
AKA rescaling or min-max scaling
"""
def min_max_normalization():
    return

"""
AKA standardization
After z-score normalization, all features will have mean of 0 and STD of 1
#quick note: axis=0 -> apply operation along each COLUMN; axis=1 -> apply operation along each ROW

notes:
- X - mu centers data for each feature around 0
- Dividing (X - mu) by sigma scales the centered data, i.e. ensures that all features have the same scale (same range of values),
  making them directly comparable. particularly useful when features have different ranges or units - ensures each feature contributes 
  proportionately to the overall variance of the data
- overall: z-score normalization transforms data into a distribution with mean of 0 and STD of 1. Makes data comparable across features 
           and helps some ML algorithms converge faster and perform better, as they are less sensitive to the scale of features (e.g. 
           gradient descent)
"""
def z_score_normalization(X):
    mu = np.mean(X, axis=0)     #(n,) vector containing feature means
    sigma = np.std(X, axis=0)   #(n,) vector containing feature STDs
    X_norm = (X-mu)/sigma       #(m,n) matrix containing normalized features
    return X_norm, mu, sigma    #keep mu and sigma for normalizing future new data/training examples

