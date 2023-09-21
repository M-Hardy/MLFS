import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
np.set_printoptions(precision=2)

X_train, y_train = [[1.24e+03, 3.00e+00, 1.00e+00, 6.40e+01]], [300.]
#X_features = ['size(sqft)','bedrooms','floors','age']

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)

sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)

b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")
print( "model parameters from previous lab: w: [110.56 -21.27 -32.71 -37.97], b: 363.16")

#make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm)

#make a prediction using w,b. 
y_pred = np.dot(X_norm, w_norm) + b_norm  
print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

print(f"Prediction on training set:\n{y_pred[:4]}" )
print(f"Target values \n{y_train[:4]}")