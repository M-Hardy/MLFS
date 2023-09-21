import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

"""
4 things to remember outside of specifying the model:
1) feature scaling (normalization) 
2) split training data into training, cross-validation/dev, and test sets
    - apply feature scaling to all 3 sets; use mean & STD from training set to scale cross-val set
3) regularization
4) feature engineering
"""

def create_test_models():

    mnist_1 = Sequential(
        [
            Dense(units=25, activation='relu', name='layer1'),
            Dense(units=15, activation='relu', name='layer2'),
            Dense(units=10, activation='linear', name='layer3')
        ], name="mnist_1"
    )

    mnist_2 = Sequential(
        [
            Dense(units=100, activation='relu', name='layer1'),
            Dense(units=500, activation='relu', name='layer2'),
            Dense(units=25, activation='relu', name='layer3'),
            Dense(units=10, activation='linear', name='layer4'),
        ], name="mnist_2"
    )

    mnist_3 = Sequential(
        [
            Dense(units=50, activation='relu', name='layer1'),
            Dense(units=25, activation='relu', name='layer2'),
            Dense(units=36, activation='relu', name='layer3'),
            Dense(units=5, activation='relu', name='layer4'),
            Dense(units=18, activation='relu', name='layer6'),
            Dense(units=10, activation='linear', name='layer5')
        ], name="mnist_3"
    )

    return [mnist_1, mnist_2, mnist_3]

#LOAD DATA (relative path given)
data = np.array(pd.read_csv('../../../datasets/kaggle_mnist/fashion-mnist_train.csv'))
m, n = data.shape
#labels/targets stored in first column of csv in this dataset
x, y = data[:,1:], data[:,0]  

model_train_error = []
model_cv_error = []
def test_models(x, y, m, n):

    #CREATE TRAINING/TEST/CROSS-VALIDATION SETS
    #60% of data in x_train, y_train; 40% of data in x_, y_
    x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.40, random_state=1)
    #20% of (total) data in x_cv, y_cv; 20% of data in x_test, y_test
    x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

    #APPLY FEATURE SCALING
    scaler_linear = StandardScaler()
    x_train_scaled = scaler_linear.fit_transform(x_train)
    x_cv_scaled = scaler_linear.transform(x_cv)
    x_test_scaled = scaler_linear.transform(x_test)

    models = create_test_models()

    for model in models:
        model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer=Adam(learning_rate=0.001))
        model.fit(x_train_scaled, y_train, epochs=50)

        #record fraction of misclassified examples for training set
        yhat = tf.math.softmax(model.predict(x_train_scaled))
        yhat = np.argmax(yhat, axis=1)  #row-wise max, i.e. max for each training example
        train_error = np.mean(yhat != y_train)
        model_train_error.append(train_error)

        #record fraction of misclassified examples for cv set
        yhat = tf.math.softmax(model.predict(x_cv_scaled))
        yhat = np.argmax(yhat, axis=1)  #row-wise max, i.e. max for each training example
        cv_error = np.mean(yhat != y_cv)
        model_cv_error.append(cv_error)


test_models(x, y, m, n)
for i in range(len(model_train_error)):
    print(f"Model {i}: Training set classification error = {model_train_error[i]:.5f}, " +
    f"CV Set Classification Error: {model_cv_error[i]:.5f}")