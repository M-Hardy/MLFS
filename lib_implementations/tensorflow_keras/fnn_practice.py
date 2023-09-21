import tensorflow as tf
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from keras.optimizers import Adam

"""
tf binary classification model v1 notes:
-   for model.summary() to work tf needs to know input shape - can provide via explicit definition or 
    pass test data into the model (tf will infer the shape from the data); input shape required to 
    compute num params of 1st layer (num units * num features), and subsequently total params of model
    params = weight and bias parameters
"""
binary_classification_model_v1 = Sequential(
    [
        tf.keras.Input(shape=(400,)),    
        Dense(units=25, activation='sigmoid', name='layer1'),
        Dense(units=15, activation='sigmoid', name='layer2'),
        Dense(units=1, activation='sigmoid', name='layer3')
    ], 
    name = 'binary_classifcation_model_v1'
)

"""
tf binary classification model v1 notes:
- hidden layers use relu activation instead of sigmoid, faster computation and learning rate 
    (tldr: sigmoid has vanishing gradient problem, significantly slows down gradient descent due to tiny derivatives -> tiny update steps)
- output layer uses linear activation function & specifies 'sigmoid activation' in loss function to avoid numerical roundoff errors
    - NOTE: outputs of model will be linear transformations, to get probabilities you have to map model's outputs through logistic/
            sigmoid function -> tf.nn.sigmoid(logit), where logit = model output
"""
binary_classification_model_v2 = Sequential(
    [ 
        Dense(units=25, activation='relu', name='layer1'),
        Dense(units=15, activation='relu', name='layer2'),
        Dense(units=1, activation='linear', name='layer3')
    ], 
    name = 'binary_classification_model_v2'
)

binary_classification_model_v2.compile(loss=BinaryCrossentropy(from_logits=True))
#TO MAKE PREDICTION:
#logit = model(X)
#f_x = tf.nn.sigmoid(logit)


"""
v1 notes:
- susceptible to numerical roundoff error
"""
multiclass_classification_model_v1 = Sequential(
    [
        Dense(units=25, activation='relu', name='layer1'),
        Dense(units=15, activation='relu', name='layer2'),
        Dense(units=4, activation='softmax', name='layer3')
    ],
    name='multiclass_classification_model_v1'

)
multiclass_classification_model_v1.compile(loss=SparseCategoricalCrossentropy(),
                                           optimizer=Adam(0.001))


"""
v2 notes:
- fix numerical roundoff error
"""
multiclass_classification_model_v2 = Sequential(
    [
        Dense(units=25, activation='relu', name='layer1'),
        Dense(units=15, activation='relu', name='layer2'),
        Dense(units=4, activation='linear', name='layer3')
    ],
    name='multiclass_classification_model_v2'

)
multiclass_classification_model_v1.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                                           optimizer=Adam(0.001))
#Fit model to training data:
#model.fit(X_train, y_train, epochs=10)
