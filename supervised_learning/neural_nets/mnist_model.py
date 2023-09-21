import numpy as np
import pandas as pd
from collections import defaultdict
import time
from ...utils import model_plt, model_io
from ...data_handling import data_loader, preprocessing, feature_scaling 
"""
Todo: 
- Add regularization to protect against overfitting
- Add batch processing to significantly improve training speed
- divide script into distinct model.py and train_model.py scripts 
    - *and any other scripts for other responsibilities - basically this script should only define model architecture
- add hyperparameter for recording model performance at X intervals
- should also add flag to print performance metrics during training
- rename CV_PROPORTION hyperparameter 
- should hyperparameters be put in a separate config file?
- QOL: Improve model plots
- test compare_prediction_with_image func - it should also just receive a prediction as opposed to 
  making one itself maybe?
"""

"""
FORWARD-PROP: 
- random initialization for param init (alternatives: Xavier/Glorot initialization, He initialization)
- layer explicitly defined: dense
- model explicitly defined: sequential, 3 dense layers
"""
def init_params(input_size, units):
    w = np.random.randn(input_size, units) * 0.01
    b = np.zeros((1, units))
    return w, b

def relu(z):
    g_z = np.maximum(0, z)
    return g_z

def softmax(z):
    exp_z = np.exp(z)             
    s_max = exp_z/np.sum(exp_z, axis=1, keepdims=True)    
    return s_max                   
    
def dense_layer(a_in, w, b, g):
    Z = np.matmul(a_in, w) + b
    A_out = g(Z)
    return Z, A_out

#sequential model, 3 dense layers
def three_layer_forward_prop(X, W1, b1, W2, b2, W3, b3, g1, g2, g3):
    Z1, A1 = dense_layer(X, W1, b1, g1)
    Z2, A2 = dense_layer(A1, W2, b2, g2)
    Z3, A3 = dense_layer (A2, W3, b3, g3)
    return Z1, A1, Z2, A2, Z3, A3

"""
BACK-PROP FUNCS: 
- One-hot encoding for y (*is there a way to avoid one-hot encoding y?)
"""
def softmax_cost(A_out, y):
    predictions = A_out[np.arange(y.size), y]
    softmax_loss = -np.log(predictions)
    return np.mean(softmax_loss)

def relu_deriv(Z):
    return Z > 0

def one_hot(y, classes):
    one_hot_y = np.zeros((y.size, classes))
    one_hot_y[np.arange(y.size), y] = 1
    return one_hot_y

def three_layer_backprop(Z1, A1, Z2, W2, A2, W3, A3, X, y):
    m = y.size
    one_hot_y = one_hot(y, A3.shape[1])
    #LAYER 3
    dj_dz3 = A3 - one_hot_y              
    dj_dw3 = np.matmul(A2.T, dj_dz3) / m   
    dj_db3 = np.sum(dj_dz3, axis=0) / m       
    #LAYER 2
    dj_dz2 = np.matmul(dj_dz3, W3.T) * relu_deriv(Z2)
    dj_dw2 = np.matmul(A1.T, dj_dz2) / m
    dj_db2 = np.sum(dj_dz2, axis=0) / m
    #LAYER 1
    dj_dz1 = np.matmul(dj_dz2, W2.T) * relu_deriv(Z1)
    dj_dw1 = np.matmul(X.T, dj_dz1) / m
    dj_db1 = np.sum(dj_dz1, axis=0) / m

    return dj_dw1, dj_db1, dj_dw2, dj_db2, dj_dw3, dj_db3

def update_params(W1, b1, W2, b2, W3, b3, dj_dw1, dj_db1, dj_dw2, dj_db2, dj_dw3, dj_db3, alpha):
    W1 -= alpha * dj_dw1
    b1 -= alpha * dj_db1
    W2 -= alpha * dj_dw2
    b2 -= alpha * dj_db2
    W3 -= alpha * dj_dw3
    b3 -= alpha * dj_db3
    return W1, b1, W2, b2, W3, b3

def run_gradient_descent(X, y, W1, b1, W2, b2, W3, b3, num_iters, alpha, print_performance): 
    model_performance = defaultdict(list)

    for i in range(num_iters + 1):
        Z1, A1, Z2, A2, Z3, A3 = three_layer_forward_prop(X, W1, b1, W2, b2, W3, b3, relu, relu, softmax)
        dj_dw1, dj_db1, dj_dw2, dj_db2, dj_dw3, dj_db3 = three_layer_backprop(Z1, A1, Z2, W2, A2, W3, A3, X, y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dj_dw1, dj_db1, dj_dw2, dj_db2, dj_dw3, dj_db3, alpha)

        if (i % 20 == 0):   #record cost & accuracy at 20th interval iterations
            cost = softmax_cost(A3, y)
            predictions = np.argmax(A3, axis=1)
            accuracy = np.sum(predictions == y) / y.size
            model_performance['iterations'].append(i)
            model_performance['cost'].append(cost)
            model_performance['accuracy'].append(accuracy) 
            if print_performance:
                print(f"Iteration: {i}, Cost: {cost:.5f}, Classification accuracy: {accuracy:.5f}")
          
    return W1, b1, W2, b2, W3, b3, model_performance


"""
MAIN FUNC: RUN MNIST MODEL
"""
def run_mnist_model(train_filepath, test_filepath, label_col_index, cv_proportion, l1_units, l2_units, l3_units, alpha, num_iters, print_performance):
    mnist_training_data = data_loader.load_data_from_csv(train_filepath)
    mnist_test_data = data_loader.load_data_from_csv(test_filepath)
    x_train, y_train, x_cv, y_cv = preprocessing.create_train_and_cv_sets(mnist_training_data, cv_proportion, label_col_index)
    normalized_x_train, train_mu, train_sigma = feature_scaling.z_score_normalization(x_train)
    normalized_x_cv = (x_cv - train_mu)/train_sigma
    x_test, y_test = mnist_test_data[:, 1:], mnist_test_data[:, 0] 
    normalized_x_test = (x_test - train_mu)/train_sigma

    n = x_train.shape[1]
    W1, b1 = init_params(n, l1_units)
    W2, b2 = init_params(l1_units, l2_units)
    W3, b3 = init_params(l2_units, l3_units)
    
    #train model on training set
    if print_performance:
        print("~~ MODEL PERFORMANCE ON TRAINING SET ~~")
    W1, b1, W2, b2, W3, b3, train_performance = run_gradient_descent(normalized_x_train, y_train, W1, b1, W2, b2, W3, b3, num_iters, alpha, print_performance)
   
    #train model on cv set
    if print_performance:
        print("\n~~ MODEL PERFORMANCE ON CROSS-VALIDATION SET ~~")
    W1, b1, W2, b2, W3, b3, cv_performance = run_gradient_descent(normalized_x_cv, y_cv, W1, b1, W2, b2, W3, b3, num_iters, alpha, print_performance)

    #test model on test set
    if print_performance:
        print("\n~~ MODEL PERFORMANCE ON TEST SET ~~")
    W1, b1, W2, b2, W3, b3, test_performance = run_gradient_descent(normalized_x_test, y_test, W1, b1, W2, b2, W3, b3, num_iters, alpha, print_performance)

    mnist_metrics = {}
    mnist_metrics['name'] = f"MLFS_MNIST_alpha={alpha}"
    mnist_metrics['l1_units'] = l1_units
    mnist_metrics['l2_units'] = l2_units
    mnist_metrics['l3_units'] = l3_units
    mnist_metrics['weight_params'] = [W1, W2, W3]
    mnist_metrics['bias_params'] = [b1, b2, b3]
    mnist_metrics['alpha'] = alpha
    mnist_metrics['num_iters'] = num_iters
    mnist_metrics['train_performance'] = train_performance
    mnist_metrics['cv_performance'] = cv_performance
    mnist_metrics['test_performance'] = test_performance

    return mnist_metrics

"""
HYPERPARAMETERS & TESTING FUNCS 
"""
MNIST_TRAIN_PATH = 'datasets/kaggle_mnist/fashion-mnist_train.csv'   
MNIST_TEST_PATH = 'datasets/kaggle_mnist/fashion-mnist_test.csv'    
METRICS_DIR_PATH = 'MLFS/supervised_learning/model_metrics'
MNIST_LABEL_COL_INDEX = 0
TEST_ALPHA_ONE = [0.1]
TEST_ALPHAS_LESS = [0.07, 0.1, 0.12]
TEST_ALPHAS_MORE = [0.01, 0.03, 0.05, 0.07, 0.1, 0.12, 0.14]
NUM_ITERS = 1
L1_UNITS, L2_UNITS, L3_UNITS = 50, 25, 10
CV_PROPORTION = 4   #1/CV_PROPORTION of training data will be used for cross-validation set
PRINT_PERFORMANCE = True

def get_predictions(A_out):
    return np.argmax(A_out, axis=1)

def make_prediction(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = three_layer_forward_prop(X, W1, b1, W2, b2, W3, b3, g1=relu, g2=relu, g3=softmax)
    predictions = get_predictions(A3)
    return predictions

def compare_prediction_with_image(i, X, y, W1, b1, W2, b2, W3, b3):
    img = X[i, 1:]
    label = y[i:0]
    prediction = make_prediction(X[i:], W1, b1, W2, b2)
    print("Prediction: ", prediction)
    print("Label: ", label)
    model_plt.show_target_image(img)  

def test_learning_rates(mnist_train_path, mnist_test_path, mnist_label_col_index, cv_proportion, l1_units, l2_units, l3_units, test_alphas, num_iters, print_performance):
    all_model_metrics = []
    for test_alpha in test_alphas:
        start_time = time.time()
        model_metrics = run_mnist_model(mnist_train_path, mnist_test_path, mnist_label_col_index, cv_proportion, l1_units, l2_units, l3_units, test_alpha, num_iters, print_performance)
        end_time = time.time()
        model_metrics['approx_training_time'] = end_time - start_time
        all_model_metrics.append(model_metrics)
    return all_model_metrics

all_model_metrics = test_learning_rates(MNIST_TRAIN_PATH, MNIST_TEST_PATH, CV_PROPORTION, L1_UNITS, L2_UNITS, L3_UNITS, TEST_ALPHAS_MORE, NUM_ITERS, PRINT_PERFORMANCE)
model_io.save_model_metrics(all_model_metrics, METRICS_DIR_PATH, 'mnist')