from ...common import np
from collections import defaultdict
import time
from ...utils import model_io
from ...data_handling import data_loader, data_splitting, feature_scaling 


"""
HYPERPARAMETERS
"""
MNIST_TRAIN_X_PATH = r'datasets\nrippner-mnist-handwritten-digits\MNIST_data_train.csv'   
MNIST_TRAIN_Y_PATH = r'datasets\nrippner-mnist-handwritten-digits\MNIST_target_train.csv'    
MNIST_TEST_X_PATH = r'datasets\nrippner-mnist-handwritten-digits\MNIST_data_test.csv' 
MNIST_TEST_Y_PATH = r'datasets\nrippner-mnist-handwritten-digits\MNIST_target_test.csv' 
MODEL_METADATA_DIR_PATH = r'MLFS\supervised_learning\model_metadata'
MNIST_LABEL_COL_INDEX = 0
TEST_ALPHA_ONE = [0.2]
TEST_ALPHAS_LESS = [0.12, 0.1, 0.07]
TEST_ALPHAS_MORE = [0.07, 0.1, 0.13, 0.16, 0.2]
NUM_ITERS = 500
L1_UNITS, L2_UNITS, L3_UNITS = 50, 25, 10
CV_PROPORTION = 5   #1/CV_PROPORTION of training data will be used for cross-validation set
PRINT_PERFORMANCE = True


"""
FORWARD-PROP: 
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

def three_layer_forward_prop(X, W1, b1, W2, b2, W3, b3, g1, g2, g3):
    Z1, A1 = dense_layer(X, W1, b1, g1)
    Z2, A2 = dense_layer(A1, W2, b2, g2)
    Z3, A3 = dense_layer (A2, W3, b3, g3)
    return Z1, A1, Z2, A2, Z3, A3


"""
BACK-PROP FUNCS: 
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
    training_performance = defaultdict(list)
    
    for i in range(num_iters + 1):
        Z1, A1, Z2, A2, Z3, A3 = three_layer_forward_prop(X, W1, b1, W2, b2, W3, b3, relu, relu, softmax)
        dj_dw1, dj_db1, dj_dw2, dj_db2, dj_dw3, dj_db3 = three_layer_backprop(Z1, A1, Z2, W2, A2, W3, A3, X, y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dj_dw1, dj_db1, dj_dw2, dj_db2, dj_dw3, dj_db3, alpha)

        if (i % 20 == 0):   #record cost & accuracy at 20th interval iterations
            cost = softmax_cost(A3, y)
            predictions = np.argmax(A3, axis=1)
            accuracy = np.sum(predictions == y) / y.size
            training_performance['iterations'].append(i)
            training_performance['cost'].append(cost)
            training_performance['accuracy'].append(accuracy) 
            if print_performance:
                print(f"Iteration: {i}, Cost: {cost:.5f}, Classification accuracy: {accuracy:.5f}")
          
    return W1, b1, W2, b2, W3, b3, training_performance


"""
MAIN FUNC: RUN MNIST MODEL
"""
def validate_mnist_model(normalized_cv_x, cv_y, W1, b1, W2, b2, W3, b3, print_performance):
    _, _, _, _, _, A_out = three_layer_forward_prop(normalized_cv_x, W1, b1, W2, b2, W3, b3, relu, relu, softmax)
    predictions = np.argmax(A_out, axis=1)
    accuracy = np.sum(predictions == cv_y) / cv_y.size
    cost = softmax_cost(A_out, cv_y)
    cv_performance = {}

    if print_performance:
        print(f"Cost: {cost}, Accuracy: {accuracy}")
    
    cv_performance['cost'] = cost
    cv_performance['accuracy'] = accuracy
    cv_performance['predictions'] = predictions
    return cv_performance

def run_mnist_model(train_x_filepath, train_y_filepath, label_col_index, cv_proportion, l1_units, l2_units, l3_units, alpha, num_iters, print_performance):
    mnist_train_x = data_loader.load_data_from_csv(train_x_filepath)
    mnist_train_y = data_loader.load_data_from_csv(train_y_filepath)
    mnist_training_data = data_splitting.merge_x_and_y_sets(mnist_train_x, mnist_train_y)
    
    #rewrite create_train_and_cv_sets to take x_vec and y_vec
    x_train, train_y, x_cv, cv_y = data_splitting.create_train_and_cv_sets(mnist_training_data, cv_proportion, label_col_index)
    normalized_train_x, train_mu, train_sigma = feature_scaling.z_score_normalization(x_train)
    normalized_cv_x = (x_cv - train_mu)/train_sigma

    #create subroutine for this block - also probably best to do this first before splitting data into multiple sets
    normalized_train_x = normalized_train_x.astype(int)
    normalized_cv_x = normalized_cv_x.astype(int)
    train_y = train_y.astype(int)
    cv_y = cv_y.astype(int)

    n = x_train.shape[1]
    W1, b1 = init_params(n, l1_units)
    W2, b2 = init_params(l1_units, l2_units)
    W3, b3 = init_params(l2_units, l3_units)

    #train model on training set
    if print_performance:
        print(f"~~ MODEL PERFORMANCE ON TRAINING SET - ALPHA={alpha} ~~")
    W1, b1, W2, b2, W3, b3, train_performance = run_gradient_descent(normalized_train_x, train_y, W1, b1, W2, b2, W3, b3, num_iters, alpha, print_performance)
    
    #test model on cv set:
    if print_performance:
        print(f"\n~~ MODEL PERFORMANCE ON CROSS-VALIDATION SET - ALPHA={alpha} ~~")
    cv_performance = validate_mnist_model(normalized_cv_x, cv_y, W1, b1, W2, b2, W3, b3, print_performance)

    mnist_metadata = {}
    mnist_metadata['name'] = f"MLFS_MNIST_alpha={alpha}"
    mnist_metadata['l1_units'] = l1_units
    mnist_metadata['l2_units'] = l2_units
    mnist_metadata['l3_units'] = l3_units
    mnist_metadata['weight_params'] = [W1, W2, W3]
    mnist_metadata['bias_params'] = [b1, b2, b3]
    mnist_metadata['alpha'] = alpha
    mnist_metadata['num_iters'] = num_iters
    mnist_metadata['train_performance'] = train_performance
    mnist_metadata['cv_performance'] = cv_performance
    mnist_metadata['train_mu'] = train_mu
    mnist_metadata['train_sigma'] = train_sigma
    return mnist_metadata

"""
TRAIN W/ DIFFERENT LEARNING RATES  
"""
def train_learning_rates(mnist_train_x_path, mnist_train_y_path, mnist_label_col_index, cv_proportion, 
                         l1_units, l2_units, l3_units, test_alphas, num_iters, print_performance):
    all_model_metadata = []
    for alpha in test_alphas:
        start_time = time.time()
        model_metadata = run_mnist_model(mnist_train_x_path, mnist_train_y_path, mnist_label_col_index, cv_proportion, 
                                        l1_units, l2_units, l3_units, alpha, num_iters, print_performance)
        end_time = time.time()
        model_metadata['approx_training_time'] = end_time - start_time
        all_model_metadata.append(model_metadata)
    return all_model_metadata

#all_model_metadata = train_learning_rates(MNIST_TRAIN_X_PATH, MNIST_TRAIN_Y_PATH, MNIST_LABEL_COL_INDEX, CV_PROPORTION, 
#                                         L1_UNITS, L2_UNITS, L3_UNITS, TEST_ALPHA_ONE, NUM_ITERS, PRINT_PERFORMANCE)
#model_io.save_model_metadata(all_model_metadata, MODEL_METADATA_DIR_PATH, 'mnist')