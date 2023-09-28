from ...common import np
from ...utils import model_io
from ...data_handling import data_loader
from ...supervised_learning.neural_nets.mnist_model import three_layer_forward_prop, relu, softmax
from ...model_visualization import model_plt
import random

TS_MODELS_METADATA_DIRPATH = r'MLFS\supervised_learning\model_metadata\mnist_2023-09-27_17-30'
MODEL_FILENAME = r'MLFS_MNIST_alpha=0.2'
TEST_X_FILEPATH = r'datasets\nrippner-mnist-handwritten-digits\MNIST_data_test.csv'
TEST_Y_FILEPATH = r'datasets\nrippner-mnist-handwritten-digits\MNIST_target_test.csv'

def get_softmax_predictions(A_out):
    return np.argmax(A_out, axis=1)

def predict_model_on_mnist_test_set(timestamped_models_metadata_dirpath, model_filename, test_x_filepath, test_y_filepath):
    model = model_io.load_model_metadata(timestamped_models_metadata_dirpath, model_filename)
    W1, W2, W3 = model['weight_params']
    b1, b2, b3 = model['bias_params']
    train_mu = model['train_mu']
    train_sigma = model['train_sigma']
    
    x_test = data_loader.load_data_from_csv(test_x_filepath)
    normalized_x_test = ((x_test - train_mu)/train_sigma).astype(int)
    y_test = data_loader.load_data_from_csv(test_y_filepath).astype(int).flatten()
    _, _, _, _, _, A_out = three_layer_forward_prop(normalized_x_test, W1, b1, W2, b2, W3, b3, relu, relu, softmax)
    predictions = get_softmax_predictions(A_out)
    accuracy = np.sum(predictions == y_test) / y_test.size
    return predictions, accuracy

def compare_prediction_with_image(example_i, test_x_filepath, test_y_filepath, timestamped_models_metadata_dirpath, model_filename):
    model = model_io.load_model_metadata(timestamped_models_metadata_dirpath, model_filename)
    W1, W2, W3 = model['weight_params']
    b1, b2, b3 = model['bias_params']
    train_mu, train_sigma = model['train_mu'], model['train_sigma']
    X = data_loader.load_data_from_csv(test_x_filepath)
    normalized_x_test = (X - train_mu)/train_sigma
    y_test = data_loader.load_data_from_csv(test_y_filepath)
    
    img_vector = normalized_x_test[example_i].astype(int)
    label = y_test[example_i].astype(int)
    
    output = three_layer_forward_prop(img_vector, W1, b1, W2, b2, W3, b3, relu, relu, softmax)
    prediction = get_softmax_predictions(output)
    print("Prediction: ", prediction)
    print("Label: ", label)
    model_plt.show_target_image(img_vector)

predictions, accuracy = predict_model_on_mnist_test_set(TS_MODELS_METADATA_DIRPATH, MODEL_FILENAME, TEST_X_FILEPATH, TEST_Y_FILEPATH)
print(f"Model: {MODEL_FILENAME}")
print(f"Test Set Accuracy = {accuracy}")
rand_img_num = random.randint(0, 10000)
print(f"Test Random Image - Image #{rand_img_num}")
compare_prediction_with_image(rand_img_num, TEST_X_FILEPATH, TEST_Y_FILEPATH, TS_MODELS_METADATA_DIRPATH, MODEL_FILENAME)