from ...common import np
from ...utils import model_io
from ...data_handling import data_loader
from ...supervised_learning.neural_nets.mnist_model import three_layer_forward_prop, relu, softmax, softmax_cost
from ...model_visualization import model_plt
import random

TS_MODELS_METADATA_DIRPATH = r'MLFS\supervised_learning\model_metadata\mnist_2023-09-27_17-30'
MODEL_FILENAME = r'MLFS_MNIST_alpha=0.2'
TEST_X_FILEPATH = r'datasets\nrippner-mnist-handwritten-digits\MNIST_data_test.csv'
TEST_Y_FILEPATH = r'datasets\nrippner-mnist-handwritten-digits\MNIST_target_test.csv'

def get_softmax_predictions(A_out):
    return np.argmax(A_out, axis=1)

def predict_model_on_mnist_test_set(model_metadata, test_x_filepath, test_y_filepath):
    W1, W2, W3 = model_metadata['weight_params']
    b1, b2, b3 = model_metadata['bias_params']
    train_mu = model_metadata['train_mu']
    train_sigma = model_metadata['train_sigma']
    
    x_test = data_loader.load_data_from_csv(test_x_filepath)
    normalized_x_test = ((x_test - train_mu)/train_sigma).astype(int)
    y_test = data_loader.load_data_from_csv(test_y_filepath).astype(int).flatten()
    _, _, _, _, _, A_out = three_layer_forward_prop(normalized_x_test, W1, b1, W2, b2, W3, b3, relu, relu, softmax)
    predictions = get_softmax_predictions(A_out)
    accuracy = np.sum(predictions == y_test) / y_test.size
    cost = softmax_cost(A_out, y_test)
    return cost, accuracy, predictions 

def compare_prediction_with_image(example_i, test_x_filepath, test_y_filepath, timestamped_models_metadata_dirpath, model_filename):
    model = model_io.load_model_metadata(timestamped_models_metadata_dirpath, model_filename)
    W1, W2, W3 = model['weight_params']
    b1, b2, b3 = model['bias_params']
    train_mu, train_sigma = model['train_mu'], model['train_sigma']
    X_test = data_loader.load_data_from_csv(test_x_filepath)
    y_test = data_loader.load_data_from_csv(test_y_filepath)
    
    normalized_x_test = (X_test - train_mu)/train_sigma
    img_vector = normalized_x_test[example_i].astype(int)
    label = y_test[example_i].astype(int)
    #print(f"img_vector = {img_vector}")
    output = three_layer_forward_prop(img_vector, W1, b1, W2, b2, W3, b3, relu, relu, softmax)
    #print(f"Output: {output}")
    prediction = get_softmax_predictions(output)
    print("Prediction: ", prediction)
    print("Label: ", label)
    model_plt.show_target_image(img_vector)

def get_model_dir_performance_metrics(timestamped_models_metadata_dirpath, test_x_filepath, test_y_filepath):
    all_models_performance_metrics = {}
    all_models_metadata = model_io.load_model_metadata_directory(timestamped_models_metadata_dirpath)
    for model_metadata in all_models_metadata:
        train_cost, train_accuracy = model_metadata['train_performance']['cost'][-1], model_metadata['train_performance']['accuracy'][-1]
        cv_cost, cv_accuracy = model_metadata['cv_performance']['cost'], model_metadata['cv_performance']['accuracy']
        
        test_cost, test_accuracy, _ = predict_model_on_mnist_test_set(model_metadata, test_x_filepath, test_y_filepath)
        
        model_name = model_metadata['name']
        all_models_performance_metrics[model_name] = {}
        all_models_performance_metrics[model_name]['train_cost'] = train_cost
        all_models_performance_metrics[model_name]['train_accuracy'] = train_accuracy
        all_models_performance_metrics[model_name]['cv_cost'] = cv_cost
        all_models_performance_metrics[model_name]['cv_accuracy'] = cv_accuracy
        all_models_performance_metrics[model_name]['test_cost'] = test_cost
        all_models_performance_metrics[model_name]['test_accuracy'] = test_accuracy

    return all_models_performance_metrics


def print_models_cv_and_test_performance(all_models_cv_test_performance):
    for model_name, model_performance_dict in all_models_cv_test_performance.items():
        print(f"~~ {model_name} PERFORMANCE METRICS ~~")
        print(f"TRAINING COST: {model_performance_dict['train_cost']}, TRAINING ACCURACY: {model_performance_dict['train_accuracy']}")
        print(f"CV COST: {model_performance_dict['cv_cost']}, CV ACCURACY: {model_performance_dict['cv_accuracy']}")
        print(f"TEST COST: {model_performance_dict['test_cost']}, TEST ACCURACY: {model_performance_dict['test_accuracy']}")
        print()


all_models_cv_test_performance = get_model_dir_performance_metrics(TS_MODELS_METADATA_DIRPATH, TEST_X_FILEPATH, TEST_Y_FILEPATH)
print(print_models_cv_and_test_performance(all_models_cv_test_performance))

# predictions, accuracy = predict_model_on_mnist_test_set(TS_MODELS_METADATA_DIRPATH, MODEL_FILENAME, TEST_X_FILEPATH, TEST_Y_FILEPATH)
# print(f"Model: {MODEL_FILENAME}")
# print(f"Test Set Accuracy = {accuracy}")
# rand_img_num = random.randint(0, 10000)
# print(f"Test Random Image - Image #{rand_img_num}")
# compare_prediction_with_image(rand_img_num, TEST_X_FILEPATH, TEST_Y_FILEPATH, TS_MODELS_METADATA_DIRPATH, MODEL_FILENAME)