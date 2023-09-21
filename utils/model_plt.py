import matplotlib.pyplot as plt

COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'pink', 'gray']

def plot_mnist_model_costs(models_diagnostics):
    for i, model in enumerate(models_diagnostics):
        test_performance = model['test_performance'] #array of tuples: (iteration, cost, accuracy)
        iterations = test_performance['iterations']
        cost = test_performance['cost']
        model_name = model['name']
        color = COLORS[i % len(COLORS)]
        plt.plot(iterations, cost, label=f"{model_name}", color=color)
    plt.xlabel('Iterations/Epochs')
    plt.ylabel('Cost')
    plt.legend()
    plt.title('Model Performance Comparison: Model Cost')
    plt.show()

def plot_mnist_model_accuracies(models_diagnostics):
    for i, model in enumerate(models_diagnostics):
        test_performance = model['test_performance'] #array of tuples: (iteration, cost, accuracy)
        iterations = test_performance['iterations']
        accuracy = test_performance['accuracy']
        model_name = model['name']
        color = COLORS[i % len(COLORS)]
        plt.plot(iterations, accuracy, label=f"{model_name}", color=color)
    plt.xlabel('Iterations/Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Performance Comparison: Model Accuracy')
    plt.show()

def show_target_image(image):
    image = image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(image, interpolation='nearest')
    plt.show()
