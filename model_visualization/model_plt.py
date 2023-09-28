import matplotlib.pyplot as plt
from ..utils import model_io

import numpy as np

COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'pink', 'gray']
PLOTS_DIR_PATH = r'MLFS\model_visualization\model_plots'
MODEL_METADATA_DIR_PATH = r'MLFS\supervised_learning\model_metadata\mnist_2023-09-27_17-30'
METRICS = ['cost', 'accuracy']

def plot_model_metric_vs_iters(all_models_metadata, metric:str, colors):
    fig, ax = plt.subplots(figsize=(18,10))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    for i, model in enumerate(all_models_metadata):
        train_performance = model['train_performance'] 
        iterations = train_performance['iterations']
        metric_train_set = train_performance[metric]
        model_name = model['name']
        color = colors[i % len(colors)]
        ax.plot(iterations, metric_train_set, label=f"{model_name}-Training_Set", color=color)
    
    ax.set_xticks(range(0, 501, 20))
    if metric == 'accuracy':
       ax.set_yticks([0.00 + i * 0.05 for i in range(21)])
    elif metric == 'cost':
       ax.set_yticks([0.00 + i * 0.1 for i in range(25)])
    
    ax.spines['left'].set_position('zero')
    ax.grid(alpha=0.25)
    ax.set_xlabel('Iterations')
    ax.set_ylabel(f"{metric.capitalize()}")
    ax.legend(bbox_to_anchor=(0.05, 0.5), loc='center left')
    ax.set_title(f'Model Training Set Performance Comparison: Model {metric.capitalize()}')
    
    return fig, ax

def plot_mnist_model_metadata_directory(model_metadata_dir_path, plots_dir_path, colors, metrics_list):
    plots = {}
    all_models_metadata = model_io.load_model_metadata_directory(model_metadata_dir_path)
    for metric in metrics_list:
        fig, ax = plot_model_metric_vs_iters(all_models_metadata, metric, colors)
        plots[metric] = fig
        plt.show()
    model_io.save_plots(plots, plots_dir_path, model_metadata_dir_path)

"""
specific to mnist dataset
"""
def show_target_image(img_vector):
    img_matrix = img_vector.reshape((28, 28))
    plt.imshow(img_matrix, cmap='gray', interpolation='nearest')
    plt.show()

#plot_mnist_model_metadata_directory(MODEL_METADATA_DIR_PATH, PLOTS_DIR_PATH, COLORS, METRICS)