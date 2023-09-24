import matplotlib.pyplot as plt
from ..utils import model_io

import numpy as np

COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'pink', 'gray']
PLOTS_DIR_PATH = r'MLFS\model_visualization\model_plots'
MODEL_METRICS_DIR_PATH = r'MLFS\supervised_learning\model_metrics\mnist_2023-09-23_13-34'
METRICS = ['cost', 'accuracy']

def plot_model_metric_vs_iters(all_model_metrics, metric:str, colors):
    fig, ax = plt.subplots(figsize=(18,10))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    for i, model in enumerate(all_model_metrics):
        train_performance = model['train_performance'] 
        cv_performance = model['cv_performance']
        iterations = train_performance['iterations']
        metric_train_set = train_performance[metric]
        metric_cv_set = cv_performance[metric]
        model_name = model['name']
        color = colors[i % len(colors)]
        ax.plot(iterations, metric_train_set, label=f"{model_name}-Training_Set", color=color)
        #shift cv training over by 400 - give appearance of 1 continuous line for each model
        ax.plot(range(400, 801, 20), metric_cv_set, label=f"{model_name}-CV_set", color=color,linestyle='--')
    
    ax.set_xticks(range(0, 801, 20))
    if metric == 'accuracy':
       ax.set_yticks([0.00 + i * 0.05 for i in range(21)])
    elif metric == 'cost':
       ax.set_yticks([0.00 + i * 0.1 for i in range(25)])
    
    ax.spines['left'].set_position('zero')
    ax.grid(alpha=0.25)
    ax.set_xlabel('Iterations')
    ax.set_ylabel(f"{metric.capitalize()}")
    ax.legend(bbox_to_anchor=(0.05, 0.5), loc='center left')
    ax.set_title(f'Model Test Set Performance Comparison: Model {metric.capitalize()}')
    
    return fig, ax

def plot_mnist_model_metrics_directory(metrics_dir_path, plots_dir_path, colors, metrics_list):
    plots = {}
    all_model_metrics = model_io.load_model_metrics_directory(metrics_dir_path)
    for metric in metrics_list:
        fig, ax = plot_model_metric_vs_iters(all_model_metrics, metric, colors)
        plots[metric] = fig
        plt.show()
    model_io.save_plots(plots, plots_dir_path, metrics_dir_path)

"""
specific to mnist dataset
"""
def show_target_image(img_vector):
    img_matrix = img_vector.reshape((28, 28))
    plt.imshow(img_matrix, cmap='gray', interpolation='nearest')
    plt.show()

#plot_mnist_model_metrics_directory(MODEL_METRICS_DIR_PATH, PLOTS_DIR_PATH, COLORS, METRICS)