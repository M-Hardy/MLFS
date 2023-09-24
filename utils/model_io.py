from datetime import datetime
import pickle
import os

def save_model_metrics(all_model_metrics, dir_path, model):
    formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")
    timestamp_folder = os.path.join(dir_path, f"{model}_{formatted_datetime}")#f"{dir_path}/{model}_{formatted_datetime}"
    os.makedirs(timestamp_folder)

    for model_dict in all_model_metrics:
        model_name = model_dict['name']
        filepath = os.path.join(timestamp_folder, f'{model_name}')
        with open(filepath, 'wb') as file:
            pickle.dump(model_dict, file)

def save_plots(figs_dict, plots_dir_path, metrics_dir_path):
    formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")
    corresponding_metrics_folder = os.path.basename(metrics_dir_path)
    timestamp_folder = os.path.join(plots_dir_path, f"{corresponding_metrics_folder}_({formatted_datetime})")
    print(timestamp_folder)
    os.makedirs(timestamp_folder)
    for metric, fig in figs_dict.items():
        fig.savefig(os.path.join(timestamp_folder, metric))
    
def load_model_metrics(dir_path, filename):
    filepath = os.path.join(dir_path, filename)
    with open(filepath, 'rb') as file:
        model_metrics = pickle.load(file)
    return model_metrics

def load_model_metrics_directory(dir_path):
    all_model_metrics = []
    for filename in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, filename)):
            model_metrics = load_model_metrics(dir_path, filename)
            all_model_metrics.append(model_metrics)
    return all_model_metrics