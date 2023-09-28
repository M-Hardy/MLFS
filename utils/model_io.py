from datetime import datetime
import pickle
import os

def save_model_metadata(all_models_metadata, dirpath, model):
    formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")
    timestamp_folder = os.path.join(dirpath, f"{model}_{formatted_datetime}")#f"{dir_path}/{model}_{formatted_datetime}"
    os.makedirs(timestamp_folder)

    for model_dict in all_models_metadata:
        model_name = model_dict['name']
        filepath = os.path.join(timestamp_folder, f'{model_name}')
        with open(filepath, 'wb') as file:
            pickle.dump(model_dict, file)

def save_plots(figs_dict, plots_dirpath, metadata_dirpath):
    formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")
    ts_metadata_folder = os.path.basename(metadata_dirpath)
    ts_plot_folder = os.path.join(plots_dirpath, f"{ts_metadata_folder}_({formatted_datetime})")
    os.makedirs(ts_plot_folder)
    for metric, fig in figs_dict.items():
        fig.savefig(os.path.join(ts_plot_folder, metric))
    
def load_model_metadata(dirpath, filename):
    filepath = os.path.join(dirpath, filename)
    with open(filepath, 'rb') as file:
        model_metadata = pickle.load(file)
    return model_metadata

def load_model_metadata_directory(dirpath):
    all_models_metadata = []
    for filename in os.listdir(dirpath):
        if os.path.isfile(os.path.join(dirpath, filename)):
            model_metadata = load_model_metadata(dirpath, filename)
            all_models_metadata.append(model_metadata)
    return all_models_metadata