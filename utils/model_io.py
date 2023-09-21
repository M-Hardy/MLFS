from datetime import datetime
import pickle
import os

def save_model_metrics(all_model_metrics, dirpath, model):
    formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")
    timestamp_folder = f"{dirpath}/{model}_{formatted_datetime}"
    os.makedirs(timestamp_folder)

    for model_dict in all_model_metrics:
        model_name = model_dict['name']
        filepath = os.path.join(timestamp_folder, f'{model_name}')
        with open(filepath, 'wb') as file:
            pickle.dump(model_dict, file)

def load_model_metrics(dirpath, filename):
    filepath = f"{dirpath}/{filename}"
    with open(filepath, 'rb') as file:
        model_metrics = pickle.load(file)
    return model_metrics