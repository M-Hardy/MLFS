from ..common import np, pd

def load_data_from_csv(file_path):
    data = np.array(pd.read_csv(file_path))
    return data