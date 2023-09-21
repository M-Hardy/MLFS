from ..common import np

def create_train_and_cv_sets(np_data, cv_proportion, label_col_index):
    m = np_data.shape[0]
    np.random.shuffle(np_data)
    batch = m // cv_proportion

    training_set = np_data[:m - batch]
    x_train = training_set[:,label_col_index+1:]
    y_train = training_set[:, label_col_index]

    cv_set = np_data[m-batch:]
    x_cv = cv_set[:, label_col_index+1:]
    y_cv = cv_set[:, label_col_index]
    return x_train, y_train, x_cv, y_cv