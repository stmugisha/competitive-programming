import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def read_data(path: str, transpose=False) -> pd.DataFrame:
    """Read a csv file into a pandas DataFrame.
    Args:
    `path`: the path where the csv is located.
    Returns:
    > The number of rows and unique columns;
    > and the top 3 records of the dataset.
    """
    dataset = pd.read_csv(path)
    if(transpose==True):
        return dataset.head(3).T
    return dataset


def analyze_data(dataset: pd.DataFrame):
    """Computes and returns insights into the data.
    Args:
    data: The dataset to be investigated/analyzed.
    """
    #print(f'There are {(dataset.columns.nunique())} unique columns and {len(dataset)} rows in the dataset\n')
    print('='*100)
    print(f'Null value check:\n {dataset.isnull().sum()}\n')
    print('='*100)
    print(f'Dtypes info: {dataset.info()}')
    print('='*100)
    return dataset.describe().T


def split_data(dataset: pd.DataFrame, train=True, test=False):
    """Split a given dataset into train and
    test sets for model training and evaluation
    Args:
    `data`: A pandas DataFrame.
    
    Returns:
    Two datasets(train & test) split from the 
    original input dataset.
    """
    features_new = []
    features_old = []
    for column in dataset.columns:
        if '2019' not in column:
            features_old.append(column)
        else:
            features_new.append(column)
    features_new.extend(['X',	'Y',	'elevation', 'LC_Type1_mode',	'Square_ID'])
    train_set = dataset[features_old]
    test_set = dataset[features_new]
    if train:
        return train_set
    if test:
        return test_set
    
def rmse(y, y_hat):
    """Compute the root mean squared error
    between true values `(y)` and the predicted values
    `(y_hat)`
    """
    return np.sqrt(mean_squared_error(y, y_hat))

