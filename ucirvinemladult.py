import json
import os
import pickle
import pandas as pd
from ucimlrepo import fetch_ucirepo, list_available_datasets

http_proxy = 'http://http.proxy.fmr.com:8000/'
http_proxy = 'http://http.proxy.fmr.com:8000/'
https_proxy = 'http://http.proxy.fmr.com:8000/'
no_proxy = '169.254.169.254'
os.environ['http_proxy'] = http_proxy
os.environ['https_proxy'] = https_proxy
os.environ['no_proxy'] = no_proxy
os.environ['HTTP_PROXY'] = http_proxy
os.environ['HTTPS_PROXY'] = https_proxy


# fetch dataset
def fetch_uc_irvine_ml_data(id):
    dataset = fetch_ucirepo(id=id)
    return dataset


def get_adult_data(id=2):  # default = 2 Adult Data Set
    adult = fetch_uc_irvine_ml_data(id)
    # metadata
    metadata = adult.metadata
    # variable information
    variables = adult.variables
    # data (as pandas dataframes)
    X = adult.data.features.copy()
    if adult.data.targets is None:
        y = pd.DataFrame()  # empty data frame
    else:
        y = adult.data.targets.copy()

    # combine data and metadata
    data = pd.concat([X, y], axis=1)
    return data, X, y, metadata, variables


def encode_labels(data, specific_columns = None):
    from sklearn.preprocessing import LabelEncoder
    # copy data before returning values
    data_internal = data.copy()
    if not specific_columns:
        specific_columns = data_internal.columns
    le = LabelEncoder()
    for col in specific_columns:
        data_internal[col] = le.fit_transform(data[col])
        # for each column in return_value get the distinct values
    for col in data_internal.columns:
        print( col, 'distinct values', data_internal[col].value_counts())
    return data_internal


"""
One-hot encoding is a data preparation technique that's best 
used when working with categorical data that doesn't have an inherent order. 
"""


def encode_one_hot_labels(data):
    """
    :param data:  data frame
    :return: data frame with one hot encoding
    """
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(sparse_output=False)
    # copy data before returning values
    data_inside = data.copy()
    print('Data inside shape', data_inside.shape)
    if data_inside.shape[1] > 1:
        data_inside = data_inside.iloc[:, 0]
        enc.fit(data_inside)
        data_inside = enc.transform(data_inside).toarray()
    else:
        enc.fit(data_inside)
        data_inside = enc.transform(data_inside)
    return pd.DataFrame(data_inside)


def encode_one_hot_labels_column_in_data(data, column):
    """
    :param data: data frame
    :param column: column name
    :return: data frame with one hot encoding
    """
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    enc = OneHotEncoder(sparse_output=False)
    # copy data before returning values
    data_inside = data.copy()
    print('Data inside shape', data_inside.shape)
    col_transformer = ColumnTransformer(
        transformers=[
            ('encoder', enc, column)
        ], remainder='passthrough'
    )
    data_inside = col_transformer.fit_transform(data_inside).astype(float)
    return_value = pd.DataFrame(data_inside)
    # rename columns to original column names and new column names
    col_names = col_transformer.get_feature_names_out()
    col_names = [col.replace('encoder__', '') for col in col_names]
    col_names = [col.replace('remainder__', '') for col in col_names]
    return_value.columns = col_names
    return return_value


# python main entry
if __name__ == '__main__':
    # if adult.csv doesn't exiet
    if not os.path.exists('adult.csv'):
        # check which datasets can be imported
        list_available_datasets()
        # fetch the dataset
        data_frame, X, Y, metadata, variables = get_adult_data()
        data_frame.to_csv('adult.csv', sep=",", index=False, header=False)
        variables.to_csv('adult_variables.csv', sep=",", index=False, header=True)
        with open('adult_metadata.json', 'w') as jsonFile:
            json.dump(metadata, jsonFile, indent=4, sort_keys=True)
        with open('adult_X_columns', 'wb') as fp:
            pickle.dump(X.columns, fp)
        X_columns = X.columns
        with open('adult_X_shape', 'wb') as fp:
            pickle.dump(X.shape, fp)
        X_shape = X.shape
        with open('adult_Y_columns', 'wb') as fp:
            pickle.dump(Y.columns, fp)
        Y_columns = Y.columns
        with open('adult_Y_shape', 'wb') as fp:
            pickle.dump(Y.shape, fp)
        Y_shape = Y.shape

    variables = pd.read_csv('adult_variables.csv')
    with open('adult_X_columns', 'rb') as fp:
        X_columns = pickle.load(fp)
    with open('adult_Y_columns', 'rb') as fp:
        Y_columns = pickle.load(fp)
    print('X Columns', X_columns)
    print('Y columns', Y_columns)
    metadata = json.load(open('adult_metadata.json'))
    adult_data = pd.read_csv('adult.csv', header=None)
    X_Data = adult_data.iloc[:, 0:adult_data.shape[1] - 1]
    Y_Data = adult_data.iloc[:, adult_data.shape[1] - 1].to_frame()
    X_Data.columns = X_columns
    Y_Data.columns = Y_columns
    all_column_names = list(X_columns) + list(Y_columns)
    adult_data.columns = all_column_names
    print('Adult Data\n', adult_data.head())
    with open('adult_X_shape', 'rb') as fp:
        X_shape = pickle.load(fp)
    print('X shape', X_shape, 'read X data', X_Data.shape)
    print('X Data\n', X_Data.head())
    print('X Data Columns without missing values\n', X_Data.columns[X_Data.isnull().mean() != 0])

    print('education-num distinct values\n', X_Data[['education-num']].value_counts())
    #trim all strings in column occupation
    X_Data['occupation'] = X_Data['occupation'].apply(lambda x: x.strip().replace('?', 'unknown') if isinstance(x, str) else x)
    X_Data['occupation'] = X_Data['occupation'].fillna('unknown')
    print('Occupation distinct values\n', X_Data[['occupation']].value_counts())
    X_Data['native-country'] = X_Data['native-country'].fillna('unknown')
    # In column native-country replace - in string value to empty space
    X_Data['native-country'] = X_Data['native-country'].apply(lambda x: x.strip().replace('?', 'unknown') if isinstance(x, str) else x)
    print('Native-country distinct values\n', X_Data[['native-country']].value_counts())
    X_Data['workclass'] = X_Data['workclass'].fillna('unknown')
    X_Data['workclass'] = X_Data['workclass'].apply(lambda x: x.strip().replace('?', 'unknown') if isinstance(x, str) else x)
    print('Workclass distinct values\n', X_Data[['workclass']].value_counts())
    X_Data.drop(['education'], axis=1, inplace=True)
    X_Data = encode_labels(X_Data, ['workclass','marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])
    X_Data.to_csv('adult_X_columns_label_encoded.csv', sep=",", index=False, header=True)

    X_Data = encode_one_hot_labels_column_in_data(X_Data,[ 'workclass','marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])
    X_Data.to_csv('adult_X_columns_one_hot_encoded.csv', sep=",", index=False, header=True)


    print('X Description\n', X_Data.describe())

    with open('adult_Y_shape', 'rb') as fp:
        Y_shape = pickle.load(fp)
    print('Y shape',Y_shape, 'read Y data', Y_Data.shape)
    print('Y Data\n', Y_Data)
    #trim a String values of every column in Y data
    Y_Data = Y_Data.map(lambda x: x.strip().replace('K.', 'K') if isinstance(x, str) else x)
    # Ordinal Map for income
    value_map ={ '<=50K': 0, '>50K': 1 }
    Y_Data['income'] = Y_Data['income'].map(value_map)
    # get distinct values of Y data
    print('Y Data distinct values\n', Y_Data.value_counts())

