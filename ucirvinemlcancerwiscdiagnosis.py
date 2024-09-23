import json
import os

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


def get_breast_cancer_wisconsin_diagnostic_data(id=17):  # default = 17 Breast Cancer Wisconsin (Diagnostic) Data Set
    breast_cancer_wisconsin_diagnostic = fetch_uc_irvine_ml_data(id)
    # data (as pandas dataframes)
    X = breast_cancer_wisconsin_diagnostic.data.features.copy()
    y = breast_cancer_wisconsin_diagnostic.data.targets.copy()
    # metadata
    metadata = breast_cancer_wisconsin_diagnostic.metadata
    # variable information
    variables = breast_cancer_wisconsin_diagnostic.variables
    # combine data and metadata
    data = pd.concat([X, y], axis=1)
    return data, X, y, metadata, variables


def encodeLabels(data):
    from sklearn.preprocessing import LabelEncoder
    # copy data before returning values
    data_internal = data.copy()
    le = LabelEncoder()
    for col in data_internal.columns:
        data_internal[col] = le.fit_transform(data[col])
    return data_internal


"""
One-hot encoding is a data preparation technique that's best 
used when working with categorical data that doesn't have an inherent order. 
"""


def encodeOneHotLabels(data):
    """
    :param data:  data frame
    :return: data frame with one hot encoding
    """
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder()
    # copy data before returning values
    data_inside = data.copy()
    print('Data inside shape', data_inside.shape)
    if data_inside.shape[1] > 1:
        data_inside = data_inside.iloc[:, 0]
        enc.fit(data_inside)
        data_inside = enc.transform(data_inside).toarray()
    else:
        enc.fit(data_inside)
        data_inside = enc.transform(data_inside).toarray()
    return pd.DataFrame(data_inside)


def encodeOneHotLabelsColumnInData(data, column):
    """
    :param data: data frame
    :param column: column name
    :return: data frame with one hot encoding
    """
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    enc = OneHotEncoder()
    # copy data before returning values
    data_inside = data.copy()
    print('Data inside shape', data_inside.shape)
    col_transformer = ColumnTransformer(
        transformers=[
            ('encoder', enc, [column])
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
    # check which datasets can be imported
    list_available_datasets()
    # fetch the dataset
    data_frame, X, Y, metadata, variables = get_breast_cancer_wisconsin_diagnostic_data()
    data_frame.to_csv('breast_cancer_wisconsin_diagnostic.csv', sep=",", index=False, header=False)
    diagnostic_data = pd.read_csv('breast_cancer_wisconsin_diagnostic.csv', header=None)
    print('X Columns', X.columns)
    print('Y columns', Y.columns)
    with open('ibreast_cancer_wisconsin_diagnostic_metadata.json', 'w') as jsonFile:
        json.dump(metadata, jsonFile, indent=4, sort_keys=True)

    variables.to_csv('breast_cancer_wisconsin_diagnostic_variables.csv', sep=",", index=False, header=False)

    X_Data = diagnostic_data.iloc[:, 0:diagnostic_data.shape[1] - 1]
    Y_Data = diagnostic_data.iloc[:, diagnostic_data.shape[1] - 1].to_frame()
    X_Data.columns = X.columns
    Y_Data.columns = Y.columns
    all_column_names = list(X.columns) + list(Y.columns)
    diagnostic_data.columns = all_column_names
    print('Diagnostic Data\n', diagnostic_data.head())

    print('X shape', X.shape, 'read X data', X_Data.shape)
    print('X Data\n', X_Data.head())
    print('X Description\n', X_Data.describe())

    print('Y shape', Y.shape, 'read Y data', Y_Data.shape)
    print('Y Data\n', Y_Data)
    # get distinct values of Y data
    print('Y Data distinct values\n', Y_Data.value_counts())

    # encode the labels
    Y_Num_Label = encodeLabels(Y_Data)
    print('Y Data encoded\n', Y_Num_Label.head())
    print('Y Data distinct values encoded \n', Y_Num_Label.value_counts())

    # One hot encoding
    Y_One_Hot_Label = encodeOneHotLabels(Y_Data)
    print('Y Data one hot encoded\n', Y_One_Hot_Label.head())
    print('Y Data distinct values of One hot encoded \n', Y_One_Hot_Label.value_counts())

    # One hot encoding for a column
    expanded_one_hot_encoding = encodeOneHotLabelsColumnInData(diagnostic_data, 'Diagnosis')
    print('Expanded one hot encoding\n', expanded_one_hot_encoding.head())
