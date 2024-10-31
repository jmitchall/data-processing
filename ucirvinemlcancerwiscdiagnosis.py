import json
import os
import pickle

import pandas as pd
from ucimlrepo import fetch_ucirepo, list_available_datasets

# fetch dataset
def fetch_uc_irvine_ml_data(id_arg):
    dataset = fetch_ucirepo(id=id_arg)
    return dataset


def get_breast_cancer_wisconsin_diagnostic_data(
        bc_w_diagnosis_id=17):  # default = 17 Breast Cancer Wisconsin (Diagnostic)
    breast_cancer_wisconsin_diagnostic = fetch_uc_irvine_ml_data(bc_w_diagnosis_id)
    # data (as pandas dataframes)
    x_data = breast_cancer_wisconsin_diagnostic.data.features.copy()
    y_data = breast_cancer_wisconsin_diagnostic.data.targets.copy()
    # metadata
    bc_w_diagnosis_meta_data = breast_cancer_wisconsin_diagnostic.metadata
    # variable information
    bc_w_diagnosis_variables = breast_cancer_wisconsin_diagnostic.variables
    # combine data and metadata
    all_data = pd.concat([x_data, y_data], axis=1)
    return all_data, x_data, y_data, bc_w_diagnosis_meta_data, bc_w_diagnosis_variables


def encode_labels(data):
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


def download_from_uc_data():
    data_frame = None
    x = None
    y = None
    m_data = None
    bc_var = None
    x_columns = None
    x_shape = None
    y_columns = None
    y_shape = None
    if not os.path.exists('breast_cancer_wisconsin_diagnostic.csv'):
        # check which datasets can be imported
        list_available_datasets()
        # fetch the dataset
        data_frame, x, y, m_data, bc_var = get_breast_cancer_wisconsin_diagnostic_data()
        data_frame.to_csv('breast_cancer_wisconsin_diagnostic.csv', sep=",", index=False, header=False)
        bc_var.to_csv('breast_cancer_wisconsin_diagnostic_variables.csv', sep=",", index=False, header=False)
        with open('breast_cancer_wisconsin_diagnostic_metadata.json', 'w') as jsonFile:
            json.dump(m_data, jsonFile, indent=4, sort_keys=True)
        with open('breast_cancer_wisconsin_diagnostic_X_columns', 'wb') as adult_x_fp:
            pickle.dump(x.columns, adult_x_fp)
        x_columns = x.columns
        with open('breast_cancer_wisconsin_diagnostic_X_shape', 'wb') as adult_shape_fp:
            pickle.dump(x.shape, adult_shape_fp)
        x_shape = x.shape
        with open('breast_cancer_wisconsin_diagnostic_Y_columns', 'wb') as adult_y_fp:
            pickle.dump(y.columns, adult_y_fp)
        y_columns = y.columns
        with open('breast_cancer_wisconsin_diagnostic_Y_shape', 'wb') as adult_fp:
            pickle.dump(y.shape, adult_fp)
        y_shape = y.shape

    return data_frame, x, y, m_data, bc_var, x_columns, x_shape, y_columns, y_shape


# python main entry
if __name__ == '__main__':
    download_from_uc_data()

    variables = pd.read_csv('breast_cancer_wisconsin_diagnostic_variables.csv')
    with open('breast_cancer_wisconsin_diagnostic_X_columns', 'rb') as fp:
        X_columns = pickle.load(fp)
    with open('breast_cancer_wisconsin_diagnostic_Y_columns', 'rb') as fp:
        Y_columns = pickle.load(fp)
    print('X Columns', X_columns)
    print('Y columns', Y_columns)
    metadata = json.load(open('breast_cancer_wisconsin_diagnostic_metadata.json'))
    diagnostic_data = pd.read_csv('breast_cancer_wisconsin_diagnostic.csv', header=None)
    X_Data = diagnostic_data.iloc[:, 0:diagnostic_data.shape[1] - 1]
    Y_Data = diagnostic_data.iloc[:, diagnostic_data.shape[1] - 1].to_frame()
    X_Data.columns = X_columns
    Y_Data.columns = Y_columns
    all_column_names = list(X_columns) + list(Y_columns)
    diagnostic_data.columns = all_column_names
    print('Diagnostic Data\n', diagnostic_data.head())
    with open('breast_cancer_wisconsin_diagnostic_X_shape', 'rb') as fp:
        X_shape = pickle.load(fp)
    print('X shape', X_shape, 'read X data', X_Data.shape)
    print('X Data\n', X_Data.head())
    print('X Description\n', X_Data.describe())

    with open('breast_cancer_wisconsin_diagnostic_Y_shape', 'rb') as fp:
        Y_shape = pickle.load(fp)
    print('Y shape', Y_shape, 'read Y data', Y_Data.shape)
    print('Y Data\n', Y_Data)
    # get distinct values of Y data
    print('Y Data distinct values\n', Y_Data.value_counts())

    # encode the labels
    Y_Num_Label = encode_labels(Y_Data)
    print('Y Data encoded\n', Y_Num_Label.head())
    print('Y Data distinct values encoded \n', Y_Num_Label.value_counts())

    # One hot encoding
    Y_One_Hot_Label = encode_one_hot_labels(Y_Data)
    print('Y Data one hot encoded\n', Y_One_Hot_Label.head())
    print('Y Data distinct values of One hot encoded \n', Y_One_Hot_Label.value_counts())

    # One hot encoding for a column
    expanded_one_hot_encoding = encode_one_hot_labels_column_in_data(diagnostic_data, 'Diagnosis')
    print('Expanded one hot encoding\n', expanded_one_hot_encoding.head())
