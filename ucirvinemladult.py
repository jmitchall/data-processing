import json
import os
import pickle

import pandas as pd
from ucimlrepo import fetch_ucirepo, list_available_datasets

http_proxy = 'http://http.proxy.fmr.com:8000/'
https_proxy = 'http://http.proxy.fmr.com:8000/'
no_proxy = '169.254.169.254'
os.environ['http_proxy'] = http_proxy
os.environ['https_proxy'] = https_proxy
os.environ['no_proxy'] = no_proxy
os.environ['HTTP_PROXY'] = http_proxy
os.environ['HTTPS_PROXY'] = https_proxy


# fetch dataset
def fetch_uc_irvine_ml_data(id_arg):
    dataset = fetch_ucirepo(id=id_arg)
    return dataset


def get_adult_data(id_value=2):  # default = 2 Adult Data Set
    adult = fetch_uc_irvine_ml_data(id_value)
    # metadata
    meta_data = adult.metadata
    # variable information
    adult_variables = adult.variables
    # data (as pandas dataframes)
    x_data = adult.data.features.copy()
    if adult.data.targets is None:
        y_data = pd.DataFrame()  # empty data frame
    else:
        y_data = adult.data.targets.copy()

    # combine data and metadata
    all_data = pd.concat([x_data, y_data], axis=1)
    return all_data, x_data, y_data, meta_data, adult_variables


def encode_binary_column_data(df, column_name, column_value_map):
    """
    :param df: data frame
    :param column_name: column name
    :param column_value_map: dictionary of values that map to binary numbers
    :return: data frame with binary encoded column
    """
    df_data = df.copy()
    df_data[column_name] = df_data[column_name].map(column_value_map)
    df_data[column_name] = df_data[column_name].apply(lambda x: format(x, 'b'))
    return df_data


def encode_frequency_count_data(df, column_name):
    """
    :param df: data frame
    :param column_name: column name
    :return: data frame with frequency count encoded column
    """
    df_data = df.copy()
    counts = df_data[column_name].value_counts()
    df_data[column_name] = df_data[column_name].map(counts)
    return df_data


def encoder_ordinal_data(df, column_name, order_value_list):
    """
    :param df: data frame
    :param column_name: column name
    :param order_value_list: list of order
    :return: data frame with ordinal encoded column
    """
    from sklearn.preprocessing import OrdinalEncoder
    # Create an instance of the OrdinalEncoder
    if not order_value_list:
        oe = OrdinalEncoder()
    else:
        oe = OrdinalEncoder(categories=[order_value_list])
    df_data = df.copy()

    # The selected line of code is performing an ordinal encoding on a specific column of a DataFrame.
    # Ordinal encoding is a type of encoding for handling categorical data, especially when there is a
    # clear ordering in the categories.
    # For example, for a feature like "size" with values "small", "medium", "large", "extra large",
    # ordinal encoding would be appropriate.

    # `df_data[column_name].values.reshape(-1, 1)` - This is reshaping the column values into a
    # 2D array, which is the required input shape for the `fit_transform` method of the OrdinalEncoder.

    # `oe.fit_transform(df_data[column_name].values.reshape(-1, 1))` -
    # This line is fitting the OrdinalEncoder on the reshaped column data and then transforming
    # the data. The transformed data, which is now ordinal encoded, is then assigned back to the
    # column in the DataFrame.

    df_data[column_name] = oe.fit_transform(df_data[column_name].values.reshape(-1, 1))

    return df_data


def encode_labels(data, specific_columns=None):
    from sklearn.preprocessing import LabelEncoder
    # copy data before returning values
    data_internal = data.copy()
    if not specific_columns:
        specific_columns = data_internal.columns
    le = LabelEncoder()
    for col in specific_columns:
        data_internal[col] = le.fit_transform(data[col])
    # # for each column in return_value get the distinct values
    # for col in data_internal.columns:
    #     print( col, 'distinct values', data_internal[col].value_counts())
    return data_internal


"""
One-hot encoding is a data preparation technique that's best 
used when working with categorical data that doesn't have an inherent order. 

By one-hot encoding them, we create a really sparse matrix and inflate the number of 
dimensions the model needs to work with, and we may fall victim to the dreaded 
Curse of Dimensionality. This is amplified when the feature has too many categories, 
most of them being useless for the prediction.
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
    if data_inside.shape[1] > 1:
        data_inside = data_inside.iloc[:, 0]
        enc.fit(data_inside)
        data_inside = enc.transform(data_inside).toarray()
    else:
        enc.fit(data_inside)
        data_inside = enc.transform(data_inside)
    return pd.DataFrame(data_inside)


def encode_dummies(data, columns):
    """
    :param data: data frame
    :param columns: column names
    :return: data frame with dummy encoding
    """
    return pd.get_dummies(data, columns=columns)


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


"""
Target Encoder - This is a technique used to encode categorical variables.
It replaces the categorical values with the mean of the target variable.
This is useful when the target variable is binary.

The target encoder is a supervised encoding technique 
to encode the categories by replacing them for a measurement of the effect 
they might have on the target.

On a binary classifier, the simplest way to do that is by calculating the 
probability p(t = 1 | x = ci) in which t denotes the target, 

x is the input and 
ci is the i-th category. 

In Bayesian statistics, this is considered the posterior probability of t=1 
given the input was the category ci.

This means we will replace the category ci 
for the value of the posterior probability of the target being 1 
on the presence of that category.

https://towardsdatascience.com/dealing-with-categorical-variables-by-using-target-encoder-a0f1733a4c69

ENCODING = (COUNT_OF_TARGET_1) / (TOTAL_OCCURRENCES_OF_CATEGORY)

"""


def encode_target_encoder(data, target_column_name, column_name):
    """
    :param data: data frame
    :param target_column_name: target column name
    :param column_name: column name
    :return: data frame with target encoding
    """
    data_internal = data.copy()
    from category_encoders import TargetEncoder
    encoder = TargetEncoder()
    data_internal[column_name + '_encoded_sklearn'] = encoder.fit_transform(
        data_internal[column_name], data_internal[target_column_name]
    )
    """
    categories = data_internal[column_name].unique()
    targets = data_internal[target_column_name].unique()
    cat_list = []
    for category in categories:
        aux_dict = {}
        aux_dict['category'] = category
        aux_df = data_internal[data_internal[column_name] == category]
        counts = aux_df[target_column_name].value_counts()
        aux_dict['count'] = sum(counts)
        for t in targets:
            aux_dict['target_' + str(t)] = counts[t]
        cat_list.append(aux_dict)
    cat_list = pd.DataFrame(cat_list)
    cat_list[column_name+'_encoded_dumb'] = cat_list['target_1'] / cat_list['count']

    Since the target of interest is the value “1”, this probability is actually the mean of the target, given a 
    category. This is the reason why this method of target encoding is also called “mean” encoding.
    
    We can calculate this mean with a simple aggregation, then:
  
    stats = data_internal[target_column_name].groupby(data_internal[column_name]).agg(['count', 'mean'])
    data_internal[column_name + '_encoded'] = data_internal[column_name].map(stats['mean'])
    smoothing_factor = 1.0 # The f of the smoothing factor equation 
    min_samples_leaf = 1 # The k of the smoothing factor equation
    
    import numpy as np
    prior = data_internal[target_column_name].mean()
    smooth = 1 / (1 + np.exp(-(stats['count'] - min_samples_leaf) / smoothing_factor))
    smoothing = prior * (1 - smooth) + stats['mean'] * smooth
    encoded = pd.Series(smoothing, name = column_name + '_encoded_complete')
    data_internal = data_internal.join(encoded, on = column_name)

    https://towardsdatascience.com/dealing-with-categorical-variables-by-using-target-encoder-a0f1733a4c69
    
    One really important effect is the Target Leakage. By using the probability of the target to encode the features 
    we are feeding them with information of the very variable we are trying to model. 
    This is like “cheating” since the model will learn from a variable that contains the target in 
    itself.
    
    Even if the mean is a good summary, we train models in a fraction of the data. The mean of this fraction may 
    not be the mean of the full population (remember the central limit theorem?), so the encoding might not be correct. 
    If the sample is different enough from the population, the model may even overfit the training data.
    """
    return data_internal


def get_memory_usage_of_data_frame(df, bytes_to_mb_div=0.000001):
    mem = round(df.memory_usage().sum() * bytes_to_mb_div, 3)
    return_str = "Memory usage is " + str(mem) + " MB"

    return return_str


def convert_to_sparse_pandas(df, exclude_columns):
    """
    https://towardsdatascience.com/working-with-sparse-data-sets-in-pandas-and-sklearn-d26c1cfbe067
    Converts columns of a data frame into SparseArrays and returns the data frame with transformed columns.
    Use exclude_columns to specify columns to be excluded from transformation.
    :param df: pandas data frame
    :param exclude_columns: list
        Columns not be converted to sparse
    :return: pandas data frame
    """
    from pandas.arrays import SparseArray
    pd.DataFrame.iteritems = pd.DataFrame.items
    df = df.copy()
    exclude_columns = set(exclude_columns)
    # get iterable tuple of column name and column data from data frame
    for (columnName, columnData) in df.iteritems():
        if columnName in exclude_columns:
            continue
        df[columnName] = SparseArray(columnData.values, dtype='uint8')
    return df


def data_frame_to_scipy_sparse_matrix(df):
    """
    Converts a sparse pandas data frame to sparse scipy csr_matrix.
    :param df: pandas data frame
    :return: csr_matrix
    """
    from scipy.sparse import lil_matrix
    import numpy as np

    # Initialize a sparse matrix with the same shape as the DataFrame `df`.
    # The `lil_matrix` is a type of sparse matrix provided by SciPy.
    # It's good for incremental construction. Row-based LIst of Lists sparse matrix.
    arr = lil_matrix(df.shape, dtype=np.float32)

    # Iterate over each column in the DataFrame.
    for i, col in enumerate(df.columns):
        # Create a boolean mask where each element is `True` if the corresponding element in the column is not zero,
        # and `False` otherwise.
        ix = df[col] != 0
        # Set the value of the sparse matrix at the positions where the mask is `True` to 1.
        # The `np.where(ix)` function returns the indices where `ix` is `True`.
        arr[np.where(ix), i] = 1

    # Convert the `lil_matrix` to a `csr_matrix` (Compressed Sparse Row matrix) and return it.
    # The `csr_matrix` is another type of sparse matrix that is efficient for arithmetic operations
    # and is suitable for machine learning algorithms in SciPy and sklearn.
    return arr.tocsr()


def get_csr_memory_usage(x_csr, bytes_to_mb_div=0.000001):
    mem = (x_csr.data.nbytes + x_csr.indptr.nbytes + x_csr.indices.nbytes) * bytes_to_mb_div
    return "Memory usage is " + str(mem) + " MB"


def download_from_uc_data():
    data_frame = None
    x = None
    y = None
    m_data = None
    a_var = None
    x_columns = None
    x_shape = None
    y_columns = None
    y_shape = None
    # if adult.csv doesn't exist
    if not os.path.exists('adult.csv'):
        # check which datasets can be imported
        list_available_datasets()
        # fetch the dataset
        data_frame, x, y, m_data, a_var = get_adult_data()
        data_frame.to_csv('adult.csv', sep=",", index=False, header=False)
        a_var.to_csv('adult_variables.csv', sep=",", index=False, header=True)
        with open('adult_metadata.json', 'w') as jsonFile:
            json.dump(m_data, jsonFile, indent=4, sort_keys=True)
        with open('adult_X_columns', 'wb') as adult_x_fp:
            pickle.dump(x.columns, adult_x_fp)
        x_columns = x.columns
        with open('adult_X_shape', 'wb') as adult_shape_fp:
            pickle.dump(x.shape, adult_shape_fp)
        x_shape = x.shape
        with open('adult_Y_columns', 'wb') as adult_y_fp:
            pickle.dump(y.columns, adult_y_fp)
        y_columns = y.columns
        with open('adult_Y_shape', 'wb') as adult_fp:
            pickle.dump(y.shape, adult_fp)
        y_shape = y.shape

    return data_frame, x, y, m_data, a_var, x_columns, x_shape, y_columns, y_shape


# python main entry
if __name__ == '__main__':
    download_from_uc_data()

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
    with open('adult_X_shape', 'rb') as fp:
        X_shape = pickle.load(fp)
    print('X Shape', X_shape)
    print('X Data Columns without missing values\n', X_Data.columns[X_Data.isnull().mean() != 0])

    X_Data['occupation'] = X_Data['occupation'].apply(
        lambda x: x.strip().replace('?', 'unknown') if isinstance(x, str) else x)
    X_Data['occupation'] = X_Data['occupation'].fillna('unknown')

    X_Data['native-country'] = X_Data['native-country'].fillna('unknown')
    X_Data['native-country'] = X_Data['native-country'].apply(
        lambda x: x.strip().replace('?', 'unknown') if isinstance(x, str) else x)

    X_Data['workclass'] = X_Data['workclass'].fillna('unknown')
    X_Data['workclass'] = X_Data['workclass'].apply(
        lambda x: x.strip().replace('?', 'unknown') if isinstance(x, str) else x)

    X_Data.drop(['education'], axis=1, inplace=True)
    X_Data = encode_labels(X_Data, ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                                    'native-country'])
    X_Data.to_csv('adult_X_columns_label_encoded.csv', sep=",", index=False, header=True)

    X_Data = encode_one_hot_labels_column_in_data(X_Data,
                                                  ['workclass', 'marital-status', 'occupation', 'relationship', 'race',
                                                   'sex', 'native-country'])
    X_Data_sparse = convert_to_sparse_pandas(X_Data, [])
    X_Data_csr = data_frame_to_scipy_sparse_matrix(X_Data)
    X_Data.to_csv('adult_X_data_one_hot_encoded.csv', sep=",", index=False, header=True)
    X_Data_sparse.to_csv('adult_X_sparse_one_hot_encoded.csv', sep=",", index=False, header=True)

    Y_Data = Y_Data.map(lambda x: x.strip().replace('K.', 'K') if isinstance(x, str) else x)
    Y_Data = encode_binary_column_data(Y_Data, 'income', {'<=50K': 0, '>50K': 1})

    Y_Data_sparse = convert_to_sparse_pandas(Y_Data, [])
    Y_Data_csr = data_frame_to_scipy_sparse_matrix(Y_Data)
    Y_Data.to_csv('adult_Y_data_one_hot_encoded.csv', sep=",", index=False, header=True)
    Y_Data_sparse.to_csv('adult_Y_sparse_one_hot_encoded.csv', sep=",", index=False, header=True)
    # get distinct values of Y data
    print('Y Data distinct values\n', Y_Data.value_counts())
    print('X Data takes up', get_memory_usage_of_data_frame(X_Data))
    print('Y Data takes up', get_memory_usage_of_data_frame(Y_Data))
    print('X sparse Data takes up', get_memory_usage_of_data_frame(X_Data_sparse))
    print('Y sparse Data takes up', get_memory_usage_of_data_frame(Y_Data_sparse))
    print('X csr Data takes up', get_csr_memory_usage(X_Data_csr))
    print('Y csr Data takes up', get_csr_memory_usage(Y_Data_csr))

    # todo : LOOK INTO
    #  https://stephenleo.github.io/data-science-blog/data-science-blog/ml/feature_engineering.html#dirty-cat

    # vector_dict = {'Pandas dataframe': [X, y],
    #                'Sparse pandas dataframe': [X_sparse, y_sparse],
    #                'Scipy sparse matrix': [X_csr, y_csr]
    #                }
    #
    # for key, item in vector_dict.items():
    #     print(key)
    #
    #     start = time.time()
    #     X_train, X_test, y_train, y_test = train_test_split(item[0], item[1], test_size=0.3, random_state=42)
    #     end = time.time()
    #     duration = round(end - start, 2)
    #     print("Train-test split: " + str(duration) + " secs")
    #
    #     start = time.time()
    #     model = LogisticRegression(random_state=0, multi_class='ovr', solver='liblinear')
    #     model.fit(X_train, y_train)
    #     end = time.time()
    #     duration = round(end - start, 2)
    #     print("Training: " + str(duration) + " secs")
    #     print("\n")

