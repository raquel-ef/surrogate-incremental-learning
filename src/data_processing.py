from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn import preprocessing


def read_arff(path):
    """
    Reads an ARFF file and converts it into a pandas DataFrame.

    Parameters:
    path (str): File path to the ARFF dataset.

    Returns:
    DataFrame: Loaded dataset.
    """
    data = arff.loadarff(path)
    return pd.DataFrame(data[0])


def split_features_target(df):
    """
    Splits a DataFrame into features (X) and target (Y).

    Parameters:
    df (DataFrame): Dataset to be split.

    Returns:
    tuple: Feature set (X) and target set (Y).
    """
    return df.iloc[:, :-1], df.iloc[:, -1:]


def reshape_timeseries(data):
    """
    Reshapes data for time series input (LSTM compatibility).

    Parameters:
    data (np.ndarray): Input data.

    Returns:
    np.ndarray: Reshaped data.
    """
    return np.reshape(data, (data.shape[0], 1, data.shape[1]))


def to_numpy(df):
    """
    Converts a pandas DataFrame to a NumPy array.

    Parameters:
    df (DataFrame): Input DataFrame.

    Returns:
    np.ndarray: Converted NumPy array.
    """
    return np.asarray(df)


def preprocess_data(df):
    """
    Preprocesses data by splitting into training, validation, and test sets,
    normalizing the data, and reshaping it for LSTM.

    Parameters:
    df (DataFrame): Input dataset.

    Returns:
    dict: Contains processed time series and normalized data splits.
    """
    # Split data into training (60%), validation (20%), and test (20%) sets
    n = len(df)
    train_set = df.iloc[: int(0.6 * n)]
    val_set = df.iloc[int(0.6 * n) : int(0.8 * n)]
    test_set = df.iloc[int(0.8 * n) :]

    # Extract features and targets
    train_X, train_Y = split_features_target(train_set)
    val_X, val_Y = split_features_target(val_set)
    test_X, test_Y = split_features_target(test_set)

    # Normalize data using MinMaxScaler
    scaler_X, scaler_Y = preprocessing.MinMaxScaler(), preprocessing.MinMaxScaler()
    train_X = scaler_X.fit_transform(train_X)
    train_Y = scaler_Y.fit_transform(train_Y)
    val_X = scaler_X.transform(val_X)
    val_Y = scaler_Y.transform(val_Y)
    test_X = scaler_X.transform(test_X)
    test_Y = scaler_Y.transform(test_Y)

    # Convert to numpy arrays
    train_X, train_Y = to_numpy(train_X), to_numpy(train_Y)
    val_X, val_Y = to_numpy(val_X), to_numpy(val_Y)
    test_X, test_Y = to_numpy(test_X), to_numpy(test_Y)

    # Reshape data for LSTM
    train_X_timeseries = reshape_timeseries(train_X)
    val_X_timeseries = reshape_timeseries(val_X)
    test_X_timeseries = reshape_timeseries(test_X)

    # Convert normalized data back to DataFrame
    train_X_df = pd.DataFrame(train_X, columns=train_set.columns[:-1])
    train_Y_df = pd.DataFrame(train_Y, columns=[train_set.columns[-1]])
    val_X_df = pd.DataFrame(val_X, columns=val_set.columns[:-1])
    val_Y_df = pd.DataFrame(val_Y, columns=[val_set.columns[-1]])
    test_X_df = pd.DataFrame(test_X, columns=test_set.columns[:-1])
    test_Y_df = pd.DataFrame(test_Y, columns=[test_set.columns[-1]])

    return {
        "timeseries": (train_X_timeseries, train_Y, val_X_timeseries, val_Y, test_X_timeseries, test_Y),
        "normalized": (train_X_df, train_Y_df, val_X_df, val_Y_df, test_X_df, test_Y_df)
    }
