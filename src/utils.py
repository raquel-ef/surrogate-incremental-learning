import numpy as np
import pandas as pd
import config.config as config
from src.evaluation import create_surrogate_model_dataset


# Applies sliding window transformation method keeping the attributes original order and deleteing the original attributes
def lags(df_original, n_lag):
    """
    Applies sliding window transformation keeping the attributes original order 
    and removing the original attributes.
    
    Parameters:
    df_original (DataFrame): Feature set for predictions.
    n_lag (int): Number of lags.
    
    Returns:
    DataFrame: New dataframe with the sliding window transformation.
    """
    df = df_original.copy()
    lagged_data = []

    # Generate lagged columns while maintaining original order
    for col in df.columns: 
        base_name = col.split(' ', 1)[0]  # Extract base name
        for j in range(1, n_lag + 1):
            lagged_data.append(df[col].shift(j).rename(f'Lag_{base_name}_{j}'))

    # Concatenate lagged columns
    df_lags = pd.concat(lagged_data, axis=1)

    # Append the last column from the original dataset
    df_lags[df.columns[-1]] = df[df.columns[-1]]

    return df_lags


def get_best_result_per_seed(df):
    """
    Selects the best result per unique seed based on the minimum 'H Val' value.
    The results are then sorted by 'H Test'.
    
    Parameters:
    df (pd.DataFrame): Input dataframe containing (at least) 'Seed', 'H Val', and 'H Test' columns.
    
    Returns:
    pd.DataFrame: Filtered dataframe containing only the best results per seed.
    """
    idxBest = df.groupby(['Seed'])['H Val'].transform("min") == df['H Val']
    return df[idxBest].sort_values(by='H Test')


def generate_random_attribute_combinations(n_attrib, n_combinations):
    """
    Generates a set of random binary attribute selection vectors, ensuring at least one attribute 
    is selected and the last N_STEPS attributes are included in at least one selection.
    
    Parameters:
    n_attributes (int): Total number of attributes available.
    n_combinations (int): Number of attribute selection combinations to generate.
    
    Returns:
    list: A list of binary numpy arrays representing attribute selections.
    """
    selected_attributes = []
    
    np.random.seed(config.SEED_VALUE) 
    
    for i in range(n_combinations - 1):
        position1 = np.zeros(n_attrib, dtype=np.int64)
        
        while np.sum(position1) == 0 or np.sum(position1[-config.N_STEPS:]) == 0:
            number_of_ones = np.random.randint(1, n_attrib + 1)
            positions = np.random.choice(n_attrib, number_of_ones, replace=False)
            position1.fill(0)
            position1[positions] = 1
        
        selected_attributes.append(position1)
    
    selected_attributes.append(np.ones(n_attrib, dtype=np.int64))  # Ensure all attributes selection case
    
    return selected_attributes


def dataset_evolution(datasets, train_X, train_Y, val_X, val_Y, test_X, test_Y):
    """
    Processes a list of datasets by generating datasets for newly added rows.

    Args:
        datasets (list of DataFrame): List of datasets containing an 'Attributes' column.
        train_X, train_Y: Training data features and labels.
        val_X, val_Y: Validation data features and labels.
        test_X, test_Y: Test data features and labels.

    Returns:
        list of DataFrame: List of datasets with surrogate data incorporated.
    """
    datasets_final = []
    dataset = pd.DataFrame()
    prev_length = 0

    for d in datasets:
        new_rows = len(d) - prev_length
        
        if new_rows > 0:
            new_data = d['Attributes'].iloc[-new_rows:].to_numpy()
            surrogate_data = create_surrogate_model_dataset(new_data, train_X, train_Y, val_X, val_Y, test_X, test_Y)
            dataset = pd.concat([dataset, surrogate_data], ignore_index=True)
        
        datasets_final.append(dataset.copy())
        prev_length = len(d)
    
    return datasets_final
