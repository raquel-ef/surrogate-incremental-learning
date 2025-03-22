import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
import tensorflow as tf # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Dropout, InputLayer # type: ignore
import config.config as config


def calculate_errors(test_y, pred):
    """
    Calculate Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Correlation Coefficient (CC).
    
    Parameters:
    test_y (array-like): True target values.
    pred (array-like): Predicted values.
    
    Returns:
    tuple: (mae, rmse, cc) where:
        - mae: Mean Absolute Error
        - rmse: Root Mean Squared Error
        - cc: Correlation Coefficient
    """
    mae = mean_absolute_error(test_y, pred)
    rmse = np.sqrt(mean_squared_error(test_y, pred))
    cc = np.corrcoef(test_y.T, pred.T)[1,0]    
    return mae, rmse, cc


def normalized_means(result_steps, sizes, maximize=False):
    """
    Normalize error metrics from h-steps-ahead predictions to allow model comparison.
    
    Parameters:
    result_steps (list of arrays): Error metrics for h-steps-ahead predictions.
    sizes (list of int): List of sizes used for splitting the data.
    maximize (bool, optional): If True, inverts the metric (for correlation coefficient normalization).
    
    Returns:
    list: Normalized mean value.
    """
    ms = []
    minim = 0
    maxim = 1

    # Concatenate the error metrics into a single array, using list accumulation
    for data in result_steps:
        ms.append(data if not maximize else 1 - data)  # Avoid double concatenation
    ms = np.concatenate(ms)

    # Normalize the array
    norm = (ms - minim) / (maxim - minim)

    # Split and calculate means more efficiently
    norm = np.array_split(norm, np.cumsum(sizes)[:-1])  # Efficient splitting based on cumulative sizes
    meanList = [np.round(np.mean(i), 6) for i in norm]

    return meanList[0]


def calculate_H(df, n_steps):
    """
    Calculate a summary metric "H" based on normalized error metrics (RMSE, MAE, CC).
    
    Parameters:
    df (DataFrame): Data containing RMSE, MAE, and CC values for different steps ahead.
    n_steps (int): Number of prediction steps.
    
    Returns:
    float: H score.
    """

    # Filter out rows where 'RMSE StepsAhead' is empty
    dfSteps = df.copy() # df[df['RMSE StepsAhead'].str.len() > 0]

    stepsRMSE = []
    stepsMAE = []
    stepsCC = []
    
    stepsRMSE.append(dfSteps['RMSE'][1:])
    stepsMAE.append(dfSteps['MAE'][1:])
    stepsCC.append(dfSteps['CC'][1:])

    sizes = [n_steps] 

    rmse = normalized_means(stepsRMSE, sizes)
    mae = normalized_means(stepsMAE, sizes)
    cc = normalized_means(stepsCC, sizes, maximize=True)

    return np.mean([rmse, mae, cc])


def predictions_h_stepsahead(testX, testy, model, n_steps):
    """
    Perform h-steps-ahead predictions using a machine learning model.
    
    Parameters:
    testX (DataFrame): Feature set for predictions.
    testy (DataFrame): True target values.
    model (Model): Trained machine learning model.
    n_steps (int): Number of steps ahead for prediction.
    
    Returns:
    tuple:
        - DataFrame: Error metrics (RMSE, MAE, CC) for each prediction step.
        - DataFrame: Predictions for each step ahead.
        - DataFrame: Updated testX with lagged predictions.
    """
    # Extract lags from column names
    predicted_attribute = testy.columns[0]
    lag_columns = testX.filter(regex=(f"{predicted_attribute}.*")).columns
    selected_lags = [int(col.split("_")[2]) for col in lag_columns]

    # Reset indices for test data
    test_X = testX.reset_index(drop=True)
    test_y = testy.reset_index(drop=True)

    # Initialize results DataFrame
    predictions = pd.DataFrame(index=test_X.index)
    results = []

    # 1-step ahead prediction
    predictions["pred1"] = model.predict(test_X).ravel()
    mae, rmse, cc = calculate_errors(test_y, predictions[["pred1"]])
    results.append({'RMSE': rmse, 'MAE': mae, 'CC': cc})

    # Handle case when no lagged variables exist
    if not selected_lags:
        print("No lagged variables found")


    # Multi-step ahead predictions (when lagged variables exist)
    for step in range(2, n_steps + 2):
        for lag in range(1, step):
            shift_lag = step - lag
            if shift_lag == 1 and 1 not in selected_lags:
                # Replace first lag with the smallest available lag
                col_name = f"Lag_{predicted_attribute}_{selected_lags[0]}"
                test_X[col_name] = predictions[f'pred{lag}'].shift(shift_lag)
            elif shift_lag in selected_lags:
                col_name = f"Lag_{predicted_attribute}_{shift_lag}"
                test_X[col_name] = predictions[f'pred{lag}'].shift(shift_lag)

        # Drop NaN values before prediction
        valid_X = test_X.dropna()
        pred = model.predict(valid_X.to_numpy()).ravel()

        # Insert NaN padding for alignment
        predictions[f'pred{step}'] = np.concatenate((np.full(step - 1, np.nan), pred))

        # Calculate errors
        valid_preds = predictions[f'pred{step}'][step - 1:]
        mae, rmse, cc = calculate_errors(test_y.iloc[step - 1:], valid_preds.to_frame())
        results.append({'RMSE': rmse, 'MAE': mae, 'CC': cc})

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return results_df, predictions, test_X


def predictions_h_stepsahead_LSTM(testX, testy, model, n_steps):
    """
    Perform h-steps-ahead predictions using an LSTM model.
    
    Parameters:
    testX (DataFrame): Feature set for predictions.
    testy (DataFrame): True target values.
    model (LSTM Model): Trained LSTM model.
    n_steps (int): Number of steps ahead for prediction.
    
    Returns:
    tuple:
        - DataFrame: Error metrics (RMSE, MAE, CC) for each prediction step.
        - DataFrame: Predictions for each step ahead.
        - DataFrame: Updated testX with lagged predictions.
    """

    test_X = testX.reset_index(drop=True)
    test_y = testy.reset_index(drop=True)
    predicted_attr = test_y.columns[0]
    
    # Extract lags from column names
    listaAtribSelected = sorted([int(col.split("_")[2]) for col in test_X.filter(regex=(predicted_attr + ".*")).columns])

    # Initialize results DataFrame
    predicciones = pd.DataFrame(index=test_X.index)
    dfResultados = []

    if listaAtribSelected:  # Ensure there are lagged target variables for step-ahead predictions
        # 1-step ahead prediction
        x_reshape = test_X.to_numpy().reshape(test_X.shape[0], 1, test_X.shape[1])
        predicciones["pred1"] = model.predict(x_reshape, verbose=0).ravel()

        rmse, mae, cc = calculate_errors(test_y, predicciones[['pred1']])
        dfResultados.append({'RMSE': rmse, 'MAE': mae, 'CC': cc})

        # h-step ahead predictions
        for i in range(2, n_steps + 2):
            for j in range(1, i):
                lag = i - j
                if lag == 1 and 1 not in listaAtribSelected:
                    first_lag = listaAtribSelected[0]
                    test_X[f"Lag_{predicted_attr}_{first_lag}"] = predicciones[f'pred{j}'].shift(lag)
                elif lag in listaAtribSelected:
                    test_X[f"Lag_{predicted_attr}_{lag}"] = predicciones[f'pred{j}'].shift(lag)

            arrayX = test_X.dropna().to_numpy().reshape(-1, 1, test_X.shape[1])
            predNa = np.insert(model.predict(arrayX, verbose=0), 0, [np.nan] * (i - 1))

            predicciones[f'pred{i}'] = predNa[:len(predNa)]

            rmse, mae, cc = calculate_errors(test_y.iloc[(i-1):], predicciones[[f'pred{i}']].iloc[(i-1):])
            dfResultados.append({'RMSE': rmse, 'MAE': mae, 'CC': cc})

    return pd.DataFrame(dfResultados), predicciones, test_X
    

def create_surrogate_model_dataset(attribute_combinations, train_X, train_Y, val_X, val_Y, test_X, test_Y):
    """
    Trains LSTM models on different attribute selections and collects evaluation metrics.
    
    Parameters:
    attribute_combinations (list): List of binary numpy arrays representing attribute selections.
    train_X, train_Y: Training data and target values.
    val_X, val_Y: Validation data and target values.
    test_X, test_Y: Test data and target values.
    
    Returns:
    pd.DataFrame: A dataframe containing performance metrics for each attribute selection.
    """
    results = []
    
    for idx, attrib_selection in enumerate(attribute_combinations, start=1):
        # print(f"{idx}/{len(attribute_combinations)}")
        
        selected_columns = train_X.columns[np.asarray(attrib_selection, dtype=bool)]
        trainX_sel, validacionX_sel, testX_sel = [df[selected_columns] for df in [train_X, val_X, test_X]]
        trainXtimeseries_sel = trainX_sel.values.reshape(trainX_sel.shape[0], 1, trainX_sel.shape[1])
        
        n_selected = np.sum(attrib_selection)
        
        # Model creation
        random.seed(config.SEED_VALUE)
        np.random.seed(config.SEED_VALUE)
        tf.random.set_seed(config.SEED_VALUE)

        model = Sequential([
            InputLayer(shape=(1, n_selected)),
            LSTM(units=config.N_NEURONS, activation='relu', return_sequences=True),
            Dropout(0.2),
            Dense(1, activation="linear")
        ])
        
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam',
                      metrics=[tf.keras.metrics.RootMeanSquaredError()])
        model.fit(trainXtimeseries_sel, train_Y, batch_size=config.BATCH_SIZE, epochs=config.EPOCHS, verbose=0)
        
        # Generate predictions
        results_dict = {
            'Attributes': attrib_selection,
            'N selected': n_selected
        }
        
        for dataset, label, target in zip([trainX_sel, validacionX_sel, testX_sel], ['Train', 'Val', 'Test'], [train_Y, val_Y, test_Y]):
            stepsAhead, _, _ = predictions_h_stepsahead_LSTM(dataset, target, model, config.N_STEPS)
            results_dict.update({
                f'RMSE StepsAhead {label}': np.round(stepsAhead['RMSE'].values, 4),
                f'MAE StepsAhead {label}': np.round(stepsAhead['MAE'].values, 4),
                f'CC StepsAhead {label}': np.round(stepsAhead['CC'].values, 4),
                f'H {label}': 1000 if (np.sum(attrib_selection[-config.N_STEPS:]) == 0 or n_selected == 0) else np.round(calculate_H(stepsAhead, config.N_STEPS), 4)
            })
        
        results.append(results_dict)
    
    return pd.DataFrame(results)


def train_incremental_real_models(results, train_X, train_Y, val_X, val_Y, test_X, test_Y, seed):
    """
    Trains real models incrementally based on attribute selections and evaluates them.
    
    Parameters:
    results (list): List of attribute selection results from optimization.
    train_X, train_Y: Training data and target values.
    val_X, val_Y: Validation data and target values.
    test_X, test_Y: Test data and target values.
    seed (int): Random seed for reproducibility.
    
    Returns:
    pd.DataFrame: A dataframe containing performance metrics for each attribute selection.
    """
    results_data = []
    
    for s in results:
        attrib_selection = [int(v[0]) for v in s.variables]
        n_selected = np.sum(attrib_selection)
        
        selected_columns = train_X.columns[np.asarray(attrib_selection, dtype=bool)]
        trainX_sel, validacionX_sel, testX_sel = [df[selected_columns] for df in [train_X, val_X, test_X]]
        trainXtimeseries_sel = trainX_sel.values.reshape(trainX_sel.shape[0], 1, trainX_sel.shape[1])
        
        # Set random seed
        random.seed(config.SEED_VALUE)
        np.random.seed(config.SEED_VALUE)
        tf.random.set_seed(config.SEED_VALUE)
        
        # Model creation
        model = Sequential([
            InputLayer(shape=(1, n_selected)),
            LSTM(units=config.N_NEURONS, activation='relu', return_sequences=True),
            Dropout(0.2),
            Dense(1, activation="linear")
        ])
        
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam',
                      metrics=[tf.keras.metrics.RootMeanSquaredError()])
        model.fit(trainXtimeseries_sel, train_Y, batch_size=config.BATCH_SIZE, epochs=config.EPOCHS, verbose=0)
        
        # Generate predictions
        result_entry = {'Seed': seed, 'Attributes': attrib_selection, 'N selected': n_selected}
        
        for dataset, label, target in zip([trainX_sel, validacionX_sel, testX_sel], ['Train', 'Val', 'Test'], [train_Y, val_Y, test_Y]):
            stepsAhead, _, _ = predictions_h_stepsahead_LSTM(dataset, target, model, config.N_STEPS)

            if stepsAhead.empty:
                result_entry.update({
                    f'RMSE StepsAhead {label}': np.nan,
                    f'MAE StepsAhead {label}': np.nan,
                    f'CC StepsAhead {label}': np.nan,
                    f'H {label}': np.nan
                })
            else:
                result_entry.update({
                    f'RMSE StepsAhead {label}': np.round(stepsAhead['RMSE'].values, 4),
                    f'MAE StepsAhead {label}': np.round(stepsAhead['MAE'].values, 4),
                    f'CC StepsAhead {label}': np.round(stepsAhead['CC'].values, 4),
                    f'H {label}': 1000 if (np.sum(attrib_selection[-config.N_STEPS:]) == 0 or n_selected == 0) else np.round(calculate_H(stepsAhead, config.N_STEPS), 4)
                })
        
        results_data.append(result_entry)
    
    return pd.DataFrame(results_data)