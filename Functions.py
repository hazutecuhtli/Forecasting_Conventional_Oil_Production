#*****************************************************************************
# Details
#*****************************************************************************

"""
Auxiliary Functions and Classes for Oil Production Forecasting

This module provides reusable components for training, evaluating, and visualizing
deep learning models (CNN, RNN, LSTM, GRU) in the context of time series forecasting,
specifically applied to conventional oil production.

Functions included:
- Sliding window generation
- Dataset creation
- Prediction plotting
- Confidence interval estimation via MC Dropout

Author: Poncho
"""

#*****************************************************************************
# Importing Libraries
#*****************************************************************************
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
#*****************************************************************************
# Classes and Functions
#*****************************************************************************

class MCModel(Model):
    
    """
    Custom wrapper for enabling Monte Carlo Dropout during inference.

    This subclass overrides the call method to ensure that dropout remains active 
    during prediction time, allowing the generation of multiple stochastic outputs
    for uncertainty estimation using Monte Carlo Dropout.

    Attributes:
        model (tf.keras.Model): The base Keras model to be wrapped.
    """

    def __init__(self, model):
        """
        Initializes the MCModel with the given trained model.

        Args:
            model (tf.keras.Model): A compiled and trained Keras model with dropout layers.
        """
        super(MCModel, self).__init__()
        self.model = model

    def call(self, inputs, training=False):
        """
        Overrides the standard call to force dropout to remain active during prediction.

        Args:
            inputs (tf.Tensor): Input tensor(s) to the model.
            training (bool): This argument is ignored. Dropout is always applied.

        Returns:
            tf.Tensor: The model output with dropout activated, even during inference.
        """
        return self.model(inputs, training=True)



def Generating_Predictions(model, X_val, n_samples, n_future_steps, scaler):

    """
    Generates future oil production forecasts and confidence intervals using Monte Carlo Dropout.

    This function leverages the MCModel wrapper to keep dropout active during inference,
    allowing the generation of multiple stochastic predictions per time step. The results 
    are used to calculate 95% confidence intervals for each forecasted month.

    Args:
        model (tf.keras.Model): Trained TensorFlow/Keras model with dropout layers.
        X_val (tf.Tensor): Validation tensor used as the last known input window for forecasting.
        n_samples (int): Number of stochastic predictions to sample per future time step.
        n_future_steps (int): Number of future time steps (e.g., months) to forecast.
        scaler (sklearn.preprocessing.MinMaxScaler): Fitted scaler used to normalize the production data.

    Returns:
        mean_rescaled (np.ndarray): Mean forecasted values, inverse-transformed to original scale.
        lower_rescaled (np.ndarray): Lower bounds of the 95% confidence interval.
        upper_rescaled (np.ndarray): Upper bounds of the 95% confidence interval.
    """
    
    # Crear wrapper con dropout activado
    mc_model = MCModel(model)
    all_preds = []

    # Última ventana real conocida
    last_window = X_val[-1].numpy().copy()

    # Bucle de predicción multistep con tqdm
    for step in tqdm(range(n_future_steps), desc="Generating forecasts"):

        preds = []

        for _ in range(n_samples):
            input_tensor = tf.convert_to_tensor([last_window], dtype=tf.float32)
            pred = mc_model.predict(input_tensor, verbose=0)[0][0]
            preds.append(pred)

        all_preds.append(preds)

        # Promedio de las muestras para generar la siguiente entrada
        mean_pred = np.mean(preds)

        # Crear nuevas características cíclicas del mes
        next_month_index = (step + 1 + X_val.shape[0]) % 12
        next_month_sin = np.sin(2 * np.pi * next_month_index / 12)
        next_month_cos = np.cos(2 * np.pi * next_month_index / 12)
        next_entry = [mean_pred, next_month_sin, next_month_cos]

        # Actualizar ventana deslizante
        last_window = np.vstack([last_window[1:], next_entry])

    # Convertir a array para calcular estadísticas
    all_preds = np.array(all_preds)

    mean_preds = all_preds.mean(axis=1)
    std_preds = all_preds.std(axis=1)

    # Intervalo de confianza 95%
    lower_bound = mean_preds - 1.96 * std_preds
    upper_bound = mean_preds + 1.96 * std_preds

    # Invertir normalización
    mean_rescaled = scaler.inverse_transform(mean_preds.reshape(-1, 1)).flatten()
    lower_rescaled = scaler.inverse_transform(lower_bound.reshape(-1, 1)).flatten()
    upper_rescaled = scaler.inverse_transform(upper_bound.reshape(-1, 1)).flatten()

    return mean_rescaled, lower_rescaled, upper_rescaled



def Plotting_Training_Losses(history):

    """
    Plots training and validation losses and MAE from a Keras training history object.

    This function extracts the loss and mean absolute error (MAE) metrics recorded 
    during model training and plots them across epochs to visually assess 
    the learning behavior and detect potential issues such as overfitting.

    Args:
        history (tf.keras.callbacks.History): History object returned by the `model.fit()` method, 
            containing the evolution of training and validation metrics.

    Returns:
        None. Displays two subplots: 
            - One for training and validation loss (MSE)
            - One for training and validation MAE
    """
    
    # Getting history data
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    epochs = range(1, len(loss) + 1)
    
    # Plotting Loss
    plt.figure(figsize=(10, 4))    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    
    # Plotting MAE
    plt.subplot(1, 2, 2)
    plt.plot(epochs, mae, 'b-', label='Training MAE')
    plt.plot(epochs, val_mae, 'r-', label='Validation MAE')
    plt.title('Model MAE Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    # Displaying Plots (No needed if using %magic method in the notebook)
    plt.tight_layout()
    plt.show()



def Validation_Prediction(model, X_val, y_val, scaler):

    """
    Generates predictions from a trained model and compares them to real values from the validation set.

    This function uses a trained TensorFlow/Keras model to predict conventional oil production 
    values from a validation dataset and plots them against the corresponding real values. 
    The predicted and actual values are inverse-transformed using the provided scaler 
    to return them to the original production scale.

    Args:
        model (tf.keras.Model): Trained model used to generate predictions.
        X_val (tf.Tensor): Input features from the validation set used for forecasting.
        y_val (np.ndarray or tf.Tensor): True target values corresponding to the validation set.
        scaler (sklearn.preprocessing.MinMaxScaler): Scaler used during training to normalize the production data.

    Returns:
        None. Displays a matplotlib plot comparing predicted and real oil production values.
    """

    # Generating Predictions
    y_pred = model.predict(X_val)
    
    # Rescaling predictions to be in the same range of values than real oil production
    y_pred_rescaled = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))
    y_val_rescaled = scaler.inverse_transform(np.array(y_val).reshape(-1, 1))
                                             
    # Plotting
    plt.figure(figsize=(12, 4))
    plt.plot(y_val_rescaled, label='Valor real', color='blue')
    plt.plot(y_pred_rescaled, label='Predicción', linestyle='--', color='darkorange')
    plt.title('Predicción de producción de petróleo (validación)')
    plt.xlabel('Mes')
    plt.ylabel('Producción (MBD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def Future_Predictions(model, df, X_val, past_samples, n_future_steps, scaler, title, window_size = 4):

    """
    Generates and plots future forecasts of oil production using a trained model.

    This function performs multistep autoregressive forecasting of conventional oil production 
    using a trained TensorFlow/Keras model. Predictions are recursively generated for a specified 
    number of future time steps, using the last known input window. The predicted values are 
    inverse-transformed and plotted along with recent historical data.

    Args:
        model (tf.keras.Model): Trained model used to generate predictions.
        df (pandas.DataFrame): DataFrame containing historical production values and dates.
        X_val (tf.Tensor): Validation input tensor used to extract the last known window.
        past_samples (int): Number of past real production values to display before the forecast.
        n_future_steps (int): Number of future time steps (e.g., months) to predict.
        scaler (sklearn.preprocessing.MinMaxScaler): Fitted scaler used for data normalization.
        title (str): Title for the resulting plot.
        window_size (int, optional): Size of the sliding window used during training and inference. Defaults to 4.

    Returns:
        None. Displays a matplotlib plot comparing recent real production values 
        with predicted values for future months.
    """
    
    # Selecting the last available window from the validation set
    last_window = X_val[-1].numpy().copy()  # Shape: (4, 3) → [scaled, month_sin, month_cos]
    
    # List to store the predicted values (scaled)
    future_preds = []
    
    # Forecasting conventional oil production using multistep autoregression
    for i in range(n_future_steps):
    
        # Creating the input tensor from the current window
        input_tensor = tf.convert_to_tensor([last_window], dtype=tf.float32)
    
        # Predicting the next production value (scaled)
        pred = model.predict(input_tensor, verbose=0)[0][0]
        future_preds.append(pred)
    
        # Generating sine/cosine features for the next month
        next_month_index = (i + 1 + X_val.shape[0]) % 12
        next_month_sin = np.sin(2 * np.pi * next_month_index / 12)
        next_month_cos = np.cos(2 * np.pi * next_month_index / 12)
    
        # Creating the next entry [scaled_pred, sin(month), cos(month)]
        next_entry = [pred, next_month_sin, next_month_cos]
    
        # Updating the window (sliding forward)
        last_window = np.vstack([last_window[1:], next_entry])
    
    # Inversely scaling the predictions using the trained MinMaxScaler
    future_preds_rescaled = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()

    
    # Creating list containing all conventional oil productio values
    oil_prod = df.Petroleo_MBD.tolist()[-past_samples:]+future_preds_rescaled.tolist()
    
    # Producción real (últimos 50 del set de validación)
    real_prod = df.Petroleo_MBD.tolist()[-past_samples:]
    pred_prod = future_preds_rescaled.tolist()
    
    dates = [df.loc[df.shape[0]-1, 'Fecha']]
    for n in range(n_future_steps):
        dates.append(dates[-1]+pd.DateOffset(months=1))
    Fechas = df.Fecha.tolist()[-past_samples:-1] + dates
    
    # Plotting predicted conventional oil production
    fig, ax = plt.subplots(1,1, figsize=(12,4)) 
    ax.plot([fecha.strftime('%Y-%m') for fecha in Fechas], oil_prod, color='darkturquoise', linestyle='--', label='Conventional Oil Production')
    ax.plot([fecha.strftime('%Y-%m') for fecha in Fechas][:-n_future_steps], real_prod, color='blue', linestyle='', marker='^', label='Real Oil Production')
    ax.plot([fecha.strftime('%Y-%m') for fecha in Fechas][-n_future_steps:], pred_prod, color='red', linestyle='', marker='^', label='Predicted Oil Production')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Oil Production (MBD)')
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
    plt.legend()
    plt.tight_layout()



# Function to implement the sliding window approach
def create_multivariate_windows(data, window_size=12):

    """
    Generates input-output pairs using a sliding window approach for time series modeling.

    This function creates overlapping sequences of features (`X`) and corresponding target values (`y`)
    using a sliding window over the input data. It is typically used to prepare datasets for training
    TensorFlow models on time series forecasting tasks.

    Args:
        data (np.ndarray): Numpy array of shape (n_samples, n_features) containing normalized input data.
        window_size (int): Number of past time steps to include in each input window. Defaults to 12.

    Returns:
        X (np.ndarray): Array of shape (n_windows, window_size, n_features) containing input sequences.
        y (np.ndarray): Array of shape (n_windows,) containing target values corresponding to each input window.
    """
    
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])  # From i-window_size to i-1
        y.append(data[i][0])               # Predicting 'scaled' at time i
    
    return np.array(X), np.array(y)
        

def Generating_datasets(df, window_size, train_per):

    """
    Generates training and validation datasets for time series forecasting using a sliding window approach.

    This function selects key features from the input DataFrame (scaled oil production and cyclical month encodings),
    applies a sliding window to create sequences for supervised learning, converts them to TensorFlow tensors,
    and splits them into training and validation sets based on the specified training proportion.

    Args:
        df (pandas.DataFrame): DataFrame containing the normalized oil production and cyclical date features.
        window_size (int): Number of time steps to include in each input sequence.
        train_per (float): Proportion of the data to use for training (between 0 and 1).

    Returns:
        X_train (tf.Tensor): Training input tensor of shape (n_train_samples, window_size, n_features).
        y_train (tf.Tensor): Training target tensor of shape (n_train_samples,).
        X_val (tf.Tensor): Validation input tensor of shape (n_val_samples, window_size, n_features).
        y_val (tf.Tensor): Validation target tensor of shape (n_val_samples,).
    """

    # Selecting features for prediction (scaled target + cyclical month encoding)
    features = df[["scaled_oil", "month_sin", "month_cos"]].values

    # Creating datasets: input sequences and target values
    X, y = create_multivariate_windows(features, window_size)
    
    # Converting to TensorFlow tensors
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
    
    # Displaying tensor shapes
    print("X shape:", X_tensor.shape)  # (samples, time_steps, features)
    print("y shape:", y_tensor.shape)  # (samples,)


    # Defining the number of elements to split the tensors into training and validation datasets
    split = int(len(X_tensor) * train_per)
    
    # Generating the training and validation datasets
    X_train, X_val = X_tensor[:split], X_tensor[split:]
    y_train, y_val = y_tensor[:split], y_tensor[split:]

    return X_train, y_train, X_val, y_val

#*****************************************************************************
# Fin
#*****************************************************************************