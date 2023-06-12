import os
import math
import csv
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from dataclasses import dataclass
from absl import logging

def plot_series(time, series, format="-", start=0, end=None,
                title=None, xlabel=None, ylabel=None, legend=None):
    
    """
    Visualizes time series data

    Args:
      time (array of int) - contains values for the x-axis
      series (array of int or tuple of arrays) - contains the values for the y-axis
      format (string) - line style when plotting the graph
      start (int) - first time step to plot
      end (int) - last time step to plot
      title (string) - title of the plot
      xlabel (string) - label for the x-axis
      ylabel (string) - label for the y-axis
      legend (list of strings) - legend for the plot
    """
    
    # Check if there are more than two series to plot
    if type(series) is tuple:

      # Loop over the series elements
      for series_curr in series:

        # Plot the time and current series values
        plt.plot(time[start:end], series_curr[start:end], format)

    else:
      # Plot the time and series values
      plt.plot(time[start:end], series[start:end], format)
    
    # Set the title
    plt.title(title)
    
    # Label the x-axis
    plt.xlabel(xlabel)

    # Label the y-axis
    plt.ylabel(ylabel)

    # Set the legend
    if legend:
      plt.legend(legend)
    
    # Overlay a grid on the graph
    plt.grid(True)

    # Draw the graph on screen
    plt.show()

def parse_data_from_df(dataframe, column=str):

    """
    Parsing datas from dataframe

    Args:
      dataframe (pandas dataframe) - contains values for the x-axis

    Returns:
      times (int) - contains values for the x-axis
      values (numpy array) - contains the values for the y-axis
    """
    
    values = dataframe.loc[:, [column]].values
    
    times = np.array([rowidx for rowidx in range(0, len(values))])
            
    return times, values

def train_val_split(time, series, time_step, time_stop):

    time_train = time[:time_step]
    series_train = series[:time_step]
    
    time_valid = time[time_step:time_stop]
    series_valid = series[time_step:time_stop]

    return time_train, series_train, time_valid, series_valid

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):

    """
    Generates dataset windows

    Args:
      series (array of float) - contains the values of the time series
      window_size (int) - the number of time steps to include in the feature
      batch_size (int) - the batch size
      shuffle_buffer(int) - buffer size to use for the shuffle method

    Returns:
      dataset (TF Dataset) - TF Dataset containing time windows
    """
    
    # Generate a TF Dataset from the series values
    ds = tf.data.Dataset.from_tensor_slices(series)
    
    # Window the data but only take those with the specified size
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    
    # Flatten the windows by putting its elements in a single batch
    ds = ds.flat_map(lambda window: window.batch(window_size + 1))

    # Shuffle the windows
    ds = ds.shuffle(shuffle_buffer)

    # Create tuples with features and labels 
    ds = ds.map(lambda window: (window[:-1], window[-1]))
    
    # Create batches of windows
    ds = ds.batch(batch_size).prefetch(1)
    
    return ds

def compute_metrics(true_series, forecast):
    
    # Make sure float32 datatype (for metric calculations)
    true_series = tf.cast(true_series, dtype=tf.float32)
    forecast = tf.cast(forecast, dtype=tf.float32)

    mse = tf.keras.metrics.mean_squared_error(true_series, forecast)
    mae = tf.keras.metrics.mean_absolute_error(true_series, forecast)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(true_series, forecast)

    # Account for different sized metrics (for longer horizons, we want to reduce metrics to a single value)
    if mae.ndim > 0:
      mae = tf.reduce_mean(mae)
      mse = tf.reduce_mean(mse)
      rmse = tf.reduce_mean(rmse)
      mape = tf.reduce_mean(mape)

    return {"mae": mae.numpy(),
          "mse": mse.numpy(),
          "rmse": rmse.numpy(),
          "mape": mape.numpy()}

def visualize_mae_loss(history):
    # Get mae and loss from history log
    mae=history.history['mae']
    loss=history.history['loss']

    # Get number of epochs
    epochs=range(len(loss)) 

    # Plot mae and loss
    plot_series(
        time=epochs, 
        series=(mae, loss), 
        title='MAE and Loss', 
        xlabel='MAE',
        ylabel='Loss',
        legend=['MAE', 'Loss']
        )

    # Only plot the last 80% of the epochs
    zoom_split = int(epochs[-1] * 0.2)
    epochs_zoom = epochs[zoom_split:]
    mae_zoom = mae[zoom_split:]
    loss_zoom = loss[zoom_split:]

    # Plot zoomed mae and loss
    plot_series(
        time=epochs_zoom, 
        series=(mae_zoom, loss_zoom), 
        title='MAE and Loss', 
        xlabel='MAE',
        ylabel='Loss',
        legend=['MAE', 'Loss']
        )

def model_forecast(model, series, window_size, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(batch_size).prefetch(1)
    forecast = model.predict(ds)
    return forecast

def evaluate_forecast(model, time, series, time_valid, series_valid,
                      time_step, time_stop, window_size, batch_size):
  # Slice the forecast to get only the predictions for the validation set
  forecast_valid_series = series[time_step - window_size:time_stop]

  # Compute the forecast for all the series
  forecast_valid = model_forecast(model, forecast_valid_series, window_size, batch_size).squeeze()
  forecast = model_forecast(model, series, window_size, batch_size).squeeze()

  # Plot the forecast
  plt.figure(figsize=(10, 6))
  plot_series(time_valid, (series_valid, forecast_valid))

  plt.figure(figsize=(10, 6))
  plot_series(time[window_size-1:], (series[window_size-1:], forecast))
  
  return forecast_valid

def make_future_forecast(values, model, into_future, window_size) -> list:
    """
    Make future forecasts into_future steps after value ends.

    Returns future forecasts as a list of floats.
    """
    # Create an empty list for future forecasts/prepare data to forecast on
    future_forecast = []
    last_window = values[-window_size:]

    # Make INTO_FUTURE number of predictions, altering the data which gets predicted on each
    for _ in range(into_future):
        # Predict on the last window then append it again, again, again (our model will eventually start to make forecasts on its own forecasts)
        future_pred = model.predict(tf.expand_dims(last_window, axis=0))
        print(f"Predicting on:\n {last_window} -> Prediction: {tf.squeeze(future_pred).numpy()}\n")

        # Append preds to future_forecast
        future_forecast.append(tf.squeeze(future_pred).numpy())

        # Update last window with new pred and get WINDOW_SIZE most recent preds (model was trained on WINDOW_SIZE windows)
        last_window = np.append(last_window, future_pred)[-window_size:]

    return future_forecast

def plot_future_forecast(timesteps, values, format=".", start=0, end=None, label=None, xlabel=None, ylabel=None):
    """
    Plots timesteps ( a series of points in time) against values (a series of values across timesteps)

    Parameters
    ----------
    timesteps: array of timestep values
    values: array of values across time
    format: style of plot, default "."
    start: where to start the plot (setting a value will index from start of timesteps & values)
    end: where to end the plot (similar to start but for the end)
    label: label to show on plot about values, default None
    """

    # Plot the series
    plt.plot(timesteps[start:end], values[start:end], format, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if label:
        plt.legend(fontsize=14) # make label bigger
    plt.grid(True)

PATH = "./saved_model"

def get_model_dir(path):
    
    if not os.path.exists(path):
        os.makedirs(path)

    model_dirs = [int(i) for i in os.listdir(path)]

    current_model = "1" if len(model_dirs)==0 else str(max(model_dirs)+1)

    return path + "/" + current_model + "/"


# Create a function to implement a ModelCheckpoint callback with a specific filename
def create_model_checkpoint(save_path="saved_model"):
  return tf.keras.callbacks.ModelCheckpoint(filepath=get_model_dir(save_path),
                                            verbose=0, # only output a limited amount of text
                                            save_best_only=True)