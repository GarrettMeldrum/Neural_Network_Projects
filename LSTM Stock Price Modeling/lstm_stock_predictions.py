# Libraries to import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf   
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
import sys

# For some reason I was getting an error with the encoding of the data, so I had to change the encoding to utf-8
sys.stdout.reconfigure(encoding='utf-8')


ticker = yf.Ticker("MSFT").history(period="10y", interval="1d")


time_start = 2015
time_end = 2023



'''
# Plotting the data
def train_test_plot (ticker, time_start, time_end):
    ticker.loc[f"{time_start}":f"{time_end}"]['Close'].plot(figsize=(16, 4), legend=True, label='Close Price history')
    ticker.loc[f"{time_end+1}":, "Close"].plot(figsize=(16, 4), legend=True, label='Close Price history')
    plt.legend([f"Train (Before {time_end+1})", f"Test ({time_end+1} and beyond)"])
    plt.title(f"{ticker} Stock Price")
    plt.show()

train_test_plot(ticker, time_start, time_end)
'''


# Splitting the data into training and testing data
def train_test_split (ticker, time_start, time_end):
    train = ticker.loc[f"{time_start}":f"{time_end}", "Close"].values
    test = ticker.loc[f"{time_end+1}":, "Close"].values
    return train, test
training_set, test_set = train_test_split(ticker, time_start, time_end)


# Normalizing (Standardize) our trainning set to avoid outliers and anomalies
sc = MinMaxScaler(feature_range=(0,1))
training_set = training_set.reshape(-1,1)
training_set_scaled = sc.fit_transform(training_set)


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


n_steps = 60
features = 1
# Split into samples
X_train, y_train = split_sequence(training_set_scaled, n_steps)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], features))


# LSTM Model
model_lstm = Sequential()
model_lstm.add(LSTM(units=125, activation='tanh', input_shape=(n_steps, features)))
model_lstm.add(Dense(units=1))

# Compile the LSTM Model
# Optimizer used can be changed, for stock predicitions I have seen Adam and RMSprop used
model_lstm.compile(optimizer='RMSprop', loss='mean_squared_error')
model_lstm.summary()


# Fit the LSTM Model
model_lstm.fit(X_train, y_train, epochs=50, batch_size=32)


dataset_total = ticker.loc[:, 'Close']
inputs = dataset_total[len(dataset_total) - len(test_set) - n_steps:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)


# Scaling the inputs
X_test, y_test = split_sequence(inputs, n_steps)
# Reshape the data
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], features))
# Prediction
predicted_stock_price = model_lstm.predict(X_test)
# Inverse transform the data
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Plotting the data
def plot_predictions(test, predicted):
    plt.plot(test, color='gray', label='Real Stock Price')
    plt.plot(predicted, color='blue', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# RMSE
def return_rmse(test, predicted):
    rmse = np.sqrt(mean_squared_error(test, predicted))
    print("The root of mean squared error is {:.2f}".format(rmse))

plot_predictions(test_set, predicted_stock_price)
return_rmse(test_set, predicted_stock_price)