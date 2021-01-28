import yfinance as yf
import pandas as pd
import math
import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras

MODEL_PATH = './models/CNN-LSTM-26-0.219.hd5' 

if len(sys.argv) != 2:
    print('Usage: predict.py <TICKER>')
    sys.exit(1)


ticker = sys.argv[1]

print('Downloading data')

# if ticker is wrong program will shut down
data = yf.Ticker(ticker).history(period='1mo', interval='1d')
# data is one month; only need last 10 days

print('Done! Processing data')

last10 = data.iloc[-10:].drop(columns=['Dividends', 'Stock Splits'])

def check_nan(df):
    if(df.isnull().values.any()):
        print('Invalid data! Select a different ticker, or try again later.')
        sys.exit(1)

check_nan(last10)

last10['DayOfWeek'] = [i.dayofweek for i in last10.index]


mean = last10['Close'].mean()  # mean of close data
std = last10['Close'].std()  # std of close data

# standardizing the data
for column in last10.columns:
    s = last10[column].std()

    last10[column] = (last10[column]-last10[column].mean()) / s if s != 0 else 0


model_in = last10.values.reshape(1, 10, 1, 6)

print('Done! Loading model')

model = keras.models.load_model(MODEL_PATH)

print('Done! Making prediction')
model_out = model.predict(model_in, batch_size=1)

output = model_out.reshape(1)

# destandardizing the input
output = output * std + mean

print(f'=== PREDICTION ({ticker}) ===')
print(f'Next Day Close:\t{output}')




