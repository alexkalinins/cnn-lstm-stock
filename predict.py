import yfinance as yf
import pandas as pd
import math
import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras

MODEL_1 = './models/day1/NEXT1-E19.h5' 
MODEL_2 = './models/day2/NEXT2-E18.h5' 
MODEL_3 = './models/day3/NEXT3-E6.h5' 

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

print('Done! Loading models')

model1 = keras.models.load_model(MODEL_1)
model2 = keras.models.load_model(MODEL_2)
model3 = keras.models.load_model(MODEL_3)

print('Done! Making prediction')
model_out_1 = model1.predict(model_in, batch_size=1)
model_out_2 = model2.predict(model_in, batch_size=1)
model_out_3 = model3.predict(model_in, batch_size=1)

output1 = model_out_1[0][0]
output2 = model_out_2[0][0]
output3 = model_out_3[0][0]

# destandardizing the input
output1 = round(output1 * std + mean, 2)
output2 = round(output2 * std + mean, 2)
output3 = round(output3 * std + mean, 2)

print('')
print(f'=== PREDICTION ({ticker}) ===')
print(f'Today+1:\t${output1}')
print(f'Today+2:\t${output2}')
print(f'Today+3:\t${output3}')




