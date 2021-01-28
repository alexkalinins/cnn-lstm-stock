import yfinance as yf
import pandas as pd
import math
from tqdm import tqdm as tqdm
import pickle
import numpy as np
import random
from multiprocessing import Pool

pd.options.mode.chained_assignment = None  # default='warn'


tickers = ['TSLA',
           'AAPL',
           'MSFT',
           'U',
           'CCL',
           'TD',
           'SPY',
           'FB',
           'V',
           'DIS',
           'CNR',
           'HD',
           'UNH',
           'MCD',
           'MMM',
           'ATVI',
           'ADBE',
           'AMD',
           'GOOG',
           'AMZN',
           'AXP',
           'BAC',
           'BA',
           'CVX',
           'C',
           'KO',
           'DOW',
           'GM',
           'GILD',
           'INTC',
           'MA',
           'NVDA',
           'TXN',
           'XRX',
           'RY.TO',
           'CP.TO',
           'TRI.TO',
           'ATD-B.TO',
           'L.TO',
           'DOL.TO',
           'BB.TO',
           'DOO.TO',
           'WEED.TO',
           'SNC.TO',
           'SHOP',
           'SU.TO',
           'CM.TO',
           'TD.TO',
           'ENB.TO',
           'APHA.TO',
           'XIU.TO', # s&p/tsx composite 60
           'AC.TO']

raws = {}

print('Downloading Data')

for ticker in tqdm(tickers):
    t = yf.Ticker(ticker)
    t_data = t.history(period = '10y', interval = '1d')
    
    raws[ticker] = t_data


print('Done')


col_d = {} # dropped columns

for ticker in tickers:
    raw = raws[ticker]

    col_d[ticker] = raw.drop(columns=['Dividends', 'Stock Splits'])



# Data standardization

# df is dataframe of 10 rows; f1, f2, f3 are ground truth future values for closign price
def z_score(df, f1, f2, f3):
    # DAY OF WEEK ALSO GETS STANDARDIZED
    df['DayOfWeek']=[i.dayofweek for i in df.index]

    for column in df.columns:
        std = df[column].std()
        mean = df[column].mean()

        # if all values are the same, then std=0, so dividing by std would make a NaN
        df[column] = (df[column] - mean) / std if std != 0 else 0

        # future value is excluded from std and mean calculations
        if column == 'Close':
            f1 = (f1-mean)/std if std != 0 else 0
            f2 = (f2-mean)/std if std != 0 else 0
            f3 = (f3-mean)/std if std != 0 else 0



    nan_flag = df.isnull().values.any() or math.isnan(f1) or math.isnan(f2) or math.isnan(f3)

    # return first 10 rows of df, excluding the future row, also nan flag to throw out nan sequences
    return df.values, f1, f2, f3, nan_flag

# lists of lists of sequences (by ticker)
a1 = []
a2 = []
a3 = []

# number of dropped sequences
dropped=0

# converts data frame into sequences; df is tickers history
def sequencify(df):
    s1,s2,s3=[],[],[]
    dropped = 0

    for i in range(len(df.index)-13):
        sequence, f1, f2, f3, nan_flag = z_score(df.iloc[i:i+10], df['Close'].iloc[i+10], df['Close'].iloc[i+11], df['Close'].iloc[i+12])

        if(nan_flag):
            dropped += 1
        else:
            s1.append([sequence, f1])
            s2.append([sequence, f2])
            s3.append([sequence, f3])

    return [s1,s2,s3, dropped]




print('Generating sequences. This may take a while...')

# you don't want to be doing this without multiprocessing...
with Pool(8) as pool:
    results = pool.imap_unordered(sequencify, list(col_d.values()))

    for res in results:
        a1.append(res[0])
        a2.append(res[1])
        a3.append(res[2])
        dropped += res[3]


print(f'Done! Dropped {dropped} sequences containing NaN')


print('Separating and shuffling sequences')

train1, test1 = [], []
train2, test2 = [], []
train3, test3 = [], []

#test : train
RATIO = 0.05


# iterating by histories of each ticker
for hist in a1:
    split = math.floor(len(hist) * RATIO)
    train1.extend(hist[:-split])
    test1.extend(hist[-split:])

for hist in a2:
    split = math.floor(len(hist) * RATIO)
    train2.extend(hist[:-split])
    test2.extend(hist[-split:])

for hist in a3:
    split = math.floor(len(hist) * RATIO)
    train3.extend(hist[:-split])
    test3.extend(hist[-split:])


random.shuffle(train1)
random.shuffle(train2)
random.shuffle(train3)

random.shuffle(test1)
random.shuffle(test2)
random.shuffle(test3)

print('Done')

# little test
for t in test1:
    seq = t[0]
    assert len(seq) == 10
    
    for r in seq:
        assert len(r) == 6


print('Saving data to file')

with open('./data/train1.pkl', 'wb') as f:
    pickle.dump(train1, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('./data/test1.pkl', 'wb') as f:
    pickle.dump(test1, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('./data/train2.pkl', 'wb') as f:
    pickle.dump(train2, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('./data/test2.pkl', 'wb') as f:
    pickle.dump(test2, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('./data/train3.pkl', 'wb') as f:
    pickle.dump(train3, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('./data/test3.pkl', 'wb') as f:
    pickle.dump(test3, f, protocol=pickle.HIGHEST_PROTOCOL)


print('Done!')


