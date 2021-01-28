import pickle5 as pickle
from tqdm import tqdm as tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, LSTM, Conv1D
from tensorflow.keras.layers import Flatten, Activation, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import activations
import random
import time

# model architecture from https://doi.org/10.1155/2020/6622927

# Data is 6x10: 6 paramaters, 10 timestamps
model = Sequential()
FILTERS=32


# trying not to use inputshape
model.add(Conv1D(filters=FILTERS, kernel_size=1, activation='swish', padding='same'))

# TODO decide batch norm
#model.add(BatchNormalization())


#model.add(Flatten())
model.add(Reshape((10, FILTERS)))

model.add(LSTM(units=64, activation='swish'))

# single output layer
model.add(Dense(units=1, activation='swish'))


#optim = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
optim = tf.keras.optimizers.Adam(lr=0.01, decay = 1e-5)

# mean absolute error loss compiling
model.compile(loss='mae', optimizer=optim, metrics=['mae'])

model.build(input_shape=(None, 10, 1, 6))
print(model.summary())




DAY = 3  # how many days into the future (1, 2, or 3)

### LOADING DATA AND TRAINING ###
train_x = []
train_y = []

test_x = []
test_y = []


#loading and merging data
with open(f'./data/train{DAY}.pkl', 'rb') as f:
    train = pickle.load(f)
    for seq in train:
        train_x.append(seq[0].reshape(10,1,6))
        train_y.append([[seq[1]]])



#loading and merging data
with open(f'./data/test{DAY}.pkl', 'rb') as f:
    test = pickle.load(f)
    for seq in test:
        test_x.append(seq[0].reshape(10,1,6))
        test_y.append([[seq[1]]])


MODEL_NAME =f'NEXT{DAY}-{int(time.time())}'
EPOCHS=20
BATCH_SIZE = 64

SUBDIR = f'day{DAY}'

# tensorboard code from tutorial
tensorboard = TensorBoard(log_dir=f'./logs/{SUBDIR}/{MODEL_NAME}')

filepath = MODEL_NAME + '-{epoch:02d}-{val_mae:.3f}'
checkpoint = ModelCheckpoint(('./models/'+SUBDIR+'/{}.h5').format(filepath, monitor='val_mae', verbose=1, save_best_only=True, mode='max'))

print(train_x[0])

# converting to numpys
train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
test_x = np.asarray(test_x)
test_y = np.asarray(test_y)



assert not np.any(np.isnan(train_x))
assert not np.any(np.isnan(train_y))
assert not np.any(np.isnan(test_x))
assert not np.any(np.isnan(test_y))


history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(test_x, test_y), callbacks=[tensorboard, checkpoint])
# dont need to save model because of model checkpoint


