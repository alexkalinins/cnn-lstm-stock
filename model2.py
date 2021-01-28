import pickle5 as pickle
from tqdm import tqdm as tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, LSTM, Conv1D, TimeDistributed
from tensorflow.keras.layers import MaxPool1D, Flatten, Activation, Input, BatchNormalization
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

# TODO would pool size 1 even do anything?
#model.add(MaxPool1D(pool_size=1, padding='same'))

# model.add(Activation('swish'))

# TODO batch norm
#model.add(BatchNormalization())


#model.add(Flatten())
model.add(Reshape((10, FILTERS)))


# Apparently tf LSTM defaults to CuDNN when the following conditions are met:
#  1. `activation` == `tanh`
#  2. `recurrent_activation` == `sigmoid`
#  3. `recurrent_dropout` == 0
#  4. `unroll` is `False`
#  5. `use_bias` is `True`
#  6. Inputs are not masked or strictly right padded.
#model.add(LSTM(units=64, input_shape=(-1,6), activation='tanh'))
#model.add(LSTM(units=64, input_shape=(None, 1, 10, 32), activation='tanh'))
model.add(LSTM(units=64, activation='swish'))
# ^ do i need to return sequences?
#model.add()


# single output layer
# paper does not specify activation; going with tanh
model.add(Dense(units=1, activation='swish'))


#optim = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
optim = tf.keras.optimizers.Adam(lr=0.01, decay = 1e-5)

# mean absolute error loss compiling
model.compile(loss='mae', optimizer=optim, metrics=['mae'])

model.build(input_shape=(None, 10, 1, 6))
print(model.summary())






### LOADING DATA AND TRAINING ###
train_x = []
train_y = []

test_x = []
test_y = []


#loading and merging data
with open('./data/new-train.pkl', 'rb') as f:
    train = pickle.load(f)
    for seq in train:
        train_x.append(seq[0].reshape(10,1,6))
        train_y.append(seq[1].reshape(1,1))



#loading and merging data
with open('./data/new-test.pkl', 'rb') as f:
    test = pickle.load(f)
    for seq in test:
        test_x.append(seq[0].reshape(10,1,6))
        test_y.append(seq[1].reshape(1,1))



MODEL_NAME = f'CNN-LSTM-v3-{int(time.time())}'
EPOCHS=100
BATCH_SIZE = 64

# tensorboard code from tutorial
tensorboard = TensorBoard(log_dir=f'./logs/{MODEL_NAME}')

filepath = 'CNN-LSTM-{epoch:02d}-{val_mae:.3f}'
checkpoint = ModelCheckpoint('./models/{}.hd5'.format(filepath, monitor='val_mae', verbose=1, save_best_only=True, mode='max'))

print(train_x[0])

# converting to numpys
train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
test_x = np.asarray(test_x)
test_y = np.asarray(test_y)




#DELETE LATER, JUST FOR TESTING
print('Answer looks like this:')
print(train_y[0])

resh_inp = train_x[0].reshape((1,10,1,6))

print('Reshape')
print(resh_inp)

pred = model.predict(resh_inp, batch_size=1)

print('And the prediction:')
print(pred)

print('The Loss')
print(tf.keras.losses.MAE(train_y[0], pred))




assert not np.any(np.isnan(train_x))
assert not np.any(np.isnan(train_y))
assert not np.any(np.isnan(test_x))
assert not np.any(np.isnan(test_y))


#train = tf.data.Dataset.from_tensor_slices((train_x, train_y))
#test = tf.data.Dataset.from_tensor_slices((test_x, test_y))

print(train_x[0])

# TODO figure out if the shape of data is valid!
history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(test_x, test_y), callbacks=[tensorboard, checkpoint])
#history = model.fit(train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=test, callbacks=[tensorboard, checkpoint])



#DELETE LATER, JUST FOR TESTING
print('Answer looks like this:')
print(train_y[0])

resh_inp = train_x[0].reshape((1,10,1,6))

print('Reshape')
print(resh_inp)

pred = model.predict(resh_inp, batch_size=1)

print('And the prediction:')
print(pred)

print('The Loss')
print(tf.keras.losses.MAE(train_y[0], pred))
















# dont need to save model because of model checkpoint


