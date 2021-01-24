import pickle
from tqdm import tqdm as tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D
from tensorflow.keras.layers import MaxPool1D, Flatten, Activation
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import activations
import random
import time

# model architecture from https://doi.org/10.1155/2020/6622927

# Data is 6x10: 6 paramaters, 10 timestamps
model = Sequential()

# sus input shape; check if shit goes wrong
model.add(Conv1D(filters=32, kernel_size=1, activation='tanh', padding='same', input_shape=(10, 6)))

# pooling
model.add(MaxPool1D(pool_size=1, padding='same'))

# paper says to add activation to MaxPool; kidna weird
model.add(Activation(activations.relu))

# paper says nothing about flattening so idk
model.add(Flatten())


# Apparently tf LSTM defaults to CuDNN when the following conditions are met:
#  1. `activation` == `tanh`
#  2. `recurrent_activation` == `sigmoid`
#  3. `recurrent_dropout` == 0
#  4. `unroll` is `False`
#  5. `use_bias` is `True`
#  6. Inputs are not masked or strictly right padded.
model.add(LSTM(units=64, activation='tanh'))
# ^ do i need to return sequences?

# single output layer
# paper does not specify activation; going with tanh
model.add(Dense(units=1, activation='tanh'))

optim = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# mean absolute error loss compiling
model.compile(loss='mae', optimizer=optim, metrics=['accuracy'])








### LOADING DATA AND TRAINING ###
train_x = []
train_y = []

test_x = []
test_y = []


#loading and merging data
with open('./data/train-sequences.pkl', 'rb') as f:
    all_seq = pickle.load(f)

    train_seq=[]

    for ticker in all_seq:
        train_seq.extend(all_seq[ticker])

    random.shuffle(train_seq)

    for seq in train_seq:
        train_x.append(seq[0])
        train_y.append(seq[1])

 
with open('./data/test-sequences.pkl', 'rb') as f:
    all_seq = pickle.load(f)

    test_seq=[]

    for ticker in all_seq:
        test_seq.extend(all_seq[ticker])

    random.shuffle(test_seq)

    for seq in test_seq:
        test_x.append(seq[0])
        test_y.append(seq[1])

 


# shuffling the tickers
random.shuffle(train_seq)
random.shuffle(test_seq)



MODEL_NAME = f'CNN-LSTM-v1-{int(time.time())}'
EPOCHS=20
BATCH_SIZE = 64

# tensorboard code from tutorial
tensorboard = TensorBoard(log_dir=f'./logs/{MODEL_NAME}')

filepath = 'CNN-LSTM-{epoch:02d}-{val_acc:.3f}'
checkpoint = ModelCheckpoint('./models/{}.model'.format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'))

# TODO figure out if the shape of data is valid!
history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(validation_x, validation_y), callbacks=[tensorboard, checkpoint])



# dont need to save model because of model checkpoint


