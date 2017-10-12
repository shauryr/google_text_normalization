path = '/home/shaurya/datasets/google_text_normalise/demo/en_train.csv'

import gc
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
import pandas as pd
from nltk import FreqDist
import numpy as np

max_num_features = 10
boundary_letter = -1
space_letter = 0
pad_size = 1

data_f = pd.read_csv(path)

x_data = []

gc.collect()

for x in data_f['before'].values:
    x_row = np.ones(max_num_features, dtype=int) * space_letter
    for xi, i in zip(list(str(x)), np.arange(max_num_features)):
        x_row[i] = ord(xi)

    x_data.append(x_row)


def context_window_transform(data, pad_size):
    pre = np.zeros(max_num_features)
    pre = [pre for x in np.arange(pad_size)]
    data = pre + data + pre
    neo_data = []
    for i in np.arange(len(data) - pad_size * 2):
        row = []
        for x in data[i: i + pad_size * 2 + 1]:
            row.append([boundary_letter])
            row.append(x)
        row.append([boundary_letter])
        neo_data.append([int(x) for y in row for x in y])
    return neo_data


x_data = np.array(context_window_transform(x_data, pad_size))
x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
print x_data.shape

vocab_size = 2000000
y_data = np.asarray(data_f["after"])
print(y_data)
thefile = open('y_data.txt', 'w')
for item in y_data:
    thefile.write(str(item) + str(type(item)) + '\n')
thefile.close()
y = [text_to_word_sequence(y) for y in y_data if
     len(y) > 0 and type(y) != float]

dist = FreqDist(np.hstack(y))
y_vocab = dist.most_common(vocab_size - 1)

y_ix_to_word = [word[0] for word in y_vocab]

y_max_len = max([len(sentence) for sentence in y])

# print y_max_len
y_word_to_ix = {word: ix for ix, word in enumerate(y_ix_to_word)}
for i, sentence in enumerate(y):
    for j, word in enumerate(sentence):
        if word in y_word_to_ix:
            y[i][j] = y_word_to_ix[word]

y_data = pad_sequences(y, maxlen=y_max_len, dtype='int32')
# y_data = np.reshape(y_data, (y_data.shape[0], y_data.shape[1], 1))
print  y_data

print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(34, 1)))
model.add(Dense(7))
model.add(Activation('relu'))

model.summary()

optimizer = RMSprop(lr=0.01)
model.compile(loss='mean_squared_error', optimizer=optimizer)

model.fit(x_data, y_data, batch_size=2000, epochs=1)
# print model.predict(x_data).shape
model.save_weights('checkpoint_epoch_{}.hdf5'.format(1))
