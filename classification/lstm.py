from __future__ import print_function
import pandas as pd
import numpy as np
import os
import gc
import re
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import RMSprop
from tqdm import tqdm
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from time import time

batch_size = 16
epochs = 40

max_num_features = 20
pad_size = 1
boundary_letter = -1
space_letter = 0
max_data_size = 10000000

out_path = r'.'
# df = pd.read_csv(r'en_train.csv')
# df = pd.read_csv(r'/data/szr207/PycharmProjects/pubmed_name_disamb/google_text_norm/datasets/en_train.csv')
df = pd.read_csv(r'/home/shaurya/datasets/google_text_normalise/en_train.csv')
path = '/home/shaurya/PycharmProjects/google_text_norm/classification/'
x_data = []
y_data = pd.factorize(df['class'])
labels = y_data[1]
y_data = y_data[0]
gc.collect()
for x in tqdm(df['before'].values):
    x_row = np.ones(max_num_features, dtype=int) * space_letter
    for xi, i in zip(list(str(x)), np.arange(max_num_features)):
        x_row[i] = ord(xi)
    x_data.append(x_row)


def context_window_transform(data, pad_size):
    pre = np.zeros(max_num_features)
    pre = [pre for x in np.arange(pad_size)]
    data = pre + data + pre
    neo_data = []
    for i in tqdm(np.arange(len(data) - pad_size * 2)):
        row = []
        for x in data[i: i + pad_size * 2 + 1]:
            row.append([boundary_letter])
            row.append(x)
        row.append([boundary_letter])
        neo_data.append([int(x) for y in row for x in y])
    return neo_data


x_data = x_data[:max_data_size]
y_data = y_data[:max_data_size]
x_data = np.array(context_window_transform(x_data, pad_size))
gc.collect()
x_data = np.array(x_data)
y_data = np.array(y_data)

print('Total number of samples:', len(x_data))

x_train = x_data
y_train = y_data
gc.collect()

print(x_train.shape)

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                      test_size=0.1, random_state=2017)
num_classes = len(labels)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))
# y_train= np.reshape(y_train,(y_train.shape[0],y_train.shape[1],1))
# y_valid = np.reshape(y_valid,(y_valid.shape[0],y_valid.shape[1],1))
print(x_train.shape)
gc.collect()

# model.add(Dense(1, activation='sigmoid'))
tensorboard = TensorBoard(log_dir=path + "logs/{}".format(time()), write_graph=False)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=2, verbose=1, mode='auto')
model = Sequential()
model.add(LSTM(64, input_shape=((max_num_features * 3) + 4, 1)))
# model.add(LSTM(64, return_sequences=True, input_shape=((max_num_features * 3) + 4, 1)))

#model.add(LSTM(32))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(labels), activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(x_train.shape, y_train.shape)

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                    validation_data=(x_valid, y_valid), callbacks=[tensorboard, early])

gc.collect()

score = model.evaluate(x_valid, y_valid, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
from matplotlib import pyplot as plt

fig, ax = plt.subplots(1, 2)
ax[0].plot(history.history['acc'], 'b')
ax[0].set_title('Accuraccy')
ax[1].plot(history.history['loss'], 'r')
ax[1].set_title('Loss')
plt.show()

pred = model.predict(x_valid)
# print 'pred', '=====', pred
pred = [labels[np.argmax(x)] for x in pred]
pred = [labels[np.argmax(x)] for x in pred]
x_valid = [[chr(x) for x in y[2 + max_num_features: 2 + max_num_features * 2]] for y in x_valid]
x_valid = [''.join(x) for x in x_valid]
x_valid = [re.sub('a+$', '', x) for x in x_valid]

gc.collect()

df_pred = pd.DataFrame(columns=['data', 'predict', 'target'])
df_pred['data'] = x_valid
df_pred['predict'] = pred
df_pred['target'] = y_valid
df_pred.to_csv(os.path.join(out_path, 'pred_lstm.csv'))

df_erros = df_pred.loc[df_pred['predict'] != df_pred['target']]
df_erros.to_csv(os.path.join(out_path, 'errors_lstm.csv'), index=False)

model.save_weights(os.path.join(out_path, 'lstm_model'))
