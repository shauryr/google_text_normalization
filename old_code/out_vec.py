path = '/home/shaurya/datasets/google_text_normalise/demo/en_train.csv'

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

import pandas as pd
from nltk import FreqDist
import numpy as np

df_y = pd.read_csv(path)
# max_len = 10
vocab_size = 2000000
y_data = np.asarray(df_y["after"])
window_out = 200
vocab = []
maximum_samples = 1000000
#
# for i in y_data:
#     vocab.extend(str(i).split(' '))
#
# vocab = list(set(vocab))
#
# columns = len(vocab)

out_matrix = np.zeros((4000000, window_out + 180))

# for i in xrange(len(y_data)):
#     for j in xrange(len(y_data[i].split(' '))):
#         out_matrix[i][j][vocab.index(y_data[i].split(' ')[j])] = 1
#
# print out_matrix.shape


# dist = FreqDist(np.hstack(y))
# y_vocab = dist.most_common(vocab_size - 1)
#
# y_ix_to_word = [word[0] for word in y_vocab]
#
# y_max_len = max([len(sentence) for sentence in y])
#
# # print y_max_len
# y_word_to_ix = {word: ix for ix, word in enumerate(y_ix_to_word)}
# for i, sentence in enumerate(y):
#     for j, word in enumerate(sentence):
#         if word in y_word_to_ix:
#             y[i][j] = y_word_to_ix[word]
#
# y = pad_sequences(y, maxlen=y_max_len, dtype='int32')
# print  y.shape
