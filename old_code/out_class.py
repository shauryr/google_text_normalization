import pandas as pd
import numpy as np
import os
import pickle
import gc
import xgboost as xgb
import re
from sklearn.model_selection import train_test_split
import glob

max_num_features = 10
pad_size = 1
boundary_letter = -1
space_letter = 0
# max_data_size = 900000

out_path = r'.'

import glob

# path = '/home/shaurya/datasets/google_text_normalise'
#
# filenames = glob.glob(path + "/output_96.csv")
# path = '/home/shaurya/PycharmProjects/google_text_norm/demo'
#
# filenames = glob.glob(path + "/*.csv")

# print filenames
#
# frame = pd.DataFrame()
# final_frame = []
# for file in filenames:
#     df_buffer = pd.read_csv(file)
#     final_frame.append(df_buffer)
#
# df = pd.concat(final_frame)
df = pd.read_csv(r'/home/shaurya/datasets/google_text_normalise/output_16.csv')

x_data = []
y_data = pd.factorize(df['Semiotic Class'])
labels = y_data[1]
y_data = y_data[0]
gc.collect()
print '--- csv read --'
for x in df['Input Token'].values:
    x_row = np.ones(max_num_features, dtype=int) * space_letter
    for xi, i in zip(list(str(x)), np.arange(max_num_features)):
        x_row[i] = ord(xi)

    x_data.append(x_row)

print x_data[0], '--', x_data[1]


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


max_data_size = 9000000
x_data = x_data[:max_data_size]
y_data = y_data[:max_data_size]
x_data = np.array(context_window_transform(x_data, pad_size))
gc.collect()
x_data = np.array(x_data)
y_data = np.array(y_data)

print('Total number of samples:', x_data.shape)
# print('Use: ', max_data_size)
# x_data = np.array(x_data)
# y_data = np.array(y_data)

print('x_data sample:')
print(x_data[0])
print('y_data sample:')
print(y_data[0])
print('labels:')
print(labels)

x_train = x_data
y_train = y_data
gc.collect()

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                      test_size=0.1, random_state=2017)
gc.collect()
num_class = len(labels)
dtrain = xgb.DMatrix(x_train, label=y_train)
dvalid = xgb.DMatrix(x_valid, label=y_valid)
watchlist = [(dvalid, 'valid'), (dtrain, 'train')]

param = {'objective': 'multi:softmax',
         'eta': '0.3', 'max_depth': 10,
         'silent': 1, 'nthread': -1,
         'num_class': num_class,
         'eval_metric': 'merror'}
model = xgb.train(param, dtrain, 50, watchlist, early_stopping_rounds=20,
                  verbose_eval=10)
gc.collect()

pred = model.predict(dvalid)
pred = [labels[int(x)] for x in pred]
y_valid = [labels[x] for x in y_valid]
x_valid = [[chr(x) for x in y[2 + max_num_features: 2 + max_num_features * 2]] for y in x_valid]
x_valid = [''.join(x) for x in x_valid]
x_valid = [re.sub('a+$', '', x) for x in x_valid]

gc.collect()

df_pred = pd.DataFrame(columns=['data', 'predict', 'target'])
df_pred['data'] = x_valid
df_pred['predict'] = pred
df_pred['target'] = y_valid
df_pred.to_csv(os.path.join(out_path, 'pred_12m.csv'))

df_erros = df_pred.loc[df_pred['predict'] != df_pred['target']]
df_erros.to_csv(os.path.join(out_path, 'errors_out11.csv'), index=False)

model.save_model(os.path.join(out_path, 'xgb_model_12m'))
