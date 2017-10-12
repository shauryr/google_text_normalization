path = '/home/shaurya/datasets/google_text_normalise/en_train.csv'

import pandas as pd

file_csv = pd.read_csv(path)
max_len = 0
literal = ''
for i in file_csv["after"]:
    if len(str(i)) > max_len:
        max_len = len(str(i))
        literal = i

print 'MAX TOKEN', max_len, literal
total_len = 0
len_list = []
character = ''
for i in file_csv["after"]:
    total_len += len(str(i))
    len_list.append(len(str(i)))
    character += str(i)
import numpy as np

print len(set(character)), len(character)
print 'AVERAGE LENGTH OF TOKEN', total_len / float(len(np.asarray(file_csv["after"])))

import matplotlib.pyplot as plt
import collections

counter = collections.Counter(len_list)
# print(counter)
# Counter({1: 4, 2: 4, 3: 2, 5: 2, 4: 1})
print('COUNT', counter.values())
# [4, 4, 2, 1, 2]
print('OCCURENCE', counter.keys())
# [1, 2, 3, 4, 5]
plt.plot(counter.keys(), counter.values())
plt.grid(linestyle='-', linewidth=1)
plt.show()
