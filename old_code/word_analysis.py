path = '/home/shaurya/datasets/google_text_normalise/en_train.csv'

import pandas as pd

file_csv = pd.read_csv(path)
max_len = 0
literal = ''
for i in file_csv["after"]:
    if len(str(i).split(" ")) > max_len:
        max_len = len(str(i).split(' '))
        literal = i

print max_len, literal

total_len = 0
len_list = []
character = []
for i in file_csv["after"]:
    total_len += len(str(i).split(' '))
    len_list.append(len(str(i).split(' ')))
    character.extend(str(i).split(' '))
import numpy as np

print len(character), total_len, len(set(character))  # set(character)
print 'AVERAGE LENGTH OF TOKEN', total_len / float(len(np.asarray(file_csv["after"])))

# print len_list

import matplotlib.pyplot as plt
import collections

counter = collections.Counter(len_list)
# print(counter)
# Counter({1: 4, 2: 4, 3: 2, 5: 2, 4: 1})
print('COUNT', counter.values())
# [4, 4, 2, 1, 2]
print('OCCURENCE', counter.keys())
# [1, 2, 3, 4, 5]
plt.plot(counter.keys()[:45], counter.values()[:45])
plt.grid(linestyle='-', linewidth=1)
plt.show()
