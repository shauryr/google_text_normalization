import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

path = '/home/shaurya/PycharmProjects/google_text_norm/classification/errors_300.csv'

df_file = pd.read_csv(path)

class_target = list(set(df_file['target']))
class_predict = list(set(df_file['predict']))

dict_target = defaultdict()
dict_count_wrong = {}
for i in sorted(class_target):
    dict_target[i] = []
    dict_count_wrong[i] = 0

for prediction, target in zip(df_file['predict'], df_file['target']):
    # if prediction in dict_target[target].keys():
    #     dict_target[target][prediction] += 1
    # else:
    #     dict_target[target] = {prediction: 1}
    dict_target[target].append(prediction)
    dict_count_wrong[target] += 1
list_count_actual = []
list_count_predict = []
list_wrong_number = []

print 'actual', 'prediction', 'wrong_number'
for target in dict_target:
    for class_label in class_predict:
        # if dict_target[target].count(class_label) != 0:
        print target, class_label, dict_target[target].count(class_label)
        list_count_actual.append(target)
        list_count_predict.append(class_label)
        list_wrong_number.append(dict_target[target].count(class_label))

n_bins = 1
# plt.plot(list_wrong_number)
# plt.show()
plt.bar(dict_count_wrong.keys(), dict_count_wrong.values(), color='g')
plt.show()
