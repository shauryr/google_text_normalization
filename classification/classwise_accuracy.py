import pandas as pd

path_1 = "/home/shaurya/datasets/google_text_normalise/s2s/test_train_k.csv"

res_path = "/home/shaurya/datasets/google_text_normalise/en_train_normalised_results_256_3L.csv"

data = pd.read_csv(path_1, names=["class", "before", "after"])
res_data = pd.read_csv(res_path, names=["prediction"])

class_names = list(set(data['class']))

dict_class_count_total = {}
dict_correct_class = {}
all = 0
correct_pred = 0

for class_name in class_names:
    dict_class_count_total[class_name] = list(data['class']).count(class_name)
    all = all + list(data['class']).count(class_name)
    dict_correct_class[class_name] = 0

for class_label, before, after, prediction in zip(data["class"], data['before'], data["after"], res_data["prediction"]):
    # print after, prediction
    if after.rstrip() == prediction.rstrip():
        dict_correct_class[class_label] += 1
        correct_pred += 1
    else:
        print class_label, before, after, '--', prediction
for count, correct in zip(dict_class_count_total, dict_correct_class):
    print count
