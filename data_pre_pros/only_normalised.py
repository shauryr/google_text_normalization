import pandas as pd
import numpy as np

path = '/home/shaurya/datasets/google_text_normalise/en_train.csv'

file_en = open(path)
df_pandas = pd.read_csv(path)
out_file = open('/home/shaurya/datasets/google_text_normalise/en_train_normalised.csv', 'w')

# for i in file_en:
#     list_row = i.split(',')
#     for i, j in enumerate(list_row):
#         list_row[i] = list_row[i].replace("\"", '')
#     if not (list_row[2] == 'PLAIN' or list_row[2] == 'PUNCT'):
#         # print list_row[3], list_row[4]
#         out_file.write(' '.join(list(list_row[3])) + '\t' + list_row[4])
#
# out_file.close()

for sent_id, class_type, before, after in zip(df_pandas['sentence_id'], df_pandas['class'], df_pandas['before'],
                                              df_pandas['after']):
    if not (class_type == 'PLAIN' or class_type == 'PUNCT'):
        # print class_type, before, after
        # print ' '.join(list(before)), '\t', after
        print sent_id, before, after
        out_file.write(' '.join(list(str(before))) + '\t' + after + '\n')

out_file.close()
