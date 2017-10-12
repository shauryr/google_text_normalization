file_ana = '/home/shaurya/PycharmProjects/google_text_norm/pred_out11.csv'

import pandas as pd
import numpy as np

df = pd.read_csv(file_ana)

mp = (df['predict'] == df['target'])

print len(df['predict']) - np.sum(np.asarray(mp))

print df.loc[~(df['predict'] == df['target'])]
