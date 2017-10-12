# -*- coding: utf-8 -*-
from multiprocessing import Pool, cpu_count
# from num2words import num2words
import pandas as pd
import numpy as np
import string
import re
import pickle
import glob

print("Train...")


def fgtn(params):
    path, ftype = params
    df = pd.DataFrame()
    if ftype == 1:
        df = pd.read_csv(path, encoding='utf-8')  # Kaggle Training
    if ftype == 2:
        df = pd.read_csv(path, encoding='utf-8', names=['class', 'before', 'after']).fillna(
            'sil')  # Kaggle Dataset - limiting for kernels
    elif ftype == 3:
        df = pd.read_csv(path, sep='\t', engine='python', error_bad_lines=False, warn_bad_lines=False, encoding='utf-8',
                         names=['class', 'before', 'after']).fillna('sil')
    # df = df[df['after'] != 'sil']
    if ftype != 1:
        df['sentence_id'] = 1
    df = df.groupby(['before', 'after'], as_index=False)['sentence_id'].count()
    df['after'] = df.apply(lambda r: r['before'] if r['after'] in ['<self>', 'sil'] else r['after'], axis=1)
    print(len(df), path)
    return df


def fdict(dfs):
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df = df.groupby(['before', 'after'], as_index=False)['sentence_id'].sum()
    df = df.sort_values(['sentence_id', 'before'], ascending=[False, True])
    df = df.drop_duplicates(['before'])
    return {key: value for (key, value) in df[['before', 'after']].values}


# train dict1
dfs = map(fgtn, [['/home/shaurya/datasets/google_text_normalise/en_train.csv', 1]])
d = fdict(dfs)
d['km2'] = 'square kilometers'
d['km'] = 'kilometers'
d['kg'] = 'kilograms'
d['lb'] = 'pounds'
d['dr'] = 'doctor'
d['m²'] = 'square meters'
d[':'] = ':'
d['-'] = '-'
# pickle.dump(d, open( "dictionary1.en", "wb" ))
# d = pickle.load(open( "dictionary1.en", "rb" ))

# train dict2
paths = [['/home/shaurya/datasets/google_text_normalise/output_' + str(i) + '.csv', 2] for i in
         [1, 6, 11, 16, 21, 91, 96]]  # [:2]
dfs = map(fgtn, paths)
d2 = fdict(dfs)
# pickle.dump(d2, open( "dictionary2.en", "wb" ))
# d2 = pickle.load(open( "dictionary2.en", "rb" ))

"""
#train dict3
paths = glob.glob('../en_with_types/**')
paths =[[p,3] for p in paths] #[:2]
dfs = map(fgtn, paths)
d3 = fdict(dfs)
pickle.dump(d3, open( "dictionary3.en", "wb" ))
#d3 = pickle.load(open( "dictionary3.en", "rb" ))
"""

for k in d2:
    if k not in d:
        d[k] = str(d2[k])
d2 = []

# for k in d3:
#    if k not in d:
#        d[k] = str(d3[k])
# d3 = []

# delete anything with '_letter' or ' sil ' in it
dl = []
for k in d:  # maybee reconsider letter?
    if '_letter' in d[k] or ' sil ' in d[k] or '<self>' in d[k] or '<eos>' in d[k]:
        dl.append(k)
for k in dl:
    del d[str(k)]

# pickle.dump(d, open( "dictionary4.en", "wb" ))
# d = pickle.load(open( "dictionary4.en", "rb" ))

print("Test...")
PUNCTUATION_TMP = r'^[' + string.punctuation + '—«¡»¿' + ']+\.*$'
DIGIT_TMP = r'^[0-9]{1}$'
NUMBER_TMP = r'^\d+$'
DECIMAL_TMP = r'^(?!0)\d*\.\d+$'  # not start with 0
LETTER_TMP = r'^[A-Z]+\.*$|^([a-zA-Z]\.){2,10}$'
DATE_TMP = r'^\d*\s*[a-zA-Z]+\s*\d*[\s,]+\d+$|^\d{4}-\d{2}-\d{2}$|^\d{2}/\d{2}/\d{2}$'
MEASURE_TMP = r'^\d*\.*\d*\s*[km%]+$'
MONEY_TMP = r'^(US){0,1}[\$£].+$'
ORDINAL_TMP = r'^(\d+[rdsthn]+|[VIX]+\.*)$'
TIME_TMP = r'^(0\d+\.\d+|\d+:\d+|\d+)[\spma\.]*$'
ELECTRONIC_TMP = r'^(::|.*(\.[a-zA-Z]+|/.*))$'
FRACTION_TMP = r'^\d+/\d+$'
TELEPHONE_TMP = r'^[\d\s-]+$'
PLAIN_TMP = r'.*'
# 'CARDINAL', 'VERBATIM', 'MEASURE', 'ADDRESS'

cascade_order = [PUNCTUATION_TMP, DIGIT_TMP, NUMBER_TMP, DECIMAL_TMP, DATE_TMP, MEASURE_TMP, MONEY_TMP, ORDINAL_TMP,
                 TIME_TMP, FRACTION_TMP, ELECTRONIC_TMP, TELEPHONE_TMP, LETTER_TMP, PLAIN_TMP]
tmp_dict = {PUNCTUATION_TMP: 'PUNCT', DIGIT_TMP: 'DIGIT', NUMBER_TMP: 'NUMBER', DECIMAL_TMP: 'DECIMAL',
            LETTER_TMP: 'LETTERS', DATE_TMP: 'DATE', MEASURE_TMP: 'MEASURE', MONEY_TMP: 'MONEY', ORDINAL_TMP: 'ORDINAL',
            TIME_TMP: 'TIME', ELECTRONIC_TMP: 'ELECTRONIC', FRACTION_TMP: 'FRACTION', TELEPHONE_TMP: 'TELEPHONE',
            PLAIN_TMP: 'PLAIN'}

d_elect = {'.': 'dot', '/': 'slash', '-': 'dash', ':': 'colon', '#': 'hash tag', '1': 'one', '2': 'two', '3': 'three',
           '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '0': 'zero'}


def felectronic(e):
    e = list(str(e).strip().lower())
    e = ' '.join([d_elect[str(l)] if str(l) in d_elect else l for l in e])
    return e


SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹①፬", "0123456789012345678914")
{'"': 'inches', "'": 'feet', 'AU': 'units', 'BAR': 'bars', 'CM': 'centimeters', 'FT': 'feet', 'G': 'grams',
 'GAL': 'gallons', 'GB': 'gigabytes', 'GHZ': 'gigahertz', 'HA': 'hectares', 'HP': 'horsepower', 'HZ': 'hertz',
 'KA': 'kilo amperes', 'KB': 'kilobytes', 'KG': 'kilograms', 'KHZ': 'kilohertz', 'KM': 'kilometers',
 'KM2': 'square kilometers', 'KM²': 'square kilometers', 'KT': 'knots', 'KV': 'kilo volts', 'KW': 'kilowatts',
 'KWH': 'kilo watt hours', 'LB': 'pounds', 'LBS': 'pounds', 'M': 'meters', 'M2': 'square meters', 'M3': 'cubic meters',
 'MA': 'mega amperes', 'MB': 'megabytes', 'MB/S': 'megabytes per second', 'MG': 'milligrams', 'MHZ': 'megahertz',
 'MI': 'miles', 'ML': 'milliliters', 'MPH': 'miles per hour', 'MS': 'milliseconds', 'MV': 'milli volts',
 'MW': 'megawatts', 'M²': 'square meters', 'M³': 'cubic meters', 'OZ': 'ounces', 'V': 'volts', 'YD': 'yards',
 'au': 'units', 'bar': 'bars', 'cm': 'centimeters', 'ft': 'feet', 'g': 'grams', 'gal': 'gallons', 'gb': 'gigabytes',
 'ghz': 'gigahertz', 'ha': 'hectares', 'hp': 'horsepower', 'hz': 'hertz', 'kWh': 'kilo watt hours',
 'ka': 'kilo amperes', 'kb': 'kilobytes', 'kg': 'kilograms', 'khz': 'kilohertz', 'km': 'kilometers',
 'km2': 'square kilometers', 'km²': 'square kilometers', 'kt': 'knots', 'kv': 'kilo volts', 'kw': 'kilowatts',
 'lb': 'pounds', 'lbs': 'pounds', 'm': 'meters', 'm2': 'square meters', 'm3': 'cubic meters', 'ma': 'mega amperes',
 'mb': 'megabytes', 'mb/s': 'megabytes per second', 'mg': 'milligrams', 'mhz': 'megahertz', 'mi': 'miles',
 'ml': 'milliliters', 'mph': 'miles per hour', 'ms': 'milliseconds', 'mv': 'milli volts', 'mw': 'megawatts',
 'm²': 'square meters', 'm³': 'cubic meters', 'oz': 'ounces', 'v': 'volts', 'yd': 'yards', 'µg': 'micrograms',
 'ΜG': 'micrograms'}


def fcase(obefore):
    obefore = str(obefore)
    if obefore in d:
        obefore = d[obefore]
    if match(obefore) == 'ELECTRONIC':
        obefore = felectronic(obefore)
    try:
        # if str(obefore).replace(',','').isdigit():
        #    r = str(obefore).translate(SUB).replace(',','')
        #    r = num2words(float(r)).replace(',','').replace('-',' ')
        #    obefore = str(r).replace('  ', ' ').strip()
        # elif str(obefore.split(' ')[0]).replace(',','').isdigit() and len(obefore.split(' '))==2:
        #    a = str(str(obefore).split(' ')[0]).translate(SUB).replace(',','')
        #    b = str(obefore.split(' ')[1])
        #    if b in m:
        #        b = m[b]
        #    r = num2words(float(a)).replace(',','').replace('-',' ') + ' ' + b
        #    obefore = str(r).replace('  ', ' ').strip()
        # elif str(obefore).isupper() and len(list(obefore))>1 and len(list(obefore))<6:
        if str(obefore).isupper() and len(list(obefore)) > 1 and len(list(obefore)) < 6:
            obefore = ' '.join(list(obefore)).lower()
    except:
        print(obefore)
        pass
    return obefore


def match(s):
    r = ''
    for pat in cascade_order:
        if re.match(pattern=pat, string=str(s)):
            r = tmp_dict[pat]
            break
    return r


def transform_df(df):
    df = pd.DataFrame(df)
    df['id'] = df['sentence_id'].astype(str) + '_' + df['token_id'].astype(str)
    df['class'] = df['before'].map(match)
    df['after'] = df['before'].map(fcase)
    return df


test = pd.read_csv('/home/shaurya/datasets/google_text_normalise/en_test.csv')
p = Pool(cpu_count())
test = p.map(transform_df, np.array_split(test, cpu_count()))
test = pd.concat(test, axis=0, ignore_index=True).reset_index(drop=True)
p.close();
p.join()

test[['id', 'after']].to_csv('submission_base2.csv', index=False)
