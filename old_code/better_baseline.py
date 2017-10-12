# Enhancing your baseline

__author__ = 'BingQing Wei'

import operator
import os
import inflect
import re

INPUT_PATH = r'.'
SUBM_PATH = r'.'

engine = inflect.engine()

NUMBER_TMP = r'^(?!0)[\d]+[\d,]*$'
DECIMAL_TMP = r'^(?!0)\d*\.\d+$'
MONEY_TMP = r'^\$([^a-zA-Z]*)\s*([a-zA-Z]*)$'


def inflect_transform(data):
    data = re.sub(r'-|,|\band\b', ' ', data)
    data = data.split(' ')
    data = [x for x in data if x is not '']
    return ' '.join(data)


def DIGIT_transform(data):
    neo_data = re.sub(r',|\s*', '', data)
    if int(neo_data) > 1000 and ',' not in data: return data
    return inflect_transform(engine.number_to_words(int(neo_data)))


def NUMBER_transform(data):
    data = re.sub(r',|\s*', '', data)
    return inflect_transform(engine.number_to_words(int(data)))


def DECIMAL_transform(data):
    data = re.sub(',|\s*', '', data)
    data = inflect_transform(engine.number_to_words(float(data)))
    return re.sub(r'^\bzero\s*', '', data)


def MONEY_transform(data):
    m = re.match(MONEY_TMP, data)
    ts = m.group(1)
    if re.match(NUMBER_TMP, ts):
        ts = NUMBER_transform(ts)
    else:
        ts = DECIMAL_transform(ts)
    if m.group(2).lower() == 'm':
        return ' '.join([ts, 'million', 'dollars'])
    elif m.group(2) is not '':
        return ' '.join([ts, m.group(2).lower(), 'dollars'])
    else:
        return ' '.join([ts, 'dollars'])


def verbose_wrapper(func, data, verbose=False):
    if verbose: print('Before: ', data)
    data = func(data)
    if verbose: print('After: ', data)
    return data


def solve():
    print('Train start...')
    changes = 0
    total = 0
    out = open('enhanced_sub.csv', "w")
    out.write('"id","after"\n')
    test = open("baseline4_en.csv")
    line = test.readline().strip()
    while 1:
        line = test.readline().strip()
        if line == '':
            break

        pos = line.find(',')
        i1 = line[:pos]
        line = line[pos + 1:]

        line = line[1:-1]
        out.write(i1 + ',')

        try:
            if re.match(DECIMAL_TMP, line):
                line = verbose_wrapper(DECIMAL_transform, data=line, verbose=False)
                changes += 1
            elif re.match(MONEY_TMP, line):
                line = verbose_wrapper(MONEY_transform, data=line, verbose=False)
                changes += 1
            elif re.match(NUMBER_TMP, line):
                line = verbose_wrapper(DIGIT_transform, data=line, verbose=False)
                changes += 1
        except Exception as ex:
            print('Exception: ', ex)
            print('Error in: ', line)
        out.write('"' + line + '"')
        out.write('\n')
        total += 1

    print('Total: {} Changed: {}'.format(total, changes))
    test.close()
    out.close()


if __name__ == '__main__':
    solve()
