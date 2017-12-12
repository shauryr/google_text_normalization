# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile
import csv
from six.moves import urllib

from tensorflow.python.platform import gfile
import tensorflow as tf
import random
import math

# maryam
import re
import codecs
import sys
from sklearn.model_selection import train_test_split
reload(sys)
sys.setdefaultencoding('utf-8')
#end maryam

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

# URLs for WMT data.
_WMT_ENFR_TRAIN_URL = "http://www.statmt.org/wmt10/training-giga-fren.tar"
_WMT_ENFR_DEV_URL = "http://www.statmt.org/wmt15/dev-v2.tgz"


def maybe_download(directory, filename, url):
  """Download filename from url unless it's already in directory."""
  if not os.path.exists(directory):
    print("Creating directory %s" % directory)
    os.mkdir(directory)
  filepath = os.path.join(directory, filename)
  if not os.path.exists(filepath):
    print("Downloading %s to %s" % (url, filepath))
    filepath, _ = urllib.request.urlretrieve(url, filepath)
    statinfo = os.stat(filepath)
    print("Successfully downloaded", filename, statinfo.st_size, "bytes")
  return filepath


def gunzip_file(gz_path, new_path):
  """Unzips from gz_path into new_path."""
  print("Unpacking %s to %s" % (gz_path, new_path))
  with gzip.open(gz_path, "rb") as gz_file:
    with open(new_path, "wb") as new_file:
      for line in gz_file:
        new_file.write(line)


def get_wmt_enfr_train_set(directory):
  """Download the WMT en-fr training corpus to directory unless it's there."""
  train_path = os.path.join(directory, "giga-fren.release2.fixed")
  if not (gfile.Exists(train_path +".fr") and gfile.Exists(train_path +".en")):
    corpus_file = maybe_download(directory, "training-giga-fren.tar",
                                 _WMT_ENFR_TRAIN_URL)
    print("Extracting tar file %s" % corpus_file)
    with tarfile.open(corpus_file, "r") as corpus_tar:
      corpus_tar.extractall(directory)
    gunzip_file(train_path + ".fr.gz", train_path + ".fr")
    gunzip_file(train_path + ".en.gz", train_path + ".en")
  return train_path


def get_wmt_enfr_dev_set(directory):
  """Download the WMT en-fr training corpus to directory unless it's there."""
  dev_name = "newstest2013"
  dev_path = os.path.join(directory, dev_name)
  if not (gfile.Exists(dev_path + ".fr") and gfile.Exists(dev_path + ".en")):
    dev_file = maybe_download(directory, "dev-v2.tgz", _WMT_ENFR_DEV_URL)
    print("Extracting tgz file %s" % dev_file)
    with tarfile.open(dev_file, "r:gz") as dev_tar:
      fr_dev_file = dev_tar.getmember("dev/" + dev_name + ".fr")
      en_dev_file = dev_tar.getmember("dev/" + dev_name + ".en")
      fr_dev_file.name = dev_name + ".fr"  # Extract without "dev/" prefix.
      en_dev_file.name = dev_name + ".en"
      dev_tar.extract(fr_dev_file, directory)
      dev_tar.extract(en_dev_file, directory)
  return dev_path


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=False):
  """Create vocabulary file (if it does not exist yet) from data file.
  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.
  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        line = tf.compat.as_bytes(line)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.
  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].
  Args:
    vocabulary_path: path to the file containing the vocabulary.
  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).
  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=False):
  """Convert a string to list of integers representing token-ids.
  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].
  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  Returns:
    a list of integers, the token-ids for the sentence.
  """

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=False):
  """Tokenize data file and turn into token-ids using given vocabulary file.
  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.
  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab,
                                            tokenizer, normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def modify_dev_train_data(data_dir,data_file,is_train):
  train_unnorm = data_dir+ "/train.unnormalized"
  dev_unnorm = data_dir + "/dev.unnormalized"
  train_norm = data_dir+ "/train.normalized"
  dev_norm = data_dir + "/dev.normalized"
  test_unnorm = data_dir + "/test.unnormalized"
  '''
  with open(data_dir +"/en_train.csv", 'rb') as f:
     rows = list(csv.reader(f))
     print (len(rows))
  '''
  if (is_train):
	line_number = 0
  	with open(data_dir + data_file, "rb") as f, open(train_unnorm, "wb+") as train_unnorm,open(train_norm, "wb+") as train_norm:
        	rows = list(csv.reader(f))
        	for i in range(0,len(rows)):
                	train_unnorm.write(rows[i][0]+' ')
                	prev_l=''
                	for l in rows[i][1].decode(('UTF-8')):
                      		if ((not l.isalpha()) and l!=' '):
                                	if (prev_l.isalpha()):
                                        	train_unnorm.write(' '+l.encode('utf-8')+' ')
                                	else:
                                        	train_unnorm.write(l.encode('utf-8')+' ')

                      		else:
                              		train_unnorm.write(l.encode('utf-8').lower())
                      		prev_l=l
                	if (prev_l.isalpha()):
                      		train_unnorm.write(' ')
                	train_unnorm.write('\n')
                	train_norm.write(rows[i][2].encode('utf-8').lower()+'\n')
  else:
	line_number = 0
        import pandas as pd
	'''
	with open(data_dir + data_file, "rb") as f, open(test_unnorm, "wb+") as test_unnorm:
                rows = list(csv.reader(f))
                for i in range(0,len(rows)):
                        test_unnorm.write(rows[i][0]+' ')
                        prev_l=''
                        for l in rows[i][1].decode(('uft-8')):
                                if ((not l.isalpha()) and l!=' '):
                                        if (prev_l.isalpha()):
                                                test_unnorm.write(' '+l.encode('utf-8')+' ')
                                        else:
                                                test_unnorm.write(l.encode('utf-8')+' ')

                                else:
                                        test_unnorm.write(l.encode('utf-8').lower())
                                prev_l=l
                        if (prev_l.isalpha()):
                                test_unnorm.write(' ')
                        test_unnorm.write('\n')
	'''
	test_unnorm = open(test_unnorm, "wb+")
	df = pd.read_csv(data_dir + data_file)
	for label,token in zip(df['class'],df['before']):
		test_unnorm.write(label+' ')
		prev_l=''
                for l in str(token):
                	if ((not l.isalpha()) and l!=' '):
                        	if (prev_l.isalpha()):
                                	test_unnorm.write(' '+l+' ')
                                else:
                                        test_unnorm.write(l+' ')
                        else:
                        	test_unnorm.write(l.lower())
                        prev_l=l
                if (prev_l.isalpha()):
                	test_unnorm.write(' ')
               	test_unnorm.write('\n')
	
	'''
		writing into dev file
                if (line_number < dev_data):
                       if (int(rows[i][0])!=line_number):
                               line_number=int(rows[i][0])
                               dev_unnorm.write("\n")
                               dev_norm.write("\n")
                        prev_l='' 
			for l in rows[i][3].decode(('UTF-8')):
			    #print (line_number,": ",l.encode('utf-8'))
			    if ((not l.isalpha()) and l!=' '):
				if (prev_l.isalpha()):
					dev_unnorm.write(' '+l.encode('utf-8')+' ')
				else:
					dev_unnorm.write(l.encode('utf-8')+' ')
				
			    else:
				dev_unnorm.write(l.encode('utf-8').lower())
			    prev_l=l
 			if (prev_l.isalpha()):
				dev_unnorm.write(' ')
			dev_norm.write(rows[i][4].lower()+' ')
		#writing into train file
	        else:
                        if (int(rows[i][0])!=line_number):
                                line_number=int(rows[i][0])
                                train_unnorm.write("\n")
				train_norm.write("\n")
			prev_l=''
                        for l in rows[i][3].decode(('UTF-8')):
                            if ((not l.isalpha()) and l!=' '):
                                if (prev_l.isalpha()):
                                        train_unnorm.write(' '+l.encode('utf-8')+' ')
                                else:
                                        train_unnorm.write(l.encode('utf-8')+' ')

                            else:
                                train_unnorm.write(l.encode('utf-8').lower())
                            prev_l=l
                        if (prev_l.isalpha()):
                                train_unnorm.write(' ')			
			train_norm.write(rows[i][4].lower()+' ')
	'''
def create_dev_data(source_path, target_path, max_size=None):
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      print ("source", source)
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]

        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set
def prepare_wmt_data(data_dir, unnormalized_vocabulary_size, normalized_vocabulary_size, tokenizer=None):
    
  modify_dev_train_data(data_dir,"/train.csv",True) 
  
  unnormalized_train_path = data_dir + "/train.unnormalized"
  normalized_train_path = data_dir + "/train.normalized"
  unnormalized_dev_path = data_dir + "/dev.unnormalized"
  normalized_dev_path = data_dir + "/dev.normalized"
  
  return prepare_data(data_dir, unnormalized_train_path, normalized_train_path, unnormalized_dev_path, normalized_dev_path, unnormalized_vocabulary_size, normalized_vocabulary_size, tokenizer)

def prepare_data(data_dir, unnormalized_train_path, normalized_train_path, unnormalized_dev_path, normalized_dev_path, unnormalized_vocabulary_size, normalized_vocabulary_size, tokenizer=None):

  # Create vocabularies of the appropriate sizes.
  normalized_vocab_path = data_dir + "/vocab.normalized"
  unnormalized_vocab_path = data_dir + "/vocab.unnormalized"
  create_vocabulary(normalized_vocab_path, normalized_train_path , normalized_vocabulary_size, tokenizer)
  create_vocabulary(unnormalized_vocab_path, unnormalized_train_path , unnormalized_vocabulary_size, tokenizer)
  
  # Create token ids for the training data.
  normalized_train_ids_path = normalized_train_path + (".ids%d" % normalized_vocabulary_size)
  unnormalized_train_ids_path = unnormalized_train_path + (".ids%d" % unnormalized_vocabulary_size)
  data_to_token_ids(normalized_train_path, normalized_train_ids_path, normalized_vocab_path, tokenizer)
  data_to_token_ids(unnormalized_train_path, unnormalized_train_ids_path, unnormalized_vocab_path, tokenizer)
  
  #create_dev_data(unnormalized_train_ids_path,normalized_train_ids_path, max_size=None)  
  
  # Create token ids for the development data
  normalized_dev_ids_path = normalized_dev_path + (".ids%d" % normalized_vocabulary_size)
  unnormalized_dev_ids_path = unnormalized_dev_path + (".ids%d" % unnormalized_vocabulary_size)
  '''
  data_to_token_ids(normalized_dev_path, normalized_dev_ids_path, normalized_vocab_path, tokenizer)
  data_to_token_ids(unnormalized_dev_path, unnormalized_dev_ids_path, unnormalized_vocab_path, tokenizer)
  ''' 
  return (unnormalized_train_ids_path, normalized_train_ids_path,
          unnormalized_dev_ids_path, normalized_dev_ids_path,
          unnormalized_vocab_path, normalized_vocab_path)
