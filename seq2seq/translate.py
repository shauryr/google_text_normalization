#Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Binary for training translation models and decoding from them.
Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.
Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.
See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tqdm import tqdm
import data_utils
import seq2seq_model

from sklearn.model_selection import train_test_split
from tensorflow.python import pywrap_tensorflow

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("from_vocab_size", 100000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("to_vocab_size", 100000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_string("from_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("from_dev_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_dev_data", None, "Training data.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
#_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
_buckets = [(20, 25)]

def read_data(source_path, target_path, max_size=None):
  """Read data from source and target files and put into buckets.
  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).
  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
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


def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.from_vocab_size,
      FLAGS.to_vocab_size,
      _buckets,
      FLAGS.size,
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      forward_only=forward_only,
      dtype=dtype)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  return model


def train():
  """Train a unnorm->norm translation model using en_train_data."""
  from_train = None
  to_train = None
  from_dev = None
  to_dev = None
  if FLAGS.from_train_data and FLAGS.to_train_data:
    from_train_data = FLAGS.from_train_data
    to_train_data = FLAGS.to_train_data
    from_dev_data = from_train_data
    to_dev_data = to_train_data
    if FLAGS.from_dev_data and FLAGS.to_dev_data:
      from_dev_data = FLAGS.from_dev_data
      to_dev_data = FLAGS.to_dev_data
    from_train, to_train, from_dev, to_dev, _, _ = data_utils.prepare_data(
        FLAGS.data_dir,
        from_train_data,
        to_train_data,
        from_dev_data,
        to_dev_data,
        FLAGS.from_vocab_size,
        FLAGS.to_vocab_size)
  else:
      # Prepare en_train data.
      print("Preparing en_train data in %s" % FLAGS.data_dir)
      from_train, to_train, from_dev, to_dev, _, _ = data_utils.prepare_wmt_data(
          FLAGS.data_dir, FLAGS.from_vocab_size, FLAGS.to_vocab_size)
  batch_idx = 0 
  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False)
  
    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    #dev_set = read_data(from_dev, to_dev) #pair of unnorm and norm ids
    data_set = read_data(from_train, to_train, FLAGS.max_train_data_size)
    data_set = np.asarray(data_set)
    X = []
    Y = []
    label = []
    for a in data_set[0]:
	X.append(a[0])
	label.append(a[0][0])
	Y.append(a[1])
    x_train, x_dev, y_train, y_dev = train_test_split(X, Y, test_size=0.1)

    train_set=[]
    for x,y in zip(x_train,y_train):
	c= []
	c.append(x)
	c.append(y)
	train_set.append(c)

    dev_set=[]
    for x,y in zip(x_dev,y_dev):
        c= []
        c.append(x)
        c.append(y)
        dev_set.append(c)
    
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]
    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    loss_array = []
    loss_eval_array = []
    epochs = 10
    for k in range(0, epochs):
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])
      # Get a batch and make a step.
      start_time = time.time()
      batch_idx = 0 
      for j in range(0, int(len(train_set) / 64)): 
      		encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          			train_set, bucket_id, batch_idx)
      		_, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   	     target_weights, bucket_id, False)
      		step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      		loss += step_loss / FLAGS.steps_per_checkpoint
      		current_step += 1
      		batch_idx += 64
      		# Once in a while, we save checkpoint, print statistics, and run evals.
      		if current_step % FLAGS.steps_per_checkpoint == 0:
        	# Print statistics for the previous epoch.
        		perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
			print ("loss",loss)
			loss_array.append(loss)
        		print ("global step %d learning rate %.4f step-time %.2f perplexity "
               			"%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         		step_time, perplexity))
        		# Decrease learning rate if no improvement was seen over last 3 times.
        		if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          			sess.run(model.learning_rate_decay_op)
        		previous_losses.append(loss)
        		# Save checkpoint and zero timer and loss.
        		checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        		model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        		step_time, loss = 0.0, 0.0
        		# Run evals on development set and print their perplexity.
        		for bucket_id in xrange(len(_buckets)):
          			if len(dev_set[bucket_id]) == 0:
            				print("  eval: empty bucket %d" % (bucket_id))
            				continue
          			encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              				dev_set, bucket_id,-1)
          			_, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
				loss_eval_array.append(eval_loss)
          			eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
              				"inf")
          			print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        		sys.stdout.flush()
      np_loss = np.asarray(loss_array)
      np.save("loss_train",np_loss)	
      np_eval_loss = np.asarray(loss_eval_array)
      np.save("loss_eval",np_eval_loss)	

def decode():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    print("hey!")
    #model.batch_size = len()  # We decode one sentence at a time.
    # Load vocabularies.
    normalized_vocab_path = FLAGS.data_dir + "/vocab.normalized"
    unnormalized_vocab_path = FLAGS.data_dir + "/vocab.unnormalized"
    en_vocab, _ = data_utils.initialize_vocabulary(unnormalized_vocab_path)
    _, rev_fr_vocab = data_utils.initialize_vocabulary(normalized_vocab_path)
    
    test_set = []
    # Get token-ids for the input sentence.
    data_utils.modify_dev_train_data(FLAGS.data_dir,"/test_train_k.csv",False)
    with open (FLAGS.data_dir + "/test.unnormalized") as f:
	for line in f:
    		token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(line), en_vocab)
		test_set.append([token_ids, []])
        
    # Which bucket does it belong to?
    bucket_id = len(_buckets) - 1
    for i, bucket in enumerate(_buckets):
    	if bucket[0] >= len(token_ids):
        	bucket_id = i
          	break
      	else:
        	logging.warning("Sentence truncated: %s", sentence)
    model.batch_size = 1  # We decode one sentence at a time.
    list_result = []
    l = ""
    # Get a 1 batch to feed the test_set to the model.
    with open(FLAGS.data_dir +"/results.csv" , "wb+") as results:
    	for i in tqdm(range (int(len(test_set)))):
    		encoder_inputs, decoder_inputs, target_weights = model.get_batch(test_set, bucket_id,i)
        	# Get output logits for the sentence.
    		_, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
    		outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

    		# If there is an EOS symbol in outputs, cut them at that point.
    		if data_utils.EOS_ID in outputs:
        		outputs = outputs[:outputs.index(data_utils.EOS_ID)]
    		# Print out French sentence corresponding to outputs.
		for output in outputs:
			results.write(tf.compat.as_str(rev_fr_vocab[output]) + ' ')			
		results.write('\n')
    		#print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
    		#print("> ", end="")
    '''
    x = np.array(output_logits)
    print ("output_logits",x[:,0,:].shape)
    k = np.einsum('ijk->jik', x)
    print ("k.shape: ", k.shape) 
    for j in range (64):
	print ("k[j,:,:].shape: ",k[j,:,:].shape)
    	#outputs = [int(np.argmax(logit, axis=1)) for logit in k[j,:,:]]
    	#outputs = k[j,:,:].argmax(axis =1)
    	# If there is an EOS symbol in outputs, cut them at that point.
    	if data_utils.EOS_ID in outputs:
     		outputs = outputs[:outputs.index(data_utils.EOS_ID)]
    	# Print out French sentence corresponding to outputs.
   	print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
    	print("> ", end="")
    #sys.stdout.flush()
    #sentence = sys.stdin.readline()  
     This is a greedy decoder - outputs are just argmaxes of output_logits.
    '''


def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.global_variables_initializer())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)


def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
