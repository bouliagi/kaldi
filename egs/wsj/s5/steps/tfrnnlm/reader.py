# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
# Copyright (C) 2017 Intellisist, Inc. (Author: Hainan Xu)
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


"""Utilities for parsing RNNLM text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import tensorflow as tf
import numpy as np

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().decode("utf-8").split()

def _build_vocab(filename):
  words = _read_words(filename)
  word_to_id = dict(zip(words, range(len(words))))
  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def rnnlm_raw_data(data_path, vocab_path):
  """Load RNNLM raw data from data directory "data_path".

  Args:
    data_path: string path to the directory where train/valid files are stored

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to RNNLMIterator.
  """

  train_path = os.path.join(data_path, "train")
  valid_path = os.path.join(data_path, "valid")
  dev_path = os.path.join(data_path, "dev")

  word_to_id = _build_vocab(vocab_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  dev_data =  _file_to_word_ids(dev_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, dev_data, vocabulary, word_to_id


def rnnlm_producer_old(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw RNNLM data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from rnnlm_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.
    And an initializer that resets to beginning of data.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "RNNLMProducer", [raw_data, batch_size, num_steps]):

    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
    print("raw_data:", raw_data)
    sys.stdout.flush()

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y

# new code based on Datasets
def rnnlm_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw RNNLM data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from rnnlm_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.
    And an initializer that resets to beginning of data.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "RNNLMProducer", [raw_data, batch_size, num_steps]):
    
    #raw_data = np.asarray(raw_data, dtype=np.int32)
    #data_len = raw_data.shape[0]
    #batch_len = data_len // batch_size
    #epoch_size = (batch_len - 1) // num_steps
    #print("batch_size, epoch_size, num_steps", batch_size, epoch_size, num_steps)
    #sys.stdout.flush()

    #features = np.reshape(raw_data[0 : batch_size * num_steps * epoch_size],
    #                      [batch_size * epoch_size, num_steps])
    #labels = np.reshape(raw_data[1 : batch_size * num_steps * epoch_size + 1],
    #                    [batch_size * epoch_size, num_steps])
    # To reproduce exactly the data order in _producer_old:
    data = np.asarray(raw_data, dtype=np.int32)
    data_len = data.shape[0]
    batch_len = data_len // batch_size
    epoch_size = batch_len // num_steps
    features = np.transpose(np.reshape(data[0:((batch_size-1)*epoch_size*num_steps)],
                                       [(batch_size-1), epoch_size, num_steps]),
                            axes=(1,0,2)).ravel().reshape([(batch_size-1)*epoch_size,num_steps])
    labels = np.transpose(np.reshape(data[1:((batch_size-1)*epoch_size*num_steps)+1],
                                       [(batch_size-1), epoch_size, num_steps]),
                            axes=(1,0,2)).ravel().reshape([(batch_size-1)*epoch_size,num_steps])

    assertion = tf.assert_positive( epoch_size,
                                    message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    # feedable iterators not yet supported?
    features_placeholder = tf.placeholder(features.dtype, features.shape)
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
   
    dataset = tf.contrib.data.Dataset.from_tensor_slices( \
                     (features_placeholder, labels_placeholder))
    #dataset = tf.contrib.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.repeat()     # repeat indefinitely
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    print("Got iterator:", iterator)
    #initializer = iterator.make_initializer(dataset)
    initializer = iterator.initializer
    print("Got initializer:", initializer)
    sys.stdout.flush()
    #tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

    # instead, we use a feedable iterator and make_one_shot()...
    #handle = tf.placeholder(tf.string, shape=[])
    #iterator = tf.contrib.data.Iterator.from_string_handle(handle, 
    #                                    dataset.output_types, dataset.output_shapes)
    next_element = iterator.get_next()
    print("Got next_element:", next_element)
    sys.stdout.flush()

    #data_iterator = dataset.make_one_shot_iterator()
    #data_handle = sess.run(data_iterator.string_handle())

    feed_dict = {features_placeholder: features, labels_placeholder: labels}
    #feed_dict = {handle: data_handle}
    return initializer, next_element, feed_dict
    
