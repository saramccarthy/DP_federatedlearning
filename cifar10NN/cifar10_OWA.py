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

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import sys
import tarfile
#import urllib.request

import urllib
#from six.moves import urllib
import tensorflow as tf
import numpy as np

from cifar10NN import cifar10_input

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--batch_size', type=int, default=50,
                    help='Number of images to process in a batch.')

parser.add_argument('--data_dir', type=str, default='./data/clients/cifar/', 
                    help='Path to the CIFAR-10 data directory.') #'./data/cifar10-batches-bin',

parser.add_argument('--use_fp16', type=bool, default=False,
                    help='Train the model using fp16.')

parser.add_argument('--full_data_dir', type=str, default='./data/cifar10-batches-bin', 
                    help='Path to the CIFAR-10 data directory.') #'./data/cifar10-batches-bin',


#parser.add_argument('--client', type=int, default=1)

FLAGS = parser.parse_args()


# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.95     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 100     # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.001  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001     # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.full_data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = FLAGS.data_dir#os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir, client="owa",
                                                  batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels

def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inference_OWA(Weights, images, NumClients):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  def norm(t):
      return t/tf.reduce_sum(t)
  
  W_tensor1 = np.asarray([ Weights[i][0] for i in range(NumClients)])
  b1 = np.asarray([ Weights[i][1] for i in range(NumClients)])
  W_tensor2 = np.asarray([ Weights[i][2] for i in range(NumClients)])
  b2 = np.asarray([ Weights[i][3] for i in range(NumClients)])
  W_tensor3 = np.asarray([ Weights[i][4] for i in range(NumClients)])
  b3 = np.asarray([ Weights[i][5] for i in range(NumClients)])
  W_tensor4 = np.asarray([ Weights[i][6] for i in range(NumClients)])
  b4 = np.asarray([ Weights[i][7] for i in range(NumClients)])
  W_tensor5 = np.asarray([ Weights[i][8] for i in range(NumClients)])
  b5 = np.asarray([ Weights[i][9] for i in range(NumClients)])
   
  with tf.variable_scope('v1_w'):
    #v1_w=tf.Variable(tf.ones([1, NumClients])*(np.random.uniform(0,1)/NumClients), trainable=True)
    v1_w=tf.Variable(tf.random_uniform([1,NumClients], 0,1/NumClients,dtype=tf.float32, seed=0), trainable=True)
  with tf.variable_scope('v2_w'):
    v2_w=tf.Variable(tf.random_uniform([1,NumClients], 0,1/NumClients,dtype=tf.float32, seed=0), trainable=True)#tf.Variable(tf.ones([1, NumClients])*(np.random.uniform(0,1)/NumClients), trainable=True)
  with tf.variable_scope('v3_w'):
    v3_w=tf.Variable(tf.random_uniform([1,NumClients], 0,1/NumClients,dtype=tf.float32, seed=0), trainable=True)#tf.Variable(tf.ones([1, NumClients])*(np.random.uniform(0,1)/NumClients), trainable=True)
  with tf.variable_scope('v4_w'):
    v4_w=tf.Variable(tf.random_uniform([1,NumClients], 0,1/NumClients,dtype=tf.float32, seed=0), trainable=True)#tf.Variable(tf.ones([1, NumClients])*(np.random.uniform(0,1)/NumClients), trainable=True)
  with tf.variable_scope('v5_w'):
    v5_w=tf.Variable(tf.random_uniform([1,NumClients], 0,1/NumClients,dtype=tf.float32, seed=0), trainable=True)#tf.Variable(tf.ones([1, NumClients])*(np.random.uniform(0,1)/NumClients), trainable=True)
  with tf.variable_scope('v1_b'):
    v1_b=tf.Variable(tf.random_uniform([1,NumClients], 0,1/NumClients,dtype=tf.float32, seed=0), trainable=True)#tf.Variable(tf.ones([1, NumClients])*(np.random.uniform(0,1)/NumClients), trainable=True)
  with tf.variable_scope('v2_b'):
    v2_b=tf.Variable(tf.random_uniform([1,NumClients], 0,1/NumClients,dtype=tf.float32, seed=0), trainable=True)#tf.Variable(tf.ones([1, NumClients])*(np.random.uniform(0,1)/NumClients), trainable=True)
  with tf.variable_scope('v3_b'):
    v3_b=tf.Variable(tf.random_uniform([1,NumClients], 0,1/NumClients,dtype=tf.float32, seed=0), trainable=True)#tf.Variable(tf.ones([1, NumClients])*(np.random.uniform(0,1)/NumClients), trainable=True)
  with tf.variable_scope('v4_b'):
    v4_b=tf.Variable(tf.random_uniform([1,NumClients], 0,1/NumClients,dtype=tf.float32, seed=0), trainable=True)#tf.Variable(tf.ones([1, NumClients])*(np.random.uniform(0,5)/NumClients), trainable=True)
  with tf.variable_scope('v5_b'):
    v5_b=tf.Variable(tf.random_uniform([1,NumClients], 0,1/NumClients,dtype=tf.float32, seed=0), trainable=True)#tf.Variable(tf.ones([1, NumClients])*(np.random.uniform(0,5)/NumClients), trainable=True)

  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  
  conv = tf.nn.conv2d(images, tf.reshape(tf.matmul(norm(v1_w),np.reshape(W_tensor1, [NumClients,-1])),W_tensor1[0].shape), [1, 1, 1, 1], padding='SAME')
  #print( tf.matmul(v1_b, b1).shape)
  #print( conv.shape)
  pre_activation = tf.add(conv, tf.reshape(tf.matmul(norm(v1_b), b1),[-1]))
  conv1 = tf.nn.relu(pre_activation)
  _activation_summary(conv1)
  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  conv = tf.nn.conv2d(norm1, tf.reshape(tf.matmul(norm(v2_w),np.reshape(W_tensor2, [NumClients,-1])),W_tensor2[0].shape), [1, 1, 1, 1], padding='SAME')
  pre_activation = tf.add(conv, tf.matmul(norm(v2_b), b2))
  conv2 = tf.nn.relu(pre_activation)
  _activation_summary(conv2)  
  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2') 

  reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
  dim = reshape.get_shape()[1].value
  local3 = tf.nn.relu(tf.matmul(reshape, tf.reshape(tf.matmul(norm(v3_w),np.reshape(W_tensor3, [NumClients,-1])),W_tensor3[0].shape)) + tf.matmul(norm(v3_b), b3))
  _activation_summary(local3)

  local4 = tf.nn.relu(tf.matmul(local3, tf.reshape(tf.matmul(norm(v4_w),np.reshape(W_tensor4, [NumClients,-1])),W_tensor4[0].shape)) + tf.matmul(norm(v4_b), b4))
  _activation_summary(local3)

  softmax_linear = tf.add(tf.matmul(local4, tf.reshape(tf.matmul(norm(v5_w),np.reshape(W_tensor5, [NumClients,-1])),W_tensor5[0].shape)), tf.matmul(norm(v5_b),b5))
  _activation_summary(softmax_linear)

  #print(softmax_linear.shape)
  #my_local = tf.get_variable("my_local", shape=(), collections=[tf.GraphKeys.LOCAL_VARIABLES])
  #print(my_local.shape)
  return softmax_linear


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)
  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  #print(tf.trainable_variables())
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op
  