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

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import csv
import numpy as np
import tensorflow as tf
import pickle
import os


from cifar10NN import cifar10

parser = cifar10.parser

parser.add_argument('--eval_dir', type=str, default='/tmp/cifar10_eval',
                    help='Directory where to write event logs.')

parser.add_argument('--eval_data', type=str, default='test',
                    help='Either `test` or `train_eval`.')

parser.add_argument('--checkpoint_dir', type=str, default='/tmp/cifar10_train',
                    help='Directory where to read model checkpoints.')

parser.add_argument('--eval_interval_secs', type=int, default=60*5,
                    help='How often to run the eval.')

parser.add_argument('--num_examples', type=int, default=10000,
                    help='Number of examples to run.')

parser.add_argument('--run_once', type=bool, default=True,
                    help='Whether to run eval only once.')
num_training = 49000
num_validation = 50000 - num_training
num_test = 10000

def unpickle(file):
    import sys
    if sys.version_info.major == 2:
        import cPickle
        with open(file, 'rb') as fo:
            dict = cPickle.load(fo)
        return dict['data'], dict['labels']
    else:
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict[b'data'], dict[b'labels']
    
def process_images():
      data, labels = unpickle("../data/cifar-10-batches-py/test_batch")
      image =  np.transpose(np.reshape(data,(10000,3,32,32)),[0,2,3,1])
      labels = np.asarray(labels)
    # Randomly crop a [height, width] section of the image.
      distorted_image = tf.random_crop(image, [24, 24, 3])
    
      # Randomly flip the image horizontally.
      distorted_image = tf.image.random_flip_left_right(distorted_image)
    
      # Because these operations are not commutative, consider randomizing
      # the order their operation.
      # NOTE: since per_image_standardization zeros the mean and makes
      # the stddev unit, this likely has no effect see tensorflow#1458.
      distorted_image = tf.image.random_brightness(distorted_image,
                                                   max_delta=63)
      distorted_image = tf.image.random_contrast(distorted_image,
                                                 lower=0.2, upper=1.8)
    
      # Subtract off the mean and divide by the variance of the pixels.
      float_image = tf.image.per_image_standardization(distorted_image)
      return float_image, labels
    
def eval_central_model(client, eval_data, eval_dir, checkpoint_dir):
    curr_client=client;

    """Eval CIFAR-10 for a number of steps."""
    FLAGS = parser.parse_args()
    
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        
        images, labels = process_images()
    
        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cifar10.inference(images)
        
        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
    
        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            cifar10.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
              # Restores from checkpoint
              saver.restore(sess, ckpt.model_checkpoint_path)
              # Assuming model_checkpoint_path looks something like:
              #   /my-favorite-path/cifar10_train/model.ckpt-0,
              # extract global_step from it.
              global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
              print('No checkpoint file found')
              return
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))
        
                num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
                true_count = 0  # Counts the number of correct predictions.
                total_sample_count = num_iter * FLAGS.batch_size
                step = 0
        
                while step < num_iter and not coord.should_stop():
                  predictions = sess.run([top_k_op])
                  true_count += np.sum(predictions)
                  step += 1
                
                # Compute precision @ 1.
                precision = true_count / total_sample_count
                print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
                write(eval_dir, precision, client)

            except Exception as e:  # pylint: disable=broad-except
              coord.request_stop(e)    
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

    
def eval_once(saver, summary_writer, top_k_op, accuracy_op, summary_op, loss, checkpoint_dir, eval_dir):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  FLAGS = parser.parse_args()

  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
      acc = sess.run([accuracy_op])
      loss = sess.run([loss])
      print("accuracy:", acc, "loss ", loss)
      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      #summary_writer.add_summary(summary, global_step)
      write(eval_dir, precision)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate(eval_data, eval_dir, model_dir, client):
  """Eval CIFAR-10 for a number of steps."""
  FLAGS = parser.parse_args()
    
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.

    images, labels = cifar10.inputs(eval_data=eval_data,  client=client)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)
    
    # Calculate predictions.
    predict = tf.argmax(logits, 1, output_type=tf.int32)
    correct = tf.equal(predict, labels)
    accuracy_op = tf.reduce_mean(tf.cast(correct, tf.float32))
    top_k_op = tf.nn.in_top_k(logits, labels, 1)    
    loss = cifar10.loss(logits, labels)                                                                                                             
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(eval_dir, g)

    while True:
        
      eval_once(saver, summary_writer, top_k_op, accuracy_op, summary_op,loss, model_dir, eval_dir)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def write(eval_dir, accuracy):
    filename = eval_dir + "accuracy.pkl"
    old = []            
    if os.path.exists(filename):
        with open(filename,'rb') as rfp: 
            old = pickle.load(rfp)
    old.append(accuracy)
    

    with open(filename, "wb") as fp:   #Pickling
        pickle.dump(old, fp)
        
def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  FLAGS = parser.parse_args()
  tf.app.run()