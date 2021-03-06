'''
Created on Nov 5, 2017

@author: Sara
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from LoadData import load_test_data, load_train_data


# Define max pooling and conv layers
def conv2d(input, kernel_size, stride, num_filter):
    stride_shape = [1, stride, stride, 1]
    filter_shape = [kernel_size, kernel_size, input.get_shape()[3], num_filter]

    W = tf.get_variable('w', filter_shape, tf.float32,  tf.random_normal_initializer(0.0, 0.02))#tf.contrib.layers.xavier_initlializer(uniform=False))
    b = tf.get_variable('b', [1, 1, 1, num_filter], initializer=tf.constant_initializer(0.0))
    return tf.nn.conv2d(input, W, stride_shape, padding='SAME') + b

def max_pool(input, kernel_size, stride):
    ksize = [1, kernel_size, kernel_size, 1]
    strides = [1, stride, stride, 1]
    return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding='SAME')

class imageModel(object):
    def __init__(self):
        self.num_epoch = 10
        self.batch_size = 256
        self.log_step = 50
        self.learning_rate=5e-4
        self.global_step=tf.Variable(0, trainable=False)
        self.learning_step = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                           500, 0.96)
        self._build_model()
        

    def _model(self):
        print('intput layer: ' + str(self.X.get_shape()))

        with tf.variable_scope('conv1'):
            self.conv1 = conv2d(self.X, 3, 1, 96)
            self.relu1 = tf.nn.relu(self.conv1)
            print('conv1 layer: ' + str(self.relu1.get_shape()))

        with tf.variable_scope('conv2'):
            
            self.conv2 = conv2d(self.relu1, 3, 1, 96)
            self.relu2 = tf.nn.relu(self.conv2)
            print('conv2 layer: ' + str(self.relu2.get_shape()))
        
        #-----#
        with tf.variable_scope('pool1'):
            self.pool1 = max_pool(self.relu2, 3, 2)            
            print('pool2 layer: ' + str(self.pool1.get_shape()))  
        #-----#
        with tf.variable_scope('conv3'):
            self.conv3 = conv2d(self.pool1, 3, 1, 192)
            self.relu3 = tf.nn.relu(self.conv3)
            print('conv4 layer: ' + str(self.relu3.get_shape()))

        with tf.variable_scope('conv4'):
            
            self.conv4 = conv2d(self.relu3, 3, 1, 192)
            self.relu4 = tf.nn.relu(self.conv4)
            print('conv4 layer: ' + str(self.relu4.get_shape()))
        #-----#
        with tf.variable_scope('pool2'):
            self.pool2 = max_pool(self.relu4, 3, 2)            
            print('pool2 layer: ' + str(self.pool2.get_shape()))         
        #-----#
        with tf.variable_scope('conv5'):
            
            self.conv5 = conv2d(self.pool2, 3, 1, 192)
            self.relu5 = tf.nn.relu(self.conv5)
            print('conv5 layer: ' + str(self.relu5.get_shape()))
        
        with tf.variable_scope('conv6'):
            
            self.conv6 = conv2d(self.relu5, 1, 1, 192)
            self.relu6 = tf.nn.relu(self.conv6)
            print('conv6 layer: ' + str(self.relu6.get_shape()))
        #-----#
        with tf.variable_scope('conv7'):
            
            self.conv7 = conv2d(self.relu6, 1, 1, 10)
            self.relu7 = tf.nn.relu(self.conv7)
            print('conv7 layer: ' + str(self.relu7.get_shape()))
        #-----#
        
        #self.gap = tf.layers.average_pooling2d(self.relu7, [6, 6])
        self.gap = tf.reduce_mean(self.relu7,axis=[1,2])
        self.sm = tf.nn.softmax(self.gap)
        
        # Return the last layer
        return self.sm

    def _input_ops(self):
        # Placeholders
        self.X = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.Y = tf.placeholder(tf.int64, [None])

        self.is_train = None
        self.keep_prob = None

    def _build_optimizer(self):
        # Adam optimizer 'self.train_op' that minimizes 'self.loss_op'
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_step).minimize(self.loss_op, global_step=self.global_step)    
        
    def _loss(self, labels, logits):
        # Softmax cross entropy loss 'self.loss_op'
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits) )       

    def _build_model(self):
        # Define input variables
        self._input_ops()

        # Convert Y to one-hot vector
        labels = tf.one_hot(self.Y, 10)

        # Build a model and get logits
        logits = self._model()

        # Compute loss
        self._loss(labels, logits)
        
        # Build optimizer
        self._build_optimizer()

        # Compute accuracy
        predict = tf.argmax(logits, 1)
        correct = tf.equal(predict, self.Y)
        self.accuracy_op = tf.reduce_mean(tf.cast(correct, tf.float32))
        
    def train(self, sess, X_train, Y_train, X_val, Y_val):
        sess.run(tf.global_variables_initializer())

        step = 0
        losses = []
        accuracies = []
        print('-' * 5 + '  Start training  ' + '-' * 5)
        for epoch in range(self.num_epoch):
            print('train for epoch %d' % epoch)
            for i in range(num_training // self.batch_size):
                X_ = X_train[i * self.batch_size:(i + 1) * self.batch_size][:]
                Y_ = Y_train[i * self.batch_size:(i + 1) * self.batch_size]

                #############################################################################
                # TODO: You can change feed data as you want                                #
                #############################################################################
                feed_dict = {self.X: X_, self.Y : Y_}
             
                #############################################################################
                #                             END OF YOUR CODE                              #
                #############################################################################
                fetches = [self.train_op, self.loss_op, self.accuracy_op]

                _, loss, accuracy = sess.run(fetches, feed_dict=feed_dict)
                losses.append(loss)
                accuracies.append(accuracy)
                if step % self.log_step == 0:
                    print('iteration (%d): loss = %.3f, accuracy = %.3f' %
                        (step, loss, accuracy))
                step += 1

            #############################################################################
            # TODO: Plot training curves                                                #
            #############################################################################
            # Graph 1. X: epoch, Y: training loss
            
            y = losses
            x = [epoch for epoch in range(len(y))]

            plt.plot(x, y)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')

            plt.title("Training Loss")
            plt.legend()
            plt.grid(True)

            plt.show()
            
            # Graph 2. X: epoch, Y: training accuracy
            y = accuracies
            x = [epoch for epoch in range(len(y))]
            plt.plot(x, y)
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')

            plt.title("Training Accuracy")
            plt.legend()
            plt.grid(True)

            plt.show()
            #############################################################################
            #                             END OF YOUR CODE                              #
            #############################################################################

            # Print validation results
            print('validation for epoch %d' % epoch)
            val_accuracy = self.evaluate(sess, X_val, Y_val)
            print('-  epoch %d: validation accuracy = %.3f' % (epoch, val_accuracy))

    def evaluate(self, sess, X_eval, Y_eval):
        eval_accuracy = 0.0
        eval_iter = 0
        for i in range(X_eval.shape[0] // self.batch_size):
            X_ = X_eval[i * self.batch_size:(i + 1) * self.batch_size][:]
            Y_ = Y_eval[i * self.batch_size:(i + 1) * self.batch_size]
                        
            #############################################################################
            # TODO: You can change feed data as you want                                #
            #############################################################################
            feed_dict = {self.X: X_, self.Y : Y_}
            #############################################################################
            #                             END OF YOUR CODE                              #
            #############################################################################
            accuracy = sess.run(self.accuracy_op, feed_dict=feed_dict)
            eval_accuracy += accuracy
            eval_iter += 1
        return eval_accuracy / eval_iter
    
    
    
    
    
num_training = 49000
num_validation = 50000 - num_training
num_test = 10000

# Load cifar-10 data
X_train, Y_train, X_val, Y_val = load_train_data()
X_test, Y_test = load_test_data()


print X_train.shape

# Clear old computation graphs
tf.reset_default_graph()

sess = tf.Session()

model = imageModel()
model.train(sess, X_train, Y_train, X_val, Y_val)
accuracy = model.evaluate(sess, X_test, Y_test)
print('***** test accuracy: %.3f' % accuracy)

# Save your model
saver = tf.train.Saver()
model_path = saver.save(sess, "lib/tf_models/problem2/csci-599_mine.ckpt")
print("Model saved in %s" % model_path)

sess.close()

