'''
Created on Nov 5, 2017

@author: Sara
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib2 import Path
from LoadData import load_test_data, load_train_data
import tensorflow.contrib.slim as slim


# Define max pooling and conv layers
def conv2d(input, kernel_size, stride, num_filter):
    stride_shape = [1, stride, stride, 1]
    print(input.shape)
    filter_shape = [kernel_size, kernel_size, input.get_shape()[3], num_filter]

    W = tf.get_variable('w', filter_shape, tf.float32,  tf.random_normal_initializer(0.0, 0.02))#tf.contrib.layers.xavier_initlializer(uniform=False))
    print "C:", W.shape
    b = tf.get_variable('b', [1, 1, 1, num_filter], initializer=tf.constant_initializer(0.0))
    return tf.nn.conv2d(input, W, stride_shape, padding='SAME') + b

def max_pool(input, kernel_size, stride):
    ksize = [1, kernel_size, kernel_size, 1]
    strides = [1, stride, stride, 1]
    return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding='SAME')
def fc(input, output_dim, act_fn, scope):
    return tf.contrib.layers.fully_connected(input,output_dim,
                                             weights_initializer =  tf.random_normal_initializer(0.0, 0.02),#tf.contrib.layers.xavier_initlializer(uniform=False),
                                             biases_initializer=tf.zeros_initializer(), scope=scope,
                                            activation_fn=act_fn)
class imageModelSmol(object):
    def __init__(self, i ):
        self.path = "./models/MNIST/client%d/model.ckpt"%i
        print self.path
        self.num_epoch = 1
        self.batch_size = 50
        self.log_step = 5
        self.learning_rate=5e-4
        self.global_step=tf.Variable(0, trainable=False)
        self.num_training=10
        #self.learning_step = tf.train.exponential_decay(self.learning_rate, self.global_step,500, 0.96)
        self._build_model()
        

    def _model(self):
        print('intput layer: ' + str(self.X.get_shape()))
        '''
        self.conv1 = slim.conv2d(self.X, 32, [5, 5], scope='conv1')
        self.pool1 = slim.max_pool2d(self.conv1, [2, 2], scope='pool1')
        self.conv2 = slim.conv2d(self.pool1, 64, [5, 5], scope='conv2')
        self.pool2 = slim.max_pool2d(self.conv2, [2, 2], scope='pool2')
        
        net = slim.flatten(net, scope='flatten3')
        '''
        with tf.variable_scope('conv1'):
            self.conv1 = conv2d(self.X, 5, 1, 32)
            print self.conv1.shape
            self.relu1 = tf.nn.relu(self.conv1)
            self.pool1 = max_pool(self.relu1, 2, 2)            
            print('conv1 layer: ' + str(self.pool1.get_shape()))

        with tf.variable_scope('conv2'):
            
            self.conv2 = conv2d(self.pool1, 5, 1, 64)
            print self.conv2.shape
            self.relu2 = tf.nn.relu(self.conv2)
            self.pool2 = max_pool(self.relu2, 3, 2)            
            print('conv2 layer: ' + str(self.pool2.get_shape()))
        
        self.flat = tf.contrib.layers.flatten(self.pool2)

        with tf.variable_scope('fc1'):
            self.fc1 = fc(self.flat,512,tf.nn.relu,'fc1')
            print('fc1 layer: ' + str(self.fc1.get_shape()))
            
        with tf.variable_scope('fc1'):
            self.fc2 = fc(self.fc1,10,tf.nn.relu,'fc2')
            print('fc2 layer: ' + str(self.fc2.get_shape()))

        #self.sm = tf.nn.softmax(self.fc1)
        
        # Return the last layer
        return self.fc1

    def _input_ops(self):
        # Placeholders
        self.X = tf.placeholder(tf.float32, [None, 28,28,1])
        self.Y = tf.placeholder(tf.int64, [None])

        self.is_train = None
        self.keep_prob = None

    def _build_optimizer(self):
        # Adam optimizer 'self.train_op' that minimizes 'self.loss_op'
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss_op, global_step=self.global_step)    
        
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
        X_Train=np.reshape(X_train, [-1,28,28,1])
        X_val=np.reshape(X_val,[-1,28,28,1])
        
        saver = tf.train.Saver()
        path = Path("./models/MNIST/central/")
        if path.exists():
            saver.restore(sess, "./models/MNIST/central/model.ckpt")
            print("success")
        else:
            sess.run(tf.global_variables_initializer())

        step = 0
        losses = []
        accuracies = []
        print('-' * 5 + '  Start training  ' + '-' * 5)
        for epoch in range(self.num_epoch):
            print('train for epoch %d' % epoch)
            for i in range(self.num_training // self.batch_size):
                X_ = X_train[i * self.batch_size:(i + 1) * self.batch_size][:]
                Y_ = Y_train[i * self.batch_size:(i + 1) * self.batch_size]

                feed_dict = {self.X: X_, self.Y : Y_}
             
                fetches = [self.train_op, self.loss_op, self.accuracy_op]

                _, loss, accuracy = sess.run(fetches, feed_dict=feed_dict)
                losses.append(loss)
                accuracies.append(accuracy)
                if step % self.log_step == 0:
                    print('iteration (%d): loss = %.3f, accuracy = %.3f' %
                        (step, loss, accuracy))
                step += 1
            plot=False    
            if (plot):
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

            # Print validation results
            print('validation for epoch %d' % epoch)
            val_accuracy = self.evaluate(sess, X_val, Y_val)
            print('-  epoch %d: validation accuracy = %.3f' % (epoch, val_accuracy))
        saver = tf.train.Saver()
        saver.save(sess, self.path)
        weights = []#np.empty([len(tf.trainable_variables())])    
        i=0
        print tf.trainable_variables()
        for t in tf.trainable_variables():
                weights.append(t.eval(session=sess))
        return weights    

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
    
    
    
'''    
    
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

model = imageModelSmol()
model.train(sess, X_train, Y_train, X_val, Y_val)
accuracy = model.evaluate(sess, X_test, Y_test)
print('***** test accuracy: %.3f' % accuracy)

# Save your model
saver = tf.train.Saver()
model_path = saver.save(sess, "lib/tf_models/problem2/csci-599_mine.ckpt")
print("Model saved in %s" % model_path)

sess.close()
'''
