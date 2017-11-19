'''
Created on Nov 5, 2017

@author: Sara
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from cifar10NN import cifar10
from numpy import dtype

num_epoch=10
class ClientCifar10(object):
    def __init__(self):
        self.global_step
    def eval(self.logits, eval_labels):
        # Compute accuracy
        predict = tf.argmax(logits, 1)
        correct = tf.equal(predict, eval_labels)
        return tf.reduce_mean(tf.cast(correct, tf.float32))
    
    def buildModel(self):
        # Placeholders
        self.images = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.labels = tf.placeholder(tf.int64, [None])
        # Build a Graph that computes the logits predictions from the
        # inference model.
        self.logits = cifar10.inference(self.images)
            
        # Calculate loss.
        self.loss = cifar10.loss(self.logits, self.labels)
                
        self.accuracy = eval(self.logits, self.labels)
                
        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        self.train_op = cifar10.train(self.loss, self.global_step)
        
                
    def train( name, X_train, Y_train, X_val, Y_val):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
    
        num_training = len(X_train)
        batch_size = cifar10.FLAGS.batch_size
        cifar10.FLAGS
        step = 0
        losses = []
        accuracies = []
        print('-' * 5 + '  Start training  ' + '-' * 5)
        #Y_train = tf.one_hot(Y_train, 10)
        #Y_val = tf.one_hot(Y_val,10)
    
        for epoch in range(num_epoch):
            print('train for epoch %d' % epoch)
            for i in range(num_training // batch_size):
                X_ = X_train[i * batch_size:(i + 1) * batch_size][:]
                Y_ = Y_train[i * batch_size:(i + 1) * batch_size]
                 
                global_step = tf.contrib.framework.get_or_create_global_step()
    
                # Get images and labels for CIFAR-10.
                # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
                # GPU and resulting in a slow down.
    
                feed_dict = {images: X_, labels : Y_}
                 
                    fetches = [self.train_op, self.loss_op, self.accuracy_op]
    
                    _, loss, accuracy = sess.run(fetches, feed_dict=feed_dict)
                    losses.append(loss)
                # Build a Graph that computes the logits predictions from the
                # inference model.
                logits = cifar10.inference(images)
            
                # Calculate loss.
                loss = cifar10.loss(logits, labels)
                
                accuracy = eval(logits, labels)
                
                # Build a Graph that trains the model with one batch of examples and
                # updates the model parameters.
                train_op = cifar10.train(loss, global_step)
                losses.append(loss)
                accuracies.append(accuracy)
                #if step % self.log_step == 0:
                #    print('iteration (%d): loss = %.3f, accuracy = %.3f' %
                #            (step, loss, accuracy))
                step += 1
    
            # Print validation results
            print('validation for epoch %d' % epoch)
            #val_accuracy = self.evaluate(sess, X_val, Y_val)
            #print('-  epoch %d: validation accuracy = %.3f' % (epoch, val_accuracy))
            saver = tf.train.Saver()
            model_path = saver.save(sess, name+".ckpt")
