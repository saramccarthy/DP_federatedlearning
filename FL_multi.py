'''[
Created on Nov 5, 2017

@author: Sara
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from LoadData import load_test_data, load_train_data, partitionCifarData, img_per_client
import os, shutil
import  cifar10NN.cifar10_eval as ev
import cifar10NN.cifar10_client as cf
import cifar10NN.cifar10_OWA_train as owa
import random as rand
from multiprocessing.pool import Pool
from contextlib import closing



re_partition_data=False

data_dir = 'data/cifar10-batches-bin/'     #'./data/cifar10-batches-bin/cifar10-batches-bin'
client_dir = 'data/clients/cifar/'
filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]
model_dir = "models/cifar/"
model_name = "model.ckpt-11.meta"
central_model_dir = "models/central/model.ckpt"

def unwrap_self_f(arg, **kwarg):
    return FL.train_one_client(*arg, **kwarg)

class FL(object):
    
    def __init__(self, model):
        self.client_model = model
        self.central_weights=[]
        self.del_old_checkpoints=False
        
    def partition_data(self, m):
        partitionCifarData(filenames, client_dir, m)

    def create_clients(self, m):
        self.m=m
        self.client_list = []
        for i in range(m):
            c = self.client_model(i)
            self.client_list.append(c)
        
        
    def train_one_client(self, client):
            print("training client %d"%i)
            client.train() 
            vars = cf.weights[:]
            cf.weights = []
            self.weights.append(vars)
            #self.evaluate_model(client)
            print("DONE")
            
    def train_clients(self):
        self.weights = []
        i=0
        
        #for client in self.client_list:
        #    print("training client %d"%i)
        #    client.train() 
        #    vars = cf.weights[:]
        #    cf.weights = []
        #    self.weights.append(vars)
        #    #self.evaluate_model(client)
        #    print("DONE")
        #    i+=1
        with closing(Pool(processes=2)) as p:
            p.map(unwrap_self_f, zip([self]*len(self.client_list),self.client_list))
    
    def optimize_weights(self):
        print(self.weights[0],self.weights[1])
        owa.train(self.weights, self.m)
        self.V = owa.weights
        self.central_weights=[]
        
        n_tensors = len(self.weights[0])
        for i in range(n_tensors):
            weight = []
            j=0
            for client_vars in self.weights:
                if len(weight) == 0: weight = np.multiply(self.V[i][0][j],client_vars[i])
                else: weight = np.add(weight, np.multiply(self.V[i][0][j],client_vars[i]))
                j+=1
            self.central_weights.append(weight)
    
    def weighted_average_weights(self):
        n_tensors = len(self.weights[0])
        self.V=[[1.0/self.m  for i in range(self.m)] for j in range(n_tensors)]
        print self.V
        self.central_weights=[]
        n_tensors = len(self.weights[0])
        for i in range(n_tensors):
            weight = []
            j=0
            for client_vars in self.weights:
                if len(weight) == 0: weight = np.multiply(self.V[i][j],client_vars[i])
                else: weight = np.add(weight, np.multiply(self.V[i][j],client_vars[i]))
                j+=1
            self.central_weights.append(weight)
        print self.central_weights[0][0][0]
    
    def combine_weights(self):
        self.central_weights=[]
        n_tensors = len(self.weights[0])
        for i in range(n_tensors):
            weight = []
            j=0
            for client_vars in self.weights:
                if len(weight) == 0: weight = np.multiply(self.V[i][0][j],client_vars[i])
                else: weight = np.add(weight, np.multiply(self.V[i][0][j],client_vars[i]))
                j+=1
            self.central_weights.append(weight)
    
    def evaluate_model(self, sess):
        #ev.eval_central_model(self.m, True, "./models/cifar/eval/central", 'models/central/')

        for client in self.client_list:
            client.eval(sess)
    def delmodels(self):
        for client in self.client_list:
            if self.del_old_checkpoints:
                folder = client.model_dir
                for the_file in os.listdir(folder):
                    file_path = os.path.join(folder, the_file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(e)              
    def save_weights(self):
        with tf.Graph().as_default():
            with tf.Session() as sess:
                new_saver = tf.train.import_meta_graph(model_dir+model_name)
                new_saver.restore(sess, tf.train.latest_checkpoint(model_dir))
                i=0
                central_vars = tf.trainable_variables()
                for tvar in central_vars:
                    assign_op = tvar.assign(self.central_weights[i])   
                    sess.run(assign_op)
                    i+=1  
                save_path = new_saver.save(sess, central_model_dir)
                self.evaluate_model(sess)
                sess.close()
    def single_pass(self):
        self.create_clients(2)
        self.train_clients()
        self.weighted_average_weights()
        #self.combine_weights()
        self.save_weights()
        
    def f_learn(self, num_clients, iterations, OWA=False):
        self.create_clients(num_clients)
        for i in range(iterations):
            self.train_clients()
            #self.delmodels()
            if OWA : self.optimize_weights()
            else: self.weighted_average_weights()
            #self.combine_weights()
            self.save_weights()
            #self.evaluate_model()
            
            
learn = FL(cf.cifar10_client)
learn.partition_data(5)
learn.f_learn(5, 2000, True)         
