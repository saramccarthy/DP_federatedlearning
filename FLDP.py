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
import cifar10NN.cifar10_client_dp as cfdp
import cifar10NN.cifar10_OWA_train as owa
import random as rand
from multiprocessing.pool import Pool
from contextlib import closing



re_partition_data=False

data_dir = 'data/cifar10-batches-bin/'     #'./data/cifar10-batches-bin/cifar10-batches-bin'
client_dir = 'data/clients/cifar/'
filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]
model_dir = "models/cifar/dp/"
model_name = "model0.ckpt.meta"
central_model_dir = "models/central/"

class FLDP(object):
    
    def __init__(self, model):
        self.client_model = model
        self.central_weights=[]
        self.del_old_checkpoints=False
        self.multi=False    
        self.network_model = "models/cifar100/"

    def partition_data(self, m):
        self.img_per_client = partitionCifarData(filenames, client_dir, m, iid=False)

    def create_clients(self, m):
        self.img_per_client=50000//m
        print(type(self.img_per_client)) 
        self.m=m
        self.client_list = []
        for i in range(m):
            c = self.client_model(i, self.img_per_client)
            self.client_list.append(c)

    def train_clients(self, round):
        self.weights = []
        i=0
        model_dir = central_model_dir
        if round==0:
        	model_dir = self.network_model
        for client in self.client_list:
            print("training client %d"%i)
            client.train(model_dir) 
            vars = cfdp.weights[:]
            cfdp.weights = []
            self.weights.append(vars)
            i+=1
        print("Done training all clients.")

    def optimize_weights(self):
        #print(self.weights[0],self.weights[1])
        owa.weights=[]
        owa.train(self.weights, self.m)
        self.V = owa.weights
        #print("weights",self.V)
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
        #print(self.V)
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
            #print(self.central_weights)
            self.central_weights.append(weight)
    
    def evaluate_model(self, sess):
        cfdp.Evaluate("models/central/","./models/cifar/eval/central")
    
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
                    #print(self.central_weights[i])
                    assign_op = tvar.assign(self.central_weights[i])
                    sess.run(assign_op)
                    i+=1  
                save_path = new_saver.save(sess, central_model_dir+"model.ckpt")
                self.evaluate_model(sess)
                sess.close()
        
    def f_learn(self, num_clients, iterations, OWA=False):
        self.create_clients(num_clients)
        for i in range(iterations):
            self.train_clients(i)
            if OWA : self.optimize_weights()
            else: self.weighted_average_weights()
            self.save_weights()
            

NUM_CLIENTS = 1
NUM_ITERATIONS = 100
USE_OWA = False
learn = FLDP(cfdp.cifar10_client)
learn.partition_data(NUM_CLIENTS)
learn.f_learn(NUM_CLIENTS, NUM_ITERATIONS, USE_OWA)
