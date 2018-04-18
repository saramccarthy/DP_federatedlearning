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
import matplotlib.gridspec as gridspec
import pickle


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
        self.multi=False        

    def partition_data(self, m, iid=False):
        partitionCifarData(filenames, client_dir, m, iid=iid)

    def create_clients(self, m):
        self.m=m
        self.client_list = []
        for i in range(m):
            c = self.client_model(i)
            self.client_list.append(c)
        
        
    def train_one_client(self, client):
        print("training client %d"%client.index)
        client.train() 
        vars = cf.weights[:]
        cf.weights = []
        self.weights.append(vars)
        #self.evaluate_model(client)
        print("DONE")
    def train_clients(self):
        self.weights = []
        i=0
        
        if self.multi:
            with closing(Pool(processes=2)) as p:
                p.map(unwrap_self_f, zip([self]*len(self.client_list),self.client_list))
        else:
            for client in self.client_list:
                print("training client %d"%i)
                client.train() 
                vars = cf.weights[:]
                cf.weights = []
                self.weights.append(vars)
                #print vars

                #self.evaluate_model(client)
                print("DONE")
                i+=1
        #with closing(Pool(processes=2)) as p:
        #    p.map(unwrap_self_f, zip([self]*len(self.client_list),self.client_list))
    
    def optimize_weights(self):
        print(self.weights[0],self.weights[1])
        owa.weights=[]
        owa.train(self.weights, self.m)
        self.V = owa.weights
        self.write("./eval", self.V)
        #self.plotHeatMap(self.V)
        print("weights",self.V)
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
            self.central_weights.append(weight)
    
    def evaluate_model(self, sess):
        #ev.eval_central_model(self.m, True, "./models/cifar/eval/central", 'models/central/')
        ev.evaluate(True, "./models/cifar/eval/central", "models/central/")
        #for client in self.client_list:
        #    client.eval(sess)
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
    
    def write(self, eval_dir, V):
        filename = eval_dir + "weights.pkl"
        old = []            
        if os.path.exists(filename):
            with open(filename,'rb') as rfp: 
                old = pickle.load(rfp)
        old.append([V])
                                                                                                                           
        with open(filename, "wb") as fp:   #Pickling
            pickle.dump(old, fp)
            fp.close()    
    def plotHeatMap(self, sol):
        plt.figure(1)
        x = len(sol)
        fig, ax = plt.subplots(x, 1)
        gs1 = gridspec.GridSpec(x, 1)
        gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes.
        i=0
        for p in ax:
            p.imshow(sol[i][0][np.newaxis,:], interpolation='none', cmap="Reds", extent=[0,self.m,0,0.1])
            p.set_xticks([])
            p.set_xticks(range(0,self.m), [])
            p.set_yticks([])
            p.grid(ls='solid')
            i+=1
        #plt.tight_layout()
        plt.show()       
            
learn = FL(cf.cifar10_client)
learn.partition_data(10, iid=False)
learn.f_learn(10, 2000, True)         
