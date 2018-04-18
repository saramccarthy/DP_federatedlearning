'''
Created on Nov 5, 2017

@author: Sara
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import os
import subprocess
from tensorflow.examples.tutorials.mnist import input_data

class DataSet(object):
    def __init__(self):
        
        self.num_training = 49000
        self.num_validation = 50000 - self.num_training
        self.num_test = 10000
    
    def load_MIST(self):
        data_dir = "../data/mnist/"
        num_train = 60000
        num_test = 10000
    
        def load_file(filename, num, shape):
            fd = open(osp.join(data_dir, filename))
            loaded = np.fromfile(file=fd, dtype=np.float32)
            return loaded[num:].reshape(shape).astype(np.float)
    
        train_image = load_file('train-images-idx3-ubyte', 16, (num_train, 28, 28, 1))
        train_label = load_file('train-labels-idx1-ubyte', 8, num_train)
        test_image = load_file('t10k-images-idx3-ubyte', 16, (num_test, 28, 28, 1))
        test_label = load_file('t10k-labels-idx1-ubyte', 8, num_test)
        #return train_image, train_label, test_image, test_label
        self.data = train_image
        self.labels = train_label
        self.data_test = test_image
        self.labels_test = test_label
        
    def load_data(self):   
        data1, labels1 = unpickle("../data/cifar-10-batches-py/data_batch_1")
        data2, labels2 = unpickle("../data/cifar-10-batches-py/data_batch_2")
        data3, labels3 = unpickle("../data/cifar-10-batches-py/data_batch_3")
        data4, labels4 = unpickle("../data/cifar-10-batches-py/data_batch_4")
        data5, labels5 = unpickle("../data/cifar-10-batches-py/data_batch_5")
        
        data = np.concatenate((data1,data2,data3,data4,data5), axis=0)
        self.labels = np.concatenate((labels1,labels2,labels3,labels4,labels5), axis=0)
        self.data = np.transpose(np.reshape(data,(50000,3,32,32)),[0,2,3,1])
        
        data, labels = unpickle("../data/cifar-10-batches-py/test_batch")
        self.data_test =  np.transpose(np.reshape(data,(10000,3,32,32)),[0,2,3,1])
        self.labels_test = np.asarray(labels)
    
    def sample_dataset(self, size):
        train, labels = self.sample_data(self.data,self.labels, size)
        i = int(0.8*size)
        return train[:i], labels[:i], train[i:], labels[i:]
    
    def sample_testset(self, size):
        test, labels = self.sample_data(self.data_test,self.labels_test, size)
        return test, labels

    def sample_data(self, data, labels, size):
        index = range(len(data))
        ri = np.random.shuffle(index)[:size]
        return data[ri], labels[ri]
    

           
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

def load_train_data():
   
    data1, labels1 = unpickle("../data/cifar-10-batches-py/data_batch_1")
    data2, labels2 = unpickle("../data/cifar-10-batches-py/data_batch_2")
    data3, labels3 = unpickle("../data/cifar-10-batches-py/data_batch_3")
    data4, labels4 = unpickle("../data/cifar-10-batches-py/data_batch_4")
    data5, labels5 = unpickle("../data/cifar-10-batches-py/data_batch_5")
    
    data = np.concatenate((data1,data2,data3,data4,data5), axis=0)
    labels = np.concatenate((labels1,labels2,labels3,labels4,labels5), axis=0)
    data = np.transpose(np.reshape(data,(50000,3,32,32)),[0,2,3,1])
    
    return data[:num_training], labels[:num_training], data[num_training:], labels[num_training:] 

def load_test_data():

    data, labels = unpickle("../data/cifar-10-batches-py/test_batch")
    data =  np.transpose(np.reshape(data,(10000,3,32,32)),[0,2,3,1])
    labels = np.asarray(labels)
    return data, labels

def sample_train_data(size_test, size_train):
    train = sample_data(load_train_data(), size_train)
    test = sample_data(load_test_data(), size_test)
    return 
def sample_data(data, size):
    index = range(len(data))
    return data[np.random.shuffle(index)[:size]]
    
def split_data(bins, train, test):
    tl = len(train)
    vl = len(test)


def img_per_client(filename_queue, client_dir, clients, iid=True):
    images = []
    bytes = 3073
    for file in filename_queue:
        f = open(file, 'rb')
        while True:
            piece = f.read(bytes)  
            if not piece:
                break
            images.append(piece)
        f.close()
    np.random.shuffle(images) 
    num_images=len(images)
    images_per_client=num_images//clients
    return images_per_client
            
def partitionCifarData(filename_queue, client_dir, clients, iid=True):
    images = []
    bytes = 3073
    for file in filename_queue:
        f = open(file, 'rb')
        while True:
            piece = f.read(bytes)  
            if not piece:
                break
            images.append(piece)
        f.close()
    np.random.shuffle(images) 
    num_images=len(images)
    images_per_client=num_images//clients
    owa_records = []
    if iid:
        for client in range(clients):
            np.random.shuffle(images)
            client_images = images[:images_per_client]
            contents = b"".join([record for record in client_images])
            filename = os.path.join(client_dir,'cifar_client_%d.bin' % client)
            open(filename, "wb").write(contents) 
        np.random.shuffle(images)
        owa_images = images[:images_per_client]
        contents = b"".join([record for record in owa_images])
        filename = os.path.join(client_dir,'cifar_owa.bin')
        open(filename, "wb").write(contents) 
    else:
        #np.random.shuffle(images)
        i=0
        for client in range(clients):
            client_images = images[i*images_per_client:(i+1)*images_per_client]
            owa_records.extend(client_images[:images_per_client//clients])
            contents = b"".join([record for record in client_images])
            filename = os.path.join(client_dir,'cifar_client_%d.bin' % client)
            open(filename, "wb").write(contents) 
        contents = b"".join([record for record in owa_records])
        filename = os.path.join(client_dir,'cifar_owa.bin')
        open(filename, "wb").write(contents)
    return images_per_client 

def load_dataMNIST(data_dir):
    num_train = 60000
    num_test = 10000

    def load_file(filename, num, shape):
        fd = open(osp.join(data_dir, filename))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        return loaded[num:].reshape(shape).astype(np.float)

    train_image = load_file('train-images-idx3-ubyte', 16, (num_train, 28, 28, 1))
    train_label = load_file('train-labels-idx1-ubyte', 8, num_train)
    test_image = load_file('t10k-images-idx3-ubyte', 16, (num_test, 28, 28, 1))
    test_label = load_file('t10k-labels-idx1-ubyte', 8, num_test)
    return train_image, train_label, test_image, test_label

def partitionMNIST(clients, iid=False):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    train_I, train_L = np.array_split(mnist.train.images, clients), np.array_split(mnist.train.labels, clients)
    val_I, val_L = np.array_split(mnist.train.images, clients), np.array_split(mnist.train.labels, clients)
    
    return train_I, train_L,val_I, val_L


partitionCifarData(["./data/cifar10-batches-bin/test_batch.bin"], './data/clients/cifar/', 10)            