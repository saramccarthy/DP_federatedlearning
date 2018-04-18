'''[
Created on Nov 5, 2017

@author: Sara
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from LoadData import load_test_data, load_train_data, partitionCifarData, img_per_client
import os, shutil
import cifar10NN.cifar10_client as cf
import cifar10NN.cifar10_OWA_train as owa
import random as rand
    
re_partition_data=False
data_dir = '/Users/Sara/Dropbox/Class/NN/Project/data/cifar10-batches-bin/'#'./data/cifar10-batches-bin/cifar10-batches-bin'
client_dir = './data/clients/cifar/'
filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
               for i in range(1, 6)]

if re_partition_data:
    partitionCifarData(filenames, client_dir, 100)

N = img_per_client(filenames, client_dir, 100)
m = 10
K = (1/N)*(100/m)
client_list = []
for i in range(m):
    c = cf.cifar10_client(i)
    client_list.append(c)

for j in range(1):    
    weights = []
    central_weights = []
    for client in client_list:
        print("training client")
        client.train() 
        vars = cf.weights #[ w*rand.randint(1,100) for w in cf.weights]
        weights.extend([vars])
        #print(vars)
    print("Optimizing Weights")
    #owa.train(weights, m)
    #optweights = owa.weights
    #print(optweights[0])
    
    #K = getClientWeights(weights)
     
    n_tensors = len(weights[0])
    for i in range(n_tensors):
        weight = []
        for client_vars in weights:
            #print client_vars[i].shape
            if len(weight) == 0: weight = np.multiply(K,client_vars[i])
            else: weight = np.add(weight, np.multiply(K,client_vars[i]))
        central_weights.append(weight)
    cf.central_weights=central_weights

    i=0
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph("/Users/Sara/Dropbox/Class/NN/Project/models/cifar/model.ckpt-11.meta")
        new_saver.restore(sess, tf.train.latest_checkpoint("/Users/Sara/Dropbox/Class/NN/Project/models/cifar/"))
        #saver = tf.train.Saver()
        #saver.restore(sess, "/Users/Sara/Dropbox/Class/NN/Project/models/cifar/tfmodel.ckpt")
    
        central_vars = tf.trainable_variables()
        for tvar in central_vars:
            #print(tvar)
            #print(central_weights[i].shape)
            assign_op = tvar.assign(central_weights[i])   
            sess.run(assign_op)
            i+=1  
        
        for client in client_list:
            new_saver = tf.train.import_meta_graph("/Users/Sara/Dropbox/Class/NN/Project/models/cifar/model.ckpt-11.meta")
            new_saver.restore(sess, tf.train.latest_checkpoint("/Users/Sara/Dropbox/Class/NN/Project/models/cifar/"))
            path = client.model_dir+"/model.ckpt"
            print(path)
            folder = client.model_dir
            for the_file in os.listdir(folder):
                file_path = os.path.join(folder, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    #elif os.path.isdir(file_path): shutil.rmtree(file_path)
                except Exception as e:
                    print(e)
            save_path = new_saver.save(sess, path)

        save_path = new_saver.save(sess, "/Users/Sara/Dropbox/Class/NN/Project/models/central/model.ckpt")
        #save_path = new_saver.save(sess, "/Users/Sara/Dropbox/Class/NN/Project/models/cifar/client1/model.ckpt")

'''    



    
print 'done '   
"""    
size=100

data = DataSet()
# Load cifar-10 data
data.load_MIST()
X_train, Y_train, X_val, Y_val = data.sample_dataset(size)
X_test, Y_test = data.sample_testset(size)


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
"""
'''