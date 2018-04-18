'''[
Created on Nov 5, 2017

@author: Sara
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from LoadData import load_test_data, load_train_data, partitionCifarData, img_per_client, partitionMNIST
import os, shutil
from CNNsmol import imageModelSmol as model
import CNNsmolOWA
import random as rand
    
m=2
tI, tL, vI, vL = partitionMNIST(100, iid=False)

client_list = []

for j in range(1):    
    weights = []
    central_weights = []
    for i in range(m):
        print("training client")
        tf.reset_default_graph()
        client = model(i)
        sess = tf.Session()
        vars = client.train(sess, tI[i], tL[i], vI[i], vL[i]) 
        weights.extend([vars])
        sess.close()

    tIfull = [item for sublist in tI for item in sublist]
    tLfull = [item for sublist in tL for item in sublist]
    vIfull = [item for sublist in vI for item in sublist]
    vLfull = [item for sublist in vL for item in sublist]

    print("Optimizing Weights")
   
    tf.reset_default_graph()
    OWA = CNNsmolOWA.imageModelSmol(m, weights)
    sess = tf.Session()
    vars = OWA.train(sess, tIfull, tLfull, vIfull, vLfull)
    #owa.train(weights, m)
    #optweights = owa.weights
    #print(optweights[0])
    K = 1.0/m
    n_tensors = len(weights[0])
    for i in range(n_tensors):
        weight = []
        for client_vars in weights:
            if len(weight) == 0: weight = np.multiply(K,client_vars[i])
            else: weight = np.add(weight, np.multiply(K,client_vars[i]))
        central_weights.append(weight)

    i=0
    with tf.Session() as sess:
        saver = tf.train.Saver()

        #new_saver = tf.train.import_meta_graph("/Users/Sara/Dropbox/Class/NN/Project/models/MNIST/client0/model.ckpt.meta")
        saver.restore(sess, "/Users/Sara/Dropbox/Class/NN/Project/models/MNIST/client0/model.ckpt")
        #saver = tf.train.Saver()
        #saver.restore(sess, "/Users/Sara/Dropbox/Class/NN/Project/models/cifar/tfmodel.ckpt")
    
        central_vars = tf.trainable_variables()
        for tvar in central_vars:
            #print(tvar)
            #print(central_weights[i].shape)
            assign_op = tvar.assign(central_weights[i])   
            sess.run(assign_op)
            i+=1  

        save_path = saver.save(sess, "/Users/Sara/Dropbox/Class/NN/Project/models/MNIST/central/model.ckpt")
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