'''
Created on Nov 24, 2017

@author: Sara
'''

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
eval_dir = "/Users/Sara/Dropbox/Class/NN/Project/models/cifar/eval/"


filename = eval_dir + "centralaccuracy.pkl"
print filename
old = []            
if os.path.exists(filename):
    with open(filename,'rb') as rfp: 
        old = pickle.load(rfp)
num_clients = 10
#old[0]=old[0][:2440]
#old[1]=old[1][:2440]
#old[2]=old[2][:2440]
rounds = len(old[0])
print rounds
ave_old = []
x=np.linspace(0,rounds/num_clients,rounds/num_clients)

def groupedAvg(myArray, N=2):
    result = np.cumsum(myArray, 0)[N-1::N]/float(N)
    result[1:] = result[1:] - result[:-1]
    return result

av_ac = groupedAvg(old[0], num_clients)
av_ac2 = groupedAvg(old[1], num_clients)
av_l = groupedAvg(old[2], num_clients)


plt.plot(x, av_ac)
plt.ylabel('Test Accuracy')
plt.xlabel('Communication Rounds')
plt.show()

plt.plot(x, av_ac2)
plt.ylabel('Test Accuracy')
plt.xlabel('Communication Rounds')
plt.show()


#plt.plot(x, old[1])
#plt.show()

plt.plot(x, av_l)
plt.ylabel('Test Loss')
plt.xlabel('Communication Rounds')
plt.show()
filename = eval_dir + "loss_client0.pkl"
old = []            
if os.path.exists(filename):
    with open(filename,'rb') as rfp: 
        old = pickle.load(rfp)
x=np.linspace(0,10,len(old))
plt.plot(x, old)
plt.show()
