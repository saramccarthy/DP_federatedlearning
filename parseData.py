'''
Created on Nov 24, 2017

@author: Sara
'''

import pickle
import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
#put the files you want to plot here

filename = "nowa_file"
filename2 = "owa_file"
print filename
old = []            
if os.path.exists(filename):
    with open(filename,'rb') as rfp: 
        old = pickle.load(rfp)
        
if os.path.exists(filename2):
    with open(filename2,'rb') as rfp: 
        old2 = pickle.load(rfp)
        
num_clients = 10
#old[0]=old[0][:2440]
#old[1]=old[1][:2440]
#old[2]=old[2][:2440]

rounds = len(old[0])
rounds2 = len(old2[0])

print rounds, rounds
ave_old = []
x=np.linspace(0,rounds/num_clients,rounds/num_clients)
x2=np.linspace(0,rounds2/num_clients,rounds2/num_clients)

def groupedAvg(myArray, N):
    result = np.cumsum(myArray, 0)[N-1::N]/float(N)
    result[1:] = result[1:] - result[:-1]
    return result

av_ac = groupedAvg(old[0], num_clients)
av_acp = groupedAvg(old[1], num_clients)
av_l = groupedAvg(old[2], num_clients)

av_ac2 = groupedAvg(old2[0], num_clients)
av_acp2 = groupedAvg(old2[1], num_clients)
av_l2 = groupedAvg(old2[2], num_clients)


owa = plt.plot(x, av_acp, color="blue", label="average")
nowa = plt.plot(x2, av_ac2, color="red", label="owa")
plt.legend()
plt.ylabel('Test Accuracy')
plt.xlabel('Communication Rounds')
plt.show()

plt.plot(x, av_acp)
plt.plot(x, av_acp2)
plt.ylabel('Test Accuracy')
plt.xlabel('Communication Rounds')
plt.show()


#plt.plot(x, old[1])
#plt.show()

plt.plot(x, av_l)
plt.plot(x, av_l2)
plt.ylabel('Test Loss')
plt.xlabel('Communication Rounds')
plt.show()

