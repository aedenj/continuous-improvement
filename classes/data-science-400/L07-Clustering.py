"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
################

def normalize(X): # max-min normalizing
    N, m = X.shape
    Y = np.zeros([N, m])
    
    for i in range(m):
        mX = min(X[:,i])
        Y[:,i] = (X[:,i] - mX) / (max(X[:,i]) - mX)
    
    return Y
################
    
def FindAll(x, y): # find all elements in x that are equal to y
    N = len(x)
    z = np.zeros(N, dtype = bool)
    
    for i in range(N):
        if x[i] == y:
            z[i] = True
    
    ind = z * (np.array(range(N)) + np.ones(N, dtype = int))
    ind = ind[ind > 0]
    n = len(ind)    
    return ind - np.ones(n, dtype = int)
################
    
def kmeans(X, k, th):
    if k < 2:
        print('k needs to be at least 2!')
        return
    if (th <= 0.0) or (th >= 1.0):
        print('th values are beyond meaningful bounds')
        return    
    
    N, m = X.shape # dimensions of the dataset
    Y = np.zeros(N, dtype=int) # cluster labels
    C = np.random.uniform(0, 1, [k,m]) # centroids
    d = th + 1.0
    dist_to_centroid = np.zeros(k) # centroid distances
    
    while d > th:
        C_ = deepcopy(C)
        
        for i in range(N): # assign cluster labels to all data points            
            for j in range(k): 
                dist_to_centroid[j] = np.sqrt(sum((X[i,] - C[j,])**2))                
            Y[i] = np.argmin(dist_to_centroid) # assign to most similar cluster            
            
        for j in range(k): # recalculate all the centroids
            ind = FindAll(Y, j) # indexes of data points in cluster j
            n = len(ind)            
            if n > 0: C[j] = sum(X[ind,]) / n
        
        d = np.mean(abs(C - C_)) # how much have the centroids shifted on average?
        
    return Y, C
#################
    
dataset = np.genfromtxt ('https://library.startlearninglabs.uw.edu/DATASCI400/Datasets/iris-values.csv', delimiter=",")
print (dataset[0:9,])
#################

X = normalize(dataset)
print (X[0:9,])
#################

# setting some parameters for k-means and for the plots
a = 0.8
s = 64
k = 2 # this is the k for the k-means clustering
th = 0.0001
cstd = 0.4 # cluster standard deviation
s2 = 100
color = 'blue'
ec = 'white' # edge color
ec2 = 'black' # edge color for second plot
seed = 0 # random seed
#################

plt.scatter(dataset[:,0], dataset[:,1], alpha=a, s=s, c=color, edgecolors=ec)
plt.show()
#################

Y, C = kmeans(X, k, th)
print (C)
print (Y)
#################

plt.scatter(X[:,0], X[:,1], c=Y, alpha=a, s=s, edgecolors=ec)
plt.show()
#################

data = np.genfromtxt ('https://library.startlearninglabs.uw.edu/DATASCI400/Datasets/CFTF-selection.csv', delimiter=",")
Y, C = kmeans(data, 2, th)
print (C)
#################

plt.scatter(data[:,0], data[:,1], c=Y, alpha=a, s=s, edgecolors=ec)
plt.show()
#################

X = normalize(data)
Y, C = kmeans(X, 2, th)
print (C)
#################

plt.scatter(X[:,0], X[:,1], c=Y, alpha=a, s=s, edgecolors=ec)
plt.show()

#################