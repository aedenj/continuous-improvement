"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

from sklearn.preprocessing import *
import pandas as pd
import numpy as np
############

NB = 3 # number of bins
x = np.array([1.0,12.0,6.0,2.0,15.0,3.0,5.0,9.0,8.0,8.0,2.0,5.0,7.0,3.0,6.0,20.0])
X = pd.DataFrame(x)
############

# one way of obtaining the boundaries of the bins
#freq, bounds = np.histogram(x, NB)
 
# more straight-forward way for obtaining the boundaries of the bins
bounds = np.linspace(np.min(x), np.max(x), NB + 1) 
############

def bin(x, b): # x = data array, b = boundaries array
    nb = len(b)
    N = len(x)
    y = np.empty(N, int) # empty integer array to store the bin numbers (output)
    
    for i in range(1, nb): # repeat for each pair of bin boundaries
        y[(x >= b[i-1])&(x < b[i])] = i
    
    y[x == b[-1]] = nb - 1 # ensure that the borderline cases are also binned appropriately
    return y
#############
    
""" NORMALIZING """
minmax_scale = MinMaxScaler().fit(X)
standardization_scale = StandardScaler().fit(X)
y = minmax_scale.transform(X)
z = standardization_scale.transform(X)
print ("\nScaled variable x using MinMax and Standardized scaling\n")
print (np.hstack((np.reshape(x, (16,1)), y, z)))

#############

# Normalizing using numpy
minmaxscaled =(x - min(x))/(max(x) - min(x))
zscaled = (x - np.mean(x))/np.std(x)
#Compare to the stack above
print ("\nScaled variable x using numpy calculations\n")
print(np.hstack(
        (np.reshape(x,(16,1)),
         np.reshape(minmaxscaled,(16,1)),
         np.reshape(zscaled, (16,1)))
          ))

#############

""" BINNING """
bx = bin(x, bounds)
print ("\n\nBinned variable x, for ", NB, "bins\n")
print ("Bin boundaries: ", bounds)
print ("Binned variable: ", bx)

#############

# Equal-width Binning using numpy
NumberOfBins = 3
BinWidth = (max(x) - min(x))/NumberOfBins
MaxBin2 = min(x) + 2 * BinWidth
MaxBin3 = min(x) + 3 * BinWidth
MaxBin4 = min(x) + 4 * BinWidth

print("Bin 1 ends at",BinWidth)
print("Bin 2 ends at",MaxBin2)
print("Bin 3 ends at",MaxBin3)

# Equal-frequency Binning
BinCount=len(x)/NumberOfBins
print("Each Bin contains",BinCount,"elements.")

##############