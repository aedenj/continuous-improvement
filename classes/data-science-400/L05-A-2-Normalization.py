"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

# L05-A-2-Normalization.py

# Compare the effect of outliers on Min Max Normalization
# with the effect of outliers on Z-Normalization

import numpy as np
import matplotlib.pyplot as plt

# Variable 1
sigma1 = 1
mu1a = 3
mu1b = 7
x1 = np.array(15)
x1 = np.append(x1, mu1a + sigma1*np.random.randn(100))
x1 = np.append(x1, mu1b + sigma1*np.random.randn(50))

# Variable 2
sigma2 = 0.3
mu2a = 8.9
mu2b = 10.1
x2 = np.array(6.5)
x2 = np.append(x2, mu2a + sigma2*np.random.randn(100))
x2 = np.append(x2, mu2b + sigma2*np.random.randn(50))

# Compare the original variables by overlaying histograms
plt.hist(x1, bins = 20, color=[0, 0, 1, 0.5])
plt.hist(x2, bins = 20, color=[1, 1, 0, 0.5])
plt.title("Compare variables without normalization")
plt.show()
# Are the distributions of the variables significantly different?  
# If so, how are they different?
# Compare the values of the histogram's x-coordinate 

# Change both variables by Min-Max Normalization
x1NormMinMax = (x1 - np.min(x1))/(np.max(x1) - np.min(x1))
x2NormMinMax = (x2 - np.min(x2))/(np.max(x2) - np.min(x2))
# Compare the Min-Max-normalized variables by overlaying histograms
plt.hist(x1NormMinMax, bins = 20, color=[0, 0, 1, 0.5])
plt.hist(x2NormMinMax, bins = 20, color=[1, 1, 0, 0.5])
plt.title("Compare variables after Min-Max Normalization")
plt.show()
# Are the distributions of the variables significantly different?  
# If so, how are they different?
# Compare the values of the histogram's x-coordinate

# Change both variables by Z-Normalization
x1NormZ = (x1 - np.mean(x1))/np.std(x1)
x2NormZ = (x2 - np.mean(x2))/np.std(x2)
# Compare the Z-normalized variables by overlaying histograms
plt.hist(x1NormZ, bins = 20, color=[0, 0, 1, 0.5])
plt.hist(x2NormZ, bins = 20, color=[1, 1, 0, 0.5])
plt.title("Compare variables after Z-normalization")
plt.show()
# Are the distributions of the variables significantly different?  
# If so, how are they different?
# Compare the values of the histogram's x-coordinate
