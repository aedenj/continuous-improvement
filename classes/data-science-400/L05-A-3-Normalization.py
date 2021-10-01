"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

# L05-A-3-Normalization.py

# Normalization continued
# What is the effect of compounding linear normalizations?

import numpy as np
import matplotlib.pyplot as plt
############

# Variable
sigma = 1
mua = 3
mub = 7
x = np.array(15)
x = np.append(x, mua + sigma*np.random.randn(100))
x = np.append(x, mub + sigma*np.random.randn(50))

# Z-Normalize the variable
xNormZ = (x - np.mean(x))/np.std(x)

# Min-Max-Normalize the variable
xNormMinMax = (x - np.min(x))/(np.max(x) - np.min(x))

plt.hist(x, bins = 20, color=[0, 0, 0, 1])
plt.title("Original Variable")
plt.show()

plt.hist(xNormZ, bins = 20, color=[1, 1, 0, 1])
plt.title("Z-normalization")
plt.show()

plt.hist(xNormMinMax, bins = 20, color=[0, 0, 1, 1])
plt.title("min-max-normalization")
plt.show()

# Compare the original variable, the Z-normalized variable and
# the Min-Max-Normalized variable amongst each other!
# Compare the values of the histograms' x-coordinate!

############

# Compound the normalizations:

# 1st min-max normalization then Z-normalization 
xNormMinMaxZ = (xNormMinMax - np.mean(xNormMinMax))/np.std(xNormMinMax)

# 1st Z-normalization then Z-normalization 
xNormZZ = (xNormZ - np.mean(xNormZ))/np.std(xNormZ)

# 1st Z-normalization then min-max normalization
xNormZMinMax = (xNormZ - np.min(xNormZ))/(np.max(xNormZ) - np.min(xNormZ))

# 1st min-max normalization then min-max normalization
xNormMinMaxMinMax = (xNormMinMax - np.min(xNormMinMax))/(np.max(xNormMinMax) - np.min(xNormMinMax))

plt.hist(xNormMinMaxZ, bins = 20, color=[1, 1, 0, 1])
plt.title("1st Min-Max then Z-normalization")
plt.show()

plt.hist(xNormZZ, bins = 20, color=[1, 1, 0, 1])
plt.title("1st Z then Z-normalization")
plt.show()

plt.hist(xNormZMinMax, bins = 20, color=[0, 0, 1, 1])
plt.title("1st Z then Min-Max-normalization")
plt.show()

plt.hist(xNormMinMaxMinMax, bins = 20, color=[0, 0, 1, 1])
plt.title("1st Min-Max then Min-Max-normalization")
plt.show()

# Compare the above 4 variables of compounded normalizations, namely
#      (1) 1st Min-Max then Z-normalization
#      (2) 1st Z then Z-normalization
#      (3) 1st Z then Min-Max-normalization
#      (4) 1st Min-Max then Z-normalization
# Which of the above four variables are are the same as the
# initial Z-normalized variable?
# Which of the above four variables are are the same as the
# initial Min-Max-normalized variable?

# What is the effect of compounding linear normalizations?

############

# Now it's Your Turn to go further
# A log "normalization" is not linear
# Compound a log "normalization" with linear Normalizations
# Use this log normalization:
xlog = np.log(x[x > 0])
# Compare the 5 distributions, namely
#      Log "Normalization"
#      Min-Max Normalization
#      Z-Normalization
#      1st log "Normalization" then Min-Max Normalization
#      1st log "Normalization" then Z-Normalization
