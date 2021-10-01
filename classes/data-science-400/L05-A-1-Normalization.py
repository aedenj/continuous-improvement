"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

# L05-A-0-Normalization.py

# Normalization
# Most normalizations are linear normalizations
# Linear normalization applies this pattern
# xNorm = (x - offset)/spread
# Where
#   x is a numeric variable
#   offset is a scalar that shifts variable x lower or higher
#   spread is a scalar that re-scales variable x to a smaller or larger spread
#   xNorm is the normalized variable

import numpy as np
############

# Normalizing a numpy array

# First we need an array that we can normalize
# We call this array a variable
x = np.array([1,11,5,3,15,3,5,9,7,9,3,5,7,3,5,21])
print(" The original variable:", x)
############

# Trivial Normalization
#   offset is 0
#   spread is 1
# In math a trivial process is one that doesn't change anything
offset = 0
spread = 1
xNormTrivial = (x - offset)/spread
print(" Trivial normalization doesn't change values: ", xNormTrivial)
############

# Min-max or Feature scaling
#   offset is min of x
#   spread is the range of x or the max of x minus the min of x
# The min of a min-max-normalized variable is zero
# The max of a min-max-normalized variable is one
offset = min(x)
spread = max(x) - min(x)
xNormMinMax = (x - offset)/spread
print(" The min-max-normalized variable:", xNormMinMax)
               
# Z-Normalization or Standard Normalization or Standard Scoring
# Standardize features by removing the mean and scaling to unit variance
# The mean of a z-normalized variable is zero
# The standard deviation of a z-normalized variable is one
# Most of the values are between -2 and +2
#   offset is mean of x
#   spread is standard deviation of x
offset = np.mean(x)
spread = np.std(x)
xNormZ = (x - offset)/spread
print(" The Z-normalized variable:", xNormZ)

# Compare the values before after the two normalizations
print ("\nScaled variable x using numpy calculations\n")
print(np.hstack(
        (np.reshape(x,(16,1)),
         np.reshape(xNormMinMax,(16,1)),
         np.reshape(xNormZ, (16,1))
        ))
    )
# What are the largest and smallest values in the min-max-normalized variable?
# What are the largest and smallest values in the z-normalized variable?
# What is the mean in the min-max-normalized variable?
# What is the mean in the z-normalized variable?
# What is the range in the min-max-normalized variable?
# What is the standard deviation in the z-normalized variable?
##############

# Now it's Your Turn to go further
# Linear Normalization using medians
#   offset is median of x
#   spread is the Median absolute deviation of x
