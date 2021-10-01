"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""
""" Remove an outlier """

# Import the Numpy library
import numpy as np
######################

# Create an array with the data
x = np.array([2, 1, 1, 99, 1, 5, 3, 1, 4, 3])
# Look at the variable explorer, how many elements in x?
#######################

# The high limit for acceptable values is the mean plus 2 standard deviations    
LimitHi = np.mean(x) + 2*np.std(x)
# The high limit is the cutoff for good values
LimitHi

# The low limit for acceptable values is the mean plus 2 standard deviations
LimitLo = np.mean(x) - 2*np.std(x)
# The low limit is the cutoff for good values
LimitLo
#######################

# Create Flag for values within limits 
FlagGood = (x >= LimitLo) & (x <= LimitHi)
# What type of variable is FlagGood? Check the Variable explorer.

# present the flag
FlagGood
#######################

# We can present the values of the items within the limits
x[FlagGood]

# Overwrite x with the selected values
x = x[FlagGood]

# present the data set
x
######################