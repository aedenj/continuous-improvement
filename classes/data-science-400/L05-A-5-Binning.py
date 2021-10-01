"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

# L05-A-5-Binning.py

# Binning or Data Binning
# Binning is important to convert numerical data into categories (strings)
# when numerical data is not desired.
# Numeric values are assigned a categorgical label
# Similar numeric values are assigned the same label
# Numeric values are similar if they fall within the same bin boundaries

# Equal-width Binning
# Equal-width binning is the standard or typical binning.
# Each bin width is the same size.  The bin width is the difference between 
# the max allowed value and the min allowed value of a bin.

import numpy as np
############

# Variable
x = np.array((81,3,3,4,4,5,5,5,5,5,5,5,5,5,5,6,6,7,7,7,7,8,8,9,12,24,24,25))
# We want a categorical array instead of the above numeric variable
# The categories in the categorical variable should be "Low", "Med", and "High" 
############

# Equal-width Binning
# Determine the boundaries of the bins
NumberOfBins = 3
BinWidth = (max(x) - min(x))/NumberOfBins
MinBin1 = float('-inf')
MaxBin1 = min(x) + 1 * BinWidth
MaxBin2 = min(x) + 2 * BinWidth
MaxBin3 = float('inf')

print(" Bin 1 is greater than", MinBin1, "up to", MaxBin1)
print(" Bin 2 is greater than", MaxBin1, "up to", MaxBin2)
print(" Bin 3 is greater than", MaxBin2, "up to", MaxBin3)

# Create the categorical variable
# Start with an empty array that is the same size as x
xBinnedEqW = np.empty(len(x), object) # np.full(len(x), "    ")

# The conditions at the boundaries should consider the difference 
# between less than (<) and less than or equal (<=) 
# and greater than (>) and greater than or equal (>=)
xBinnedEqW[(x > MinBin1) & (x <= MaxBin1)] = "Low"
xBinnedEqW[(x > MaxBin1) & (x <= MaxBin2)] = "Med"
xBinnedEqW[(x > MaxBin2) & (x <= MaxBin3)] = "High"
print(" x binned into 3 equal-width bins:", xBinnedEqW)
############

# Equal-frequency Binning

# Equal-frequency binning is less common than standard or typical binning.
# In equal-frequency binning, each bin has approximately the same number of
# items. The bin widths usually differ from bin to bin. 

ApproxBinCount = np.round(len(x)/NumberOfBins)
print(" Each bin should contain approximately", ApproxBinCount, "elements.")

# Set bin bounds so that each has bin has approximately the same number
# of items.
# Find boundaries that divide the sorted variable into bins of approximately
# the right number of elements.
print(" Sort the variable:", np.sort(x))

# One solution would be bins with 4, 12, and 12 elements:
# 3,3,4,4,| 5,5,5,5,5,5,5,5,5,5,6,6,| 7,7,7,7,8,8,9,12,24,24,25,81
MinBin1 = float('-inf')
MaxBin1 = 4.5
MaxBin2 = 6.5
MaxBin3 = float('+inf')

# A better solution would be bins with 14, 6, and 8 elements:
# 3,3,4,4,5,5,5,5,5,5,5,5,5,5,| 6,6,7,7,7,7,| 8,8,9,12,24,24,25,81
MinBin1 = float('-inf')
MaxBin1 = 5.5
MaxBin2 = 7.5
MaxBin3 = float('+inf')

# Create the categorical variable
# Start with an empty array that is the same size as x
xBinnedEqF = np.empty(len(x), object) # np.full(len(x), "    ")
xBinnedEqF[(MinBin1 < x) & (x <= MaxBin1)] = "Low"
xBinnedEqF[(MaxBin1 < x) & (x <= MaxBin2)] = "Med"
xBinnedEqF[(MaxBin2 < x) & (x  < MaxBin3)] = "High"

print(" x binned into 3 equal-freq1uency bins: ")
print(xBinnedEqF)

# Now it's Your Turn to go further
# Define a function called EqualFreqBin that uses percentile to bin an array
# into equal frequency bins
percentiles = np.linspace(0, 100, NumberOfBins + 1)
bounds = np.percentile(x, percentiles)
