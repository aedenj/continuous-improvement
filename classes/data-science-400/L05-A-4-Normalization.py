"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

# L06-A-4-Normalization.py

# Normalization continued
# De-normalize linear normalizations

import numpy as np
import matplotlib.pyplot as plt

# Variable
sigma = 1
mua = 3
mub = 7
x = np.array(15)
x = np.append(x, mua + sigma*np.random.randn(100))
x = np.append(x, mub + sigma*np.random.randn(50))

# xNorm = (x - offset)/spread
# Where
#   x is a numeric variable
#   offset is a scalar that shifts variable x lower or higher
#   spread is a scalar that re-scales variable x to a smaller or larger spread
#   xNorm is the normalized variable

# Z-Normalize the variable
offset = np.mean(x)
spread = np.std(x)
xNorm = (x - offset)/spread

plt.hist(x, bins = 20, color=[0, 0, 0, 1])
plt.title("Original Distribution")
plt.show()

plt.hist(xNorm, bins = 20, color=[1, 1, 0, 1])
plt.title("Z-normalization")
plt.show()

# Get Denormalization with some algebra:
# x = xNorm*spread + offset

# Incorrect Denormilaztion
offset = np.mean(xNorm)
spread = np.std(xNorm)
x_wrong = xNorm*np.std(xNorm) + np.mean(xNorm)

plt.hist(x_wrong, bins = 20, color=[1, 1, 0, 1])
plt.title("Incorrectly De-normalized")
plt.show()

# Why is above denormalization attempt incorrect?
# Is the denormalized variable the same as the normalized variable?
# Compare the values of the histograms' x-coordinates
# What are the values of the spread and offset?

# Correct Denormilaztion
offset = np.mean(x)
spread = np.std(x)
x_correct = xNorm*spread + offset

plt.hist(x_correct, bins = 20, color=[0, 0, 0, 1])
plt.title("Correctly De-normalized")
plt.show()

# Is the denormalized variable the same as the normalized variable?
# Compare the values of the histograms' x-coordinates
