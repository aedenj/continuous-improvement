"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

""" Missing Values -- Removal """

# Import NumPy
import numpy as np

# Create an array
x = np.array([2, 1, " ", 1, 99, 1, 5, 3, "?", 1, 4, 3])
################

# Attempt to tally values that are larger than 4
sum(x > 4)

# Find out the data type for x:
type(x)

# Find out the data type for the elements in the array
x.dtype.name
#################

# Do not allow specific texts
FlagGood = (x != "?") & (x != " ")

# Find elements that are numbers
FlagGood = [element.isdigit() for element in x]
##################

# Select only the values that look like numbers
x = x[FlagGood]

x
##################
 
# Attempt to tally values that are larger than 4
sum(x > 4)
##################

# Need to cast the numbers from text (string) to real numeric values
x = x.astype(int)

x
##################

# tally values that are larger than 4
sum(x > 4)

#################