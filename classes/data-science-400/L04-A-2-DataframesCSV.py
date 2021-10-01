"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

import pandas as pd

# Origin of data:
# https://archive.ics.uci.edu/ml/datasets/car+evaluation

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
################

# download data
Cars = pd.read_csv(url)
Cars.head()
#################

# Inform read_csv that the data contain no column headers
Cars = pd.read_csv(url, header=None)
Cars.head()
##################

# Add headers from https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.c45-names
Cars.columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "evaluation"]
Cars.head()

###################

# Write a local copy of the file. index=False does not create a new column for the indices
Cars.to_csv('cars.csv', sep=",", index=False)

# Where is the file located?
import os
os.getcwd()
os.listdir()

# Check the file displays the same dataframe as before
Cars2=pd.read_csv('cars.csv')
Cars2.head()

###################