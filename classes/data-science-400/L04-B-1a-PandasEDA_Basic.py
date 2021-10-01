"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

# Basic EDA of pandas

# Import library for pandas data frame
import pandas as pd
####################

# Read in data as a pandas data frame
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
Auto = pd.read_csv(url,sep='\s+', header=None)
# Find proper column names here:
# https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.names
Auto.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 
               'acceleration', 'model year', 'origin', 'car name']
####################

# Basic EDA of numeric data

# Get size of data set
Auto.shape
####################

# View first few rows
Auto.head()
####################

# Present values from the column
Auto.loc[:,'mpg']
####################

# Present sorted values
Auto.loc[:,'mpg'].sort_values()
####################

# Make Histogram of miles per gallon
import matplotlib.pyplot as plt
plt.hist(Auto.loc[:,'mpg'])
####################

# Your turn:  Make histograms of the other columns in this dataset
