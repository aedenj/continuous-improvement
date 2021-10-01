"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

# Make the pandas package available
import pandas as pd

# Create an empty data frame called Cars
Cars = pd.DataFrame()

# View the data frame
Cars.head()
#################

# Create a column of price categories that has values for the first 4 cars
Cars.loc[:,"buying"]  = ["vhigh", "high", "low", "med"]

# Create a column of number of doors that has values for the first 4 cars
Cars.loc[:,"doors"]  = [2, 2, 4, 4]

# View the data frame
Cars.head()
##################

# Add a fifth row of data
Cars.loc[4]  = ["vhigh", 3]

# View the data frame
Cars.head()
##################

# View the data types of the columns in the data frame
Cars.dtypes

####################