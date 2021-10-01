"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

# import package
import pandas as pd

# Download the data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data"
Mamm = pd.read_csv(url, header=None)
Mamm.columns = ["BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"]
##############

# Check the first rows of the data frame
Mamm.head()
##############
# Check the number of rows and columns
Mamm.shape
##############

# Replace >Question Marks< with NaNs
Mamm = Mamm.replace(to_replace="?", value=float("NaN"))

# Check the first rows of the data frame
Mamm.head()

# Check the number of rows and columns
Mamm.shape
##############

# Count NaNs
Mamm.isnull().sum()
##############

# Remove rows that contain one or more NaN
Mamm_FewerRows = Mamm.dropna(axis=0)

# Check the first rows of the data frame
Mamm_FewerRows.head()

# Check the number of rows and columns
Mamm_FewerRows.shape
##############

# Remove columns that contain one or more NaN
Mamm_FewerCols = Mamm.dropna(axis=1)

# Check the first rows of the data frame
Mamm_FewerCols.head()

# Check the number of rows and columns
Mamm_FewerCols.shape

###############
