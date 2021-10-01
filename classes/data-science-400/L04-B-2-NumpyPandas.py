"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

# import packages
import numpy as np
import pandas as pd
###################

# The url for the data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data"

# Download the data
Mamm = pd.read_csv(url, header=None)

# Check the first rows of the data frame
Mamm.head()
####################

# Replace the default column names 
Mamm.columns = ["BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"]

# Check the first rows of the data frame
Mamm.head()
####################

# Check data types of columns
Mamm.dtypes

# Check distinct values 
Mamm.loc[:,"BI-RADS"].unique()
####################

# Convert BI-RADS to numeric data including nans
Mamm.loc[:, "BI-RADS"] = pd.to_numeric(Mamm.loc[:, "BI-RADS"], errors='coerce')

# Check the data types of the columns in the data frame
Mamm.dtypes

# Check distinct values 
Mamm.loc[:,"BI-RADS"].unique()
####################

# Calculation on a column with nan values
np.median(Mamm.loc[:,"BI-RADS"])
####################

# Determine the elements in the "BI-RADS" column that have nan values
HasNan = np.isnan(Mamm.loc[:,"BI-RADS"])

# Determine the median from a column with NANs
np.median(Mamm.loc[~HasNan,"BI-RADS"])
######################

# Determine the median from a column with NANs
np.nanmedian(Mamm.loc[:,"BI-RADS"])
#####################

# The replacement value for NaNs is Median
Median = np.nanmedian(Mamm.loc[:,"BI-RADS"])

# Median imputation of nans
Mamm.loc[HasNan, "BI-RADS"] = Median

#####################