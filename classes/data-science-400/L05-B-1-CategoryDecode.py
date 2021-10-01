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
#################

# Check the data types
Mamm.dtypes
#################
# Check the first rows of the data frame
Mamm.head()
#################
# Check the unique values
Mamm.loc[:, "Shape"].unique()
##################

# Decode Shape
Replace = Mamm.loc[:, "Shape"] == "1"
Mamm.loc[Replace, "Shape"] = "round"

Replace = Mamm.loc[:, "Shape"] == "2"
Mamm.loc[Replace, "Shape"] = "oval"

Replace = Mamm.loc[:, "Shape"] == "3"
Mamm.loc[Replace, "Shape"] = "lobular"

Replace = Mamm.loc[:, "Shape"] == "4"
Mamm.loc[Replace, "Shape"] = "irregular"
###################

# Get the counts for each value
Mamm.loc[:,"Shape"].value_counts()
###################

# Specify all the locations that have a missing value
MissingValue = Mamm.loc[:, "Shape"] == "?"

# Impute missing values
Mamm.loc[MissingValue, "Shape"] = "irregular"
###################

# Decode Margin
Replace = Mamm.loc[:, "Margin"] == "1"
Mamm.loc[Replace, "Margin"] = "circumscribed"

Replace = Mamm.loc[:, "Margin"] == "2"
Mamm.loc[Replace, "Margin"] = "microlobulated"

Replace = Mamm.loc[:, "Margin"] == "3"
Mamm.loc[Replace, "Margin"] = "obscured"

Replace = Mamm.loc[:, "Margin"] == "4"
Mamm.loc[Replace, "Margin"] = "ill-defined"

Replace = Mamm.loc[:, "Margin"] == "5"
Mamm.loc[Replace, "Margin"] = "spiculated"
###################

# Get the counts for each value
Mamm.loc[:,"Margin"].value_counts()

# Specify all the locations that have a missing value
MissingValue = Mamm.loc[:, "Margin"] == "?"

# Impute missing values
Mamm.loc[MissingValue, "Margin"] = "circumscribed"
###################

# Get the counts for each value
Mamm.loc[:,"Shape"].value_counts()

# Get the counts for each value
Mamm.loc[:,"Margin"].value_counts()

####################
